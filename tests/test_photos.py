"""Photos: EXIF time and orientation, bursts, analysis, rendering and the review app."""

from datetime import UTC, datetime

import pytest

from reelsift import candidates, photos, proxies
from reelsift.analysis import analyze
from reelsift.ingest import ingest
from reelsift.media import probe
from reelsift.project import Trip
from reelsift.render import Output, render
from reelsift.select import Item
from reelsift.sync import clock, sync
from reelsift.timebase import Provenance

from .conftest import make_clip, make_photo, needs_ffmpeg
from .fakes import FakeDescriber, FakeEmbedder, FakeTranscriber


def test_exif_time_with_zone_and_upright_size(tmp_path):
    info = photos.probe(make_photo(tmp_path / "a.jpg", orientation=6, size=(400, 300)))["photo"]
    assert (info["width"], info["height"]) == (300, 400)
    [ev] = photos.time_evidence(info)
    assert ev.provenance == Provenance.METADATA_WITH_ZONE and ev.zone_offset_min == 120
    assert ev.utc_at_zero == datetime(2026, 10, 11, 8, 20, tzinfo=UTC)
    assert photos.camera_name(info) == "Google Pixel 10"


def test_exif_time_without_zone_is_unsure(tmp_path):
    info = photos.probe(make_photo(tmp_path / "a.jpg", offset=None))["photo"]
    [ev] = photos.time_evidence(info)
    assert ev.provenance == Provenance.METADATA_LOCAL_UNKNOWN_ZONE and ev.confidence < 0.5
    assert (
        photos.time_evidence(photos.probe(make_photo(tmp_path / "b.jpg", taken=None))["photo"])
        == []
    )


def test_upright_proxy(tmp_path):
    src = make_photo(tmp_path / "a.jpg", orientation=6, size=(2000, 1000))
    w, h = photos.make_proxy(src, tmp_path / "p.jpg")
    assert (w, h) == (640, 1280)


@pytest.fixture(scope="module")
def photo_trip(tmp_path_factory):
    root = tmp_path_factory.mktemp("photos")
    footage = root / "footage"
    make_clip(footage / "cam" / "walk.mp4", 4.0, creation_time="2026-10-11T08:19:00Z")
    pics = footage / "phone"
    # A burst of three near-identical shots, the middle one sharpest (more stripes).
    for i, sec in enumerate(("10", "11", "12")):
        make_photo(pics / f"PXL_burst_{i}.jpg", taken=f"2026:10:11 10:21:{sec}")
    make_photo(pics / "PXL_lake.jpg", taken="2026:10:11 10:30:00", color=(10, 200, 90))
    make_photo(pics / "PXL_tall.jpg", taken="2026:10:11 10:40:00", orientation=6,
               color=(200, 60, 40))  # fmt: skip
    with Trip.create(root / "trip", [footage]) as trip:
        ingest(trip)
        proxies.generate(trip)
        sync(trip)
        analyze(trip, FakeEmbedder, FakeTranscriber, say=lambda _: None,
                describer=(FakeDescriber.name, FakeDescriber))  # fmt: skip
        candidates.generate(trip)
        yield trip


@needs_ffmpeg
def test_photos_become_recordings_with_proxies_and_time(photo_trip):
    db = photo_trip.db
    rows = db.execute("SELECT * FROM recordings WHERE kind = 'photo'").fetchall()
    assert len(rows) == 5
    labels = {r["label"] for r in db.execute("SELECT label FROM cameras")}
    assert "Google Pixel 10 (phone)" in labels
    for r in rows:
        p = db.execute("SELECT * FROM proxies WHERE recording_id = ?", (r["id"],)).fetchone()
        assert p["path"].endswith(".jpg") and p["upright"] == 1
        clk = clock(photo_trip, r["id"])
        assert clk.zone_offset_min == 120
    tall = next(r for r in rows if r["height"] > r["width"])
    assert tall is not None


@needs_ffmpeg
def test_photos_are_analyzed_and_bursts_collapse(photo_trip):
    db = photo_trip.db
    photo_ids = [r["id"] for r in db.execute("SELECT id FROM recordings WHERE kind = 'photo'")]
    for rid in photo_ids:
        assert (
            db.execute("SELECT count FROM embeddings WHERE recording_id = ?", (rid,)).fetchone()[0]
            == 1
        )
        assert (
            db.execute("SELECT COUNT(*) FROM vlm_blocks WHERE recording_id = ?", (rid,)).fetchone()[
                0
            ]
            == 1
        )
    photo_candidates = db.execute(
        "SELECT COUNT(*) FROM candidates WHERE source = 'photo'"
    ).fetchone()[0]
    assert photo_candidates == 3  # the burst of three counts once


@needs_ffmpeg
def test_photo_renders_as_a_zooming_still(photo_trip, tmp_path):
    db = photo_trip.db
    rid = db.execute(
        "SELECT r.id FROM recordings r JOIN chapters c ON c.recording_id = r.id"
        " JOIN assets a ON a.id = c.asset_id WHERE a.path LIKE '%PXL_tall.jpg'"
    ).fetchone()["id"]
    video = db.execute("SELECT id FROM recordings WHERE kind = 'video'").fetchone()["id"]
    items = [
        Item(video, 0.0, 2.0, -1, 0.0, None, "window"),
        Item(rid, 0.0, 3.0, -1, 0.0, None, "photo"),
    ]
    out = render(photo_trip, items, tmp_path / "e.mp4", Output(640, 360))
    p = probe(out)
    assert (p.main_video.width, p.main_video.height) == (640, 360)
    assert abs(p.main_video.frames - 5 * 25) <= 2
    assert p.audio
