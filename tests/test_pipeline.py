"""End-to-end: ingest, proxies, sync and timeline on generated clips."""

import json
import shutil
from datetime import UTC, datetime
from fractions import Fraction

import pytest

from reelsift import proxies, timeline
from reelsift.ingest import ingest
from reelsift.media import probe
from reelsift.project import Trip
from reelsift.sync import add_anchor, clock, sync

from .conftest import needs_ffmpeg

pytestmark = needs_ffmpeg


@pytest.fixture
def trip(tmp_path, footage):
    with Trip.create(tmp_path / "trip", [footage]) as t:
        yield t


def issues(trip, severity=None):
    sql = "SELECT severity, message, path FROM issues"
    rows = trip.db.execute(sql).fetchall()
    return [r for r in rows if severity is None or r["severity"] == severity]


def test_ingest_groups_files_and_pairs_matching_proxies(trip):
    s = ingest(trip)
    assert (s.files, s.new, s.recordings) == (4, 4, 4)
    assert s.camera_proxies == 2  # the third .LRF has the wrong length
    cams = {r["label"]: r["device"] for r in trip.db.execute("SELECT label, device FROM cameras")}
    assert cams == {"DJI (DJI_001)": "dji", "Unknown camera (phone)": "generic"}
    warnings = [r["message"] for r in issues(trip, "warning")]
    assert any("duration differs" in m for m in warnings)


def test_rotated_phone_clip_is_recorded_upright(trip):
    ingest(trip)
    row = trip.db.execute(
        "SELECT r.width, r.height, r.rotation FROM recordings r JOIN cameras c ON c.id = r.camera_id"
        " WHERE c.device = 'generic'"
    ).fetchone()
    # -display_rotation 90 is counterclockwise, so the clip must be turned 270 degrees
    assert (row["width"], row["height"], row["rotation"]) == (180, 320, 270)


def test_reingest_is_unchanged_and_moved_files_are_relinked(trip, footage):
    ingest(trip)
    again = ingest(trip)
    assert (again.unchanged, again.new, again.recordings) == (4, 0, 0)
    moved = footage / "moved"
    moved.mkdir()
    shutil.move(footage / "phone" / "PXL_20261011_081000.mp4", moved / "renamed.mp4")
    third = ingest(trip)
    assert (third.relinked, third.new, third.recordings, third.missing) == (1, 0, 0, 0)


def test_missing_files_are_reported(trip, footage):
    ingest(trip)
    (footage / "phone" / "PXL_20261011_081000.mp4").unlink()
    s = ingest(trip)
    assert s.missing == 1
    assert any("no longer" in r["message"] for r in issues(trip, "warning"))


def test_proxies_are_generated_upright_with_offsets(trip):
    ingest(trip)
    assert len(proxies.pending(trip)) == 2  # the phone clip and the DJI clip with a bad .LRF
    result = proxies.generate(trip)
    assert result.ok
    assert proxies.pending(trip) == []
    rows = trip.db.execute("SELECT * FROM proxies WHERE origin = 'generated'").fetchall()
    for row in rows:
        path = proxies.resolve(trip, row["path"])
        made = probe(path).main_video
        assert made.rotation == 0 and row["upright"] == 1
        assert (made.color_primaries, made.color_transfer, made.color_space) == ("bt709",) * 3
        assert max(made.width, made.height) <= proxies.LONG_SIDE
        assert json.loads(row["settings"])["version"] == proxies.SETTINGS_VERSION
        assert Fraction(row["offset"]) == 0
    phone = [r for r in rows if r["height"] > r["width"]]
    assert len(phone) == 1  # rotated source became a portrait proxy


def test_sync_places_recordings_and_anchor_corrects_camera(trip):
    ingest(trip)
    s = sync(trip)
    assert s.recordings == 4 and s.without_time == 0
    rec = trip.db.execute(
        "SELECT c.recording_id FROM chapters c JOIN assets a ON a.id = c.asset_id"
        " WHERE a.path LIKE '%0001_D.MP4'"
    ).fetchone()["recording_id"]
    clk = clock(trip, rec)
    assert clk.zone_offset_min == 120
    assert clk.to_utc(Fraction(0)) == datetime(2026, 10, 11, 7, 51, 34, tzinfo=UTC)

    add_anchor(trip, "DJI_20261011095133_0001_D.MP4", "1", utc="2026-10-11T09:51:40+02:00")
    sync(trip)
    moved = clock(trip, rec)
    assert moved.to_utc(Fraction(1)) == datetime(2026, 10, 11, 7, 51, 40, tzinfo=UTC)
    other = trip.db.execute(
        "SELECT c.recording_id FROM chapters c JOIN assets a ON a.id = c.asset_id"
        " WHERE a.path LIKE '%0002_D.MP4'"
    ).fetchone()["recording_id"]
    assert clock(trip, other).to_utc(Fraction(0)) == datetime(2026, 10, 11, 8, 0, 6, tzinfo=UTC)


def test_unknown_zone_can_be_set_in_config(tmp_path, footage):
    root = tmp_path / "trip"
    with Trip.create(root, [footage]):
        pass
    with (root / "reelsift.toml").open("a") as f:
        f.write('\n[cameras."Unknown camera (phone)"]\nzone = "+02:00"\n')
    with Trip(root, write=True) as t:
        ingest(t)
        sync(t)
        zones = {r["zone_offset_min"] for r in t.db.execute("SELECT zone_offset_min FROM clocks")}
    assert zones == {120}


def test_timeline_page_lists_every_recording(trip):
    ingest(trip)
    sync(trip)
    page = timeline.render(trip)
    assert page.count("<rect") == 4
    assert "UTC+02:00" in page
    assert "Issues" in page


def test_clip_without_time_is_placed_by_modified_time_and_flagged(tmp_path):
    from .conftest import make_clip

    make_clip(tmp_path / "f" / "untimed.mp4", 2.0)
    with Trip.create(tmp_path / "trip", [tmp_path / "f"]) as t:
        ingest(t)
        s = sync(t)
        assert s.by_provenance == {"file_modified": 1}
        assert any("modified time" in r["message"] for r in issues(t, "warning"))
