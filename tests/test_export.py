"""Timeline export: FCPXML and OpenTimelineIO."""

import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

from reelsift import candidates, proxies
from reelsift.analysis import analyze
from reelsift.edits import build
from reelsift.export import export, timecode_frames
from reelsift.ingest import ingest
from reelsift.project import Trip
from reelsift.render import Output
from reelsift.sync import sync

from .conftest import make_clip, make_photo, needs_ffmpeg
from .fakes import FakeDescriber, FakeEmbedder, FakeTranscriber


def seconds(value: str) -> Fraction:
    return Fraction(value.removesuffix("s")) if value != "0s" else Fraction(0)


def test_timecode_frames():
    assert timecode_frames("10:48:47:17", Fraction(25)) == ((10 * 60 + 48) * 60 + 47) * 25 + 17
    assert timecode_frames("00:01:00;02", Fraction(30000, 1001)) == 1800  # drop-frame skips 2
    assert timecode_frames("00:10:00;00", Fraction(30000, 1001)) == 17982
    assert timecode_frames("garbage", Fraction(25)) == 0


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("export")
    footage = root / "footage"
    make_clip(footage / "cam" / "a.mp4", 6.0, creation_time="2026-10-11T08:00:00Z",
              timecode="01:00:00:00")  # fmt: skip
    make_clip(footage / "cam" / "b.mp4", 6.0, creation_time="2026-10-11T08:05:00Z")
    make_photo(footage / "phone" / "p.jpg", taken="2026:10:11 10:03:00")
    with Trip.create(root / "trip", [footage]) as trip:
        ingest(trip)
        proxies.generate(trip)
        sync(trip)
        analyze(trip, FakeEmbedder, FakeTranscriber, say=lambda _: None,
                describer=(FakeDescriber.name, FakeDescriber))  # fmt: skip
        candidates.generate(trip)
        [sel] = build(trip, [12])
        paths = export(trip, sel.items, "test edit", root / "out", Output(640, 360))
        yield trip, sel, paths


@needs_ffmpeg
def test_fcpxml_cuts_are_frame_exact_and_contiguous(exported):
    _, sel, paths = exported
    root = ET.parse(paths[0]).getroot()
    assert root.tag == "fcpxml" and root.get("version") == "1.10"
    formats = {f.get("id"): f for f in root.iter("format")}
    assets = {a.get("id"): a for a in root.iter("asset")}
    sequence = next(root.iter("sequence"))
    frame = seconds(formats[sequence.get("format")].get("frameDuration"))
    offset = Fraction(0)
    clips = [e for e in next(root.iter("spine"))]
    assert clips
    for clip in clips:
        assert seconds(clip.get("offset")) == offset, "no gaps or overlaps"
        duration = seconds(clip.get("duration"))
        assert (duration / frame).denominator == 1, "durations are whole frames"
        asset = assets[clip.get("ref")]
        path = Path(unquote(urlparse(asset.find("media-rep").get("src")).path.lstrip("/")))
        assert path.is_file() or Path("/" + str(path)).is_file()
        if clip.tag == "asset-clip":
            start = seconds(clip.get("start"))
            assert seconds(asset.get("start")) <= start
            assert (
                start + duration
                <= seconds(asset.get("start")) + seconds(asset.get("duration")) + frame
            )
        offset += duration
    assert seconds(sequence.get("duration")) == offset
    assert abs(float(offset) - sel.duration) < 0.1 * len(clips)


@needs_ffmpeg
def test_timecode_is_used_for_clip_positions(exported):
    _, _, paths = exported
    root = ET.parse(paths[0]).getroot()
    starts = {a.get("name"): seconds(a.get("start")) for a in root.iter("asset")}
    if "a.mp4" in starts:
        assert starts["a.mp4"] == 3600  # 01:00:00:00
    if "b.mp4" in starts:
        assert starts["b.mp4"] == 0


@needs_ffmpeg
def test_otio_reads_back(exported):
    import opentimelineio as otio

    _, sel, paths = exported
    timeline = otio.adapters.read_from_file(str(paths[1]))
    video, audio = timeline.tracks
    assert len(video) == len(sel.items)
    assert abs(video.duration().to_seconds() - audio.duration().to_seconds()) < 0.05
    for clip in video:
        assert clip.media_reference.target_url.startswith("file:")


@needs_ffmpeg
def test_photo_and_timecode_with_chosen_clips(exported, tmp_path):
    import opentimelineio as otio

    from reelsift.select import Item

    trip, _, _ = exported
    rec = {
        Path(r["path"]).name: r["recording_id"]
        for r in trip.db.execute(
            "SELECT c.recording_id, a.path FROM chapters c JOIN assets a ON a.id = c.asset_id"
        )
    }
    items = [
        Item(rec["a.mp4"], 1.0, 3.0, -1, 0.0, None, "window"),
        Item(rec["p.jpg"], 0.0, 3.0, -1, 0.0, None, "photo"),
    ]
    fcp, ot = export(trip, items, "chosen", tmp_path, Output(640, 360))
    root = ET.parse(fcp).getroot()
    spine = list(next(root.iter("spine")))
    assert [e.tag for e in spine] == ["asset-clip", "video"]
    assert seconds(spine[0].get("start")) == 3601  # timecode 01:00:00:00 plus 1 s in
    assert seconds(spine[1].get("offset")) == 2
    timeline = otio.adapters.read_from_file(str(ot))
    assert [c.name for c in timeline.tracks[0]] == ["a.mp4", "p.jpg"]
    assert timeline.tracks[0][0].source_range.start_time.to_seconds() == pytest.approx(3601)
