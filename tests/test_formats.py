"""Containers, codecs and file types beyond MP4 and MOV, found by running real test sets."""

import re
import subprocess

import pytest

from reelsift.project import Trip
from reelsift.tools import run, tool

from .conftest import needs_ffmpeg

pytestmark = needs_ffmpeg

WHEN = "2026-07-04T09:30:00Z"


def make(path, *codec: str, seconds: float = 2.0, vf: str = "", fmt: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    run([tool("ffmpeg"), "-v", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=25:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         *(["-vf", vf] if vf else []), *codec, "-metadata", f"creation_time={WHEN}",
         *(["-f", fmt] if fmt else []), str(path)])  # fmt: skip
    return path


def test_common_containers_and_codecs_are_read(tmp_path):
    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    make(footage / "a.webm", "-c:v", "libvpx-vp9", "-c:a", "libopus")
    make(footage / "b.avi", "-c:v", "mjpeg", "-c:a", "pcm_s16le")
    make(footage / "c.mpg", "-c:v", "mpeg2video", "-c:a", "mp2")
    make(footage / "d.mts", "-c:v", "libx264", "-c:a", "ac3", fmt="mpegts")
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        summary = ingest(trip)
        assert not summary.failed
        assert trip.db.execute("SELECT COUNT(*) FROM recordings").fetchone()[0] == 4


def test_files_reelsift_cannot_read_are_mentioned_once(tmp_path):
    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    make(footage / "a.mp4", "-c:v", "libx264", "-c:a", "aac")
    for name in ("notes.docx", "track.gpx", "GOPR0001.THM", "other.docx", "raw.cr3"):
        (footage / name).write_bytes(b"x")
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        messages = [r["message"] for r in trip.db.execute("SELECT message FROM issues")]
    skipped = [m for m in messages if m.startswith("Skipped files")]
    assert skipped == ["Skipped files of types Reelsift doesn't read: 1 .cr3, 2 .docx"]


def test_files_without_a_time_get_one_warning_per_camera(tmp_path):
    from reelsift.ingest import ingest
    from reelsift.sync import sync

    from .conftest import make_photo

    footage = tmp_path / "footage"
    for n in range(4):
        make_photo(footage / f"p{n}.jpg", taken=None, color=(n * 50, 20, 30))
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        sync(trip)
        warnings = [r["message"] for r in trip.db.execute(
            "SELECT message FROM issues WHERE severity = 'warning'")]  # fmt: skip
    assert len(warnings) == 1 and warnings[0].startswith("4 recording(s)")


def test_interlaced_video_is_deinterlaced(tmp_path):
    from fractions import Fraction

    from reelsift.ingest import ingest
    from reelsift.media import probe
    from reelsift.render import Output, pieces, render, video_filter
    from reelsift.select import Item

    footage = tmp_path / "footage"
    source = make(footage / "i.mts", "-c:v", "libx264", "-flags", "+ildct+ilme", "-x264opts",
                  "tff=1", "-c:a", "ac3", vf="fps=50,interlace", fmt="mpegts")  # fmt: skip
    assert probe(source).main_video.interlaced
    out = tmp_path / "out.mp4"
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        rec = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
        item = Item(rec, 0.2, 1.8, -1, 0.0, None, "window")
        (piece,) = pieces(trip, item)
        assert piece.interlaced and "bwdif" in video_filter(piece, Output(), Fraction(25))
        render(trip, [item], out, Output(320, 240))
    assert probe(out).main_video.field_order == "progressive"


def test_rendering_an_empty_edit_says_so(tmp_path, capsys):
    from reelsift.cli import _render
    from reelsift.edits import save
    from reelsift.select import Selection

    with Trip.create(tmp_path / "trip", [tmp_path]) as trip:
        save(trip, Selection(30, [], 30.0), {})
        assert _render(trip, [30], vertical=False) == []
    assert "edit is empty" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["clip.MTS", "clip.M2TS", "clip.WebM"])
def test_extensions_are_case_insensitive(tmp_path, name):
    from reelsift.ingest import scan

    footage = tmp_path / "footage"
    make(footage / name, *(["-c:v", "libvpx-vp9", "-c:a", "libopus"] if name.endswith("WebM") else ["-c:v", "libx264", "-c:a", "aac"]),
         fmt="webm" if name.endswith("WebM") else "mpegts")  # fmt: skip
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        videos, _, _ = scan(trip)
    assert [p.name for p in videos] == [name]


def test_a_cut_inside_an_mpeg_ts_file_lands_on_its_frame(tmp_path):
    """MPEG-TS has no index: seeking straight to a cut can land past the keyframe it needs."""
    import numpy as np

    from reelsift.ingest import ingest
    from reelsift.render import Output, render
    from reelsift.select import Item

    footage = tmp_path / "footage"
    source = make(footage / "long_gop.ts", "-c:v", "libx264", "-c:a", "ac3", seconds=6.0,
                  fmt="mpegts")  # fmt: skip
    out = tmp_path / "cut.mp4"
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        rec = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
        render(trip, [Item(rec, 1.0, 2.0, -1, 0.0, None, "window")], out, Output(320, 240))

    def frame(path, n):
        raw = run([tool("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:v:0", "-vf",
                   rf"select=eq(n\,{n}),scale=320:240", "-frames:v", "1", "-f", "rawvideo",
                   "-pix_fmt", "rgb24", "-"]).stdout  # fmt: skip
        return np.frombuffer(raw, np.uint8).astype(float)

    first = frame(out, 0)
    assert first.size, "the cut has frames"
    errors = [float(np.abs(first - frame(source, n)).mean()) for n in (24, 25, 26)]
    assert int(np.argmin(errors)) == 1, errors


def test_cuts_in_an_mpeg_ts_file_with_long_gops_land_on_their_frames(tmp_path):
    """Seeking in MPEG-TS can land on the keyframe after the seek point, ten seconds later
    here; the cut must still start at its own frame, wherever it is in the file."""
    from fractions import Fraction

    import numpy as np

    from reelsift.media import probe
    from reelsift.render import Output, Piece, clip_command

    source = tmp_path / "long_gop.ts"
    run([tool("ffmpeg"), "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25:duration=70",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=70",
         "-c:v", "libx264", "-g", "250", "-keyint_min", "250", "-sc_threshold", "0",
         "-c:a", "ac3", "-f", "mpegts", str(source)])  # fmt: skip
    info = probe(source)

    def frames(path, n=None):
        raw = run([tool("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:v:0",
                   "-vf", "scale=160:120", *(["-frames:v", str(n)] if n else []),
                   "-f", "rawvideo", "-pix_fmt", "gray", "-"]).stdout  # fmt: skip
        return np.frombuffer(raw, np.uint8).reshape(-1, 120, 160).astype(float)

    every = frames(source)
    for cut in (1, 29, 31, 35, 41, 61):
        piece = Piece(source, Fraction(cut), Fraction(1), Fraction(25), False, "bt709", True,
                      160, 120, indexed=False,
                      origin=max(Fraction(0), info.main_video.start - info.start))  # fmt: skip
        clip = tmp_path / f"cut{cut}.mp4"
        run(clip_command(piece, Output(160, 120), Fraction(25), clip, gpu_decode=False, cpu=True))
        first = frames(clip, 1)
        assert len(first), f"the cut at {cut} s has frames"
        nearest = int(np.argmin([np.abs(first[0] - f).mean() for f in every]))
        assert nearest == cut * 25, (cut, nearest)


def test_a_cut_far_from_any_keyframe_seeks_back_far_enough(tmp_path):
    """With a keyframe only every 60 s, seeking 30 s before a cut lands after it; the cut must
    still start at its own frame."""
    from fractions import Fraction

    import numpy as np

    from reelsift.media import probe
    from reelsift.render import Output, Piece, clip_command, seek_margin

    source = tmp_path / "gop60.ts"
    run([tool("ffmpeg"), "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25:duration=90",
         "-c:v", "libx264", "-preset", "ultrafast", "-g", "1500", "-keyint_min", "1500",
         "-sc_threshold", "0", "-an", "-f", "mpegts", str(source)])  # fmt: skip
    info = probe(source)

    def frames(path, n=None):
        raw = run([tool("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:v:0",
                   *(["-frames:v", str(n)] if n else []), "-f", "rawvideo", "-pix_fmt", "gray",
                   "-"]).stdout  # fmt: skip
        return np.frombuffer(raw, np.uint8).reshape(-1, 120, 160).astype(float)

    every = frames(source)
    for cut in (35, 75):
        piece = Piece(source, Fraction(cut), Fraction(1), Fraction(25), False, "bt709", False,
                      160, 120, indexed=False,
                      origin=max(Fraction(0), info.main_video.start - info.start))  # fmt: skip
        clip = tmp_path / f"cut{cut}.mp4"
        run(clip_command(piece, Output(160, 120), Fraction(25), clip, gpu_decode=False,
                         cpu=True, margin=seek_margin(piece)))  # fmt: skip
        first = frames(clip, 1)[0]
        nearest = int(np.argmin([np.abs(first - f).mean() for f in every]))
        assert nearest == cut * 25, (cut, nearest)


def test_proxies_made_before_deinterlacing_are_made_again(tmp_path):
    import json

    from reelsift.ingest import ingest
    from reelsift.proxies import reconcile

    footage = tmp_path / "footage"
    make(footage / "i.mts", "-c:v", "libx264", "-flags", "+ildct+ilme", "-x264opts", "tff=1",
         "-c:a", "ac3", vf="fps=50,interlace", fmt="mpegts")  # fmt: skip
    make(footage / "p.mp4", "-c:v", "libx264", "-c:a", "aac")
    old = json.dumps({"version": 1, "long_side": 1280, "tonemap": False})
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        for rec in [r["id"] for r in trip.db.execute("SELECT id FROM recordings")]:
            proxy = tmp_path / f"proxy{rec}.mp4"
            proxy.write_bytes(b"old proxy")
            trip.db.execute(
                "INSERT OR REPLACE INTO proxies VALUES (?, 0, 'generated', ?, NULL, '0/1', 320,"
                " 240, 1, ?)", (rec, str(proxy), old))  # fmt: skip
            trip.db.execute("INSERT INTO stage_runs VALUES (?, 'visual', 'v', 'now')", (rec,))
        assert reconcile(trip) == 1
        left = [r["path"] for r in trip.db.execute(
            "SELECT a.path FROM proxies p JOIN chapters c ON c.recording_id = p.recording_id"
            " JOIN assets a ON a.id = c.asset_id")]  # fmt: skip
        analysed = trip.db.execute("SELECT COUNT(*) FROM stage_runs").fetchone()[0]
    assert [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in left] == ["p.mp4"]
    assert analysed == 1, "the deinterlaced recording is analysed again"


def _mean_luma(path) -> float:
    out = subprocess.run([tool("ffmpeg"), "-v", "error", "-i", str(path), "-vf",
                          "signalstats,metadata=print:file=-", "-frames:v", "10", "-f", "null",
                          "-"], capture_output=True, text=True).stdout  # fmt: skip
    values = [float(v) for v in re.findall(r"YAVG=([\d.]+)", out)]
    return sum(values) / len(values)


@pytest.mark.parametrize(("colour", "expected"), [("black", 16), ("white", 235)])
def test_full_range_video_and_photos_come_out_in_video_range(tmp_path, colour, expected):
    """Phones record 0-255 levels; the output is 16-235, so they must be converted."""
    from PIL import Image

    from reelsift.ingest import ingest
    from reelsift.render import Output, render
    from reelsift.select import Item

    footage = tmp_path / "footage"
    footage.mkdir()
    run([tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
         f"color=c={colour}:size=320x240:rate=25:duration=2", "-pix_fmt", "yuvj420p",
         "-color_range", "pc", "-c:v", "libx264", "-metadata", f"creation_time={WHEN}",
         str(footage / "phone.mp4")])  # fmt: skip
    Image.new("RGB", (400, 300), colour).save(footage / "photo.jpg")
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        for rec in trip.db.execute("SELECT id, kind FROM recordings").fetchall():
            out = tmp_path / f"{rec['kind']}.mp4"
            render(trip, [Item(rec["id"], 0.0, 1.5, -1, 0.0, None, rec["kind"])], out,
                   Output(320, 240))  # fmt: skip
            assert abs(_mean_luma(out) - expected) <= 3, rec["kind"]
