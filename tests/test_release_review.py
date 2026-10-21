"""Regressions from the reviews before the first release: stopping the app, music for the ride,
lengths, running make again, model downloads, timestamps, MJPEG video and camera clocks."""

import json
import threading
import time
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from reelsift.cli import app
from reelsift.project import Trip
from reelsift.tools import run, tool

from .conftest import make_photo, needs_ffmpeg

runner = CliRunner()


def _ride_trip(tmp_path) -> Path:
    """A trip with one 12 s recording and a ride (stopped from 4 to 10 s)."""
    from reelsift import ride
    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    footage.mkdir()
    run([tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=160x120:rate=30:duration=12", "-c:v", "libx264",
         "-metadata", "creation_time=2026-07-04T09:30:00Z", str(footage / "a.mp4")])  # fmt: skip
    root = tmp_path / "trip"
    with Trip.create(root, [footage]) as trip:
        ingest(trip)
        rec = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
        kmh = [40.0] * 3 + [0.0] * 7 + [40.0] * 2
        trip.db.executemany("INSERT INTO travel VALUES (?, ?, ?, NULL)",
                            [(rec, s, v) for s, v in enumerate(kmh)])  # fmt: skip
        trip.db.execute("INSERT OR REPLACE INTO stage_runs VALUES (?, 'travel', ?, 'now')",
                        (rec, ride._version()))  # fmt: skip
    return root


# --- T1: nothing writes to the trip after the review app stopped ---------------------------------


@needs_ffmpeg
def test_a_change_being_saved_when_the_app_stops_finishes_before_the_trip_is_released(
    tmp_path, monkeypatch
):
    import uvicorn

    from reelsift import cli
    from reelsift.web import app as web_app

    root = _ride_trip(tmp_path)
    assert runner.invoke(app, ["ride", str(root), "--speed", "4"]).exit_code == 0
    seen = {}
    real_create_app = web_app.create_app

    def create_app(t, review, port):
        seen["review"] = review
        return real_create_app(t, review, port)

    def stopping_server(*_args, **_kwargs):
        review = seen["review"]

        def slow_change():  # a request thread the server no longer waits for
            with review.lock:
                time.sleep(0.5)
                review.trip.db.execute("UPDATE edits SET approved = 1")
                seen["saved"] = time.monotonic()

        threading.Thread(target=slow_change).start()
        time.sleep(0.05)

    monkeypatch.setattr(web_app, "create_app", create_app)
    monkeypatch.setattr(uvicorn, "run", stopping_server)
    monkeypatch.setattr(cli, "_review_models", lambda t: (None, None))
    result = runner.invoke(app, ["review", str(root), "--no-browser", "--port", "8797"])
    stopped = time.monotonic()
    assert result.exit_code == 0, result.output
    assert "Review app stopped." in result.output
    assert seen["saved"] <= stopped, "the change was saved before the command ended"
    with pytest.raises(Exception, match="closed"):
        seen["review"].trip.db.execute("SELECT 1")


# --- T2: music for the ride alone ------------------------------------------------------------------


def test_music_for_the_ride_alone_or_with_every_length(tmp_path, monkeypatch):
    from reelsift import musicgen

    asked = []
    monkeypatch.setattr(musicgen, "generate_many",
                        lambda t, keys, **_kw: asked.append(list(keys)) or {})  # fmt: skip
    root = tmp_path / "trip"
    (tmp_path / "footage").mkdir()
    assert (
        runner.invoke(app, ["init", str(root), "--footage", str(tmp_path / "footage")]).exit_code
        == 0
    )
    r = runner.invoke(app, ["music", "generate", str(root), "--ride", "--genre", "synthwave"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["music", "generate", str(root), "--all", "--ride"])
    assert r.exit_code == 0, r.output
    assert asked == [[-1], [30, 180, 600, -1]]
    with Trip(root) as trip:
        assert trip.config.music_for(-1).genre == "synthwave"


# --- T3: lengths chosen when the trip is made --------------------------------------------------------


def test_a_trips_lengths_are_chosen_when_it_is_made(tmp_path):
    (tmp_path / "footage").mkdir()
    root = tmp_path / "trip"
    r = runner.invoke(app, ["init", str(root), "--footage", str(tmp_path / "footage"),
                            "--length", "60", "--length", "15"])  # fmt: skip
    assert r.exit_code == 0, r.output
    with Trip(root) as trip:
        assert trip.config.lengths == [15, 60]
    bad = runner.invoke(app, ["init", str(tmp_path / "t2"), "--footage", str(tmp_path / "footage"),
                              "--length", "0"])  # fmt: skip
    assert bad.exit_code == 1


# --- T4: running make again reuses what it made ----------------------------------------------------


@needs_ffmpeg
def test_running_make_again_keeps_the_ride_and_its_music(tmp_path, monkeypatch):
    from reelsift import cli, edits, musicgen, ride, soundtrack
    from reelsift import config as config_mod

    root = _ride_trip(tmp_path)
    generated = []

    def generate(t, keys, given, per_edit, *_a):
        generated.append(list(keys))
        for n in keys:  # as if a track had been made and chosen
            t.config.music_lengths[n] = given.over(t.config.music_lengths.get(n, type(given)()))
            config_mod.write_trip(t.root, t.config)
            track = t.root / "music" / f"{len(generated)}.flac"
            track.parent.mkdir(exist_ok=True)
            track.write_bytes(b"flac")
            asked = json.dumps({"genre": given.genre, "mood": given.mood})
            asset = t.db.execute(
                "INSERT INTO music_assets (path, sha256, duration, sample_rate, channels, length,"
                " analysis, request, created_at) VALUES (?, '', 3, 48000, 2, ?, '{}', ?, 'now')",
                (f"music/{track.name}", n, asked)).lastrowid  # fmt: skip
            st = soundtrack.get(t, edits.latest(t, n).id)
            st.source, st.asset_id = "generated", asset
            soundtrack.put(t, st)

    monkeypatch.setattr(musicgen, "installed", lambda: object())
    monkeypatch.setattr(cli, "_generate_music", generate)
    monkeypatch.setattr(cli, "_render_ride", lambda t, stored, v: Path(f"ride-r{stored.revision}"))
    args = ["make", str(root), "--only-ride", "--ride-speed", "4", "--music", "synthwave"]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    again = runner.invoke(app, args)
    assert again.exit_code == 0, again.output
    assert "Ride unchanged (revision 1)" in again.output and "already plays music" in again.output
    assert generated == [[-1]]
    with Trip(root) as trip:
        assert edits.latest(trip, ride.RIDE).revision == 1
    other = runner.invoke(app, [*args[:-1], "lo-fi"])
    assert other.exit_code == 0, other.output
    assert generated == [[-1], [-1]], "another genre makes new music"


# --- R1: an interrupted model download isn't taken as downloaded -------------------------------------


def test_a_model_counts_as_downloaded_only_when_all_of_it_is(tmp_path):
    from reelsift import models

    repo = tmp_path / "models--org--model"
    sha = "a" * 40
    snap = repo / "snapshots" / sha
    snap.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(sha)
    (repo / "blobs").mkdir()
    (snap / "config.json").write_text("{}")
    files = ["config.json", "model.safetensors"]
    assert not models.is_cached("org/model", files, cache_dir=tmp_path)
    assert not models.is_cached("org/model", cache_dir=tmp_path), "offline: no weights yet"
    (snap / "model.safetensors").write_bytes(b"weights")
    assert models.is_cached("org/model", files, cache_dir=tmp_path)
    assert models.is_cached("org/model", cache_dir=tmp_path)
    (snap / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model.safetensors", "b": "model-2.safetensors"}}'
    )
    assert not models.is_cached("org/model", cache_dir=tmp_path), "a shard is missing"
    (snap / "model-2.safetensors").write_bytes(b"more")
    assert models.is_cached("org/model", cache_dir=tmp_path)
    (repo / "blobs" / "abc.incomplete").write_bytes(b"half")
    assert not models.is_cached("org/model", cache_dir=tmp_path), "a download left half done"


# --- R2, R3: timestamps before zero, MJPEG movies ----------------------------------------------------


def test_a_file_whose_timestamps_start_before_zero_keeps_that_start():
    from reelsift.media import parse_probe

    probe = parse_probe(Path("x.mkv"), {"format": {"start_time": "-1.000000"}, "streams": []})
    assert probe.start == -1


@needs_ffmpeg
def test_an_mjpeg_movie_is_video_and_a_cover_picture_is_not(tmp_path):
    from reelsift.media import _is_thumbnail, probe

    path = tmp_path / "webcam.mkv"
    run([tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=160x90:rate=25:duration=2", "-c:v", "mjpeg", str(path)])  # fmt: skip
    assert probe(path).main_video is not None
    assert _is_thumbnail({"codec_name": "mjpeg", "nb_frames": "1"})
    assert _is_thumbnail({"codec_name": "png"}), "one picture, no duration"
    assert _is_thumbnail({"codec_name": "mjpeg", "disposition": {"attached_pic": 1}})


# --- R4: a camera set to UTC keeps its times ------------------------------------------------------


def test_a_camera_set_to_write_utc_isnt_shifted_by_its_zone(tmp_path):
    from reelsift import config as config_mod
    from reelsift.ingest import ingest
    from reelsift.sync import clock, sync

    footage = tmp_path / "footage"
    make_photo(footage / "cam" / "a.jpg", taken="2026:10:11 10:00:00", offset=None)
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        label = trip.db.execute("SELECT label FROM cameras").fetchone()["label"]
        trip.config.cameras[label] = config_mod.CameraConfig(zone="+02:00", creation_time="utc")
        sync(trip)
        rec = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
        clk = clock(trip, rec)
        assert clk.to_utc(Fraction(0)) == datetime(2026, 10, 11, 10, 0, tzinfo=UTC)
        assert clk.local(Fraction(0)).hour == 12


# --- the AGENTS.md trial ----------------------------------------------------------------------------


@needs_ffmpeg
def test_photos_get_thumbnails(tmp_path):
    from reelsift import proxies
    from reelsift.ingest import ingest
    from reelsift.web.media import thumbnail

    footage = tmp_path / "footage"
    make_photo(footage / "a.jpg", color=(200, 30, 30))
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        proxies.generate(trip)
        rec = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
        path = thumbnail(trip, rec, 1.0)
        assert path is not None and path.stat().st_size > 0


def test_command_output_is_utf8_when_saved_to_a_file():
    import os
    import subprocess
    import sys

    script = "from reelsift.cli import main_options; main_options(); print('a \u2014 b')"
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}  # as when output goes to a file on Windows
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, env=env, check=True)
    assert out.stdout.decode("utf-8").strip() == "a — b"


# --- the bug review ----------------------------------------------------------------------------------


def _track(t, name: str, genre: str) -> int:
    track = t.root / "music" / name
    track.parent.mkdir(exist_ok=True)
    track.write_bytes(b"flac")
    return t.db.execute(
        "INSERT INTO music_assets (path, sha256, duration, sample_rate, channels, length,"
        " analysis, request, created_at) VALUES (?, '', 3, 48000, 2, -1, '{}', ?, 'now')",
        (f"music/{name}", json.dumps({"genre": genre, "mood": None})),
    ).lastrowid


def test_music_made_before_counts_only_if_it_was_asked_for_that_way(tmp_path, monkeypatch):
    """A failed generation saves the new genre as the setting, but the old track still plays."""
    from reelsift import cli, edits, musicgen, ride, soundtrack
    from reelsift.config import MusicSettings

    root = tmp_path / "trip"
    (tmp_path / "footage").mkdir()
    made = runner.invoke(app, ["init", str(root), "--footage", str(tmp_path / "footage")])
    assert made.exit_code == 0
    with Trip(root, write=True) as t:
        stored = edits.save(t, edits.Selection(ride.RIDE, [], 0.0), {"note": "ride"})
        st = soundtrack.get(t, stored.id)
        st.source, st.asset_id = "generated", _track(t, "old.flac", "techno")
        soundtrack.put(t, st)

        def lost(*_a, **_k):
            raise RuntimeError("GPU lost")

        monkeypatch.setattr(musicgen, "generate_many", lost)
        synthwave = MusicSettings(genre="synthwave", mood="sunny")
        with pytest.raises(typer.Exit):
            cli._generate_music(t, [ride.RIDE], synthwave, per_edit=True)
        assert t.config.music_for(ride.RIDE).genre == "synthwave", "the setting was saved"
        assert not cli._has_generated(t, ride.RIDE, synthwave), "but techno is still playing"
        assert cli._has_generated(t, ride.RIDE, MusicSettings(genre="techno"))
        (root / "music" / "old.flac").unlink()
        assert not cli._has_generated(t, ride.RIDE, MusicSettings(genre="techno")), "file gone"


@needs_ffmpeg
def test_widescreen_footage_in_narrow_pixels_keeps_its_shape(tmp_path):
    import numpy as np

    from reelsift.ingest import ingest
    from reelsift.render import Output, render
    from reelsift.select import Item

    footage = tmp_path / "footage"
    footage.mkdir()
    run([tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
         "color=c=red:size=720x576:rate=25:duration=3", "-vf", "setsar=64/45",
         "-c:v", "libx264", "-metadata", "creation_time=2026-07-04T09:30:00Z",
         str(footage / "dv.mp4")])  # fmt: skip
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        rec = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
        out = render(trip, [Item(rec, 0.5, 2.5, -1, 0.0, None, "test")], tmp_path / "r.mp4",
                     Output(320, 180, fps="25"))  # fmt: skip
    raw = run([tool("ffmpeg"), "-v", "error", "-i", str(out), "-frames:v", "1", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-"]).stdout  # fmt: skip
    frame = np.frombuffer(raw, np.uint8).reshape(180, 320, 3)
    red = (frame[:, :, 0] > 150) & (frame[:, :, 1] < 80)
    columns = np.flatnonzero(red.any(axis=0))
    assert columns.min() <= 2 and columns.max() >= 317, "16:9 footage fills a 16:9 frame"


def test_a_render_that_cant_open_the_trip_ends_with_an_error(tmp_path):
    import sqlite3

    from reelsift.web.app import Renders

    class Unreachable:
        root = tmp_path

        def worker(self):
            raise sqlite3.OperationalError("unable to open database file")

    renders = Renders(Unreachable())
    renders.start(30, False, [], 1, "trip", None)
    renders.thread.join(5)
    assert renders.state["running"] is False
    assert "unable to open" in renders.state["error"]
    renders.start(30, False, [], 1, "trip", None)  # and a new render can start
    renders.thread.join(5)


def test_a_search_left_out_stays_left_out_at_zero():
    from reelsift.prefs import Preferences

    p = Preferences(searches=[{"text": "beach", "value": 0}], exclusions=["search:beach"]).clean()
    assert p.exclusions == ["search:beach"] and [x["text"] for x in p.searches] == ["beach"]
    q = Preferences(searches=[{"text": "beach", "value": 0}]).clean()
    assert q.searches == [], "a search at zero that leaves nothing out is dropped"
