"""Music: beats, snapping cuts to them, the final mix, and 9:16 framing."""

import re
import shutil

import numpy as np
import pytest

from reelsift import config as config_mod
from reelsift import music
from reelsift.project import Trip
from reelsift.render import Output, gain_expression, render
from reelsift.review import Review
from reelsift.select import Item, snap_to_beats
from reelsift.tools import run, tool

from .conftest import make_music, needs_ffmpeg
from .test_review import analyzed_root  # noqa: F401  (module fixture reused here)
from .test_select import trip_data


@needs_ffmpeg
def test_beats_of_a_click_track(tmp_path):
    track = music.analyze_file(make_music(tmp_path / "m.wav", bpm=128, seconds=20))
    assert abs(track.bpm - 128) < 3
    truth = np.arange(0.25, 19.9, 60 / 128)
    errors = [np.min(np.abs(truth - b)) for b in track.beats]
    assert np.median(errors) < 0.02
    assert len(track.beats) >= len(truth) - 2


def test_beat_grid_repeats_tracks():
    t = music.Track(None, 10.0, 120, np.arange(0, 10, 0.5))
    grid = music.beat_grid([t], 25)
    assert grid[0] == 0 and 20.0 in grid and grid.max() <= 26


def test_cuts_snap_to_beats_within_limits():
    data = trip_data(n_recordings=1, duration=100.0)
    c = data.candidates
    items = [
        Item(1, 10.0, 12.2, int(c.id[4]), 1, None, "window"),
        Item(1, 30.0, 31.9, int(c.id[14]), 1, None, "window"),
        Item(1, 50.0, 52.0, int(c.id[24]), 1, None, "window", locked=True),
    ]
    beats = np.arange(0, 20, 0.5)
    out = snap_to_beats(items, beats, data)
    assert out[0].end == pytest.approx(12.0)  # 2.2 -> 2.0, a 0.2 s move
    assert out[0].duration + out[1].duration == pytest.approx(4.0)  # second cut on a beat
    assert out[2] == items[2], "locked clips never move"


def test_snapping_never_moves_more_than_the_limit():
    data = trip_data(n_recordings=1, duration=100.0)
    items = [Item(1, 10.0, 12.25, int(data.candidates.id[4]), 1, None, "window")]
    out = snap_to_beats(items, np.array([0.0, 3.0]), data)
    assert out == items


def test_gain_expression_ramps():
    expr = gain_expression([(1.0, 2.0)], 0.1, 0.5)
    assert expr.startswith("0.5+") and "clip(" in expr
    assert gain_expression([], 0.1, 0.5) == "0.5"


@pytest.fixture
def review_with_music(analyzed_root, tmp_path):  # noqa: F811
    root = tmp_path / "trip"
    shutil.copytree(analyzed_root, root, ignore=shutil.ignore_patterns("reelsift.lock"))
    song = make_music(tmp_path / "song.wav", bpm=120, seconds=30)
    with Trip(root, write=True) as trip:
        trip.config.music = [str(song)]
        config_mod.write_trip(root, trip.config)
        yield Review(trip)


@needs_ffmpeg
def test_selection_with_music_lands_on_beats(review_with_music):
    review = review_with_music
    review.select_all()
    beats = music.beat_grid(review.music(), 20)
    for length in review.lengths:
        t = 0.0
        on_beat = 0
        items = review.edit(length).items
        for i in items:
            t += i.duration
            on_beat += np.min(np.abs(beats - t)) < 0.02
        assert on_beat >= len(items) // 2, "most cuts land on beats"


def _loudness(path) -> float:
    out = run([tool("ffmpeg"), "-v", "info", "-nostats", "-i", str(path), "-map", "0:a:0",
               "-af", "ebur128", "-f", "null", "-"], text=True).stderr  # fmt: skip
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", out)[-1])


@needs_ffmpeg
def test_render_with_music_is_normalized(review_with_music, tmp_path):
    review = review_with_music
    edit = review.edit(8)
    out = render(review.trip, edit.items, tmp_path / "m.mp4", Output(320, 180),
                 music=review.music())  # fmt: skip
    assert abs(_loudness(out) - (-16)) < 1.5


@needs_ffmpeg
def test_vertical_crop_position_moves_the_window(review_with_music, tmp_path):
    from dataclasses import replace

    from PIL import Image

    review = review_with_music
    item = next(i for i in review.edit(8).items if review.data.recordings[i.recording_id].width
                > review.data.recordings[i.recording_id].height)  # fmt: skip
    frames = []
    for x in (0.0, 1.0):
        out = render(
            review.trip, [replace(item, crop_x=x)], tmp_path / f"v{x}.mp4", Output(90, 160)
        )
        png = tmp_path / f"v{x}.png"
        run([tool("ffmpeg"), "-v", "error", "-y", "-i", str(out), "-frames:v", "1", str(png)])
        frames.append(np.asarray(Image.open(png).convert("RGB"), float))
    assert frames[0].shape == (160, 90, 3)
    assert np.abs(frames[0] - frames[1]).mean() > 5, "left and right crops show different parts"


@needs_ffmpeg
def test_crop_api_saves_a_revision(review_with_music):
    from fastapi.testclient import TestClient

    from reelsift.web.app import create_app

    trip = review_with_music.trip
    shared = Review(trip.worker(shared=True))
    client = TestClient(create_app(trip, shared, 8765), base_url="http://127.0.0.1:8765")
    item = client.get("/api/edits/8").json()["items"][0]
    r = client.post("/api/edits/8/crop", json={"recording_id": item["recording_id"],
                                                 "start": item["start"], "end": item["end"],
                                                 "crop_x": 0.2})  # fmt: skip
    assert r.status_code == 200
    assert r.json()["items"][0]["crop_x"] == 0.2
    assert client.get("/api/trip").json()["music"][0]["bpm"] in range(110, 131)
    shared.trip.close()
