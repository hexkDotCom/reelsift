"""Rides: sped-up rendering, the clips' sound level, stop finding and building a ride."""

import itertools
from fractions import Fraction

import numpy as np
import pytest

from reelsift.project import Trip
from reelsift.select import Item
from reelsift.tools import run, tool

from .conftest import needs_ffmpeg


def frames_of_file(path, size=(160, 120)):
    w, h = size
    raw = run([tool("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:v:0",
               "-fps_mode", "passthrough", "-vf",
               f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "-"]).stdout  # fmt: skip
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(float)


@pytest.fixture(scope="module")
def numbered(tmp_path_factory):
    """A 20 s, 30 fps clip whose frames are all different, ingested into a trip."""
    from reelsift.ingest import ingest

    root = tmp_path_factory.mktemp("ride")
    footage = root / "footage"
    footage.mkdir()
    run([tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=320x240:rate=30:duration=20", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=20", "-c:v", "libx264", "-g", "30", "-c:a", "aac",
         "-metadata", "creation_time=2026-07-04T09:30:00Z", str(footage / "ride.mp4")])  # fmt: skip
    with Trip.create(root / "trip", [footage]) as trip:
        ingest(trip)
        rec = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
    return root, rec, frames_of_file(footage / "ride.mp4")


@needs_ffmpeg
@pytest.mark.parametrize("speed", [2, 8, 16])
def test_a_sped_up_clip_shows_every_nth_frame_from_its_first(numbered, tmp_path, speed):
    from reelsift.render import Output, render

    root, rec, source = numbered
    item = Item(rec, 1.0, 17.0, -1, 0.0, None, "ride", speed=float(speed))
    with Trip(root / "trip", write=True) as trip:
        out = render(trip, [item], tmp_path / "r.mp4", Output(320, 240, fps="30"))
    got = frames_of_file(out)
    expected = list(range(30, 30 * 17, speed))  # one source frame per output frame
    assert len(got) == len(expected)
    for k in (0, 1, len(got) // 2, len(got) - 1):
        nearest = int(np.argmin([np.abs(got[k] - f).mean() for f in source]))
        assert nearest == expected[k], (speed, k, nearest)


@needs_ffmpeg
def test_sped_up_sound_lasts_as_long_as_the_picture(numbered, tmp_path):
    from reelsift.render import Output, render

    root, rec, _ = numbered
    item = Item(rec, 0.0, 16.0, -1, 0.0, None, "ride", speed=8.0)
    with Trip(root / "trip", write=True) as trip:
        out = render(trip, [item], tmp_path / "s.mp4", Output(320, 240, fps="30"))

    def length(stream):
        return float(run([tool("ffprobe"), "-v", "error", "-select_streams", stream,
                          "-show_entries", "stream=duration", "-of", "csv=p=0", str(out)],
                         text=True).stdout.strip())  # fmt: skip

    assert length("v:0") == pytest.approx(2.0, abs=0.05)
    assert length("a:0") == pytest.approx(2.0, abs=0.1)


def test_tempo_steps_multiply_to_the_speed():
    from reelsift.render import tempo

    steps = [float(s.split("=")[1]) for s in tempo(Fraction(48)).strip(",").split(",")]
    assert all(s <= 2 for s in steps) and np.prod(steps) == pytest.approx(48)
    assert tempo(Fraction(1)) == ""


def test_sped_up_clips_keep_one_frame_grid_across_pieces():
    from reelsift.render import CHUNK_SOURCE_S, Piece, _sample_pieces

    def piece(start, duration):
        return Piece(None, Fraction(start), Fraction(duration), Fraction(30), False, "bt709",
                     True, speed=Fraction(16))  # fmt: skip

    # Two chapters, 7.3 s and 300 s: samples every 16/30 s from the clip's start, all along.
    out = _sample_pieces([piece(10, Fraction(73, 10)), piece(0, 300)], Fraction(16), Fraction(30))
    step = Fraction(16, 30)
    assert sum(p.frames for p in out) == int(np.ceil((Fraction(3073, 10)) / step))
    assert out[0].start == 10 and all(p.duration <= CHUNK_SOURCE_S for p in out)
    second = out[1]  # the second chapter starts at the next sample after 7.3 s
    assert second.start == 14 * step - Fraction(73, 10)


def test_the_clips_sound_can_be_switched_off_completely():
    from reelsift.render import mix_graph
    from reelsift.soundtrack import Mix

    off = Mix(sound=-60.0)
    assert "volume=0.0" in mix_graph(10.0, [(1.0, 2.0)], 0, off)
    with_music = mix_graph(10.0, [(1.0, 2.0)], 1, off)
    assert with_music.split("[src]")[0].count("0.0*(") == 1, "speech can't come through either"
    loud = mix_graph(10.0, [], 1, Mix(music=0.0, sound=-60.0))
    assert "volume='1.0'" in loud


def test_music_can_be_switched_off_completely():
    from reelsift.soundtrack import Mix

    assert Mix(music=-60.0).gains()[0] == 0.0 and Mix(music=-60.0).gains()[2] == 0.0


# --- stops ------------------------------------------------------------------------------------


def test_gps_stops_of_six_seconds_or_more_are_cut_with_a_pad():
    from reelsift import travel

    kmh = np.array([50.0] * 10 + [0.0] * 10 + [40.0] * 5 + [0.0] * 3 + [30.0] * 5)
    states = travel.classify_gps(kmh)
    assert travel.runs(states, travel.STOPPED) == [(11, 19)], "a 10 s stop less 1 s each side"
    assert (states[20:28] == travel.MOVING).all(), "a 3 s pause is part of the ride"


def test_no_gps_fix_is_unknown_never_a_stop():
    from reelsift import travel

    states = travel.classify_gps(np.array([np.nan] * 20 + [0.0] * 20))
    assert (states[:20] == travel.UNKNOWN).all() and travel.STOPPED in states[20:]


def test_the_picture_only_calls_clear_stops():
    from reelsift import travel

    rng = np.random.default_rng(0)
    driving = rng.uniform(0.04, 0.12, 300)
    waiting = rng.uniform(0.001, 0.004, 60)
    values = np.concatenate([driving[:150], waiting, driving[150:]])
    t = travel.thresholds(values)
    states = travel.classify(values, t)
    assert (states[:150] != travel.STOPPED).all() and (states[210:] != travel.STOPPED).all()
    assert (states[165:205] == travel.STOPPED).all(), "the stop, less its uncertain edges"
    assert travel.thresholds(driving) is None, "all driving: nothing is called a stop"


def test_open_road_counts_as_travel_from_its_busiest_tiles():
    from reelsift import travel

    # Most tiles barely move on an open road (sky, far road); the roadside does.
    second = np.array([0.001] * 12 + [0.08, 0.1, 0.12, 0.09])
    tiles = np.tile(second, (30, 1))
    assert travel.travel_values(tiles)[0] > 0.05


# --- building -----------------------------------------------------------------------------------


@pytest.fixture
def ride_trip(numbered):
    """The numbered clip with GPS-like travel: driving, a 6 s stop at 8-14 s, driving."""
    root, rec, _ = numbered
    with Trip(root / "trip", write=True) as trip:
        trip.db.execute("DELETE FROM travel")
        kmh = [40.0] * 7 + [0.0] * 8 + [40.0] * 5
        trip.db.executemany("INSERT INTO travel VALUES (?, ?, ?, NULL)",
                            [(rec, s, v) for s, v in enumerate(kmh)])  # fmt: skip
        trip.db.execute("INSERT OR REPLACE INTO stage_runs VALUES (?, 'travel', ?, 'now')",
                        (rec, __import__("reelsift.ride", fromlist=["x"])._version()))  # fmt: skip
        trip.db.execute("DELETE FROM edits WHERE length = -1")
        yield trip, rec


def test_a_ride_cuts_the_stop_and_speeds_up_the_rest(ride_trip):
    from reelsift import ride

    trip, _ = ride_trip
    items, report = ride.build(trip, ride.RideSettings(speed="4"), say=lambda _: None)
    assert [(i.start, i.end, i.speed) for i in items] == [(0.0, 8.0, 4.0), (14.0, 20.0, 4.0)]
    assert report.stopped == 6 and report.duration == pytest.approx(14 / 4)
    assert report.options()[1] == (2, pytest.approx(7.0))


def test_a_ride_fits_a_length(ride_trip):
    from reelsift import ride

    trip, _ = ride_trip
    items, report = ride.build(trip, ride.RideSettings(target=7.0, speed="fit"),
                               say=lambda _: None)  # fmt: skip
    assert report.speed == pytest.approx(2.0) and sum(i.duration for i in items) == pytest.approx(7)


def test_a_length_the_route_cant_fit_says_what_it_needs(ride_trip):
    from reelsift import ride

    trip, _ = ride_trip
    with pytest.raises(ride.RideError, match="at least"):
        ride.build(trip, ride.RideSettings(target=0.1, speed="fit"), say=lambda _: None)


def test_corrections_drop_stretches_and_keep_stops_at_normal_speed(ride_trip):
    from reelsift import ride

    trip, rec = ride_trip
    settings = ride.RideSettings(speed="4", overrides=[
        {"kind": "keep", "recording_id": rec, "start": 9.0, "end": 12.0},
        {"kind": "drop", "recording_id": rec, "start": 16.0, "end": 20.0},
    ])  # fmt: skip
    items, report = ride.build(trip, settings, say=lambda _: None)
    assert [(i.start, i.end, i.speed) for i in items] == [
        (0.0, 8.0, 4.0),
        (9.0, 12.0, 1.0),
        (14.0, 16.0, 4.0),
    ]
    assert report.kept_stops == 3 and report.dropped == 4


@needs_ffmpeg
def test_a_ride_renders_and_keeps_its_soundtrack_levels(ride_trip, tmp_path):
    from reelsift import ride, soundtrack
    from reelsift.render import Output, render

    trip, _ = ride_trip
    settings = ride.RideSettings(speed="4")
    items, report = ride.build(trip, settings, say=lambda _: None)
    stored = ride.save(trip, items, settings, report, "test")
    st = soundtrack.get(trip, stored.id)
    assert (st.mix.music, st.mix.sound) == (0.0, -20.0), "the ride's own default levels"
    out = render(trip, stored.items, tmp_path / "ride.mp4", Output(320, 240, fps="30"),
                 music=soundtrack.resolve(trip, stored.id))  # fmt: skip
    assert len(frames_of_file(out)) == 105  # (8 + 6) s of footage at 4x, 30 fps


@needs_ffmpeg
def test_a_rides_export_says_how_fast_each_clip_plays(ride_trip, tmp_path):
    import opentimelineio as otio

    from reelsift import ride
    from reelsift.export import export

    trip, _ = ride_trip
    items, _ = ride.build(trip, ride.RideSettings(speed="4"), say=lambda _: None)
    export(trip, items, "ride", tmp_path)
    timeline = otio.adapters.read_from_file(str(tmp_path / "ride.otio"))
    clips = list(timeline.tracks[0].find_clips())
    assert [c.effects[0].time_scalar for c in clips] == [4.0, 4.0]
    assert timeline.duration().to_seconds() == pytest.approx(3.5, abs=0.05)
    assert "<timeMap>" in (tmp_path / "ride.fcpxml").read_text(encoding="utf-8")


@needs_ffmpeg
@pytest.mark.parametrize("speed,vfr", [
    (2.75, None), (3.1, None),
    (2.75, "lt(n,30)+not(mod(n,3))"),  # 30 fps for a second, then 10 fps
    (2.0, "lt(n,180)+gte(n,270)"),  # a 3 s gap with no frames, across a chunk start
])  # fmt: skip
def test_chunks_show_the_same_frames_as_one_piece(tmp_path, monkeypatch, speed, vfr):
    """Render chunks start between source frames at most speeds, and anywhere in variable-rate
    footage; the frames chosen must be the same as without chunks, at every chunk start."""
    from reelsift import render as render_mod
    from reelsift.ingest import ingest
    from reelsift.render import Output, render

    footage = tmp_path / "footage"
    footage.mkdir()
    rate = ["-vf", f"select='{vfr}'", "-fps_mode", "vfr"] if vfr else []
    run([tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=160x120:rate=30:duration=40", *rate, "-c:v", "libx264", "-g", "30",
         "-metadata", "creation_time=2026-07-04T09:30:00Z", str(footage / "f.mp4")])  # fmt: skip
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        rec = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
        item = Item(rec, 0.0, 39.0, -1, 0.0, None, "ride", speed=speed)
        whole = render(trip, [item], tmp_path / "whole.mp4", Output(160, 120, fps="30"))
        monkeypatch.setattr(render_mod, "CHUNK_SOURCE_S", 7)
        chunked = render(trip, [item], tmp_path / "chunked.mp4", Output(160, 120, fps="30"))
    size = (80, 60)
    source = frames_of_file(footage / "f.mp4", size).reshape(-1, size[0] * size[1])
    a = frames_of_file(whole, size).reshape(-1, size[0] * size[1])
    b = frames_of_file(chunked, size).reshape(-1, size[0] * size[1])
    assert len(a) == len(b)

    def shown(frames):  # which source frame each shows (chunks are encoded on their own)
        return [int(np.argmin(np.abs(source - f).mean(axis=1))) for f in frames]

    per_chunk = 7 * 30 / speed
    near = sorted(
        {k for c in range(1, 6) for k in range(round(c * per_chunk) - 4, round(c * per_chunk) + 5)}
        | {0, 1}
    )
    near = [k for k in near if k < len(a)]
    assert shown(a[near]) == shown(b[near])


def test_a_fitted_ride_lasts_its_length_to_the_frame():
    from reelsift.render import Piece, plan_frames

    def piece(duration):
        return Piece(None, Fraction(0), Fraction(duration), Fraction(30), False, "bt709", True,
                     speed=Fraction(321, 10))  # fmt: skip

    # 152 stretches of 3 min of footage at 32.1x: each rounded on its own adds up
    stretches = [[piece(Fraction(180) + Fraction(k, 7))] for k in range(152)]
    plan_frames(stretches, Fraction(30))
    total = sum(p.frames for ps in stretches for p in ps)
    exact = sum(Fraction(180) + Fraction(k, 7) for k in range(152)) / Fraction(321, 10) * 30
    assert abs(total - exact) <= 1


# --- review round 1 -------------------------------------------------------------------------------


def _travel(trip, rec, kmh=None, optical=None, n=20):
    from reelsift import ride

    trip.db.execute("DELETE FROM travel WHERE recording_id = ?", (rec,))
    trip.db.executemany("INSERT INTO travel VALUES (?, ?, ?, ?)", [
        (rec, s, None if kmh is None or kmh[s] is None else kmh[s],
         None if optical is None else optical[s]) for s in range(n)])  # fmt: skip
    trip.db.execute("INSERT OR REPLACE INTO stage_runs VALUES (?, 'travel', ?, 'now')",
                    (rec, ride._version()))  # fmt: skip


def test_gps_decides_every_second_it_covers(ride_trip):
    from reelsift import ride, travel

    trip, rec = ride_trip
    _travel(trip, rec, kmh=[50.0] * 10 + [None] * 10, optical=[0.001] * 10 + [0.1] * 10)
    states, _ = ride.states(trip, [rec])[rec]
    assert travel.STOPPED not in states[:10], "known travel is never cut"


def test_corrections_apply_in_order_and_are_never_merged_away(ride_trip):
    from reelsift import ride

    trip, rec = ride_trip
    drop = {"kind": "drop", "recording_id": rec, "start": 4.0, "end": 4.3}
    items, _ = ride.build(trip, ride.RideSettings(speed="4", overrides=[drop]), say=print)
    assert (0.0, 4.0) in [(i.start, i.end) for i in items] and all(
        not (i.start < 4.3 and i.end > 4.0) for i in items)  # fmt: skip
    keep = {"kind": "keep", "recording_id": rec, "start": 8.0, "end": 14.0}
    again = {"kind": "drop", "recording_id": rec, "start": 8.0, "end": 14.0}
    items, _ = ride.build(trip, ride.RideSettings(speed="4", overrides=[keep, again, keep]),
                          say=print)  # fmt: skip
    assert (8.0, 14.0, 1.0) in [(i.start, i.end, i.speed) for i in items], "the last keep wins"
    starts = sorted((i.start, i.end) for i in items)
    assert all(a[1] <= b[0] for a, b in itertools.pairwise(starts)), "no overlaps"


def test_a_drop_from_every_edit_reaches_the_ride_and_undo_brings_it_back(ride_trip):
    from reelsift import edits, ride
    from reelsift.review import Review

    trip, rec = ride_trip
    items, report = ride.build(trip, ride.RideSettings(speed="4"), say=print)
    ride.save(trip, items, ride.RideSettings(speed="4"), report, "first")
    review = Review(trip)
    review.decide("drop", rec, 4.0, 6.0)
    spans = [(i.start, i.end, i.speed) for i in edits.latest(trip, ride.RIDE).items]
    assert (0.0, 4.0, 4.0) in spans and (6.0, 8.0, 4.0) in spans
    review.undo()
    first = edits.latest(trip, ride.RIDE).items[0]
    assert (first.start, first.end) == (0.0, 8.0)


def test_restoring_a_ride_keeps_its_speeds_and_settings(ride_trip):
    from reelsift import edits, ride
    from reelsift.review import Review

    trip, rec = ride_trip
    slow = ride.RideSettings(speed="4")
    items, report = ride.build(trip, slow, say=print)
    first = ride.save(trip, items, slow, report, "4x")
    fast = ride.RideSettings(speed="8")
    items, report = ride.build(trip, fast, say=print)
    ride.save(trip, items, fast, report, "8x")
    review = Review(trip)
    review.decide("drop", rec, 4.0, 6.0)
    review.restore(ride.RIDE, first.id)
    restored = edits.latest(trip, ride.RIDE)
    assert [(i.start, i.end, i.speed) for i in restored.items][:2] == [
        (0.0, 4.0, 4.0),
        (6.0, 8.0, 4.0),
    ]
    assert ride.settings_of(trip).speed == "4"


def test_gps_is_read_without_proxies_and_missing_proxies_are_retried(ride_trip, monkeypatch):
    from reelsift import ride
    from reelsift import speed as speed_mod

    trip, rec = ride_trip
    trip.db.execute("DELETE FROM stage_runs WHERE stage = 'travel'")
    monkeypatch.setattr(speed_mod, "samples", lambda p, i: [(float(t), 30.0) for t in range(20)])
    ride.analyse_travel(trip, [rec], say=print)
    kmh = [r[0] for r in trip.db.execute("SELECT kmh FROM travel WHERE recording_id = ?", (rec,))]
    assert kmh[:20] == [30.0] * 20
    monkeypatch.setattr(speed_mod, "samples", lambda p, i: None)
    trip.db.execute("DELETE FROM stage_runs WHERE stage = 'travel'")
    ride.analyse_travel(trip, [rec], say=print)  # the picture is needed, and there's no proxy
    assert not trip.db.execute("SELECT 1 FROM stage_runs WHERE stage = 'travel'").fetchone()


def test_levels_set_for_the_ride_beforehand_are_kept(ride_trip):
    from reelsift import ride, soundtrack
    from reelsift.config import MusicSettings

    trip, _ = ride_trip
    trip.config.music_lengths[ride.RIDE] = MusicSettings(sound=-60.0, music=-60.0)
    items, report = ride.build(trip, ride.RideSettings(speed="4"), say=print)
    stored = ride.save(trip, items, ride.RideSettings(speed="4"), report, "first")
    trip.config.music_lengths.pop(ride.RIDE)
    st = soundtrack.get(trip, stored.id)
    assert (st.mix.sound, st.mix.music, st.mix.background) == (-60.0, -60.0, 0.0)


def test_music_energy_follows_the_edits_own_time():
    from types import SimpleNamespace

    from reelsift.musicgen import edit_energy

    motion = np.zeros(200)
    motion[0:200:10] = 1.0  # each kept second moves; the seconds in between don't
    data = SimpleNamespace(observations={1: {"motion": motion, "activity": motion}}, words={})
    items = [Item(1, float(k * 10), float(k * 10 + 1), -1, 0, None, "ride", speed=8.0)
             for k in range(10)]  # fmt: skip
    energy, _ = edit_energy(data, items)
    assert len(energy) == 2 and np.nanmin(energy) == pytest.approx(1.0)


@needs_ffmpeg
def test_a_stop_split_between_two_files_is_still_a_stop(tmp_path):
    from reelsift import ride, travel
    from reelsift.ingest import ingest
    from reelsift.sync import sync

    footage = tmp_path / "footage"
    footage.mkdir()
    for name, when in (("a.mp4", "09:30:00"), ("b.mp4", "09:30:20")):
        run([tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
             "testsrc2=size=160x120:rate=30:duration=20", "-c:v", "libx264",
             "-metadata", f"creation_time=2026-07-04T{when}Z", str(footage / name)])  # fmt: skip
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        sync(trip)
        recs = [r for r, _, _ in ride.recordings_of(trip, ride.cameras(trip)[0][0])]
        _travel(trip, recs[0], kmh=[40.0] * 16 + [0.0] * 4)
        _travel(trip, recs[1], kmh=[0.0] * 4 + [40.0] * 16)
        states = ride.states(trip, recs)
    together = np.concatenate([states[r][0] for r in recs])
    assert travel.STOPPED in together, "4 s at the end of one file and 4 s at the start of the next"


def test_a_build_finishing_after_the_ride_changed_isnt_saved(ride_trip, monkeypatch):
    import time

    from fastapi.testclient import TestClient

    from reelsift import edits, ride
    from reelsift.review import Review
    from reelsift.web.app import create_app

    trip, _ = ride_trip
    items, report = ride.build(trip, ride.RideSettings(speed="4"), say=print)
    ride.save(trip, items, ride.RideSettings(speed="4"), report, "first")
    shared = Review(trip.worker(shared=True))
    client = TestClient(create_app(trip, shared, 8765), base_url="http://127.0.0.1:8765")
    real_build = ride.build

    def slow_build(t, settings, say=print):
        found = real_build(t, settings, say)
        other = trip.worker()  # meanwhile, the ride changes (a restore, another tab)
        two = ride.RideSettings(speed="2")
        changed, why = real_build(other, two, print)
        ride.save(other, changed, two, why, "meanwhile")
        other.close()
        return found

    monkeypatch.setattr(ride, "build", slow_build)
    r = client.post("/api/ride/build", json={"speed": "16"})
    assert r.status_code == 200, r.text
    for _ in range(100):
        job = client.get("/api/ride").json()["job"]
        if not job.get("running"):
            break
        time.sleep(0.05)
    assert "changed" in (job.get("error") or "")
    assert edits.latest(trip, ride.RIDE).settings["note"] == "meanwhile"


# --- review round 2 -------------------------------------------------------------------------------


@needs_ffmpeg
def test_a_dark_picture_is_unknown_never_a_stop(tmp_path):
    from reelsift import travel

    path = tmp_path / "dark.mp4"
    run([tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
         "color=c=black:size=320x240:rate=30:duration=8", "-c:v", "libx264", str(path)])  # fmt: skip
    tiles = travel.tile_flow(path, 320, 240)
    assert tiles and all(np.isnan(travel.travel_values(np.array([v]))[0]) for v in tiles.values())
    # a tunnel in the middle of a picture-only ride, with stops elsewhere to learn from
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.uniform(0.04, 0.12, 100), rng.uniform(0.001, 0.004, 30),
                             rng.uniform(0.04, 0.12, 100), [np.nan] * 20,
                             rng.uniform(0.04, 0.12, 100)])  # fmt: skip
    states = travel.classify(values, travel.thresholds(values))
    assert travel.STOPPED in states[100:130], "the real stop is still found"
    assert (states[230:250] == travel.UNKNOWN).all(), "the dark stretch is unknown"


@needs_ffmpeg
def test_the_ride_lists_every_stop_it_cut_even_whole_files(tmp_path):
    from fastapi.testclient import TestClient

    from reelsift import ride
    from reelsift.ingest import ingest
    from reelsift.review import Review
    from reelsift.sync import sync
    from reelsift.web.app import create_app

    footage = tmp_path / "footage"
    footage.mkdir()
    for name, when in (("a.mp4", "09:30:00"), ("b.mp4", "09:30:20"), ("c.mp4", "09:30:40")):
        run([tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
             "testsrc2=size=160x120:rate=30:duration=20", "-c:v", "libx264",
             "-metadata", f"creation_time=2026-07-04T{when}Z", str(footage / name)])  # fmt: skip
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        sync(trip)
        recs = [r for r, _, _ in ride.recordings_of(trip, ride.cameras(trip)[0][0])]
        _travel(trip, recs[0], kmh=[40.0] * 10 + [0.0] * 10)
        _travel(trip, recs[1], kmh=[0.0] * 20)
        _travel(trip, recs[2], kmh=[0.0] * 10 + [40.0] * 10)
        settings = ride.RideSettings(speed="4")
        items, report = ride.build(trip, settings, say=print)
        ride.save(trip, items, settings, report, "ride")
        assert recs[1] not in {i.recording_id for i in items}
        client = TestClient(create_app(trip, Review(trip.worker(shared=True)), 8765),
                            base_url="http://127.0.0.1:8765")  # fmt: skip
        stops = [
            (s["recording_id"], s["start"], s["end"])
            for s in client.get("/api/ride").json()["stops"]
        ]
    assert stops == [(recs[0], 11.0, 20.0), (recs[1], 0.0, 20.0), (recs[2], 0.0, 9.0)]


def test_a_ride_of_kept_stops_only_fits_its_own_length(ride_trip):
    from reelsift import ride

    trip, rec = ride_trip
    overrides = [{"kind": "drop", "recording_id": rec, "start": 0.0, "end": 8.0},
                 {"kind": "drop", "recording_id": rec, "start": 14.0, "end": 20.0},
                 {"kind": "keep", "recording_id": rec, "start": 8.0, "end": 14.0}]  # fmt: skip
    items, _ = ride.build(trip, ride.RideSettings(speed="fit", target=6.0, overrides=overrides),
                          say=print)  # fmt: skip
    assert [(i.start, i.end, i.speed) for i in items] == [(8.0, 14.0, 1.0)]
    with pytest.raises(ride.RideError, match="longer than"):
        ride.build(trip, ride.RideSettings(speed="fit", target=4.0, overrides=overrides),
                   say=print)  # fmt: skip


def test_resetting_a_rides_levels_from_the_command_line_gives_the_rides_own(numbered):
    from typer.testing import CliRunner

    from reelsift import edits, ride, soundtrack
    from reelsift.cli import app

    root, rec, _ = numbered
    with Trip(root / "trip", write=True) as trip:
        trip.db.execute("DELETE FROM edits WHERE length = -1")
        _travel(trip, rec, kmh=[40.0] * 7 + [0.0] * 8 + [40.0] * 5)
        items, report = ride.build(trip, ride.RideSettings(speed="4"), say=print)
        ride.save(trip, items, ride.RideSettings(speed="4"), report, "ride")
    result = CliRunner().invoke(app, ["music", "mix", str(root / "trip"), "--ride", "--reset"])
    assert result.exit_code == 0, result.output
    with Trip(root / "trip") as trip:
        mix = soundtrack.get(trip, edits.latest(trip, ride.RIDE).id).mix
    assert mix == ride.RIDE_MIX


# --- review round 3 -------------------------------------------------------------------------------


@needs_ffmpeg
def test_a_sped_up_piece_starting_in_a_gap_shows_the_frame_before_it(tmp_path):
    from reelsift.ingest import ingest
    from reelsift.render import Output, render

    footage = tmp_path / "footage"
    footage.mkdir()
    run([tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=160x120:rate=30:duration=24", "-vf", "select='lt(n,180)+gte(n,270)'",
         "-fps_mode", "vfr", "-c:v", "libx264", "-g", "30",
         "-metadata", "creation_time=2026-07-04T09:30:00Z", str(footage / "f.mp4")])  # fmt: skip
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        rec = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
        item = Item(rec, 7.5, 12.0, -1, 0.0, None, "ride", speed=2.0)
        out = render(trip, [item], tmp_path / "r.mp4", Output(160, 120, fps="30"))
    source = frames_of_file(footage / "f.mp4")
    first = frames_of_file(out)[0]
    shown = int(np.argmin([np.abs(first - f).mean() for f in source]))
    assert shown == 179, "the last frame before the gap, not the first after it"


def test_a_fitted_speed_is_kept_to_the_frame(ride_trip):
    from reelsift.render import pieces

    trip, rec = ride_trip
    (piece,) = pieces(trip, Item(rec, 0.0, 20.0, -1, 0.0, None, "ride", speed=1.000363))
    assert piece.speed != 1 and float(piece.speed) == pytest.approx(1.000363, abs=1e-9)


def test_music_for_every_edit_reaches_the_ride(numbered, tmp_path):
    from typer.testing import CliRunner

    from reelsift import edits, ride, soundtrack
    from reelsift.cli import app

    from .conftest import make_music

    root, rec, _ = numbered
    with Trip(root / "trip", write=True) as trip:
        trip.db.execute("DELETE FROM edits WHERE length = -1")
        _travel(trip, rec, kmh=[40.0] * 7 + [0.0] * 8 + [40.0] * 5)
        items, report = ride.build(trip, ride.RideSettings(speed="4"), say=print)
        ride.save(trip, items, ride.RideSettings(speed="4"), report, "ride")
    song = make_music(tmp_path / "s.wav", seconds=3)
    runner = CliRunner()
    result = runner.invoke(app, ["music", "set", str(root / "trip"), str(song)])
    assert result.exit_code == 0, result.output
    with Trip(root / "trip") as trip:
        st = soundtrack.get(trip, edits.latest(trip, ride.RIDE).id)
    assert st.source == "files" and st.files == [str(song.resolve())]
    assert runner.invoke(app, ["music", "clear", str(root / "trip")]).exit_code == 0
    with Trip(root / "trip") as trip:
        assert soundtrack.get(trip, edits.latest(trip, ride.RIDE).id).source == "none"


def test_music_energy_weighs_footage_by_its_time_in_the_edit():
    from types import SimpleNamespace

    from reelsift.musicgen import edit_energy

    motion = np.zeros(20)
    motion[0:4] = 1.0
    data = SimpleNamespace(observations={1: {"motion": motion, "activity": motion}}, words={})
    items = [
        Item(1, 0.0, 4.0, -1, 0, None, "ride", speed=8.0),  # half a second, busy
        Item(1, 10.0, 10.5, -1, 0, None, "ride", speed=1.0),
    ]  # half a second, still
    energy, _ = edit_energy(data, items)
    assert energy[0] == pytest.approx(0.5)


@needs_ffmpeg
def test_the_frame_before_a_gap_is_found_when_timestamps_dont_start_at_zero(tmp_path):
    from reelsift.render import frame_before

    path = tmp_path / "gap.ts"  # MPEG-TS timestamps start well after zero
    run([tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=160x120:rate=30:duration=24", "-vf", "select='lt(n,180)+gte(n,270)'",
         "-fps_mode", "vfr", "-c:v", "libx264", "-output_ts_offset", "40", str(path)])  # fmt: skip
    # at most a few frames early (B-frames are stored out of order), never after the frame
    assert 5.8 <= float(frame_before(path, 0, Fraction(15, 2))) <= 179 / 30 + 0.001
    assert 9.8 <= float(frame_before(path, 0, Fraction(10))) <= 10.001


# --- no review ------------------------------------------------------------------------------------


@needs_ffmpeg
def test_make_does_a_ride_alone_with_no_review(tmp_path):
    from typer.testing import CliRunner

    from reelsift.cli import app

    footage = tmp_path / "footage"
    footage.mkdir()
    run([tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=320x240:rate=30:duration=12", "-c:v", "libx264",
         "-metadata", "creation_time=2026-07-04T09:30:00Z", str(footage / "a.mp4")])  # fmt: skip
    trip = str(tmp_path / "trip")
    runner = CliRunner()
    assert runner.invoke(app, ["init", trip, "--footage", str(footage)]).exit_code == 0
    result = runner.invoke(app, ["make", trip, "--only-ride", "--ride-speed", "4"])
    assert result.exit_code == 0, result.output
    videos = sorted((tmp_path / "trip" / "output").glob("*.mp4"))
    assert len(videos) == 1 and videos[0].name.startswith("trip-ride-r1")
    assert str(videos[0]) in result.output
    result = runner.invoke(app, ["render", trip, "--ride", "--vertical"])
    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "trip" / "output").glob("trip-ride-r1*-vertical.mp4"))) == 1


def test_make_with_music_says_to_install_it_first(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from reelsift.cli import app

    monkeypatch.setenv("REELSIFT_MUSIC_DIR", str(tmp_path / "no-music"))
    result = CliRunner().invoke(app, ["make", str(tmp_path / "trip"), "--music", "synthwave"])
    assert result.exit_code == 1 and "reelsift music install" in result.output
