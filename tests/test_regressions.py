"""Regression tests: review decisions, ingest, caches, media streams, export and sound."""

import shutil

import pytest
from fastapi.testclient import TestClient

from reelsift import edits as edits_mod
from reelsift.project import Trip
from reelsift.review import Review, overlap
from reelsift.select import Item
from reelsift.web.app import create_app

from .conftest import needs_ffmpeg
from .test_review import LENGTHS, analyzed_root  # noqa: F401  (module fixture reused here)

pytestmark = needs_ffmpeg


@pytest.fixture
def review(analyzed_root, tmp_path):  # noqa: F811
    root = tmp_path / "trip"
    shutil.copytree(analyzed_root, root, ignore=shutil.ignore_patterns("reelsift.lock"))
    with Trip(root, write=True) as trip:
        yield Review(trip)


def walk(review: Review) -> int:
    """The 12-second recording, long enough for interval scenarios."""
    return max(review.data.recordings, key=lambda r: review.data.recordings[r].duration)


def touches(items, rec, a, b) -> list[Item]:
    return [i for i in items if i.recording_id == rec and overlap(i.start, i.end, a, b) > 0]


def consistent(review: Review) -> bool:
    return review.problems() == []


def test_drop_overrides_an_earlier_keep(review):
    rec = walk(review)
    review.decide("keep", rec, 2.0, 4.0, length=4)
    review.decide("drop", rec, 2.0, 4.0)
    review.select_all()
    review.refill(8)
    for n in LENGTHS:
        assert not touches(review.edit(n).items, rec, 2.0, 4.0)
    review.undo()
    assert any(d["kind"] == "keep" for d in review.decisions()), (
        "undoing the drop restores the keep"
    )


def test_dropped_footage_stays_out_of_grown_and_snapped_clips(review):
    rec = walk(review)
    review.decide("drop", rec, 3.0, 4.0)
    review.select_all()
    review.refill(4)
    for n in LENGTHS:
        assert not [i for i in touches(review.edit(n).items, rec, 3.0, 4.0) if not i.locked]
    assert consistent(review)


def test_adding_across_two_clips_keeps_both_whole(review):
    rec = walk(review)
    items = [
        Item(rec, 0.0, 2.0, -1, 0.0, None, "window"),
        Item(rec, 4.0, 8.0, -1, 0.0, None, "window", locked=True),
    ]
    placed = review._place(8, items, rec, 1.0, 5.0, None)
    assert [(i.start, i.end) for i in placed if i.recording_id == rec] == [(0.0, 8.0)]


def test_a_trim_in_one_edit_leaves_the_other_edits_keep(review):
    rec = walk(review)
    review.decide("keep", rec, 5.0, 7.0, length=4)
    review.decide("keep", rec, 5.0, 7.0, length=8)
    review.decide("keep", rec, 5.0, 6.0, length=8, replaces=(5.0, 7.0))
    keeps = sorted((d["length"], d["start"], d["end"]) for d in review.decisions())
    assert keeps == [(4, 5.0, 7.0), (8, 5.0, 6.0)]
    review.select_all()
    assert touches(review.edit(4).items, rec, 6.5, 7.0)
    assert not touches(review.edit(8).items, rec, 6.2, 7.0)
    assert consistent(review)
    review.undo()
    keeps = sorted((d["length"], d["start"], d["end"]) for d in review.decisions())
    assert keeps == [(4, 5.0, 7.0), (8, 5.0, 7.0)], "undo brings the original keep back"


def test_restoring_a_revision_brings_back_that_edit_and_its_decisions(review):
    old_short = review.edit(4)
    rec = walk(review)
    review.decide("force", rec, 9.0, 10.5, length=4)
    review.decide("force", rec, 9.0, 10.5, length=8)
    review.select_all()
    restored = review.restore(4, old_short.id)
    assert [(i.recording_id, i.start, i.end) for i in restored.items] == [
        (i.recording_id, i.start, i.end) for i in old_short.items
    ]
    assert consistent(review)
    assert [d["length"] for d in review.decisions()] == [8], (
        "the 4 s edit's later add is switched off; the 8 s edit's stays"
    )


def test_a_failing_save_leaves_no_half_written_family(review, monkeypatch):
    review.select_all()
    before = review.trip.db.execute("SELECT COUNT(*) FROM edits").fetchone()[0]
    real_save = edits_mod.save
    calls = []

    def failing(trip, sel, settings):
        calls.append(sel.length)
        if len(calls) == 2:
            raise RuntimeError("disk full")
        return real_save(trip, sel, settings)

    monkeypatch.setattr(edits_mod, "save", failing)
    with pytest.raises(RuntimeError):
        review.select_all()
    after = review.trip.db.execute("SELECT COUNT(*) FROM edits").fetchone()[0]
    assert after == before
    monkeypatch.setattr(edits_mod, "save", real_save)
    old = review.weights
    calls.clear()
    calls.append(None)  # the next save is the second call, which fails
    monkeypatch.setattr(edits_mod, "save", failing)
    from reelsift.prefs import Preferences

    with pytest.raises(RuntimeError):
        review.apply(8, Preferences(adjustments={"people": 2}))
    assert review.weights == old, "in-memory scoring is restored too"
    assert review.trip.db.execute("SELECT COUNT(*) FROM preferences").fetchone()[0] == 0


def test_select_with_a_length_keeps_decisions(review):
    from reelsift.cli import _select

    rec = walk(review)
    review.decide("drop", rec, 0.0, 12.0)
    _select(review.trip, [8], "combined")
    assert not touches(edits_mod.latest(review.trip, 8).items, rec, 0.0, 12.0)


def test_select_rejects_unknown_lengths(review):
    import typer

    from reelsift.cli import _select

    with pytest.raises(typer.Exit):
        _select(review.trip, [7], "combined")


def test_intervals_are_checked_after_clamping(review):
    rec = walk(review)
    with pytest.raises(ValueError):
        review.decide("force", rec, 130.0, 140.0, length=8)
    with pytest.raises(ValueError):
        review.decide("keep", rec, 1.0, 2.0, length=8, replaces=(1.0,))
    with pytest.raises(ValueError):
        review.decide("force", rec, float("nan"), 2.0, length=8)
    assert not review.decisions()


def test_kept_clips_keep_their_crop(review):
    item = review.edit(8).items[0]
    review.decide("keep", item.recording_id, item.start, item.end, length=8)
    kept = next(i for i in review.edit(8).items if i.recording_id == item.recording_id
                and abs(i.start - item.start) < 1e-3)  # fmt: skip
    review.set_crop(8, kept.recording_id, kept.start, kept.end, 0.1)
    review.select_all()
    again = next(i for i in review.edit(8).items if i.recording_id == item.recording_id
                 and overlap(i.start, i.end, item.start, item.end) > 0)  # fmt: skip
    assert again.crop_x == 0.1


@pytest.fixture
def client(review):
    shared = Review(review.trip.worker(shared=True))
    yield TestClient(create_app(review.trip, shared, 8765), base_url="http://127.0.0.1:8765")
    shared.trip.close()


def test_changes_against_a_stale_edit_are_refused(client):
    edit = client.get("/api/edits/8").json()
    item = edit["items"][0]
    body = {"kind": "keep", "recording_id": item["recording_id"], "start": item["start"],
            "end": item["end"], "length": 8, "board": 8, "expected_edit_id": edit["edit_id"]}  # fmt: skip
    assert client.post("/api/decisions", json=body).status_code == 200
    stale = dict(body, kind="drop", length=None)  # same, old edit id
    assert client.post("/api/decisions", json=stale).status_code == 409
    assert (
        client.post(f"/api/edits/8/approve?expected_edit_id={edit['edit_id']}").status_code == 409
    )


def test_api_rejects_bad_numbers(client):
    item = client.get("/api/edits/8").json()["items"][0]
    base = {"kind": "keep", "recording_id": item["recording_id"], "length": 8}
    r = client.post("/api/decisions", json=dict(base, start=1, end=2, replaces=[1, 2, 3]))
    assert r.status_code == 422
    r = client.post("/api/decisions", json=dict(base, start="inf", end=2))
    assert r.status_code == 422
    r = client.post("/api/decisions", json=dict(base, start=500, end=600))
    assert r.status_code == 400


def test_preview_spans_cross_chapters():
    from reelsift.web.media import split_spans

    chapters = [(0, 0.0, 10.0), (1, 10.0, 10.0)]
    assert split_spans(chapters, 9.0, 12.0) == [
        {"chapter": 0, "start": 9.0, "end": 10.0},
        {"chapter": 1, "start": 0.0, "end": 2.0},
    ]
    assert split_spans(chapters, 2.0, 4.0) == [{"chapter": 0, "start": 2.0, "end": 4.0}]
    assert split_spans([(0, 0.0, 0.0)], 0.0, 3.0) == [{"chapter": 0, "start": 0.0, "end": 3.0}]


def test_api_returns_every_span(client):
    item = client.get("/api/edits/8").json()["items"][0]
    assert item["spans"] and item["preview"] == item["spans"][0]
    total = sum(s["end"] - s["start"] for s in item["spans"])
    assert abs(total - item["duration"]) < 0.01


# --- ingest, proxies and caches ------------------------------------------------------------------


def prepared(tmp_path, footage, analyzed=False):
    from reelsift import candidates, proxies
    from reelsift.analysis import analyze
    from reelsift.ingest import ingest
    from reelsift.sync import sync

    from .fakes import FakeEmbedder, FakeTranscriber

    trip = Trip.create(tmp_path / "trip", [footage]).__enter__()
    ingest(trip)
    proxies.generate(trip)
    sync(trip)
    if analyzed:
        analyze(trip, FakeEmbedder, FakeTranscriber, say=lambda _: None)
        candidates.generate(trip)
    return trip


def test_overwritten_file_is_ingested_as_new_footage(tmp_path):
    from reelsift.ingest import ingest

    from .conftest import make_photo

    footage = tmp_path / "footage"
    make_photo(footage / "a.jpg", color=(10, 20, 30))
    trip = prepared(tmp_path, footage)
    old = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
    make_photo(footage / "a.jpg", color=(200, 30, 30), size=(500, 300))
    summary = ingest(trip)
    assert not summary.failed
    rows = trip.db.execute("SELECT id, width FROM recordings").fetchall()
    assert len(rows) == 1 and rows[0]["id"] != old and rows[0]["width"] == 500
    assert trip.db.execute("SELECT COUNT(*) FROM assets WHERE kind = 'photo'").fetchone()[0] == 1
    assert any(
        "content changed" in r["message"] for r in trip.db.execute("SELECT message FROM issues")
    )
    trip.__exit__(None, None, None)


def test_moved_camera_proxies_follow_their_footage(tmp_path):
    from reelsift import config as config_mod
    from reelsift.ingest import ingest
    from reelsift.proxies import resolve

    from .conftest import make_clip

    footage = tmp_path / "footage"
    make_clip(footage / "DJI_001" / "DJI_20261011095133_0001_D.MP4", 2.0,
              creation_time="2026-10-11T07:51:34Z")  # fmt: skip
    make_clip(footage / "DJI_001" / "DJI_20261011095133_0001_D.LRF", 2.0, size="160x90")
    trip = prepared(tmp_path, footage)
    moved = tmp_path / "moved"
    shutil.move(footage, moved)
    trip.config.footage = [str(moved)]
    config_mod.write_trip(trip.root, trip.config)
    ingest(trip)
    row = trip.db.execute("SELECT * FROM proxies").fetchone()
    assert row["origin"] == "camera"
    assert resolve(trip, row["path"]).is_file()
    assert str(moved) in row["path"]
    trip.__exit__(None, None, None)


def test_deleted_generated_proxies_are_made_again(tmp_path):
    from reelsift import proxies
    from reelsift.ingest import ingest

    from .conftest import make_clip

    footage = tmp_path / "footage"
    make_clip(footage / "cam" / "a.mp4", 2.0, creation_time="2026-10-11T07:51:34Z")
    trip = prepared(tmp_path, footage)
    shutil.rmtree(trip.root / "cache" / "proxies")
    ingest(trip)
    assert proxies.pending(trip), "the missing proxy is due again"
    proxies.generate(trip)
    row = trip.db.execute("SELECT path FROM proxies").fetchone()
    assert proxies.resolve(trip, row["path"]).is_file()
    trip.__exit__(None, None, None)


def test_damaged_embeddings_are_redone(tmp_path):
    from reelsift.analysis import analyze, embeddings_ok
    from reelsift.tripdata import load

    from .conftest import make_clip
    from .fakes import FakeEmbedder, FakeTranscriber

    footage = tmp_path / "footage"
    make_clip(footage / "cam" / "a.mp4", 3.0, creation_time="2026-10-11T07:51:34Z")
    trip = prepared(tmp_path, footage, analyzed=True)
    rec = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
    path = trip.root / trip.db.execute("SELECT path FROM embeddings").fetchone()["path"]
    path.write_bytes(b"truncated")
    assert not embeddings_ok(trip, rec)
    assert rec not in load(trip).embeddings, "a damaged file is unknown, not a crash"
    summary = analyze(trip, FakeEmbedder, FakeTranscriber, say=lambda _: None)
    assert summary.visual == 1 and embeddings_ok(trip, rec)
    assert not list(path.parent.glob("*.partial.npy"))
    trip.__exit__(None, None, None)


# --- media streams, chapters, decoding -----------------------------------------------------------


def test_cover_picture_first_is_not_mistaken_for_the_video(tmp_path):
    from PIL import Image

    from reelsift import proxies
    from reelsift.media import probe
    from reelsift.render import Output, render
    from reelsift.tools import run, tool

    from .conftest import make_clip

    plain = make_clip(tmp_path / "plain.mp4", 2.0, creation_time="2026-10-11T07:51:34Z")
    cover = tmp_path / "cover.jpg"
    Image.new("RGB", (64, 64), (255, 0, 0)).save(cover)
    footage = tmp_path / "footage"
    footage.mkdir()
    covered = footage / "covered.mp4"
    # A one-frame MJPEG picture as video stream 0, the movie second, as DJI cameras write it.
    run([tool("ffmpeg"), "-v", "error", "-y", "-i", str(cover), "-i", str(plain),
         "-map", "0:v", "-map", "1:v", "-map", "1:a", "-c", "copy", "-f", "mp4", str(covered)])  # fmt: skip
    assert probe(covered).main_video.frames == 50
    assert probe(covered).main_video.index != 0, "the movie is not the first video stream"
    trip = prepared(tmp_path, footage)
    row = trip.db.execute("SELECT path FROM proxies").fetchone()
    assert probe(proxies.resolve(trip, row["path"])).main_video.frames >= 40
    rec = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
    out = render(trip, [Item(rec, 0.0, 2.0, -1, 0.0, None, "window")], tmp_path / "o.mp4",
                 Output(320, 180))  # fmt: skip
    assert probe(out).main_video.frames >= 45
    trip.__exit__(None, None, None)


def test_samples_land_on_the_recording_grid():
    import numpy as np

    from reelsift.analysis import fill_rows, recording_second

    seconds = [recording_second(0.6 * k, 0.0) for k in range(10)]  # ten 0.6 s chapters
    assert sorted(set(seconds)) == [0, 1, 2, 3, 4, 5]
    assert recording_second(1234.56, 2.0) == 1236
    rows = fill_rows({0: np.array([1.0]), 2: np.array([3.0])})
    assert rows.tolist() == [[1.0], [1.0], [3.0]]


def test_a_noisy_decoder_cannot_deadlock(tmp_path):
    import sys
    import threading

    from reelsift.analysis import stream_chunks

    script = tmp_path / "noisy.py"
    script.write_text(
        "import sys\n"
        "sys.stderr.write('decode error' * 200000)\n"
        "sys.stderr.flush()\n"
        "sys.stdout.buffer.write(bytes(64))\n"
    )
    result = []

    def consume():
        result.extend(stream_chunks([sys.executable, str(script)], 16, 2, "noisy"))

    worker = threading.Thread(target=consume, daemon=True)
    worker.start()
    worker.join(timeout=30)
    assert not worker.is_alive(), "decoding hung"
    assert b"".join(result) == bytes(64)


def test_stopping_early_stops_the_decoder(tmp_path):
    import sys

    from reelsift.analysis import stream_chunks

    script = tmp_path / "endless.py"
    script.write_text("import sys\nwhile True:\n    sys.stdout.buffer.write(bytes(1024))\n")
    chunks = stream_chunks([sys.executable, str(script)], 16, 4, "endless")
    assert next(chunks)
    chunks.close()  # must return promptly, with the child stopped


# --- export, sound, music, config, photos, model answers -----------------------------------------


def test_otio_keeps_timeline_lengths_for_other_frame_rates(tmp_path):
    from fractions import Fraction
    from pathlib import Path

    from reelsift.export import Clip, otio_timeline

    clips = [
        Clip(
            "a.mp4",
            Path(tmp_path / "a.mp4"),
            False,
            Fraction(24),
            0,
            240,
            3 * k,
            2,
            3,
            64,
            36,
            True,
        )
        for k in range(10)
    ]
    timeline = otio_timeline(clips, "mixed", Fraction(30))
    video, audio = timeline.tracks
    assert abs(video.duration().to_seconds() - 1.0) < 1e-9
    assert abs(audio.duration().to_seconds() - 1.0) < 1e-9


def test_zone_setting_converts_local_photo_time(tmp_path):
    from datetime import UTC, datetime
    from fractions import Fraction

    from reelsift import config as config_mod
    from reelsift.ingest import ingest
    from reelsift.sync import clock, sync

    from .conftest import make_photo

    footage = tmp_path / "footage"
    make_photo(footage / "phone" / "a.jpg", taken="2026:10:11 10:00:00", offset=None)
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        label = trip.db.execute("SELECT label FROM cameras").fetchone()["label"]
        trip.config.cameras[label] = config_mod.CameraConfig(zone="+02:00")
        sync(trip)
        rec = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
        clk = clock(trip, rec)
        assert clk.to_utc(Fraction(0)) == datetime(2026, 10, 11, 8, 0, tzinfo=UTC)
        assert clk.local(Fraction(0)).hour == 10


def test_photo_descriptions_count_in_scores(tmp_path):
    import numpy as np

    from reelsift import candidates, proxies
    from reelsift.analysis import analyze
    from reelsift.ingest import ingest
    from reelsift.scoring import features, load_prompt_bank
    from reelsift.sync import sync
    from reelsift.tripdata import load

    from .conftest import make_photo
    from .fakes import FakeDescriber, FakeEmbedder, FakeTranscriber

    footage = tmp_path / "footage"
    make_photo(footage / "p.jpg", color=(220, 220, 220))
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        proxies.generate(trip)
        sync(trip)
        analyze(trip, FakeEmbedder, FakeTranscriber, say=lambda _: None,
                describer=(FakeDescriber.name, FakeDescriber))  # fmt: skip
        candidates.generate(trip)
        data = load(trip)
        feats = features(data, load_prompt_bank(trip))
        j = feats.names.index("interest")
        assert np.isfinite(feats.raw[:, j]).all()
        assert feats.raw[0, feats.names.index("tag:scenery")] == 1.0


@pytest.mark.parametrize(
    "answer",
    [
        {"description": "view", "interest": 5, "tags": None, "quality": "blur"},
        {"description": "view", "interest": float("inf"), "tags": []},
        {"description": "view", "interest": 11, "tags": []},
        {"description": "view", "interest": 6.5, "tags": []},
        {"description": "view", "interest": "7", "tags": []},
        {"description": "view", "interest": True, "tags": []},
        {"description": "view", "interest": 7, "tags": [1, 2]},
        ["not", "an", "object"],
    ],
)
def test_malformed_answers_become_unknown(answer):
    from reelsift.vlm import validate

    assert validate(answer) is None


def test_valid_answers_pass_and_unknown_tags_are_dropped():
    from reelsift.vlm import validate

    v = validate({"description": " a view ", "interest": 7.0, "tags": ["scenery", "nope"]})
    assert v == {"description": "a view", "tags": ["scenery"], "interest": 7.0, "quality": []}


def test_loudness_is_measured_before_normalizing(tmp_path):
    from reelsift.render import measure_loudness, mix_graph
    from reelsift.tools import tool

    from .conftest import make_clip

    clip = make_clip(tmp_path / "tone.mp4", 3.0)
    stats = measure_loudness([tool("ffmpeg"), "-v", "error", "-y", "-i", str(clip)],
                             mix_graph(3.0, [], 0))  # fmt: skip
    assert stats is not None and float(stats["input_i"]) < 0


def test_saving_music_keeps_the_preset_and_unknown_settings(tmp_path):
    from reelsift import config as config_mod

    (tmp_path / config_mod.TRIP_CONFIG).write_text(
        '[trip]\nname = "Lake"\nfootage = []\npreset = "small"\n[notes]\nwho = "me"\n',
        encoding="utf-8",
    )
    cfg = config_mod.load(tmp_path, user_path=tmp_path / "none.toml")
    cfg.music = ["song.mp3"]
    config_mod.write_trip(tmp_path, cfg)
    again = config_mod.load(tmp_path, user_path=tmp_path / "none.toml")
    assert again.preset == "small" and again.music == ["song.mp3"]
    assert 'who = "me"' in (tmp_path / config_mod.TRIP_CONFIG).read_text(encoding="utf-8")
    again.music = []
    config_mod.write_trip(tmp_path, again)
    assert "render" not in (tmp_path / config_mod.TRIP_CONFIG).read_text(encoding="utf-8")


def test_very_short_music_has_no_beats_instead_of_crashing():
    import numpy as np

    from reelsift import music

    env, rate = music.onset_envelope(np.zeros(3000, np.float32))
    assert len(env) >= 1
    period = music.estimate_period(env, rate)
    assert len(music.track_beats(env, period)) <= 1


def test_clip_sizing_and_growth_stop_at_dropped_footage():
    import numpy as np

    from reelsift.select import clip_interval, clip_length, expand, snap_to_beats
    from reelsift.tripdata import Candidates, Recording, TripData

    data = TripData(
        recordings={1: Recording(1, 1, "cam", "video", 20.0, 1e6, 0.8, 120)},
        observations={},
        embeddings={},
        candidates=Candidates(
            *(np.array([v]) for v in (1, 1, 0.0, 4.0, 1.0, 3.0, 0.0, 7.0)),
            source=np.array(["window"], dtype=object),
        ),
    )
    dropped = {1: [(3.0, 4.0)]}
    _start, end = clip_interval(data, 0, clip_length(600), dropped)
    assert end <= 3.0, "a long edit's clip would otherwise reach 4.99 s"
    grown = expand(Item(1, 1.0, 3.0, 1, 0.0, None, "window"), 6.0, data, [], dropped)
    assert grown.end <= 3.0
    snapped = snap_to_beats([Item(1, 1.0, 2.9, 1, 0.0, None, "window")], np.array([2.2]), data,
                            forbidden=dropped)  # fmt: skip
    assert snapped[0].end <= 3.0
