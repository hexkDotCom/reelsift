"""Hardening: camera identity, fallbacks, caches, lifecycle, invariants and unusual media."""

import os
import random
import shutil
import threading
import time

import numpy as np
import pytest

from reelsift import candidates, proxies
from reelsift.analysis import analyze
from reelsift.ingest import ingest
from reelsift.media import probe
from reelsift.project import Trip
from reelsift.review import BrokenEdits, Review
from reelsift.sync import sync
from reelsift.tools import filter_available, run, tool

from .conftest import make_clip, needs_ffmpeg
from .fakes import FakeEmbedder, FakeTranscriber
from .test_review import LENGTHS, analyzed_root  # noqa: F401  (module fixture reused here)

pytestmark = needs_ffmpeg


def prepared(root, footage, analyzed=False) -> Trip:
    trip = Trip.create(root / "trip", [footage]).__enter__()
    ingest(trip)
    proxies.generate(trip)
    sync(trip)
    if analyzed:
        analyze(trip, FakeEmbedder, FakeTranscriber, say=lambda _: None)
        candidates.generate(trip)
    return trip


@pytest.fixture
def review(analyzed_root, tmp_path):  # noqa: F811
    root = tmp_path / "trip"
    shutil.copytree(analyzed_root, root, ignore=shutil.ignore_patterns("reelsift.lock"))
    with Trip(root, write=True) as trip:
        yield Review(trip)


# --- camera identity and configuration ------------------------------------------------------


def test_same_folder_name_from_two_people_stays_two_cameras(tmp_path):
    footage = tmp_path / "footage"
    for person, when in (("Anna", "07:51:34"), ("Ben", "07:52:10")):
        make_clip(footage / person / "DJI_001" / "DJI_20261011095133_0001_D.MP4", 2.0,
                  creation_time=f"2026-10-11T{when}Z")  # fmt: skip
    trip = prepared(tmp_path, footage)
    labels = sorted(r["label"] for r in trip.db.execute("SELECT label FROM cameras"))
    assert labels == ["DJI (Anna/DJI_001)", "DJI (Ben/DJI_001)"]
    trip.__exit__(None, None, None)


@pytest.mark.parametrize("bad", ["+25:00", "+02:75", "-13:00", "nonsense"])
def test_impossible_time_zones_are_rejected(tmp_path, bad):
    from reelsift import config

    (tmp_path / config.TRIP_CONFIG).write_text(f'[cameras."Cam"]\nzone = "{bad}"\n')
    with pytest.raises(ValueError):
        config.load(tmp_path, user_path=tmp_path / "none.toml")


# --- fallbacks, lazy models, running out of memory -------------------------------------------


def test_proxy_generation_falls_back_to_x264(tmp_path, monkeypatch):
    from reelsift import proxies as proxies_mod

    footage = tmp_path / "footage"
    make_clip(footage / "cam" / "a.mp4", 2.0, creation_time="2026-10-11T07:51:34Z")
    real = proxies_mod.command
    used = []

    def flaky(source, out, gpu_decode=False, cpu=False):
        used.append(cpu)
        if not cpu:  # as if NVENC refused this file
            return [
                tool("ffmpeg"),
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "nullsrc",
                "-c:v",
                "no_such_codec",
                "-f",
                "null",
                "-",
            ]
        return real(source, out, gpu_decode, cpu)

    monkeypatch.setattr(proxies_mod, "command", flaky)
    trip = prepared(tmp_path, footage)
    assert used[-1] is True and trip.db.execute("SELECT COUNT(*) FROM proxies").fetchone()[0] == 1
    trip.__exit__(None, None, None)


def test_models_load_only_when_there_is_work(tmp_path):
    footage = tmp_path / "footage"
    make_clip(footage / "cam" / "a.mp4", 2.0, creation_time="2026-10-11T07:51:34Z")
    trip = prepared(tmp_path, footage, analyzed=True)
    loads = []

    def embedder():
        loads.append("embedder")
        return FakeEmbedder()

    def transcriber():
        loads.append("transcriber")
        return FakeTranscriber()

    embedder.model_name = FakeEmbedder.name
    transcriber.model_name = FakeTranscriber.name
    analyze(trip, embedder, transcriber, say=lambda _: None)
    assert loads == [], "nothing to redo, so no model is loaded"
    trip.__exit__(None, None, None)


def test_running_out_of_memory_stops_the_job(review):
    from reelsift.jobs import Job, is_out_of_memory

    class OutOfMemoryError(RuntimeError):
        pass

    calls = []

    def work(item):
        calls.append(item)
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    with pytest.raises(OutOfMemoryError) as caught:
        Job(review.trip, "test", {}, ["a", "b", "c"]).run(work)
    assert calls == ["a"], "the remaining items are not tried"
    assert is_out_of_memory(caught.value)
    assert not is_out_of_memory(ValueError("bad file"))


# --- compatible embeddings, cleaning ---------------------------------------------------------


def test_embeddings_from_another_model_are_ignored(review):
    from reelsift.tripdata import load

    rec = next(iter(review.data.embeddings))
    review.trip.db.execute(
        "UPDATE embeddings SET model = 'other-model' WHERE recording_id = ?", (rec,)
    )
    data = load(review.trip)
    assert rec not in data.embeddings
    review.reload()  # scoring and selection still work
    review.select_all()
    assert not review.problems()


def test_clean_frees_rebuildable_caches(review, tmp_path):
    from reelsift.project import clean
    from reelsift.render import Output, render

    edit = review.edit(4)
    render(review.trip, edit.items, tmp_path / "a.mp4", Output(320, 180))
    assert any((review.trip.root / "cache" / "render").rglob("*.mp4"))
    freed = clean(review.trip)
    assert freed["render"] > 0
    assert not (review.trip.root / "cache" / "render").exists()
    assert (review.trip.root / "cache" / "proxies").exists(), "proxies kept unless asked"
    render(review.trip, edit.items, tmp_path / "b.mp4", Output(320, 180))  # rebuilds
    clean(review.trip, proxies=True)
    assert proxies.pending(review.trip), "deleted proxies are due again"


# --- background work ------------------------------------------------------------------------


def test_closing_the_app_stops_a_render(review, monkeypatch):
    from fastapi.testclient import TestClient

    from reelsift import render as render_mod
    from reelsift.web.app import create_app

    started = threading.Event()

    def slow_render(trip, items, dest, out, progress, music=None):
        started.set()
        for n in range(1000):
            progress(n, 1000, str(n))
            time.sleep(0.01)

    monkeypatch.setattr(render_mod, "render", slow_render)
    shared = Review(review.trip.worker(shared=True))
    app = create_app(review.trip, shared, 8765)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        assert client.post("/api/render", json={"length": 8}).status_code == 200
        assert started.wait(10)
    renders = app.state.renders
    assert not renders.thread.is_alive(), "shutdown waited for the render to stop"
    assert "cancelled" in (renders.state.get("error") or "")
    shared.trip.close()


def test_reading_preferences_does_not_block_the_app(review):
    from fastapi.testclient import TestClient

    from reelsift.web.app import create_app

    release = threading.Event()
    entered = threading.Event()

    def slow_model(prompt):
        entered.set()
        release.wait(20)
        return '{"more": {"water": 2}}'

    shared = Review(review.trip.worker(shared=True), None, slow_model)
    client = TestClient(create_app(review.trip, shared, 8765), base_url="http://127.0.0.1:8765")
    worker = threading.Thread(
        target=lambda: client.post("/api/preferences/8/text", json={"text": "more water"})
    )
    worker.start()
    assert entered.wait(10)
    t0 = time.perf_counter()
    assert client.get("/api/edits/8").status_code == 200
    assert time.perf_counter() - t0 < 5, "the edit loads while the model is still thinking"
    release.set()
    worker.join(30)
    assert shared.prefs[8].adjustments == {"water": 2.0}
    shared.trip.close()


# --- rules every change keeps ------------------------------------------------------------


def test_a_change_that_breaks_the_rules_is_rolled_back(review, monkeypatch):
    item = review.edit(8).items[0]
    before = review.trip.db.execute("SELECT COUNT(*) FROM edits").fetchone()[0]

    def broken_place(length, items, rec, start, end, replaces, dropped=()):
        return [*items, *items]  # every clip twice: overlapping

    monkeypatch.setattr(review, "_place", broken_place)
    with pytest.raises(BrokenEdits):
        review.decide("keep", item.recording_id, item.start, item.end, length=8)
    assert review.trip.db.execute("SELECT COUNT(*) FROM edits").fetchone()[0] == before
    assert not review.decisions()


@pytest.mark.parametrize(
    "seed", [int(s) for s in os.environ.get("REELSIFT_SEEDS", "7 11 23 42").split()]
)
def test_random_sequences_of_changes_keep_the_rules(review, seed):
    rng = random.Random(seed)
    rec_ids = list(review.data.recordings)
    for step in range(150):
        action = rng.choice(["keep", "drop", "force", "trim", "refill", "select", "undo",
                             "restore", "crop"])  # fmt: skip
        n = rng.choice(LENGTHS)
        items = review.edit(n).items
        try:
            if action in ("keep", "drop", "trim", "crop") and items:
                i = rng.choice(items)
                if action == "keep":
                    review.decide("keep", i.recording_id, i.start, i.end, length=n)
                elif action == "drop":
                    everywhere = rng.random() < 0.3
                    review.decide("drop", i.recording_id, i.start, i.end,
                                  length=None if everywhere else n)  # fmt: skip
                elif action == "trim":
                    d = rng.choice([-0.5, 0.5])
                    review.decide("keep", i.recording_id, i.start, max(i.start + 0.6, i.end + d),
                                  length=n, replaces=(i.start, i.end))  # fmt: skip
                else:
                    review.set_crop(n, i.recording_id, i.start, i.end, rng.random())
            elif action == "force":
                rec = rng.choice(rec_ids)
                duration = review.data.recordings[rec].duration
                a = rng.uniform(0, max(0.0, duration - 1.0))
                review.decide("force", rec, a, min(duration, a + rng.uniform(0.5, 2.0)), length=n)
            elif action == "refill":
                review.refill(n)
            elif action == "select":
                review.select(n)
            elif action == "undo":
                review.undo()
            elif action == "restore":
                revisions = review.revisions(n)
                review.restore(n, rng.choice(revisions)["id"])
        except ValueError:
            continue  # an interval the change refuses (too short, outside the recording)
        assert review.problems() == [], f"step {step}: {action} broke the edits"


# --- unusual media --------------------------------------------------------------------------


@pytest.mark.skipif(not filter_available("zscale"), reason="FFmpeg without zscale")
def test_hdr_video_is_tone_mapped_to_bt709(tmp_path):
    from reelsift.render import Output, render
    from reelsift.select import Item

    footage = tmp_path / "footage"
    footage.mkdir()
    hdr = footage / "hdr.mp4"
    # Tags go on the frames: FFmpeg 7 and later take them from there, not from output options.
    tags = "setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc"
    run([tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25",
         "-t", "2", "-vf", tags, "-c:v", "libx264", "-pix_fmt", "yuv420p10le", "-metadata",
         "creation_time=2026-10-11T07:51:34Z", str(hdr)])  # fmt: skip
    trip = prepared(tmp_path, footage)
    assert trip.db.execute("SELECT hdr FROM recordings").fetchone()["hdr"] == 1
    row = trip.db.execute("SELECT path FROM proxies").fetchone()
    p = probe(proxies.resolve(trip, row["path"])).main_video
    assert (p.color_transfer, p.color_primaries) == ("bt709", "bt709")
    rec = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
    out = render(trip, [Item(rec, 0.0, 2.0, -1, 0.0, None, "window")], tmp_path / "o.mp4",
                 Output(320, 180))  # fmt: skip
    v = probe(out).main_video
    assert v.color_transfer == "bt709" and v.frames == 50
    trip.__exit__(None, None, None)


def test_variable_frame_rate_video_renders_whole_frames(tmp_path):
    from reelsift.render import Output, render
    from reelsift.select import Item

    footage = tmp_path / "footage"
    footage.mkdir()
    vfr = footage / "vfr.mp4"
    # 30 fps for the first second, then every other frame (15 fps).
    run([tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30",
         "-t", "3", "-vf", "select='lt(n\\,30)+not(mod(n\\,2))'", "-fps_mode", "vfr",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-metadata",
         "creation_time=2026-10-11T07:51:34Z", str(vfr)])  # fmt: skip
    assert probe(vfr).main_video.vfr
    trip = prepared(tmp_path, footage)
    rec = trip.db.execute("SELECT id, vfr FROM recordings").fetchone()
    assert rec["vfr"] == 1
    out = render(trip, [Item(rec["id"], 0.5, 2.5, -1, 0.0, None, "window")], tmp_path / "o.mp4",
                 Output(320, 180, fps="30"))  # fmt: skip
    assert probe(out).main_video.frames == 60
    trip.__exit__(None, None, None)


def test_a_damaged_file_does_not_stop_the_rest(tmp_path):
    footage = tmp_path / "footage"
    make_clip(footage / "cam" / "good.mp4", 2.0, creation_time="2026-10-11T07:51:34Z")
    (footage / "cam" / "broken.mp4").write_bytes(np.random.default_rng(0).bytes(50_000))
    good = make_clip(tmp_path / "src.mp4", 4.0, creation_time="2026-10-11T07:55:00Z")
    data = good.read_bytes()
    (footage / "cam" / "truncated.mp4").write_bytes(data[: len(data) // 2])
    trip = prepared(tmp_path, footage, analyzed=True)
    issues = [r["message"] for r in trip.db.execute("SELECT message FROM issues")]
    assert any("Could not read" in m for m in issues)
    assert (
        trip.db.execute("SELECT COUNT(*) FROM stage_runs WHERE stage = 'visual'").fetchone()[0] >= 1
    )
    trip.__exit__(None, None, None)
