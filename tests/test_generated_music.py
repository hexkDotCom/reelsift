"""Generated music: the worker and its protocol, keeping and choosing tracks, soundtracks across
edit revisions, the mix levels, fitting cuts, export, and the GPU lock. ACE-Step itself is
replaced by a stub that writes click tracks, so none of this needs a GPU."""

import json
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from reelsift import config as config_mod
from reelsift import gpu, music, musicgen, soundtrack
from reelsift.project import Trip
from reelsift.review import Review
from reelsift.tools import Cancelled, cancel, cancellable, stream

from .conftest import make_music, needs_ffmpeg
from .test_review import analyzed_root  # noqa: F401  (module fixture reused here)

PROFILE = musicgen.Profile("test", "dit", None, (), 0, 600, 0, True)


@pytest.fixture(autouse=True)
def own_gpu_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("REELSIFT_LOCK_DIR", str(tmp_path / "locks"))


@pytest.fixture
def review(analyzed_root, tmp_path):  # noqa: F811
    root = tmp_path / "trip"
    shutil.copytree(analyzed_root, root, ignore=shutil.ignore_patterns("reelsift.lock"))
    with Trip(root, write=True) as trip:
        r = Review(trip)
        r.select_all()
        yield r


def fake_worker(short_by: float = 3.0, fail_seed: int | None = None, calls: list | None = None):
    """Stands in for run_worker: click tracks at the requested tempo, ending short_by seconds
    before the requested duration, as ACE-Step's do."""

    def run(profile, candidates, out, log, on_event=None):
        events = [{"event": "ready", "seconds": 0.1, "vram_gb": 0}]
        out.mkdir(parents=True, exist_ok=True)
        for c in candidates:
            if calls is not None:
                calls.append(c)
            if c["seed"] == fail_seed:
                e = {"event": "error", "stage": "generating", "id": c["id"], "message": "boom"}
            else:
                path = make_music(out / f"{c['id']}.wav", bpm=c["bpm"],
                                  seconds=max(5.0, c["duration"] - short_by))  # fmt: skip
                e = {"event": "candidate", "id": c["id"], "path": str(path), "seconds": 0.1,
                     "seed": c["seed"]}  # fmt: skip
            events.append(e)
            if on_event:
                on_event(e)
        events.append({"event": "done"})
        return events

    return run


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(musicgen, "installed", lambda: PROFILE)
    monkeypatch.setattr(musicgen, "run_worker", fake_worker())


# --- the worker process ----------------------------------------------------------------------


def stub_acestep(folder: Path) -> Path:
    """A stand-in for ACE-Step's Python package: it prints to standard output (as the real one
    does) and writes a short WAV per generation; seed 13 fails."""
    pkg = folder / "acestep"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "handler.py").write_text(
        textwrap.dedent("""
        class AceStepHandler:
            def initialize_service(self, **kw):
                print("loading the DiT, chatty")
                return "ok", True
    """)
    )
    (pkg / "llm_inference.py").write_text(
        textwrap.dedent("""
        class LLMHandler:
            def initialize(self, **kw):
                return "ok", True
    """)
    )
    (pkg / "inference.py").write_text(
        textwrap.dedent("""
        import os, struct, wave
        from dataclasses import dataclass, field

        @dataclass
        class GenerationParams:
            task_type: str = ""; caption: str = ""; lyrics: str = ""; instrumental: bool = False
            bpm: int = None; keyscale: str = ""; timesignature: str = ""; duration: float = 10
            thinking: bool = True; inference_steps: int = 8; shift: float = 1.0; seed: int = -1

        @dataclass
        class GenerationConfig:
            batch_size: int = 1; use_random_seed: bool = True; seeds: list = None
            audio_format: str = "wav"

        @dataclass
        class Result:
            success: bool; audios: list = field(default_factory=list); error: str = ""
            status_message: str = ""

        def generate_music(dit, llm, params, config, save_dir):
            print("generating... (stdout noise)")
            if params.seed == 13:
                return Result(False, error="out of memory")
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, "x.wav")
            with wave.open(path, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
                w.writeframes(struct.pack("<h", 1000) * int(8000 * params.duration))
            return Result(True, [{"path": path}])
    """)
    )
    torch = folder / "torch"
    torch.mkdir()
    (torch / "__init__.py").write_text(
        textwrap.dedent("""
        class cuda:
            @staticmethod
            def max_memory_allocated():
                return 0
    """)
    )
    return folder


def test_the_worker_speaks_only_its_protocol_on_stdout(tmp_path):
    source = stub_acestep(tmp_path / "src")
    request = {"protocol": 1, "job": "j", "source": str(source), "checkpoints": str(tmp_path),
               "dit": "d", "lm": None, "out": str(tmp_path / "out"),
               "candidates": [{"id": "a", "caption": "x", "bpm": 120, "duration": 1, "seed": 1},
                              {"id": "b", "caption": "x", "bpm": 120, "duration": 1, "seed": 13}]}  # fmt: skip
    lines = []
    worker = Path(musicgen.__file__).with_name("music_worker.py")
    code = stream([sys.executable, "-I", str(worker)], json.dumps(request), lines.append,
                  tmp_path / "worker.log")  # fmt: skip
    events = [json.loads(line) for line in lines]  # every line is protocol: no noise
    assert code == 0
    assert [e["event"] for e in events] == ["stage", "ready", "stage", "candidate", "stage",
                                            "error", "done"]  # fmt: skip
    assert Path(events[3]["path"]).is_file() and events[5]["id"] == "b"
    assert "chatty" in (tmp_path / "worker.log").read_text(encoding="utf-8")


def test_the_worker_refuses_another_protocol(tmp_path):
    lines = []
    worker = Path(musicgen.__file__).with_name("music_worker.py")
    code = stream([sys.executable, "-I", str(worker)], json.dumps({"protocol": 2}), lines.append,
                  tmp_path / "w.log")  # fmt: skip
    assert code == 2 and json.loads(lines[0])["stage"] == "request"


def test_cancelling_kills_the_worker_promptly(tmp_path):
    event = threading.Event()
    script = tmp_path / "slow.py"
    script.write_text("import time\nprint('started', flush=True)\ntime.sleep(60)\n")
    started = time.monotonic()
    threading.Timer(0.5, lambda: cancel(event)).start()
    with pytest.raises(Cancelled), cancellable(event):
        stream([sys.executable, str(script)], "", lambda _: None, tmp_path / "s.log")
    assert time.monotonic() - started < 15


def test_a_silent_worker_is_stopped(tmp_path):
    script = tmp_path / "quiet.py"
    script.write_text("import time\ntime.sleep(60)\n")
    with pytest.raises(RuntimeError, match="stopped responding"):
        stream([sys.executable, str(script)], "", lambda _: None, tmp_path / "q.log", stall_s=1)


# --- generating, keeping and choosing --------------------------------------------------------


@needs_ffmpeg
def test_generated_tracks_are_kept_and_the_best_plays(review, fake_model):
    trip = review.trip
    report = musicgen.generate(trip, 8, candidates=2, seed=5, say=lambda _: None, data=review.data)
    assert len(report.made) == 2 and report.assigned in report.made
    a = soundtrack.asset(trip, report.assigned)
    assert a.path.is_file() and a.path.suffix == ".flac" and a.path.parent == trip.root / "music"
    assert a.request["seed"] in (5, 6) and "instrumental, no vocals" in a.request["caption"]
    st = soundtrack.get(trip, review.edit(8).id)
    assert st.source == "generated" and st.asset_id == report.assigned
    assert st.fit["end"] <= musicgen.edit_duration(trip, review.edit(8).items) + 1e-6
    assert not (trip.root / "music" / ".partial").exists() or not any(
        (trip.root / "music" / ".partial").iterdir())  # fmt: skip


@needs_ffmpeg
def test_music_made_for_an_older_revision_is_kept_but_not_played(review, monkeypatch):
    trip = review.trip
    monkeypatch.setattr(musicgen, "installed", lambda: PROFILE)
    inner = fake_worker()

    def worker_while_editing(*args, **kw):
        review.refill(8)  # the user changes the edit while the music is being made
        return inner(*args, **kw)

    monkeypatch.setattr(musicgen, "run_worker", worker_while_editing)
    report = musicgen.generate(trip, 8, candidates=1, say=lambda _: None, data=review.data)
    assert report.made and report.assigned is None and "changed" in report.note
    assert soundtrack.get(trip, review.edit(8).id).source != "generated"


@needs_ffmpeg
def test_a_failed_candidate_doesnt_lose_the_others(review, monkeypatch):
    monkeypatch.setattr(musicgen, "installed", lambda: PROFILE)
    monkeypatch.setattr(musicgen, "run_worker", fake_worker(fail_seed=10))
    report = musicgen.generate(review.trip, 8, candidates=2, seed=10, say=lambda _: None,
                               data=review.data)  # fmt: skip
    assert len(report.made) == 1 and report.errors == ["boom"]


def test_generating_without_the_model_says_how_to_install(review, monkeypatch):
    monkeypatch.setattr(musicgen, "installed", lambda: None)
    with pytest.raises(musicgen.MusicUnavailable, match="reelsift music install"):
        musicgen.generate(review.trip, 8, say=lambda _: None)


def test_edits_longer_than_the_model_can_make_are_refused(review, monkeypatch):
    monkeypatch.setattr(musicgen, "installed", lambda: replace(PROFILE, max_duration=2))
    with pytest.raises(musicgen.MusicUnavailable, match="at most 2 s"):
        musicgen.generate(review.trip, 8, say=lambda _: None)


# --- soundtracks across revisions ------------------------------------------------------------


@needs_ffmpeg
def test_new_revisions_keep_the_music_and_restore_brings_back_the_old(review, fake_model):
    trip = review.trip
    first = musicgen.generate(trip, 8, candidates=2, seed=1, say=lambda _: None, data=review.data)
    other = next(i for i in first.made if i != first.assigned)
    review.set_mix(8, music=-12.0)
    old_edit = review.edit(8).id
    review.refill(8)
    st = soundtrack.get(trip, review.edit(8).id)
    assert (st.asset_id, st.mix.music) == (first.assigned, -12.0), "inherited"
    review.use_track(8, other)
    review.set_mix(8, music=0.0)
    review.restore(8, old_edit)
    st = soundtrack.get(trip, review.edit(8).id)
    assert (st.asset_id, st.mix.music) == (first.assigned, -12.0), "restored with the revision"


def test_older_revisions_without_a_soundtrack_use_the_trips_music_files(review, tmp_path):
    song = make_music(tmp_path / "s.wav", seconds=5)
    review.trip.config.music = [str(song)]
    st = soundtrack.get(review.trip, review.edit(8).id)
    assert st.source == "files" and not st.stored and st.files == [str(song)]
    review.trip.config.music = []
    assert soundtrack.get(review.trip, review.edit(8).id).source == "none"


@needs_ffmpeg
def test_a_missing_generated_track_is_reported_not_replaced(review, fake_model):
    report = musicgen.generate(review.trip, 8, candidates=1, say=lambda _: None, data=review.data)
    soundtrack.asset(review.trip, report.assigned).path.unlink()
    with pytest.raises(RuntimeError, match="missing"):
        soundtrack.resolve(review.trip, review.edit(8).id)


# --- the mix ---------------------------------------------------------------------------------


def test_default_levels_are_the_usual_mix():
    from reelsift.render import mix_graph

    graph = mix_graph(10.0, [], 1)
    assert "volume='0.35'" in graph and "volume='0.5'" in graph
    spoken = mix_graph(10.0, [(1.0, 2.0)], 1)
    assert spoken.count("0.12") == 0 or "0.12" in spoken  # music under speech is 0.12
    assert "-0.38" in spoken, "music drops from 0.5 to 0.12 while people speak"


def test_mix_levels_set_the_gains():
    from reelsift.render import mix_graph
    from reelsift.soundtrack import Mix

    graph = mix_graph(10.0, [], 1, Mix(music=0.0, background=-20.0, duck=0.0))
    assert "volume='1.0'" in graph and "volume='0.1'" in graph


def test_no_music_leaves_the_clips_sound_alone():
    from reelsift.render import mix_graph

    assert "volume" not in mix_graph(10.0, [(1.0, 2.0)], 0)


def test_generated_music_plays_once_and_stops_at_its_end():
    from reelsift.render import mix_graph

    resolved = soundtrack.Resolved([music.Track(Path("t.flac"), 30, 120, np.zeros(0))],
                                   soundtrack.Mix(), True, end=8.0, fade=1.5, gain_db=-3.0)  # fmt: skip
    graph = mix_graph(10.0, [], 1, resolved.mix, resolved)
    assert "concat=n=1" in graph and "atrim=0:8.000" in graph
    assert "afade=t=out:st=6.500:d=1.500" in graph and "volume=-3.00dB" in graph


@needs_ffmpeg
def test_a_render_with_generated_music_is_normalized(review, fake_model, tmp_path):
    from reelsift.render import Output, render

    musicgen.generate(review.trip, 8, candidates=1, say=lambda _: None, data=review.data)
    edit = review.edit(8)
    out = render(review.trip, edit.items, tmp_path / "g.mp4", Output(320, 180),
                 music=soundtrack.resolve(review.trip, edit.id))  # fmt: skip
    from .test_music import _loudness

    assert abs(_loudness(out) - (-16)) < 1.5


# --- fitting ----------------------------------------------------------------------------------


def test_the_end_follows_the_tracks_own_ending():
    a = {"sound_end": 29.0, "bpm": 120}
    assert musicgen.fit_end(a, 30.0) == {"end": 29.0, "fade": 0.5, "ends_early": True}
    assert musicgen.fit_end({"sound_end": 30.2, "bpm": 120}, 30.0) == {"end": 30.0, "fade": 0.5}
    assert musicgen.fit_end({"sound_end": 40.0, "bpm": 120}, 30.0) == {"end": 30.0, "fade": 2.0}


def test_what_is_asked_for_follows_the_edit():
    s = config_mod.MusicSettings(genre="techno", mood="dark").over(soundtrack.DEFAULTS)
    short = musicgen.ask(s, "preview", 30.0, np.ones(30), 0.0, 600)
    assert "final hit" in short.caption and "dark" in short.caption and short.duration == 36.0
    assert short.bpm == 133, "a preview sits high in the genre's range"
    story = musicgen.ask(s, "story", 598.0, np.ones(598), 0.7, 600)
    assert "outro" in story.caption and "room for voices" in story.caption
    assert story.duration == 600 and story.bpm == 127
    rising = musicgen.ask(s, "highlights", 120.0, np.linspace(0, 1, 120), 0.0, 600)
    assert "big finish" in rising.caption


def test_snapping_never_moves_a_cut_into_a_word():
    from reelsift.select import Item, snap_to_beats

    from .test_select import trip_data

    data = trip_data(n_recordings=1, duration=100.0)
    item = Item(1, 10.0, 12.2, int(data.candidates.id[4]), 1, None, "window")
    beats = np.arange(0, 20, 0.5)
    assert snap_to_beats([item], beats, data)[0].end == pytest.approx(12.0)
    data.words[1] = np.array([[11.9, 12.1]])  # a word across the beat
    assert snap_to_beats([item], beats, data)[0].end == pytest.approx(12.2)


@needs_ffmpeg
def test_fitting_cuts_again_starts_from_the_unfitted_edit(review, fake_model):
    musicgen.generate(review.trip, 8, candidates=1, say=lambda _: None, data=review.data)
    before = [(i.start, i.end) for i in review.edit(8).items]
    review.fit_cuts(8)
    once = [(i.start, i.end) for i in review.edit(8).items]
    review.fit_cuts(8)
    assert [(i.start, i.end) for i in review.edit(8).items] == once
    st = soundtrack.get(review.trip, review.edit(8).id)
    unfitted = [(i["start"], i["end"]) for i in st.fit["unfitted"]]
    assert st.fit["fitted_cuts"] and unfitted == before


@needs_ffmpeg
def test_beat_confidence_tells_clicks_from_noise(tmp_path):
    clicks = music.analyze_file(make_music(tmp_path / "c.wav", seconds=15))
    noise = tmp_path / "n.wav"
    import wave

    with wave.open(str(noise), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes((np.random.default_rng(1).normal(0, 0.2, 22050 * 15) * 32767)
                      .clip(-32767, 32767).astype(np.int16).tobytes())  # fmt: skip
    assert clicks.confidence > 0.5 > music.analyze_file(noise).confidence


@needs_ffmpeg
def test_tempo_is_measured_between_envelope_frames(tmp_path):
    track = music.analyze_file(make_music(tmp_path / "m.wav", bpm=124, seconds=30))
    assert abs(track.bpm - 124) < 0.6


# --- settings ---------------------------------------------------------------------------------


def test_music_settings_layer_by_length_and_round_trip(tmp_path):
    (tmp_path / "reelsift.toml").write_text(
        textwrap.dedent("""
        [trip]
        name = "t"
        footage = []
        [music]
        genre = "techno"
        [music.mix]
        music = -8.0
        [music.lengths."30"]
        genre = "drum and bass"
        [music.lengths."30".mix]
        duck = -20.0
        [other]
        kept = 1
    """),
        encoding="utf-8",
    )
    cfg = config_mod.load(tmp_path, user_path=tmp_path / "none.toml")
    thirty, three = cfg.music_for(30), cfg.music_for(180)
    assert (thirty.genre, thirty.music, thirty.duck) == ("drum and bass", -8.0, -20.0)
    assert three.genre == "techno" and three.duck == pytest.approx(soundtrack.DEFAULTS.duck)
    config_mod.write_trip(tmp_path, cfg)
    again = config_mod.load(tmp_path, user_path=tmp_path / "none.toml")
    assert again.music_for(30) == thirty and "kept = 1" in (tmp_path / "reelsift.toml").read_text()


def test_bad_mix_levels_are_refused(tmp_path):
    (tmp_path / "reelsift.toml").write_text('[trip]\nname = "t"\nfootage = []\n[music.mix]\n'
                                            'duck = 5.0\n', encoding="utf-8")  # fmt: skip
    with pytest.raises(ValueError, match="duck"):
        config_mod.load(tmp_path, user_path=tmp_path / "none.toml")


# --- export, cleaning, the GPU lock and the app ----------------------------------------------


@needs_ffmpeg
def test_export_puts_the_music_on_its_own_track(review, fake_model, tmp_path):
    import opentimelineio as otio

    from reelsift.export import export

    musicgen.generate(review.trip, 8, candidates=1, say=lambda _: None, data=review.data)
    edit = review.edit(8)
    paths = export(review.trip, edit.items, "e", tmp_path, music=soundtrack.resolve(
        review.trip, edit.id))  # fmt: skip
    wav = tmp_path / "e-music.wav"
    assert wav in paths and wav.stat().st_size > 1000
    timeline = otio.adapters.read_from_file(str(tmp_path / "e.otio"))
    assert [t.name for t in timeline.tracks] == ["Video", "Audio", "Music"]
    assert 'lane="-1"' in (tmp_path / "e.fcpxml").read_text(encoding="utf-8")


@needs_ffmpeg
def test_clean_keeps_generated_music_and_prune_removes_unused(review, fake_model):
    from reelsift.project import clean

    report = musicgen.generate(review.trip, 8, candidates=2, say=lambda _: None, data=review.data)
    clean(review.trip)
    assert all(soundtrack.asset(review.trip, i).path.is_file() for i in report.made)
    assert musicgen.prune(review.trip)[0] == 1
    left = [r["id"] for r in review.trip.db.execute("SELECT id FROM music_assets")]
    assert left == [report.assigned]


def test_the_gpu_lock_is_exclusive_across_processes(tmp_path):
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
        import sys, time
        from reelsift import gpu
        with gpu.hold("the other process"):
            print("held", flush=True)
            time.sleep(30)
    """),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )  # inherits REELSIFT_LOCK_DIR
    try:
        assert holder.stdout.readline().strip() == "held"
        seen = []
        deadline = time.monotonic() + 2
        with pytest.raises(Cancelled):
            with gpu.hold("me", cancelled=lambda: time.monotonic() > deadline,
                          waiting=seen.append):  # fmt: skip
                pass
        assert any(s and "the other process" in s for s in seen)
    finally:
        holder.kill()
        holder.wait()
    with gpu.hold("me now"):  # released when its holder ended
        pass


@needs_ffmpeg
def test_the_app_generates_chooses_and_mixes(review, fake_model):
    from fastapi.testclient import TestClient

    from reelsift.web.app import create_app

    trip = review.trip
    shared = Review(trip.worker(shared=True))
    client = TestClient(create_app(trip, shared, 8765), base_url="http://127.0.0.1:8765")
    edit = client.get("/api/edits/8").json()
    assert edit["music"]["source"] == "none" and edit["music"]["tracks"] == []
    r = client.post("/api/edits/8/music/generate", json={"genre": "synthwave", "candidates": 2,
                                                         "expected_edit_id": edit["edit_id"]})  # fmt: skip
    assert r.status_code == 200, r.text
    for _ in range(200):
        job = client.get("/api/music/job").json()
        if not job["running"]:
            break
        time.sleep(0.1)
    assert job["error"] is None and len(job["made"]) == 2, job
    edit = client.get("/api/edits/8").json()
    m = edit["music"]
    assert m["source"] == "generated" and m["genre"] == "synthwave" and len(m["tracks"]) == 2
    other = next(t["id"] for t in m["tracks"] if t["id"] != m["asset_id"])
    m = client.post("/api/edits/8/music/use", json={"asset_id": other}).json()["music"]
    assert m["asset_id"] == other
    m = client.post("/api/edits/8/music/mix", json={"music": -3}).json()["music"]
    assert m["mix"]["music"] == -3.0
    m = client.post("/api/edits/8/music/mix", json={"reset": True}).json()["music"]
    assert m["mix"]["music"] == pytest.approx(-6.0, abs=0.05)
    m = client.post("/api/edits/8/music/source", json={"source": "none"}).json()["music"]
    assert m["source"] == "none"
    assert client.get(f"/media/music/{other}").status_code == 200
    assert "synthwave" in (trip.root / "reelsift.toml").read_text(encoding="utf-8")


@needs_ffmpeg
def test_rendering_again_with_another_mix_reuses_the_clips(
    review, fake_model, tmp_path, monkeypatch
):
    from reelsift import render as render_mod
    from reelsift.render import Output, render

    musicgen.generate(review.trip, 8, candidates=1, say=lambda _: None, data=review.data)
    edit = review.edit(8)
    render(review.trip, edit.items, tmp_path / "a.mp4", Output(320, 180),
           music=soundtrack.resolve(review.trip, edit.id))  # fmt: skip
    made = []
    real = render_mod.clip_command
    monkeypatch.setattr(render_mod, "clip_command", lambda *a, **k: made.append(1) or real(*a, **k))
    review.set_mix(8, music=-20.0)
    render(review.trip, edit.items, tmp_path / "b.mp4", Output(320, 180),
           music=soundtrack.resolve(review.trip, edit.id))  # fmt: skip
    assert made == [] and (tmp_path / "b.mp4").is_file()


# --- review round 1 ---------------------------------------------------------------------------


@needs_ffmpeg
def test_a_choice_made_while_generating_is_kept(review, monkeypatch):
    monkeypatch.setattr(musicgen, "installed", lambda: PROFILE)
    inner = fake_worker()

    def choose_meanwhile(*args, **kw):
        review.set_source(8, "none")  # the user picks "No music" while it runs
        return inner(*args, **kw)

    monkeypatch.setattr(musicgen, "run_worker", choose_meanwhile)
    report = musicgen.generate(review.trip, 8, candidates=1, say=lambda _: None, data=review.data)
    assert report.made and report.assigned is None
    assert soundtrack.get(review.trip, review.edit(8).id).source == "none"


@needs_ffmpeg
def test_switching_tracks_keeps_the_edit_from_before_fitting(review, fake_model):
    first = musicgen.generate(review.trip, 8, candidates=2, seed=3, say=lambda _: None,
                              data=review.data)  # fmt: skip
    original = [(i.start, i.end) for i in review.edit(8).items]
    review.fit_cuts(8)
    other = next(i for i in first.made if i != first.assigned)
    review.use_track(8, other)
    st = soundtrack.get(review.trip, review.edit(8).id)
    assert [(i["start"], i["end"]) for i in st.fit["unfitted"]] == original


@needs_ffmpeg
def test_undo_keeps_music_chosen_since(review, fake_model):
    item = review.edit(8).items[0]
    review.decide("keep", item.recording_id, item.start, item.end, 8)
    report = musicgen.generate(review.trip, 8, candidates=1, say=lambda _: None, data=review.data)
    review.set_mix(8, music=-15.0)
    review.undo()
    st = soundtrack.get(review.trip, review.edit(8).id)
    assert (st.asset_id, st.mix.music) == (report.assigned, -15.0)


@needs_ffmpeg
def test_a_mix_change_from_a_stale_page_is_refused(review):
    from fastapi.testclient import TestClient

    from reelsift.web.app import create_app

    shared = Review(review.trip.worker(shared=True))
    client = TestClient(create_app(review.trip, shared, 8765), base_url="http://127.0.0.1:8765")
    seen = client.get("/api/edits/8").json()["edit_id"]
    client.post("/api/edits/8/refill")
    r = client.post("/api/edits/8/music/mix", json={"music": -23, "expected_edit_id": seen})
    assert r.status_code == 409


def test_a_failed_reinstall_leaves_the_working_one(tmp_path, monkeypatch):
    monkeypatch.setenv("REELSIFT_MUSIC_DIR", str(tmp_path / "music"))
    working = tmp_path / "music" / "v0.1.8-old"
    (working / "venv" / ("Scripts" if sys.platform == "win32" else "bin")).mkdir(parents=True)
    (working / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")).touch()
    (working / "manifest.json").write_text(json.dumps({"commit": musicgen.ACESTEP["commit"],
                                                        "profile": "xl"}))  # fmt: skip
    (tmp_path / "music" / "active.json").write_text(json.dumps({"slot": working.name}))
    assert musicgen.installed() is musicgen.PROFILES["xl"]

    def broken(target, profile, say):
        target.mkdir(parents=True)
        raise musicgen.MusicUnavailable("uv sync failed")

    monkeypatch.setattr(musicgen, "_install_into", broken)
    monkeypatch.setattr(musicgen, "_checkpoints_present", lambda p: p.disk_gb * 2**30)
    with pytest.raises(musicgen.MusicUnavailable):
        musicgen.install(musicgen.PROFILES["xl"], say=lambda _: None)
    assert musicgen.installed() is musicgen.PROFILES["xl"] and musicgen.install_dir() == working
    assert sorted(p.name for p in (tmp_path / "music").iterdir() if p.is_dir()) == [working.name]


def test_clip_sound_comes_back_after_generated_music_ends():
    from reelsift.render import mix_graph

    resolved = soundtrack.Resolved([music.Track(Path("t.flac"), 30, 120, np.zeros(0))],
                                   soundtrack.Mix(), True, end=30.0, fade=1.0)  # fmt: skip
    longer = mix_graph(60.0, [], 1, resolved.mix, resolved)
    source = longer.split("[src]")[0]
    assert "29.800" in source, "the clips' sound rises back to full as the music ends"
    assert "clip(" not in mix_graph(30.0, [], 1, resolved.mix, resolved).split("[src]")[0]


def test_weak_tracks_in_a_playlist_give_no_beats():
    strong = music.Track(None, 5.0, 120, np.arange(0, 5, 0.5), 0.8)
    weak = music.Track(None, 5.0, 120, np.arange(0.25, 5, 0.5), 0.0)
    grid = music.beat_grid([strong, weak], 10, music.WEAK_BEAT)
    assert grid.max() < 5 and len(grid) == 10


# --- review round 2 ---------------------------------------------------------------------------


def test_a_playlist_without_a_clear_beat_gives_no_beats():
    weak = music.Track(None, 5.0, 120, np.arange(0.25, 5, 0.5), 0.0)
    strong = music.Track(None, 5.0, 120, np.arange(0, 5, 0.5), 0.8)
    assert len(music.beat_grid([weak], 10, music.WEAK_BEAT)) == 0
    assert music.beat_grid([weak, strong], 4, music.WEAK_BEAT).size == 0


def strong_files(review, tmp_path):
    song = make_music(tmp_path / "song.wav", bpm=120, seconds=40)
    review.trip.config.music = [str(song)]
    review.set_source(8, "files")


@needs_ffmpeg
def test_fitting_again_after_a_trim_keeps_the_trim(review, tmp_path):
    strong_files(review, tmp_path)
    review.fit_cuts(8)
    item = review.edit(8).items[0]
    review.decide("keep", item.recording_id, item.start, item.end - 0.3, 8,
                  replaces=(item.start, item.end))  # fmt: skip
    trimmed = next(i for i in review.edit(8).items if i.recording_id == item.recording_id
                   and abs(i.start - item.start) < 1e-3)  # fmt: skip
    review.fit_cuts(8)
    again = next(i for i in review.edit(8).items if i.recording_id == item.recording_id
                 and abs(i.start - item.start) < 1e-3)  # fmt: skip
    assert again.locked and again.end == pytest.approx(trimmed.end)


@needs_ffmpeg
def test_fitting_again_keeps_locks_and_crops_made_since(review, fake_model):
    musicgen.generate(review.trip, 8, candidates=1, say=lambda _: None, data=review.data)
    review.fit_cuts(8)
    first, second = review.edit(8).items[:2]
    review.decide("keep", first.recording_id, first.start, first.end, 8)
    review.set_crop(8, second.recording_id, second.start, second.end, 0.9)
    review.fit_cuts(8)
    by_start = {(i.recording_id, round(i.start, 3)): i for i in review.edit(8).items}
    kept = by_start[(first.recording_id, round(first.start, 3))]
    cropped = by_start[(second.recording_id, round(second.start, 3))]
    assert kept.locked and kept.end == pytest.approx(first.end)
    assert cropped.crop_x == pytest.approx(0.9)


@needs_ffmpeg
def test_a_choice_racing_the_assignment_waits_and_wins(review, monkeypatch):
    from reelsift.project import transaction

    monkeypatch.setattr(musicgen, "installed", lambda: PROFILE)
    monkeypatch.setattr(musicgen, "run_worker", fake_worker())
    real_use = musicgen.use
    other = review.trip.worker(shared=True)
    edit_id = review.edit(8).id
    errors = []

    def choose():
        try:
            with transaction(other.db):
                st = soundtrack.get(other, edit_id)
                st.source, st.asset_id, st.fit = "none", None, {}
                soundtrack.put(other, st)
                musicgen.chosen(other, 8)
        except Exception as e:
            errors.append(e)

    def racing_use(*args, **kw):
        t = threading.Thread(target=choose)
        t.start()
        t.join(0.5)  # it can't write while the assignment holds the write lock
        assert t.is_alive()
        real_use(*args, **kw)
        racing_use.thread = t

    monkeypatch.setattr(musicgen, "use", racing_use)
    musicgen.generate(review.trip, 8, candidates=1, say=lambda _: None, data=review.data)
    racing_use.thread.join(10)
    other.close()
    assert not errors
    assert soundtrack.get(review.trip, review.edit(8).id).source == "none", "the later choice"


def test_the_source_setting_overrides_the_music_files(review, tmp_path):
    song = make_music(tmp_path / "s.wav", seconds=5)
    review.trip.config.music = [str(song)]
    review.trip.config.music_settings = config_mod.MusicSettings(source="none")
    assert soundtrack.get(review.trip, review.edit(8).id).source == "none"
    review.trip.config.music_lengths[8] = config_mod.MusicSettings(source="files")
    assert soundtrack.get(review.trip, review.edit(8).id).source == "files"


def test_saving_a_trip_doesnt_copy_the_users_music_defaults(tmp_path):
    user = tmp_path / "user.toml"
    user.write_text('[music]\ngenre = "house"\n[music.mix]\nmusic = -8.0\n', encoding="utf-8")
    (tmp_path / "reelsift.toml").write_text('[trip]\nname = "t"\nfootage = []\n[music]\n'
                                            'mood = "dark"\n', encoding="utf-8")  # fmt: skip
    cfg = config_mod.load(tmp_path, user_path=user)
    assert (cfg.music_for(30).genre, cfg.music_for(30).mood) == ("house", "dark")
    config_mod.write_trip(tmp_path, cfg)
    user.write_text('[music]\ngenre = "ambient"\n', encoding="utf-8")
    again = config_mod.load(tmp_path, user_path=user)
    assert (again.music_for(30).genre, again.music_for(30).mood) == ("ambient", "dark")
    assert again.music_for(30).music == pytest.approx(soundtrack.DEFAULTS.music)


# --- review round 3 ---------------------------------------------------------------------------


@needs_ffmpeg
def test_a_choice_made_while_a_generation_starts_up_wins(review, monkeypatch):
    monkeypatch.setattr(musicgen, "installed", lambda: PROFILE)
    monkeypatch.setattr(musicgen, "run_worker", fake_worker())
    started = musicgen.begin(review.trip, 8)  # the app accepts the request...
    review.set_source(8, "none")  # ...and the user chooses before the job looks at the edit
    report = musicgen.generate(review.trip, 8, candidates=1, say=lambda _: None,
                               data=review.data, started=started)  # fmt: skip
    assert report.made and report.assigned is None
    assert soundtrack.get(review.trip, review.edit(8).id).source == "none"


def test_a_generation_for_a_revision_no_longer_shown_is_refused(review):
    with pytest.raises(ValueError, match="changed"):
        musicgen.begin(review.trip, 8, expected_edit_id=review.edit(8).id - 1)


def test_generated_isnt_a_source_setting(tmp_path):
    (tmp_path / "reelsift.toml").write_text('[trip]\nname = "t"\nfootage = []\n[music]\n'
                                            'source = "generated"\n', encoding="utf-8")  # fmt: skip
    with pytest.raises(ValueError, match="chosen per edit"):
        config_mod.load(tmp_path, user_path=tmp_path / "none.toml")


def test_music_set_and_clear_hold_for_edits_not_selected_yet(tmp_path):
    from typer.testing import CliRunner

    from reelsift.cli import app

    song = make_music(tmp_path / "s.wav", seconds=3)
    root = tmp_path / "trip"
    with Trip.create(root, [tmp_path]) as trip:
        trip.config.music_settings = config_mod.MusicSettings(source="none")
        trip.config.music_lengths[30] = config_mod.MusicSettings(source="none")
        config_mod.write_trip(root, trip.config)
    runner = CliRunner()
    assert runner.invoke(app, ["music", "set", str(root), str(song)]).exit_code == 0
    cfg = config_mod.load(root, user_path=tmp_path / "none.toml")
    assert cfg.music_for(30).source == "files" and cfg.music_for(180).source == "files"
    assert runner.invoke(app, ["music", "clear", str(root), "-l", "30"]).exit_code == 0
    cfg = config_mod.load(root, user_path=tmp_path / "none.toml")
    assert cfg.music_for(30).source == "none" and cfg.music_for(180).source == "files"


def test_a_worker_that_closes_its_output_but_hangs_is_stopped(tmp_path):
    script = tmp_path / "hang.py"
    script.write_text("import os, time\nos.close(1)\ntime.sleep(60)\n")
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="didn't exit"):
        stream([sys.executable, str(script)], "", lambda _: None, tmp_path / "h.log", stall_s=1)
    assert time.monotonic() - started < 20


def test_what_a_worker_started_dies_with_it(tmp_path):
    marker = tmp_path / "child.pid"
    script = tmp_path / "parent.py"
    script.write_text(
        textwrap.dedent(f"""
        import subprocess, sys
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open({str(marker)!r}, "w").write(str(child.pid))
    """)
    )  # the parent exits at once; its child keeps the output pipe open
    with pytest.raises(RuntimeError):
        stream([sys.executable, str(script)], "", lambda _: None, tmp_path / "p.log", stall_s=1)
    pid = int(marker.read_text())
    deadline = time.monotonic() + 10
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(pid)


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # query limited info
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    import os

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
