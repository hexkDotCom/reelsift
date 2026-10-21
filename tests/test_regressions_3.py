"""Regression tests: resumed ingest, stream start times, drops, cameras in several footage roots, damaged caches, responsiveness."""

import shutil
import threading

import numpy as np
import pytest

from reelsift.project import Trip
from reelsift.select import Item

from .conftest import make_photo, needs_ffmpeg
from .test_regressions import client, review, walk  # noqa: F401  (fixtures reused here)
from .test_review import analyzed_root  # noqa: F401

# --- resumed ingest ---------------------------------------------------------------------------


def test_a_file_changed_after_an_unfinished_ingest_is_read_again(tmp_path):
    from reelsift.ingest import ingest
    from reelsift.media import fingerprint

    footage = tmp_path / "footage"
    a = make_photo(footage / "a.jpg", color=(10, 20, 30))
    (footage / "b.jpg").write_bytes(b"not a picture")
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        assert ingest(trip).failed  # b.jpg keeps the job unfinished
        make_photo(a, color=(200, 30, 30), size=(500, 300))
        assert ingest(trip).failed
        row = trip.db.execute("SELECT fingerprint FROM assets WHERE path = ?", (str(a),)).fetchone()
        assert row["fingerprint"] == fingerprint(a)
        assert trip.db.execute("SELECT width FROM recordings").fetchone()["width"] == 500


# --- audio and video that start at different times --------------------------------------------


def _render_piece(tmp_path, source, start, end):
    from reelsift.ingest import ingest
    from reelsift.render import Output, clip_command, pieces
    from reelsift.tools import run

    with Trip.create(tmp_path / "trip", [source.parent]) as trip:
        ingest(trip)
        rec = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
        (piece,) = pieces(trip, Item(rec, start, end, -1, 0.0, None, "window"))
    out = tmp_path / "cut.mp4"
    run(clip_command(piece, Output(160, 90), piece.fps, out, gpu_decode=False, cpu=True))
    return out


def _frame(path, n):
    from reelsift.tools import run, tool

    raw = run([tool("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:v:0",
               "-vf", f"select=eq(n\\,{n})", "-frames:v", "1", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-"]).stdout  # fmt: skip
    return np.frombuffer(raw, np.uint8).astype(float)


def _audio(path):
    from reelsift.tools import run, tool

    raw = run([tool("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1",
               "-ar", "48000", "-f", "f32le", "-"]).stdout  # fmt: skip
    return np.frombuffer(raw, np.float32)


@needs_ffmpeg
def test_a_cut_lands_on_its_frame_when_the_video_starts_after_the_audio(tmp_path):
    from reelsift.tools import run, tool

    source = tmp_path / "footage" / "late-video.mp4"
    source.parent.mkdir(parents=True)
    run([tool("ffmpeg"), "-v", "error", "-y", "-itsoffset", "1",
         "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=25:duration=3",
         "-f", "lavfi", "-i", "sine=sample_rate=48000:duration=4",
         "-c:v", "libx264", "-preset", "ultrafast", "-fps_mode", "passthrough", "-c:a", "aac",
         str(source)])  # fmt: skip
    out = _render_piece(tmp_path, source, 1.0, 2.0)
    first = _frame(out, 0)
    wanted, wrong = (float(np.abs(first - _frame(source, n)).mean()) for n in (25, 0))
    assert wanted < wrong / 3, (wanted, wrong)


@needs_ffmpeg
def test_sound_that_starts_late_stays_late(tmp_path):
    from reelsift.tools import run, tool

    source = tmp_path / "footage" / "late-audio.mp4"
    source.parent.mkdir(parents=True)
    run([tool("ffmpeg"), "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=25:duration=3",
         "-itsoffset", "1", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(source)])  # fmt: skip
    samples = _audio(_render_piece(tmp_path, source, 0.0, 3.0))
    rms = [float(np.sqrt(np.mean(samples[int(a * 48000):int(b * 48000)] ** 2)))
           for a, b in ((0.1, 0.8), (1.3, 2.5))]  # fmt: skip
    assert rms[0] < 0.01 < rms[1], rms


# --- dropping inside a shared longer clip ------------------------------------------------------


@needs_ffmpeg
def test_dropping_part_of_a_kept_clip_keeps_the_rest(review):  # noqa: F811
    rec = walk(review)
    review.decide("keep", rec, 0.0, 8.0, length=8)
    review.decide("force", rec, 5.0, 6.0, length=4)
    review.decide("drop", rec, 1.0, 2.0)
    assert review.problems() == []
    short = [i for i in review.edit(4).items if i.recording_id == rec]
    assert any(i.start <= 5.0 + 1e-3 and 6.0 <= i.end + 1e-3 for i in short)
    long = [i for i in review.edit(8).items if i.recording_id == rec]
    assert any(i.start <= 2.0 + 1e-3 and 8.0 <= i.end + 1e-3 for i in long)
    assert not [i for i in long if i.start < 2.0 and i.end > 1.0]
    # What the keep kept outside the drop stays kept, and undo brings the keep back whole.
    review.select_all()
    assert any(i.start <= 2.0 + 1e-3 and 8.0 <= i.end + 1e-3
               for i in review.edit(8).items if i.recording_id == rec)  # fmt: skip
    review.undo()
    assert (0.0, 8.0, 8) in [(d["start"], d["end"], d["length"]) for d in review.decisions()]


# --- cameras in several footage roots ---------------------------------------------------------


def _cameras(tmp_path, roots):
    from reelsift.ingest import ingest

    with Trip.create(tmp_path / "trip", roots) as trip:
        ingest(trip)
        return sorted(r["label"] for r in trip.db.execute("SELECT label FROM cameras"))


def test_the_same_folder_in_two_footage_roots_is_two_cameras(tmp_path):
    make_photo(tmp_path / "Anna" / "DCIM" / "a.jpg", color=(10, 20, 30))
    make_photo(tmp_path / "Ben" / "DCIM" / "b.jpg", color=(200, 30, 30))
    labels = _cameras(tmp_path, [tmp_path / "Anna", tmp_path / "Ben"])
    assert len(labels) == 2, labels
    assert any("Anna/DCIM" in label for label in labels)


def test_two_footage_roots_with_the_same_name_are_two_cameras(tmp_path):
    make_photo(tmp_path / "card1" / "photos" / "a.jpg", color=(10, 20, 30))
    make_photo(tmp_path / "card2" / "photos" / "b.jpg", color=(200, 30, 30))
    labels = _cameras(tmp_path, [tmp_path / "card1" / "photos", tmp_path / "card2" / "photos"])
    assert len(labels) == 2, labels


# --- a damaged prompt bank ----------------------------------------------------------------------


@needs_ffmpeg
def test_analysis_rebuilds_a_damaged_prompt_bank(tmp_path):
    from reelsift import proxies
    from reelsift.analysis import analyze
    from reelsift.ingest import ingest
    from reelsift.scoring import load_prompt_bank, prompt_bank_path
    from reelsift.sync import sync

    from .fakes import FakeEmbedder, FakeTranscriber

    footage = tmp_path / "footage"
    make_photo(footage / "a.jpg")
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        proxies.generate(trip)
        sync(trip)
        analyze(trip, FakeEmbedder, FakeTranscriber, say=lambda _: None)
        bank = prompt_bank_path(trip)
        bank.write_bytes(bank.read_bytes()[:40])  # what an interrupted write leaves
        assert load_prompt_bank(trip)[0] == [], "a damaged bank reads as missing"
        analyze(trip, FakeEmbedder, FakeTranscriber, say=lambda _: None)
        assert load_prompt_bank(trip)[0]
        assert not list(bank.parent.glob("*.partial.npz"))


# --- the proxies command ------------------------------------------------------------------------


@needs_ffmpeg
def test_the_proxies_command_remakes_a_deleted_proxy(tmp_path):
    from reelsift import proxies
    from reelsift.cli import _proxies
    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    make_photo(footage / "a.jpg")
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        proxies.generate(trip)
        shutil.rmtree(trip.root / "cache" / "proxies")
        _proxies(trip)
        row = trip.db.execute("SELECT path FROM proxies").fetchone()
        assert row and proxies.resolve(trip, row["path"]).is_file()


# --- searches added with the sliders ------------------------------------------------------------


@needs_ffmpeg
def test_a_new_search_from_the_sliders_is_embedded_outside_the_lock(client):  # noqa: F811
    shared = _shared_review(client)
    dim = shared.feats.embedding.shape[1]
    started, release = threading.Event(), threading.Event()

    def slow_encoder(texts):
        started.set()
        release.wait(10)
        return np.ones((len(texts), dim), np.float32) / np.sqrt(dim)

    shared.text_encoder = slow_encoder
    result = {}

    def put():
        result["response"] = client.put("/api/preferences/8", json={
            "searches": [{"text": "a dog on the trail", "value": 2}],
        })  # fmt: skip

    thread = threading.Thread(target=put)
    thread.start()
    assert started.wait(10)
    got_lock = shared.lock.acquire(timeout=2)
    if got_lock:
        shared.lock.release()
    release.set()
    thread.join(20)
    assert got_lock, "other requests can use the review while the search is embedded"
    assert result["response"].status_code == 200


def _shared_review(test_client):
    """The Review the app was created with (create_app keeps it in its route closures)."""
    for route in test_client.app.routes:
        endpoint = getattr(route, "endpoint", None)
        for cell in getattr(endpoint, "__closure__", None) or ():
            from reelsift.review import Review

            if isinstance(cell.cell_contents, Review):
                return cell.cell_contents
    pytest.fail("no Review found in the app")
