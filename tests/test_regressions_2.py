"""Regression tests: review decisions, old trips, resumed jobs, stopping renders. The frontend tests are in tests/js and run through test_frontend.py."""

import shutil
import sqlite3
import sys
import threading
import time
from datetime import UTC, datetime
from fractions import Fraction

import numpy as np
import pytest

from reelsift.project import Trip
from reelsift.review import BrokenEdits
from reelsift.select import Item

from .conftest import make_photo, needs_ffmpeg
from .test_regressions import review, touches, walk  # noqa: F401  (fixtures reused here)
from .test_review import analyzed_root  # noqa: F401

# --- review decisions ----------------------------------------------------------------------------


@needs_ffmpeg
def test_trimming_a_short_clip_keeps_the_longer_edits_keep(review):  # noqa: F811
    rec = walk(review)
    review.decide("keep", rec, 2.0, 6.0, length=8)
    review.decide("keep", rec, 2.0, 6.0, length=4)
    review.decide("keep", rec, 2.0, 5.0, length=4, replaces=(2.0, 6.0))
    review.select_all()
    assert [i for i in review.edit(8).items
            if i.recording_id == rec and i.start <= 2.0 + 1e-3 and 6.0 <= i.end + 1e-3]  # fmt: skip
    assert review.problems() == []


@needs_ffmpeg
def test_undoing_a_trim_brings_back_exactly_the_original_keep(review):  # noqa: F811
    rec = walk(review)
    review.decide("keep", rec, 2.0, 6.0, length=4)
    review.decide("keep", rec, 2.0, 5.0, length=4, replaces=(2.0, 6.0))
    review.undo()
    kept = [(d["start"], d["end"], d["length"]) for d in review.decisions()]
    assert kept == [(2.0, 6.0, 4)]


@needs_ffmpeg
def test_adding_next_to_a_clip_does_not_join_across_dropped_footage(review):  # noqa: F811
    rec = walk(review)
    review.decide("drop", rec, 3.0, 3.5)
    review.decide("force", rec, 1.0, 3.0, length=8)
    review.decide("force", rec, 3.5, 5.0, length=8)
    assert not touches(review.edit(8).items, rec, 3.0, 3.5)
    assert review.problems() == []


@needs_ffmpeg
def test_kept_clips_over_still_dropped_footage_are_refused(review):  # noqa: F811
    rec = walk(review)
    review.decide("drop", rec, 3.0, 3.5)
    review.decide("force", rec, 1.0, 3.0, length=8)
    others = [i for i in review.edit(8).items if i.recording_id != rec]
    with pytest.raises(BrokenEdits) as e, review.operation():
        review._save(8, [*others, review._item(8, rec, 1.0, 5.0, "kept")], "test")
    assert any("dropped footage" in p for p in e.value.problems)
    # A later keep of the dropped footage itself brings it back.
    review.decide("keep", rec, 3.0, 3.5, length=8)
    assert review.problems() == []


def test_merging_keeps_the_details_of_the_clip_listed_first():
    from reelsift.select import merge

    kept = Item(1, 2.0, 4.0, -1, 0.0, None, "kept", locked=True, crop_x=0.1)
    inherited = Item(1, 1.0, 3.0, -1, 0.0, None, "window", crop_x=0.5)
    (joined,) = merge([kept, inherited])
    assert (joined.start, joined.end, joined.crop_x, joined.locked) == (1.0, 4.0, 0.1, True)


# --- ingest and old trips ------------------------------------------------------------------------


def as_pre_fix_trip(root) -> None:
    """Turn a trip back into what a version before these fixes saved (schema 5)."""
    db = sqlite3.connect(root / "reelsift.db")
    db.execute("DROP TABLE travel")
    db.execute("DROP TABLE soundtracks")
    db.execute("DROP TABLE music_assets")
    db.execute("ALTER TABLE assets DROP COLUMN changed")
    db.execute("ALTER TABLE preferences DROP COLUMN length")
    db.execute("ALTER TABLE preferences DROP COLUMN settings")
    db.execute("DELETE FROM meta WHERE key = 'last_recording_id'")
    db.execute("UPDATE time_evidence SET provenance = 'metadata_unknown_zone'"
               " WHERE provenance = 'metadata_local_unknown_zone'")  # fmt: skip
    db.execute("PRAGMA user_version = 5")
    db.commit()
    db.close()


def test_upgraded_trips_never_reuse_a_recording_id(tmp_path):
    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    make_photo(footage / "a.jpg", color=(10, 20, 30))
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        old = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
    as_pre_fix_trip(tmp_path / "trip")
    make_photo(footage / "a.jpg", color=(200, 30, 30), size=(500, 300))
    with Trip(tmp_path / "trip", write=True) as trip:
        ingest(trip)
        new = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
    assert new != old


def test_ids_still_named_by_old_edits_are_never_reused(tmp_path):
    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    make_photo(footage / "a.jpg")
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        trip.db.execute(
            "INSERT INTO edits (length, revision, items, settings, created_at)"
            " VALUES (30, 1, ?, '{}', '')",
            ('[{"recording_id": 7, "start": 0, "end": 1, "candidate_id": -1, "score": 0,'
             ' "utc": null, "source": "window"}]',),
        )  # fmt: skip
    as_pre_fix_trip(tmp_path / "trip")
    with Trip(tmp_path / "trip", write=True) as trip:
        ingest(trip)
        assert trip.db.execute("SELECT id FROM recordings").fetchone()["id"] > 7


@needs_ffmpeg
def test_an_unreadable_replacement_keeps_the_old_recording(tmp_path):
    from reelsift.ingest import ingest
    from reelsift.render import pieces

    footage = tmp_path / "footage"
    photo = make_photo(footage / "a.jpg")
    original = photo.read_bytes()
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        rec = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
        trip.db.execute(
            "INSERT INTO decisions (kind, recording_id, start, end, edits_before, created_at)"
            " VALUES ('keep', ?, 0, 0, '{}', '')", (rec,),
        )  # fmt: skip
        photo.write_bytes(b"not a picture")
        assert ingest(trip).failed
        assert trip.db.execute("SELECT id FROM recordings").fetchone()["id"] == rec
        assert trip.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
        with pytest.raises(RuntimeError, match="replaced by a file that can't be read"):
            pieces(trip, Item(rec, 0.0, 2.0, -1, 0.0, None, "photo"))
        photo.write_bytes(original)  # the original is back (with a new modified time)
        assert not ingest(trip).failed
        assert trip.db.execute("SELECT id FROM recordings").fetchone()["id"] == rec
        assert pieces(trip, Item(rec, 0.0, 2.0, -1, 0.0, None, "photo"))


def test_old_local_photo_times_follow_the_zone_after_upgrading(tmp_path):
    from reelsift import config as config_mod
    from reelsift.ingest import ingest
    from reelsift.sync import clock, sync

    footage = tmp_path / "footage"
    make_photo(footage / "phone" / "a.jpg", taken="2026:10:11 10:00:00", offset=None)
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
    as_pre_fix_trip(tmp_path / "trip")
    with Trip(tmp_path / "trip", write=True) as trip:
        label = trip.db.execute("SELECT label FROM cameras").fetchone()["label"]
        trip.config.cameras[label] = config_mod.CameraConfig(zone="+02:00")
        sync(trip)
        rec = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
        clk = clock(trip, rec)
        assert clk.to_utc(Fraction(0)) == datetime(2026, 10, 11, 8, 0, tzinfo=UTC)
        assert clk.local(Fraction(0)).hour == 10


# --- resumed jobs and model switches -------------------------------------------------------------


def test_a_resumed_proxy_job_remakes_a_proxy_deleted_since(tmp_path, monkeypatch):
    from reelsift import photos, proxies
    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    make_photo(footage / "a.jpg", color=(10, 20, 30))
    make_photo(footage / "b.jpg", color=(200, 30, 30))
    real = photos.make_proxy

    def failing(path, out):
        if path.name == "b.jpg":
            raise RuntimeError("disk full")
        return real(path, out)

    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        monkeypatch.setattr(photos, "make_proxy", failing)
        assert proxies.generate(trip).failed  # the job stays unfinished
        monkeypatch.setattr(photos, "make_proxy", real)
        shutil.rmtree(trip.root / "cache" / "proxies")
        result = proxies.generate(trip)
        assert not result.failed and proxies.pending(trip) == []
        for row in trip.db.execute("SELECT path FROM proxies"):
            assert proxies.resolve(trip, row["path"]).is_file()


def test_a_resumed_visual_job_redoes_embeddings_deleted_since(tmp_path, monkeypatch):
    from reelsift import analysis, proxies
    from reelsift.analysis import analyze, embeddings_ok
    from reelsift.ingest import ingest
    from reelsift.sync import sync

    from .fakes import FakeEmbedder, FakeTranscriber

    footage = tmp_path / "footage"
    make_photo(footage / "a.jpg", color=(10, 20, 30))
    make_photo(footage / "b.jpg", color=(200, 30, 30))
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        proxies.generate(trip)
        sync(trip)
        first, second = (r["id"] for r in trip.db.execute("SELECT id FROM recordings ORDER BY id"))
        real = analysis.run_visual

        def failing(trip_, recording_id, embedder):
            if recording_id == second:
                raise RuntimeError("decoder crashed")
            return real(trip_, recording_id, embedder)

        monkeypatch.setattr(analysis, "run_visual", failing)
        analyze(trip, FakeEmbedder, FakeTranscriber, say=lambda _: None)
        monkeypatch.setattr(analysis, "run_visual", real)
        (trip.root / trip.db.execute(
            "SELECT path FROM embeddings WHERE recording_id = ?", (first,)
        ).fetchone()["path"]).unlink()  # fmt: skip
        analyze(trip, FakeEmbedder, FakeTranscriber, say=lambda _: None)
        assert embeddings_ok(trip, first) and embeddings_ok(trip, second)


def test_bursts_never_compare_embeddings_from_different_models(tmp_path):
    from reelsift import proxies
    from reelsift.analysis import analyze
    from reelsift.candidates import burst_leaders
    from reelsift.ingest import ingest
    from reelsift.sync import sync

    from .fakes import FakeEmbedder, FakeTranscriber

    footage = tmp_path / "footage"
    make_photo(footage / "a.jpg", taken="2026:10:11 10:00:00")
    make_photo(footage / "b.jpg", taken="2026:10:11 10:00:01")
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        proxies.generate(trip)
        sync(trip)
        analyze(trip, FakeEmbedder, FakeTranscriber, say=lambda _: None)
        ids = [r["id"] for r in trip.db.execute("SELECT id FROM recordings ORDER BY id")]
        assert len(burst_leaders(trip, ids)) == 1, "the same picture a second apart is a burst"
        rows = trip.db.execute("SELECT recording_id, path FROM embeddings ORDER BY recording_id")
        a, b = (trip.root / r["path"] for r in rows.fetchall())
        # Same size, other model: equal vectors must still not join the photos.
        shutil.copyfile(a, b)
        trip.db.execute("UPDATE embeddings SET model = 'other' WHERE recording_id = ?", (ids[1],))
        assert burst_leaders(trip, ids) == set(ids)
        # Other size as well: no crash.
        np.save(b, np.ones((1, 7), np.float32) / np.sqrt(7))
        assert burst_leaders(trip, ids) == set(ids)


# --- stopping renders ----------------------------------------------------------------------------


def sleeper(seconds: float = 30) -> list[str]:
    return [sys.executable, "-c", f"import time; time.sleep({seconds})"]


def test_cancelling_stops_the_process_and_starts_no_fallback(tmp_path):
    from reelsift.render import _first_working
    from reelsift.tools import Cancelled, cancel, cancellable

    marker = tmp_path / "fallback-ran"
    event = threading.Event()
    outcome = []

    def work():
        try:
            with cancellable(event):
                _first_working([
                    lambda: sleeper(),
                    lambda: [sys.executable, "-c", f"open(r'{marker}', 'w')"],
                ])  # fmt: skip
        except Cancelled:
            outcome.append("cancelled")
        except Exception as e:  # recorded for the assertion
            outcome.append(repr(e))

    thread = threading.Thread(target=work)
    thread.start()

    time.sleep(1.0)  # let the first command start
    cancel(event)
    thread.join(10)
    assert not thread.is_alive()
    assert outcome == ["cancelled"]
    assert not marker.exists(), "no fallback command starts after cancelling"


def test_stopping_the_app_waits_until_the_render_has_stopped(tmp_path):
    from reelsift.render import _first_working
    from reelsift.tools import cancellable
    from reelsift.web.app import Renders

    marker = tmp_path / "fallback-ran"
    renders = Renders.__new__(Renders)
    renders.cancel = threading.Event()

    def work():
        with cancellable(renders.cancel):
            _first_working([
                lambda: sleeper(),
                lambda: [sys.executable, "-c", f"open(r'{marker}', 'w')"],
            ])  # fmt: skip

    def run():
        try:
            work()
        except BaseException:  # the thread just ends
            pass

    renders.thread = threading.Thread(target=run)
    renders.thread.start()

    time.sleep(1.0)
    renders.stop()
    assert not renders.thread.is_alive(), "stop returns only once the render has stopped"
    assert not marker.exists()


def test_a_cancelled_job_is_marked_cancelled(tmp_path):
    from reelsift.jobs import Job
    from reelsift.tools import Cancelled

    with Trip.create(tmp_path / "trip", [tmp_path]) as trip:
        job = Job(trip, "render", {"key": "x"}, ["0", "1"])

        def work(item):
            raise Cancelled()

        with pytest.raises(Cancelled):
            job.run(work)
        state = trip.db.execute("SELECT state FROM jobs WHERE id = ?", (job.id,)).fetchone()[0]
        assert state == "cancelled"


# --- third review --------------------------------------------------------------------------------


def covering(review, length, rec, a, b) -> list[Item]:  # noqa: F811 (not the fixture)
    return [i for i in review.edit(length).items
            if i.recording_id == rec and i.start <= a + 1e-3 and b <= i.end + 1e-3]  # fmt: skip


@needs_ffmpeg
def test_trimming_a_merged_clip_stays_trimmed_after_selecting_again(review):  # noqa: F811
    rec = walk(review)
    review.decide("force", rec, 2.0, 6.0, length=8)
    review.decide("force", rec, 1.0, 3.0, length=8)
    merged = covering(review, 8, rec, 1.0, 6.0)[0]
    review.decide("keep", rec, merged.start, 5.0, length=8, replaces=(merged.start, merged.end))
    review.select_all()
    assert not covering(review, 8, rec, 5.5, 6.0), "the trimmed-off end stays out"
    assert covering(review, 8, rec, 1.0, 5.0)
    assert review.problems() == []


@pytest.fixture
def review3(analyzed_root, tmp_path):  # noqa: F811
    """A trip with three lengths: 4, 8 and 12 seconds."""
    from reelsift.review import Review

    root = tmp_path / "trip3"
    shutil.copytree(analyzed_root, root, ignore=shutil.ignore_patterns("reelsift.lock"))
    toml = root / "reelsift.toml"
    toml.write_text(toml.read_text(encoding="utf-8").replace(
        "lengths = [4, 8]", "lengths = [4, 8, 12]", 1), encoding="utf-8")  # fmt: skip
    with Trip(root, write=True) as trip:
        review = Review(trip)
        assert review.lengths == [4, 8, 12]
        yield review


@needs_ffmpeg
def test_refill_after_trimming_merged_clips_near_dropped_footage(review3):

    rec = walk(review3)
    review3.decide("drop", rec, 7.5, 8.0)
    review3.decide("force", rec, 8.0, 12.0, length=12)
    review3.decide("keep", rec, 8.0, 11.5, length=12, replaces=(8.0, 12.0))
    review3.decide("force", rec, 7.0, 9.0, length=12)
    merged = covering(review3, 12, rec, 7.0, 11.5)[0]
    review3.decide("keep", rec, merged.start, 11.0, length=12, replaces=(merged.start, merged.end))
    review3.decide("force", rec, 4.0, 7.0, length=12)
    review3.refill(12)
    assert review3.problems() == []
    assert not covering(review3, 12, rec, 11.2, 11.5)


def test_a_joined_clip_keeps_the_time_of_its_start():
    from reelsift.select import merge

    kept = Item(1, 2.0, 4.0, -1, 0.0, 102.0, "kept", locked=True, crop_x=0.1)
    inherited = Item(1, 1.0, 3.0, -1, 0.0, 101.0, "window", crop_x=0.5)
    (joined,) = merge([kept, inherited])
    assert (joined.start, joined.utc, joined.crop_x) == (1.0, 101.0, 0.1)
