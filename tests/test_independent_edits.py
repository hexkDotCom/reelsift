"""Independent edits: converting older trips, history across edits, and preferences changed
while a model is busy."""

import json
import shutil
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from reelsift import config as config_mod
from reelsift import drafts
from reelsift.project import Trip
from reelsift.review import Review
from reelsift.select import Item
from reelsift.web.app import create_app

from .conftest import needs_ffmpeg
from .test_review import LENGTHS, analyzed_root, covers  # noqa: F401  (module fixture reused here)

pytestmark = needs_ffmpeg


@pytest.fixture
def trip(analyzed_root, tmp_path):  # noqa: F811
    root = tmp_path / "trip"
    shutil.copytree(analyzed_root, root, ignore=shutil.ignore_patterns("reelsift.lock"))
    with Trip(root, write=True) as t:
        yield t


def walk(trip) -> int:
    """The 12-second recording, long enough for interval scenarios."""
    from fractions import Fraction

    rows = trip.db.execute("SELECT id, duration FROM recordings").fetchall()
    return max(rows, key=lambda r: Fraction(r["duration"]))["id"]


def describe_footage(trip) -> None:
    """One scene description, so drafting has something to read."""
    trip.db.execute(
        "INSERT INTO vlm_blocks (recording_id, start, end, description, tags, interest, quality,"
        " model) VALUES (?, 0, 6, 'riders by a lake', '[]', 8, '[]', 'fake')", (walk(trip),),
    )  # fmt: skip


def legacy(trip, kind, rec, start, end, length, suppressed=(), created=()) -> int:
    """A decision as an older version saved it (before edits were independent)."""
    trip.db.execute("DELETE FROM meta WHERE key = 'independent_edits'")
    return trip.db.execute(
        "INSERT INTO decisions (kind, recording_id, start, end, length, edits_before, suppressed,"
        " created, created_at) VALUES (?, ?, ?, ?, ?, '{}', ?, ?, '')",
        (kind, rec, start, end, length, json.dumps(list(suppressed)), json.dumps(list(created))),
    ).lastrowid  # fmt: skip


def test_restoring_an_old_revision_keeps_the_converted_keep(trip):
    rec = walk(trip)
    keep = legacy(trip, "keep", rec, 2.0, 3.0, 4)
    clip = Item(rec, 2.0, 3.0, -1, 0.0, None, "kept", locked=True)
    edit_id = trip.db.execute(
        "INSERT INTO edits (length, revision, items, settings, created_at) VALUES (8, 99, ?, ?, '')",
        (json.dumps([clip.__dict__]), json.dumps({"decisions": [keep], "op": "old"})),
    ).lastrowid  # fmt: skip
    review = Review(trip)
    review.restore(8, edit_id)
    assert covers(review.forced(8), rec, 2.0, 3.0), "the 8 s edit's copy of the keep is active"
    review.select(8)
    assert covers(review.edit(8).items, rec, 2.0, 3.0)


def test_undo_after_converting_takes_back_the_latest_action_first(trip):
    rec = walk(trip)
    legacy(trip, "keep", rec, 2.0, 3.0, 4)
    legacy(trip, "drop", rec, 10.0, 11.0, None)
    review = Review(trip)
    review.undo()
    active = [(d["kind"], d["length"]) for d in review.decisions()]
    assert active == [("keep", 8), ("keep", 4)], "the drop was the latest action"
    review.undo()
    assert review.decisions() == [], "the keep goes back together with its copy"


def test_a_converted_drop_still_switches_off_the_keeps_copies(trip):
    rec = walk(trip)
    keep = legacy(trip, "keep", rec, 2.0, 3.0, 4)
    trip.db.execute("UPDATE decisions SET active = 0 WHERE id = ?", (keep,))
    legacy(trip, "drop", rec, 2.0, 3.0, None, suppressed=[keep])
    review = Review(trip)
    assert [d["kind"] for d in review.decisions()] == ["drop"]
    review.undo()
    assert sorted(d["length"] for d in review.decisions()) == [4, 8], "undo brings back both"


def test_undo_skips_an_edit_no_longer_configured(trip):
    rec = walk(trip)
    review = Review(trip)
    review.edit(4), review.edit(8)
    review.decide("force", rec, 9.0, 10.5, length=8)
    trip.config.lengths = [4]
    config_mod.write_trip(trip.root, trip.config)
    trip.config = config_mod.load(trip.root)
    review = Review(trip)
    review.undo()
    assert review.decisions() == []


# --- preferences changed while a model is busy --------------------------------------------------


def blocking_model(answers):
    """A language model whose first call waits until released."""
    started, release = threading.Event(), threading.Event()
    calls = []

    def complete(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            started.set()
            release.wait(20)
        if "maps each video's key" in prompt:
            return json.dumps(answers)
        return '{"more": {"water": 2}, "exclude": ["tag:road", "water"]}'

    return complete, started, release


def test_a_redraft_never_overwrites_settings_saved_meanwhile(trip):
    describe_footage(trip)
    complete, started, release = blocking_model({"8": "the lake"})
    shared = Review(trip.worker(shared=True), None, complete)
    client = TestClient(create_app(trip, shared, 8765), base_url="http://127.0.0.1:8765")
    result = {}
    worker = threading.Thread(
        target=lambda: result.update(r=client.post("/api/preferences/8/draft"))
    )
    worker.start()
    assert started.wait(10)
    saved = client.put("/api/preferences/8", json={"style": "preview", "clip": [1, 2]})
    assert saved.status_code == 200
    release.set()
    worker.join(30)
    assert result["r"].status_code == 409
    after = client.get("/api/preferences/8").json()
    assert after["style"] == "preview" and after["clip"] == [1.0, 2.0]
    shared.trip.close()


def test_a_redraft_from_the_app_never_excludes_footage(trip):
    describe_footage(trip)
    complete, _, release = blocking_model({"8": "the lake, never roads"})
    release.set()
    shared = Review(trip.worker(shared=True), None, complete)
    client = TestClient(create_app(trip, shared, 8765), base_url="http://127.0.0.1:8765")
    body = client.post("/api/preferences/8/draft").json()
    assert body["text"] == "the lake, never roads" and body["drafted"]
    assert body["exclusions"] == []
    assert body["adjustments"]["water"] > 0
    shared.trip.close()


def test_draft_searches_from_another_model_are_not_cached(trip):
    from reelsift.prefs import Preferences

    drafted = {8: Preferences(text="x", searches=[{"text": "red bridge", "value": 2}],
                              style="story")}  # fmt: skip
    drafts.save_drafts(trip, drafted, lambda texts: np.ones((len(texts), 7), np.float32))
    assert trip.db.execute("SELECT COUNT(*) FROM text_embeddings").fetchone()[0] == 0
    review = Review(trip)  # opens fine; the search just isn't available yet
    assert review.prefs[8].searches
