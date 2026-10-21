"""Review decisions and the web API, on a small analyzed trip with stand-in models."""

import shutil

import pytest
from fastapi.testclient import TestClient

from reelsift import candidates, proxies
from reelsift.analysis import analyze
from reelsift.ingest import ingest
from reelsift.prefs import Preferences
from reelsift.project import Trip
from reelsift.review import Review, overlap
from reelsift.sync import sync
from reelsift.web.app import create_app

from .conftest import make_clip, needs_ffmpeg, write_footage
from .fakes import FakeEmbedder, FakeTranscriber

pytestmark = needs_ffmpeg
LENGTHS = [4, 8]


@pytest.fixture(scope="module")
def analyzed_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("review")
    footage = write_footage(root / "footage")
    make_clip(footage / "long" / "walk.mp4", 12.0, creation_time="2026-10-11T08:30:00Z")
    with Trip.create(root / "trip", [footage]) as trip:
        ingest(trip)
        proxies.generate(trip)
        sync(trip)
        analyze(trip, FakeEmbedder, FakeTranscriber, say=lambda _: None)
        candidates.generate(trip)
    with (root / "trip" / "reelsift.toml").open("a", encoding="utf-8") as f:
        f.write("")
    text = (root / "trip" / "reelsift.toml").read_text(encoding="utf-8")
    text = text.replace("[trip]\n", f"[trip]\nlengths = {LENGTHS}\n", 1)
    (root / "trip" / "reelsift.toml").write_text(text, encoding="utf-8")
    return root / "trip"


@pytest.fixture
def review(analyzed_root, tmp_path):
    root = tmp_path / "trip"
    shutil.copytree(analyzed_root, root, ignore=shutil.ignore_patterns("reelsift.lock"))
    with Trip(root, write=True) as trip:
        yield Review(trip)


def covers(items, rec, start, end) -> bool:
    return any(i.recording_id == rec and overlap(i.start, i.end, start, end) > 0 for i in items)


def test_first_look_selects_every_length(review):
    for length in LENGTHS:
        edit = review.edit(length)
        assert edit.items and edit.revision == 1


def test_drop_everywhere_removes_from_every_length_and_stays_out(review):
    target = review.edit(8).items[0]
    review.decide("drop", target.recording_id, target.start, target.end)
    for length in LENGTHS:
        assert not covers(review.edit(length).items, target.recording_id, target.start, target.end)
        review.select(length)
        assert not covers(
            review.edit(length).items, target.recording_id, target.start + 0.2, target.end - 0.2
        )


def test_drop_from_one_edit_leaves_the_others(review):
    rec = max(review.data.recordings, key=lambda r: review.data.recordings[r].duration)
    review.decide("force", rec, 2.0, 3.0, length=4)
    review.decide("force", rec, 2.0, 3.0, length=8)
    changed = review.decide("drop", rec, 2.0, 3.0, length=8)
    assert set(changed) == {8}
    assert not covers(review.edit(8).items, rec, 2.0, 3.0)
    assert covers(review.edit(4).items, rec, 2.0, 3.0)
    review.select(8)
    assert not covers(review.edit(8).items, rec, 2.2, 2.8)


def test_force_goes_into_its_own_edit_only(review):
    rec = max(review.data.recordings, key=lambda r: review.data.recordings[r].duration)
    four, _ = review.edit(4).revision, review.edit(8)
    changed = review.decide("force", rec, 9.0, 10.5, length=8)
    assert set(changed) == {8}
    assert covers(review.edit(8).items, rec, 9.0, 10.5)
    assert review.edit(4).revision == four
    review.select(8)
    assert covers(review.edit(8).items, rec, 9.0, 10.5), "forced clips survive re-selection"
    with pytest.raises(ValueError):
        review.decide("force", rec, 9.0, 10.5)  # an add needs its edit


def test_trim_replaces_and_undo_restores(review):
    item = review.edit(8).items[0]
    before = review.edit(8)
    review.decide("keep", item.recording_id, item.start, item.end - 0.4, length=8,
                  replaces=(item.start, item.end))  # fmt: skip
    trimmed = [i for i in review.edit(8).items if i.recording_id == item.recording_id
               and abs(i.start - item.start) < 1e-3]  # fmt: skip
    assert trimmed and trimmed[0].locked and abs(trimmed[0].end - (item.end - 0.4)) < 1e-3
    review.undo()
    assert [(i.start, i.end) for i in review.edit(8).items] == [
        (i.start, i.end) for i in before.items
    ]
    assert not review.decisions()


def test_refill_keeps_current_clips(review):
    current = review.edit(8).items
    target = current[0]
    review.decide("drop", target.recording_id, target.start, target.end, length=8)
    remaining = review.edit(8).items
    refilled = review.refill(8).items
    for i in remaining:
        assert covers(refilled, i.recording_id, i.start, i.end)


def test_suggestions_skip_dropped_and_current(review):
    target = review.edit(8).items[0]
    review.decide("drop", target.recording_id, target.start, target.end)
    c = review.data.candidates
    for index in review.suggestions(8, target.recording_id, target.start):
        assert review.allowed(8)[index]
        assert not covers(
            review.edit(8).items, int(c.recording[index]), c.start[index], c.end[index]
        )


def test_approve_marks_latest_revision(review):
    review.approve(4)
    assert review.edit(4).approved
    review.select(4)
    assert not review.edit(4).approved


@pytest.fixture
def client(review):
    trip = review.trip
    shared = Review(trip.worker(shared=True))  # the server answers on several threads
    app = create_app(trip, shared, 8765)
    yield TestClient(app, base_url="http://127.0.0.1:8765")
    shared.trip.close()


def test_api_round_trip(client):
    info = client.get("/api/trip").json()
    assert info["lengths"] == LENGTHS and info["recording_list"]
    edit = client.get("/api/edits/8").json()
    item = edit["items"][0]
    assert item["preview"] and item["thumb"].startswith("/media/thumb/")
    r = client.post("/api/decisions", json={"kind": "drop", "recording_id": item["recording_id"],
                                             "start": item["start"], "end": item["end"]})  # fmt: skip
    assert r.status_code == 200 and 8 in r.json()["changed"]
    assert client.post("/api/undo").json()["changed"]
    assert client.get("/api/candidates/8?limit=5").json()
    assert len(client.get("/api/edits/8/revisions").json()) >= 3


def test_media_endpoints(client):
    item = client.get("/api/edits/8").json()["items"][0]
    thumb = client.get(item["thumb"])
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    proxy = client.get(f"/media/proxy/{item['recording_id']}/{item['preview']['chapter']}",
                       headers={"Range": "bytes=0-99"})  # fmt: skip
    assert proxy.status_code == 206 and len(proxy.content) == 100
    assert client.get("/media/proxy/99999/0").status_code == 404
    assert client.get("/media/output/..%2Freelsift.toml").status_code == 404
    assert "Reelsift" in client.get("/").text


def test_foreign_host_and_origin_are_rejected(client):
    assert client.get("/api/trip", headers={"host": "evil.example"}).status_code == 403
    r = client.post("/api/undo", headers={"origin": "http://evil.example"})
    assert r.status_code == 403


def test_bad_requests(client):
    assert client.get("/api/edits/7").status_code == 404
    bad = {"kind": "explode", "recording_id": 1, "start": 0, "end": 1}
    assert client.post("/api/decisions", json=bad).status_code == 400
    empty = {"kind": "drop", "recording_id": 1, "start": 2, "end": 1}
    assert client.post("/api/decisions", json=empty).status_code == 400


def test_edits_are_chosen_independently(review):
    review.edit(4), review.edit(8)
    eight = review.edit(8).revision
    review.select(4)
    item = review.edit(4).items[0]
    review.decide("drop", item.recording_id, item.start, item.end, length=4)
    review.refill(4)
    review.apply(4, Preferences(adjustments={"motion": 3}))
    assert review.edit(8).revision == eight, "nothing done to the 4 s edit touches the 8 s edit"


def test_each_edit_has_its_own_style_and_preferences(review):
    assert review.prefs[4].style == "preview" and review.prefs[8].style == "story"
    review.apply(4, Preferences(adjustments={"motion": 3}))
    assert review.prefs[4].adjustments == {"motion": 3.0}
    assert review.prefs[8].adjustments == {}
    assert review.weights[4] != review.weights[8]
    again = Review(review.trip)
    assert again.prefs[4].adjustments == {"motion": 3.0} and again.prefs[4].style == "preview"
    assert again.own_prefs[4] and not again.own_prefs[8]


def test_clip_lengths_set_by_hand(review):
    review.apply(8, Preferences(clip=[3.0, 3.5]))
    chosen = [i for i in review.edit(8).items if not i.locked]
    assert chosen and all(i.duration <= 3.5 + 1e-3 for i in chosen)
    assert review.prefs[8].clip == [3.0, 3.5]
    with pytest.raises(ValueError):
        review.apply(8, Preferences(clip=[5.0, 2.0]))
    assert review.prefs[8].clip == [3.0, 3.5], "a refused change leaves the old one"


def test_old_trips_give_each_edit_its_own_copy_of_a_keep(analyzed_root, tmp_path):
    root = tmp_path / "trip"
    shutil.copytree(analyzed_root, root, ignore=shutil.ignore_patterns("reelsift.lock"))
    with Trip(root, write=True) as trip:
        rec = trip.db.execute("SELECT id FROM recordings LIMIT 1").fetchone()["id"]
        # As an older version saved it: a keep for the 4 s edit applied to the 8 s edit too.
        trip.db.execute(
            "INSERT INTO decisions (kind, recording_id, start, end, length, edits_before,"
            " created_at) VALUES ('keep', ?, 0.5, 1.5, 4, '{}', '')", (rec,),
        )  # fmt: skip
        trip.db.execute("DELETE FROM meta WHERE key = 'independent_edits'")
        trip.db.execute(
            "INSERT INTO preferences (text, weights, searches, exclusions, interpretation,"
            " created_at) VALUES ('more water', '{\"water\": 2}', '[]', '[]', '{}', '')"
        )  # fmt: skip
        review = Review(trip)
        lengths = sorted(d["length"] for d in review.decisions())
        assert lengths == [4, 8]
        assert all(covers(review.edit(n).items, rec, 0.5, 1.5) for n in LENGTHS)
        assert review.prefs[4].adjustments == review.prefs[8].adjustments == {"water": 2}
        Review(trip)  # converting happens once
        assert len(review.decisions()) == 2


def test_preferences_rescore_and_reselect(analyzed_root, tmp_path):
    from .fakes import FakeEmbedder

    root = tmp_path / "trip"
    shutil.copytree(analyzed_root, root, ignore=shutil.ignore_patterns("reelsift.lock"))
    with Trip(root, write=True) as trip:
        review = Review(trip, FakeEmbedder().encode_text, None)
        before = review.edit(8).revision
        prefs = review.set_text(8, "more scenery, less road, more of the red bridge")
        assert prefs.engine == "keywords"
        assert prefs.adjustments["scenery"] > 0
        assert any(s["text"] == "red bridge" for s in prefs.searches)
        assert "search:red bridge" in review.feats.names
        assert review.edit(8).revision > before
        cached = trip.db.execute("SELECT COUNT(*) FROM text_embeddings").fetchone()[0]
        assert cached == 1
        # A new session picks the preferences up again, using the cached search embedding.
        again = Review(trip, None, None)
        assert "search:red bridge" in again.feats.names
        assert again.prefs[8].text == prefs.text
        assert again.prefs[8].style == "story", "reading text keeps the edit's style"


def test_exclusion_keeps_strong_matches_out(review):
    name = "motion"
    j = review.feats.names.index(name)
    review.apply(8, Preferences(adjustments={name: 1}, exclusions=[name]))
    excluded = review.feats.z[:, j] >= 1.0
    assert not review.allowed(8)[excluded].any()
    assert review.allowed(4)[excluded].any(), "the other edit has no exclusion"


def test_search_without_encoder_is_reported(review):
    prefs = review.apply(8, Preferences(searches=[{"text": "red bridge", "value": 2}]))
    assert prefs.searches == []
    assert any("can't search" in line for line in prefs.interpretation)


def test_preferences_api(client):
    r = client.post("/api/preferences/8/text", json={"text": "more water, less road"})
    assert r.status_code == 200
    body = r.json()
    assert body["adjustments"]["water"] > 0 and body["vocabulary"] and body["length"] == 8
    assert body["style"] == "story" and len(body["auto_clip"]) == 2
    r = client.put("/api/preferences/8", json={"adjustments": {"people": 2}, "searches": [],
                                                "exclusions": [], "style": "highlights",
                                                "clip": [2.5, 4]})  # fmt: skip
    body = r.json()
    assert body["adjustments"] == {"people": 2.0} and body["style"] == "highlights"
    assert body["clip"] == [2.5, 4.0]
    assert client.get("/api/preferences/4").json()["adjustments"] == {}
    bad = client.put("/api/preferences/8", json={"clip": [9, 1]})
    assert bad.status_code == 400
    assert client.put("/api/preferences/8", json={"style": "opera"}).status_code == 400
    assert client.get("/api/preferences/7").status_code == 404


def test_draft_api_without_a_language_model_uses_the_style_template(client):
    from reelsift.prefs import STYLES

    body = client.post("/api/preferences/4/draft").json()
    assert body["text"] == STYLES["preview"].template and not body["drafted"]
    assert body["adjustments"]["action"] > 0


def test_a_rendered_video_plays_or_downloads(client, review):
    out = review.trip.root / "output"
    out.mkdir(exist_ok=True)
    (out / "trip-30s-r1.mp4").write_bytes(b"not really a video")
    played = client.get("/media/output/trip-30s-r1.mp4")
    assert played.status_code == 200 and "attachment" not in played.headers.get(
        "content-disposition", "")  # fmt: skip
    saved = client.get("/media/output/trip-30s-r1.mp4?download=true")
    assert saved.status_code == 200 and saved.content == b"not really a video"
    assert 'attachment; filename="trip-30s-r1.mp4"' in saved.headers["content-disposition"]


def test_export_api_writes_downloadable_timelines(client):
    r = client.post("/api/export", json={"length": 8})
    assert r.status_code == 200
    files = r.json()["files"]
    assert {f.rsplit(".", 1)[1] for f in files} == {"fcpxml", "otio"}
    download = client.get(f"/media/output/{files[0]}")
    assert download.status_code == 200
    assert "attachment" in download.headers.get("content-disposition", "")


def test_a_browser_closing_or_the_app_stopping_isnt_reported_as_an_error(caplog):
    import asyncio
    import logging
    import sys

    from reelsift.web.app import quiet_disconnects, quiet_shutdown

    loop = asyncio.new_event_loop()
    try:
        quiet_disconnects(loop)
        with caplog.at_level(logging.ERROR, logger="asyncio"):
            loop.call_exception_handler({"message": "lost", "exception": ConnectionResetError()})
            assert not caplog.records, "a browser closed while a video was loading"
            loop.call_exception_handler({"message": "real", "exception": ValueError("bug")})
            assert "real" in caplog.text, "other errors are still reported"
    finally:
        loop.close()

    def record(exc):
        try:
            raise exc
        except BaseException:
            info = sys.exc_info()
        return logging.LogRecord("uvicorn.error", logging.ERROR, "", 0,
                                 "Exception in ASGI application", None, info)  # fmt: skip

    assert not quiet_shutdown(record(asyncio.CancelledError()))
    assert quiet_shutdown(record(ValueError("bug")))


def test_settings_shown_before_another_tab_saved_cant_overwrite_that_save(client):
    shown = client.get("/api/preferences/8").json()
    body = {"text": shown["text"], "adjustments": {}, "searches": [], "exclusions": ["people"],
            "style": shown["style"], "clip": None}  # fmt: skip
    first = client.put("/api/preferences/8", json={**body, "expected_version": shown["version"]})
    assert first.status_code == 200, first.text
    stale = client.put("/api/preferences/8", json={**body, "exclusions": [], "style": "story",
                                                   "expected_version": shown["version"]})  # fmt: skip
    assert stale.status_code == 409
    assert client.get("/api/preferences/8").json()["exclusions"] == ["people"]
