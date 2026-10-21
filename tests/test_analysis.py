"""Analysis, candidates, selection and rendering on generated clips with stand-in models."""

import numpy as np
import pytest

from reelsift import candidates, proxies
from reelsift.analysis import analyze
from reelsift.edits import build, latest, save
from reelsift.ingest import ingest
from reelsift.media import probe
from reelsift.project import Trip
from reelsift.render import Output, render
from reelsift.scoring import load_prompt_bank
from reelsift.sync import sync
from reelsift.tripdata import load

from .conftest import make_clip, needs_ffmpeg, write_footage
from .fakes import FakeDescriber, FakeEmbedder, FakeTranscriber

pytestmark = needs_ffmpeg


@pytest.fixture(scope="module")
def analyzed(tmp_path_factory):
    """One analyzed trip shared by the tests in this module; tests must not change it."""
    root = tmp_path_factory.mktemp("analysis")
    footage = write_footage(root / "footage")
    make_clip(footage / "silent" / "quiet.mp4", 6.0, creation_time="2026-10-11T08:20:00Z",
              audio=False)  # fmt: skip
    with Trip.create(root / "trip", [footage]) as trip:
        ingest(trip)
        proxies.generate(trip)
        sync(trip)
        analyze(trip, FakeEmbedder, FakeTranscriber, say=lambda _: None,
                describer=(FakeDescriber.name, FakeDescriber))  # fmt: skip
        candidates.generate(trip)
        yield trip


def test_every_recording_gets_observations_and_embeddings(analyzed):
    db = analyzed.db
    recordings = db.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]
    for stage in ("visual", "speech"):
        done = db.execute("SELECT COUNT(*) FROM stage_runs WHERE stage = ?", (stage,)).fetchone()[0]
        assert done == recordings
    rows = db.execute("SELECT * FROM observations WHERE motion IS NOT NULL").fetchall()
    assert rows and all(r["blur"] is not None for r in rows)
    emb = db.execute("SELECT * FROM embeddings").fetchall()
    assert len(emb) == recordings
    assert load_prompt_bank(analyzed)[0]


def test_speech_and_loudness(analyzed):
    db = analyzed.db
    assert db.execute("SELECT COUNT(*) FROM words").fetchone()[0] > 0
    quiet = db.execute(
        "SELECT r.id FROM recordings r JOIN chapters c ON c.recording_id = r.id"
        " JOIN assets a ON a.id = c.asset_id WHERE a.path LIKE '%quiet.mp4'"
    ).fetchone()["id"]
    assert (
        db.execute("SELECT COUNT(*) FROM words WHERE recording_id = ?", (quiet,)).fetchone()[0] == 0
    )
    speech_seconds = db.execute("SELECT COUNT(*) FROM observations WHERE speech > 0").fetchone()[0]
    assert speech_seconds > 0


def test_analysis_is_not_repeated(analyzed):
    calls = []

    class Counting(FakeEmbedder):
        def encode_images(self, frames):
            calls.append(len(frames))
            return super().encode_images(frames)

    analyze(analyzed, FakeEmbedder, FakeTranscriber, say=lambda _: None)
    assert calls == []
    analyze(analyzed, Counting, FakeTranscriber, say=lambda _: None)
    assert calls == []  # same model name and version: nothing to do


def test_candidates_cover_every_recording(analyzed):
    data = load(analyzed)
    assert set(data.candidates.recording) == set(data.recordings)
    assert "speech" in set(data.candidates.source)
    assert np.all(data.candidates.min_start <= data.candidates.start)


def test_selection_fills_length_without_overlaps(analyzed):
    [sel] = build(analyzed, [8])
    assert 6 <= sel.duration <= 9.5
    by_rec: dict[int, list] = {}
    for item in sel.items:
        for other in by_rec.get(item.recording_id, []):
            assert item.end <= other.start or other.end <= item.start
        by_rec.setdefault(item.recording_id, []).append(item)
    utc = [i.utc for i in sel.items]
    assert utc == sorted(utc)


def test_rendered_edit_has_exact_frames_and_sound(analyzed, tmp_path):
    [sel] = build(analyzed, [8])
    stored = save(analyzed, sel, {"mode": "combined"})
    assert latest(analyzed, 8).revision == stored.revision
    out = render(analyzed, sel.items, tmp_path / "edit.mp4", Output(640, 360))
    p = probe(out)
    v = p.main_video
    assert (v.width, v.height) == (640, 360)
    assert p.audio, "silent sources still get an audio track"
    expected = sum(max(1, round(i.duration * 25)) for i in sel.items)
    assert abs(v.frames - expected) <= len(sel.items)
    assert abs(float(p.duration) - expected / 25) < 0.2


def test_vertical_render_letterboxes(analyzed, tmp_path):
    [sel] = build(analyzed, [4])
    out = render(analyzed, sel.items, tmp_path / "v.mp4", Output(360, 640))
    assert (probe(out).main_video.width, probe(out).main_video.height) == (360, 640)


def test_vlm_blocks_are_stored_and_validated(analyzed):
    rows = analyzed.db.execute("SELECT * FROM vlm_blocks").fetchall()
    assert rows
    import json

    for r in rows:
        assert "nonsense" not in json.loads(r["tags"])
        assert 1 <= r["interest"] <= 10
    data = load(analyzed)
    assert data.vlm
    from reelsift.scoring import features

    feats = features(data, load_prompt_bank(analyzed))
    j = feats.names.index("interest")
    assert np.isfinite(feats.raw[:, j]).any()


def test_vlm_is_not_repeated(analyzed):
    from reelsift import vlm

    recordings = [r["id"] for r in analyzed.db.execute("SELECT id FROM recordings")]
    loaded = []
    done, failed = vlm.analyze(analyzed, recordings, FakeDescriber.name, lambda: loaded.append(1))
    assert (done, failed, loaded) == (0, {}, [])
