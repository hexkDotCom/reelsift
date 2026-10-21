"""Drafting each edit's description from the trip's scene descriptions."""

import json
import shutil

import pytest

from reelsift import drafts
from reelsift import prefs as prefs_mod
from reelsift.prefs import STYLES
from reelsift.project import Trip
from reelsift.review import Review
from reelsift.tripdata import load

from .conftest import needs_ffmpeg
from .test_review import LENGTHS, analyzed_root  # noqa: F401  (module fixture reused here)


def fake_model(answers: dict[str, str] | str):
    """A language model that drafts the given texts and reads any description as 'more action'."""
    prompts = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        if "maps each video's key" in prompt:
            return answers if isinstance(answers, str) else json.dumps(answers)
        return '{"more": {"action": 2}, "less": {"tag:road": 1}}'

    complete.prompts = prompts
    return complete


def test_drafts_follow_each_edits_style():
    model = fake_model(
        {"30": "the jumps and the fast descent", "600": "the whole day, start to end"}
    )
    texts = drafts.write("09:00 [interest 8] riders on a trail", {30: "preview", 600: "story"},
                         model)  # fmt: skip
    assert texts == {30: "the jumps and the fast descent", 600: "the whole day, start to end"}
    prompt = model.prompts[0]
    assert STYLES["preview"].brief in prompt and STYLES["story"].brief in prompt
    assert '"30": a 30-second video' in prompt and '"600": a 10-minute video' in prompt
    assert "riders on a trail" in prompt


@pytest.mark.parametrize("answer", ["no json at all", '{"30": 5}', '{"other": "x"}'])
def test_unusable_drafts_fall_back_to_the_style_templates(answer):
    texts = drafts.write("09:00 [interest 8] riders", {30: "preview"}, fake_model(answer))
    assert texts == {30: STYLES["preview"].template}


def test_without_a_model_or_descriptions_the_templates_are_used():
    assert drafts.write("anything", {30: "preview"}, None) == {30: STYLES["preview"].template}
    model = fake_model({"30": "x"})
    assert drafts.write("", {30: "preview"}, model) == {30: STYLES["preview"].template}
    assert not model.prompts, "nothing to read: the model isn't asked"


@pytest.fixture
def described(analyzed_root, tmp_path):  # noqa: F811
    root = tmp_path / "trip"
    shutil.copytree(analyzed_root, root, ignore=shutil.ignore_patterns("reelsift.lock"))
    with Trip(root, write=True) as trip:
        recs = [r["id"] for r in trip.db.execute("SELECT id FROM recordings WHERE kind = 'video'")]
        rows = []
        for k in range(200):
            rec = recs[k % len(recs)]
            text = "a quiet road" if k % 2 else f"scene {k}: riders by a lake"
            rows.append((rec, k * 0.01, k * 0.01 + 6, text, "[]", k % 10 + 1, "[]", "fake"))
        trip.db.executemany(
            "INSERT OR REPLACE INTO vlm_blocks (recording_id, start, end, description, tags,"
            " interest, quality, model) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows,
        )  # fmt: skip
        yield trip


@needs_ffmpeg
def test_summary_is_short_in_time_order_and_keeps_the_interesting_parts(described):
    text = drafts.summary(described, load(described))
    lines = text.splitlines()
    assert len(lines) - 1 <= drafts.SUMMARY_LINES
    assert "videos" in lines[0] and "photos" in lines[0]
    assert "interest 10" in text


@needs_ffmpeg
def test_drafted_preferences_are_saved_per_edit_and_used(described):
    from reelsift.cli import _draft

    assert drafts.missing(described) == LENGTHS
    model = fake_model({"4": "the fast riding", "8": "the whole ride with the group"})

    class Describer:
        def complete(self, prompt):
            return model(prompt)

    _draft(described, drafts.missing(described), ("fake", Describer), None)
    assert drafts.missing(described) == []
    review = Review(described)
    assert review.prefs[4].text == "the fast riding" and review.prefs[4].drafted
    assert review.prefs[4].style == "preview" and review.prefs[8].style == "story"
    assert review.prefs[8].adjustments == {"action": 2.0, "tag:road": -1.0}
    assert review.own_prefs[4] and review.own_prefs[8]


@needs_ffmpeg
def test_a_description_you_wrote_is_never_drafted_over(described):
    from reelsift.cli import _draft

    review = Review(described)
    review.apply(8, prefs_mod.Preferences(text="mine", adjustments={"water": 2}, clip=[3, 4]))
    assert drafts.missing(described) == [4]
    _draft(described, drafts.missing(described), None, None)
    review = Review(described)
    assert review.prefs[8].text == "mine"
    assert review.prefs[4].text == STYLES["preview"].template and not review.prefs[4].drafted
    # Drafting it again on purpose replaces the description but keeps its clip lengths.
    _draft(described, [8], None, None)
    assert Review(described).prefs[8].clip == [3.0, 4.0]
