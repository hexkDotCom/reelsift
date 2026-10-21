import json

from reelsift.prefs import Preferences, compile_llm, compile_rules, compile_text, search_feature


def test_keywords_turn_into_signed_sliders():
    p = compile_rules("lots more scenery and the kids, less road, fewer stops")
    assert p.adjustments["scenery"] == 3
    assert p.adjustments["kids"] == 2
    assert p.adjustments["road"] == -2
    assert p.adjustments["tag:stop"] == -2
    assert any("scenic" in line for line in p.interpretation)


def test_unknown_phrases_become_searches():
    p = compile_rules("more of the red bridge, no traffic cones")
    texts = {s["text"]: s["value"] for s in p.searches}
    assert texts["red bridge"] == 2
    assert texts["traffic cones"] == -2


def test_language_model_answer_is_validated():
    answer = json.dumps(
        {
            "more": {"scenery": 3, "made_up": 2},
            "less": {"road": 2},
            "search_more": ["red bridge"],
            "search_less": [],
            "exclude": ["tag:boring", "selfie stick"],
        }
    )
    p = compile_llm("whatever", lambda prompt: answer)
    assert p.adjustments == {"scenery": 3.0, "road": -2.0, "tag:boring": 0.0} or (
        p.adjustments["scenery"] == 3
        and p.adjustments["road"] == -2
        and "made_up" not in p.adjustments
    )
    assert "tag:boring" in p.exclusions
    assert search_feature("selfie stick") in p.exclusions
    assert {s["text"] for s in p.searches} == {"red bridge", "selfie stick"}


def test_unusable_model_answer_falls_back_to_keywords():
    p = compile_text("more water", lambda prompt: "I am not JSON")
    assert p.engine == "keywords" and p.adjustments == {"water": 2.0}

    def broken(prompt):
        raise RuntimeError("out of memory")

    assert compile_text("more water", broken).adjustments == {"water": 2.0}


def test_weights_add_to_base_and_searches_get_their_own_feature():
    p = Preferences(adjustments={"scenery": 2}, searches=[{"text": "Red Bridge", "value": -1}])
    w = p.weights({"scenery": 0.5, "people": 0.7})
    assert w["scenery"] == 0.5 + 1.2
    assert w["people"] == 0.7
    assert w[search_feature("Red Bridge")] < 0


def test_clean_clamps_and_drops_unknowns():
    p = Preferences(
        adjustments={"scenery": 9, "nonsense": 1, "people": 0},
        searches=[{"text": "  ", "value": 2}, {"text": "A", "value": 2}, {"text": "a", "value": 1}],
        exclusions=["tag:boring", "nope"],
    ).clean()
    assert p.adjustments == {"scenery": 3.0}
    assert p.searches == [{"text": "A", "value": 2.0}]
    assert p.exclusions == ["tag:boring"]
