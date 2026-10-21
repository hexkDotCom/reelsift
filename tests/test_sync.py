from datetime import UTC, datetime, timedelta
from fractions import Fraction

import pytest

from reelsift.sync import Anchor, parse_offset, solve_corrections
from reelsift.timebase import ClockTransform, Provenance

T0 = datetime(2026, 10, 11, 8, 0, 0, tzinfo=UTC)


def clock(seconds_after_t0: float, confidence: float = 0.8) -> ClockTransform:
    return ClockTransform(
        T0 + timedelta(seconds=seconds_after_t0),
        Provenance.METADATA_ZONE_FROM_FILENAME,
        confidence,
        120,
    )


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("75.5", Fraction(151, 2)),
        ("1:15.5", Fraction(151, 2)),
        ("0:01:15", 75),
    ],
)
def test_parse_offset(text, seconds):
    assert parse_offset(text) == seconds


@pytest.mark.parametrize("bad", ["", "1:2:3:4", "abc", "-5"])
def test_parse_offset_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_offset(bad)


def test_pair_anchor_moves_the_less_trusted_camera():
    # Recording 1 (camera 10) and recording 2 (camera 20) both claim to start at T0, but the
    # moment at 30 s in recording 1 is the moment at 20 s in recording 2: camera 20 is 10 s early.
    camera_of = {1: 10, 2: 20, 3: 20}
    base = {1: clock(0, 0.9), 2: clock(0, 0.5), 3: clock(600, 0.5)}
    anchors = [Anchor(1, 1, Fraction(30), 2, Fraction(20), None)]
    corrections, residuals = solve_corrections(camera_of, base, anchors)
    assert corrections == {10: 0.0, 20: 10.0}
    assert residuals[1] == pytest.approx(0)


def test_utc_anchor_sets_absolute_time():
    camera_of = {1: 10}
    base = {1: clock(0)}
    anchors = [Anchor(1, 1, Fraction(5), None, None, T0 + timedelta(seconds=2))]
    corrections, _ = solve_corrections(camera_of, base, anchors)
    assert corrections[10] == pytest.approx(-3)


def test_contradicting_anchors_show_residuals():
    camera_of = {1: 10, 2: 20}
    base = {1: clock(0, 0.9), 2: clock(0, 0.5)}
    anchors = [
        Anchor(1, 1, Fraction(30), 2, Fraction(20), None),  # says camera 20 is 10 s early
        Anchor(2, 1, Fraction(60), 2, Fraction(56), None),  # says 4 s
    ]
    corrections, residuals = solve_corrections(camera_of, base, anchors)
    assert corrections[20] == pytest.approx(7)
    assert residuals[1] == pytest.approx(3)
    assert residuals[2] == pytest.approx(3)


def test_cameras_without_anchors_are_untouched():
    corrections, residuals = solve_corrections({1: 10}, {1: clock(0)}, [])
    assert corrections == {} and residuals == {}
