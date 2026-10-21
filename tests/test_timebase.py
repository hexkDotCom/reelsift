from datetime import UTC, datetime, timedelta
from fractions import Fraction

import pytest

from reelsift.timebase import (
    ClockTransform,
    Interval,
    Provenance,
    format_zone,
    frame_at,
    from_db,
    parse_zone,
    to_db,
)


def test_fraction_round_trips_through_database_text():
    value = Fraction(1001, 30000) * 7
    assert from_db(to_db(value)) == value


def test_interval_is_half_open():
    a = Interval(Fraction(0), Fraction(5))
    assert a.contains(Fraction(0))
    assert not a.contains(Fraction(5))
    assert not a.overlaps(Interval(Fraction(5), Fraction(6)))
    assert a.intersect(Interval(Fraction(3), Fraction(9))) == Interval(Fraction(3), Fraction(5))
    assert a.intersect(Interval(Fraction(5), Fraction(9))) is None
    assert a.covers(Interval(Fraction(1), Fraction(5)))


def test_interval_rejects_negative_length():
    with pytest.raises(ValueError):
        Interval(Fraction(2), Fraction(1))


def test_frame_at_handles_ntsc_rates_exactly():
    fps = Fraction(30000, 1001)
    assert frame_at(Fraction(1001, 30000) * 1000, fps) == 1000


@pytest.mark.parametrize(
    ("text", "minutes"),
    [
        ("+02:00", 120),
        ("-0530", -330),
        ("UTC+2", 120),
        ("Z", 0),
        ("+0545", 345),
    ],
)
def test_parse_zone(text, minutes):
    assert parse_zone(text) == minutes


def test_format_zone():
    assert format_zone(-330) == "UTC-05:30"
    assert format_zone(None) == "unknown"


def test_clock_transform_maps_both_ways():
    zero = datetime(2026, 10, 11, 7, 51, 34, tzinfo=UTC)
    clock = ClockTransform(zero, Provenance.GPS, 0.9, 120)
    assert clock.to_utc(Fraction(90)) == zero + timedelta(seconds=90)
    assert clock.from_utc(zero + timedelta(seconds=90)) == 90
    assert clock.local(Fraction(0)).hour == 9


def test_clock_transform_requires_aware_time():
    with pytest.raises(ValueError):
        ClockTransform(datetime(2026, 1, 1), Provenance.GPS, 1.0)
