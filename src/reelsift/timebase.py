"""Time in Reelsift.

Source time is a position inside one recording, in seconds, kept as an exact Fraction so that
stream time bases such as 1/90000 or 1001/30000 never accumulate rounding error. Intervals are
half-open: [start, end).

A clock transform maps source time of one recording to UTC.
"""

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, timezone
from enum import StrEnum
from fractions import Fraction


def to_db(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def from_db(text: str) -> Fraction:
    return Fraction(text)


def from_pts(pts: int, time_base: Fraction) -> Fraction:
    return pts * time_base


def frame_at(t: Fraction, fps: Fraction) -> int:
    """Index of the constant-frame-rate frame showing at source time t."""
    return int(t * fps)


@dataclass(frozen=True, order=True)
class Interval:
    start: Fraction
    end: Fraction

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"interval ends before it starts: {self.start} > {self.end}")

    @property
    def duration(self) -> Fraction:
        return self.end - self.start

    def contains(self, t: Fraction) -> bool:
        return self.start <= t < self.end

    def covers(self, other: "Interval") -> bool:
        return self.start <= other.start and other.end <= self.end

    def overlaps(self, other: "Interval") -> bool:
        return self.start < other.end and other.start < self.end

    def intersect(self, other: "Interval") -> "Interval | None":
        start, end = max(self.start, other.start), min(self.end, other.end)
        return Interval(start, end) if start < end else None

    def shift(self, by: Fraction) -> "Interval":
        return Interval(self.start + by, self.end + by)


class Provenance(StrEnum):
    """Where a recording's UTC time came from, best first."""

    MANUAL = "manual"
    GPS = "gps"
    METADATA_WITH_ZONE = "metadata_with_zone"
    METADATA_ZONE_FROM_FILENAME = "metadata_zone_from_filename"
    METADATA_ZONE_FROM_CONFIG = "metadata_zone_from_config"
    METADATA_ZONE_FROM_CAMERA = "metadata_zone_from_camera"
    """The zone of the same camera's nearest recording that knows it (a phone's photo for its
    video)."""
    METADATA_UNKNOWN_ZONE = "metadata_unknown_zone"
    """True UTC, but the local zone is unknown."""
    METADATA_LOCAL_UNKNOWN_ZONE = "metadata_local_unknown_zone"
    """Local wall-clock time with no zone, provisionally read as UTC."""
    FILE_MODIFIED = "file_modified"


def zone(offset_minutes: int | None) -> timezone | None:
    return None if offset_minutes is None else timezone(timedelta(minutes=offset_minutes))


def format_zone(offset_minutes: int | None) -> str:
    if offset_minutes is None:
        return "unknown"
    sign = "+" if offset_minutes >= 0 else "-"
    hours, minutes = divmod(abs(offset_minutes), 60)
    return f"UTC{sign}{hours:02d}:{minutes:02d}"


def parse_zone(text: str) -> int:
    """Parse "+02:00", "-0530" or "UTC+2" into minutes east of UTC."""
    value = text.strip().upper().removeprefix("UTC").strip()
    if value in ("", "Z"):
        return 0
    sign = -1 if value.startswith("-") else 1
    value = value.lstrip("+-")
    if ":" in value:
        hours, minutes = value.split(":", 1)
    elif len(value) > 2:
        hours, minutes = value[:-2], value[-2:]
    else:
        hours, minutes = value, "0"
    try:
        h, m = int(hours), int(minutes)
    except ValueError:
        raise ValueError(f"not a time zone: {text!r} (use a form like +02:00)") from None
    # Real zones run from UTC-12:00 to UTC+14:00.
    if not 0 <= m < 60 or h * 60 + m > 14 * 60 or (sign < 0 and h * 60 + m > 12 * 60):
        raise ValueError(f"not a time zone: {text!r} (zones run from -12:00 to +14:00)")
    return sign * (h * 60 + m)


@dataclass(frozen=True)
class ClockTransform:
    """Maps source time of a recording to UTC: utc = utc_at_zero + t * (1 + drift_ppm / 1e6)."""

    utc_at_zero: datetime
    provenance: Provenance
    confidence: float
    zone_offset_min: int | None = None
    drift_ppm: float = 0.0

    def __post_init__(self) -> None:
        if self.utc_at_zero.tzinfo is None:
            raise ValueError("utc_at_zero must be timezone-aware")

    def to_utc(self, t: Fraction) -> datetime:
        seconds = float(t) * (1 + self.drift_ppm / 1e6)
        return self.utc_at_zero.astimezone(UTC) + timedelta(seconds=seconds)

    def from_utc(self, when: datetime) -> Fraction:
        seconds = (when.astimezone(UTC) - self.utc_at_zero.astimezone(UTC)).total_seconds()
        return Fraction(seconds / (1 + self.drift_ppm / 1e6)).limit_denominator(1_000_000)

    def local(self, t: Fraction) -> datetime:
        """Wall-clock time where the recording was made, when the zone is known."""
        tz = zone(self.zone_offset_min)
        return self.to_utc(t).astimezone(tz) if tz else self.to_utc(t)

    def shifted(
        self, seconds: float, provenance: Provenance, confidence: float
    ) -> "ClockTransform":
        return replace(
            self,
            utc_at_zero=self.utc_at_zero + timedelta(seconds=seconds),
            provenance=provenance,
            confidence=confidence,
        )
