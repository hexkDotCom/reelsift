"""Clock sync: put every recording on one UTC timeline.

Each recording starts from its best time evidence (see devices). Manual anchors then correct whole
cameras, because a wrong camera clock is wrong for all its recordings:

- a UTC anchor says "this moment in recording A happened at this UTC time";
- a pair anchor says "this moment in A is the same instant as this moment in B".

Corrections are solved per connected group of cameras by least squares, so redundant anchors
average out and contradicting ones show up as residuals.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path

import numpy as np

from .project import Trip, now, transaction
from .timebase import ClockTransform, Provenance, from_db, to_db

STAGE = "sync"
RANK = list(Provenance)
ZONE_FROM_CAMERA_S = 2 * 86400
"""A recording takes its camera's zone from another recording at most this far away in time."""
RESIDUAL_WARNING_S = 1.0


@dataclass
class Anchor:
    id: int
    recording_a: int
    time_a: Fraction
    recording_b: int | None
    time_b: Fraction | None
    utc: datetime | None


@dataclass
class SyncSummary:
    recordings: int = 0
    by_provenance: dict[str, int] = field(default_factory=dict)
    without_time: int = 0
    unknown_zone: int = 0
    anchors: int = 0
    max_residual_s: float = 0.0


def parse_offset(text: str) -> Fraction:
    """Parse a position in a recording: "75.5", "1:15.5" or "0:01:15.5" seconds."""
    parts = text.strip().split(":")
    if not 1 <= len(parts) <= 3 or not all(re.fullmatch(r"\d+(\.\d+)?", p) for p in parts):
        raise ValueError(f"not a time position: {text!r} (use seconds, m:ss or h:mm:ss)")
    total = Fraction(0)
    for part in parts:
        total = total * 60 + Fraction(part)
    return total


def find_recording(trip: Trip, ref: str) -> int:
    """Find a recording by id, or by the name or path of one of its files."""
    db = trip.db
    if (
        ref.isdigit()
        and db.execute("SELECT 1 FROM recordings WHERE id = ?", (int(ref),)).fetchone()
    ):
        return int(ref)
    name = Path(ref).name.lower()
    stem = Path(ref).stem.lower()
    matches = {
        r["recording_id"]
        for r in db.execute(
            "SELECT c.recording_id, a.path FROM chapters c JOIN assets a ON a.id = c.asset_id"
        )
        if Path(r["path"]).name.lower() == name or Path(r["path"]).stem.lower() == stem
    }
    if len(matches) == 1:
        return matches.pop()
    if not matches:
        raise ValueError(f"no recording matches {ref!r}")
    raise ValueError(f"{ref!r} matches several recordings ({sorted(matches)}); use the id")


def add_anchor(
    trip: Trip,
    a: str,
    time_a: str,
    *,
    b: str | None = None,
    time_b: str | None = None,
    utc: str | None = None,
    note: str = "",
) -> int:
    if (b is None) == (utc is None):
        raise ValueError("an anchor needs either a second recording or a UTC time")
    rec_a = find_recording(trip, a)
    values = [rec_a, to_db(parse_offset(time_a)), None, None, None]
    if b is not None:
        if time_b is None:
            raise ValueError("a pair anchor needs a time in both recordings")
        values[2], values[3] = find_recording(trip, b), to_db(parse_offset(time_b))
    else:
        when = datetime.fromisoformat(utc.replace("Z", "+00:00"))
        if when.tzinfo is None:
            raise ValueError("give the UTC time with a zone, such as 2026-10-11T10:05:00+02:00")
        values[4] = when.astimezone(UTC).isoformat()
    return trip.db.execute(
        "INSERT INTO anchors (recording_a, time_a, recording_b, time_b, utc, note, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (*values, note, now()),
    ).lastrowid


def _anchors(trip: Trip) -> list[Anchor]:
    return [
        Anchor(
            r["id"],
            r["recording_a"],
            from_db(r["time_a"]),
            r["recording_b"],
            from_db(r["time_b"]) if r["time_b"] else None,
            datetime.fromisoformat(r["utc"]) if r["utc"] else None,
        )
        for r in trip.db.execute("SELECT * FROM anchors ORDER BY id")
    ]


def base_transforms(trip: Trip) -> dict[int, ClockTransform]:
    """Best clock per recording from its own evidence and the camera settings in the config."""
    db = trip.db
    cameras = {r["id"]: r["label"] for r in db.execute("SELECT id, label FROM cameras")}
    evidence = defaultdict(list)
    for row in db.execute("SELECT * FROM time_evidence"):
        evidence[row["recording_id"]].append(row)
    transforms = {}
    for rec in db.execute("SELECT id, camera_id FROM recordings"):
        rows = evidence.get(rec["id"])
        if not rows:
            continue
        best = min(rows, key=lambda r: (RANK.index(Provenance(r["provenance"])), -r["confidence"]))
        clock = ClockTransform(
            utc_at_zero=datetime.fromisoformat(best["utc_at_zero"]),
            provenance=Provenance(best["provenance"]),
            confidence=best["confidence"],
            zone_offset_min=best["zone_offset_min"],
        )
        cam = trip.config.camera(cameras[rec["camera_id"]])
        if (
            cam.creation_time == "utc"
            and clock.provenance == Provenance.METADATA_LOCAL_UNKNOWN_ZONE
        ):
            # Set to write true UTC: the recorded time stands, and a zone only gives local time.
            clock = ClockTransform(clock.utc_at_zero, Provenance.METADATA_UNKNOWN_ZONE,
                                   clock.confidence, clock.zone_offset_min)  # fmt: skip
        if cam.creation_time == "local" and clock.provenance != Provenance.GPS:
            if cam.zone_offset_min is None:
                trip.issue(
                    STAGE,
                    "warning",
                    f'Camera "{cameras[rec["camera_id"]]}" is set to local creation_time but '
                    "has no zone in the config",
                    recording_id=rec["id"],
                )
            else:
                clock = ClockTransform(
                    clock.utc_at_zero - timedelta(minutes=cam.zone_offset_min),
                    Provenance.METADATA_ZONE_FROM_CONFIG,
                    0.6,
                    cam.zone_offset_min,
                )
        if (
            clock.provenance == Provenance.METADATA_LOCAL_UNKNOWN_ZONE
            and cam.zone_offset_min is not None
        ):
            # Local wall-clock time: knowing the zone turns it into UTC once.
            clock = ClockTransform(
                clock.utc_at_zero - timedelta(minutes=cam.zone_offset_min),
                Provenance.METADATA_ZONE_FROM_CONFIG, 0.6, cam.zone_offset_min,
            )  # fmt: skip
        if clock.zone_offset_min is None and cam.zone_offset_min is not None:
            clock = ClockTransform(
                clock.utc_at_zero,
                Provenance.METADATA_ZONE_FROM_CONFIG,
                max(clock.confidence, 0.6),
                cam.zone_offset_min,
            )
        transforms[rec["id"]] = clock
    return _zones_from_camera(
        transforms,
        {r["id"]: r["camera_id"] for r in db.execute("SELECT id, camera_id FROM recordings")},
    )


def _zones_from_camera(
    transforms: dict[int, ClockTransform], camera_of: dict[int, int]
) -> dict[int, ClockTransform]:
    """Recordings without a zone take it from the same camera's nearest recording that has one:
    phones record their zone in photos (and iPhones in videos), Android videos only in UTC."""
    known: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for rec_id, clock in transforms.items():
        if clock.zone_offset_min is not None:
            known[camera_of[rec_id]].append((clock.utc_at_zero.timestamp(), clock.zone_offset_min))
    out = dict(transforms)
    for rec_id, clock in transforms.items():
        candidates = known.get(camera_of[rec_id])
        if clock.zone_offset_min is not None or not candidates:
            continue

        def utc_with(zone_min: int, clock: ClockTransform = clock) -> datetime:
            if clock.provenance == Provenance.METADATA_LOCAL_UNKNOWN_ZONE:
                # Local wall-clock time: what it is in UTC depends on the zone tried.
                return clock.utc_at_zero - timedelta(minutes=zone_min)
            return clock.utc_at_zero

        distance, zone_min = min(
            (abs(utc_with(z).timestamp() - t), z) for t, z in candidates
        )  # fmt: skip
        if distance > ZONE_FROM_CAMERA_S:
            continue
        if clock.provenance == Provenance.FILE_MODIFIED:
            # The zone is only for showing local time; the time itself is still a guess from
            # the file's modified time, and stays marked as one.
            out[rec_id] = ClockTransform(clock.utc_at_zero, clock.provenance, clock.confidence,
                                         zone_min)  # fmt: skip
            continue
        out[rec_id] = ClockTransform(
            utc_with(zone_min), Provenance.METADATA_ZONE_FROM_CAMERA, max(clock.confidence, 0.5),
            zone_min,
        )  # fmt: skip
    return out


def _seconds(dt: datetime) -> float:
    return dt.astimezone(UTC).timestamp()


def solve_corrections(
    camera_of: dict[int, int],
    base: dict[int, ClockTransform],
    anchors: list[Anchor],
) -> tuple[dict[int, float], dict[int, float]]:
    """Return (correction in seconds per camera, residual in seconds per anchor)."""
    edges: dict[int, set[int]] = defaultdict(set)
    usable = []
    for a in anchors:
        if a.recording_a not in base or (a.recording_b is not None and a.recording_b not in base):
            continue
        usable.append(a)
        cam_a = camera_of[a.recording_a]
        edges[cam_a].add(cam_a)
        if a.recording_b is not None:
            cam_b = camera_of[a.recording_b]
            edges[cam_a].add(cam_b)
            edges[cam_b].add(cam_a)

    corrections: dict[int, float] = {}
    residuals: dict[int, float] = {}
    seen: set[int] = set()
    for start in list(edges):
        if start in seen:
            continue
        component, stack = [], [start]
        while stack:
            cam = stack.pop()
            if cam in seen:
                continue
            seen.add(cam)
            component.append(cam)
            stack.extend(edges[cam] - seen)
        index = {cam: i for i, cam in enumerate(component)}
        rows, rhs, which = [], [], []
        absolute = False
        for a in usable:
            cam_a = camera_of[a.recording_a]
            if cam_a not in index:
                continue
            row = np.zeros(len(component))
            seen_a = _seconds(base[a.recording_a].to_utc(a.time_a))
            if a.recording_b is None:
                row[index[cam_a]] = 1
                rhs.append(_seconds(a.utc) - seen_a)
                absolute = True
            else:
                cam_b = camera_of[a.recording_b]
                row[index[cam_b]] += 1
                row[index[cam_a]] -= 1
                rhs.append(seen_a - _seconds(base[a.recording_b].to_utc(a.time_b)))
            rows.append(row)
            which.append(a.id)
        if not absolute:
            # Without any UTC anchor, keep the most trusted camera where it is.
            def trust(cam: int) -> float:
                values = [
                    base[r].confidence for r, c in camera_of.items() if c == cam and r in base
                ]
                return sum(values) / len(values) if values else 0.0

            reference = max(component, key=trust)
            row = np.zeros(len(component))
            row[index[reference]] = 1
            rows.append(row)
            rhs.append(0.0)
        matrix, target = np.array(rows), np.array(rhs)
        solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        for cam, i in index.items():
            corrections[cam] = round(float(solution[i]), 6)
        fitted = matrix @ solution - target
        for anchor_id, err in zip(which, fitted[: len(which)], strict=True):
            residuals[anchor_id] = float(abs(err))
    return corrections, residuals


def sync(trip: Trip) -> SyncSummary:
    db = trip.db
    summary = SyncSummary()
    trip.clear_issues(STAGE)
    camera_of = {
        r["id"]: r["camera_id"] for r in db.execute("SELECT id, camera_id FROM recordings")
    }
    labels = {r["id"]: r["label"] for r in db.execute("SELECT id, label FROM cameras")}
    base = base_transforms(trip)
    anchors = _anchors(trip)
    corrections, residuals = solve_corrections(camera_of, base, anchors)
    summary.anchors = len(anchors)

    unplaced: dict[int, int] = {}  # camera -> recordings placed by their modified time
    with transaction(db):
        db.execute("DELETE FROM clocks")
        for anchor_id, residual in residuals.items():
            db.execute("UPDATE anchors SET residual_s = ? WHERE id = ?", (residual, anchor_id))
            summary.max_residual_s = max(summary.max_residual_s, residual)
            if residual > RESIDUAL_WARNING_S:
                trip.issue(
                    STAGE,
                    "warning",
                    f"Anchor {anchor_id} disagrees with the others by {residual:.1f} s",
                )
        for rec_id, cam_id in camera_of.items():
            summary.recordings += 1
            clock = base.get(rec_id)
            if clock is None:
                summary.without_time += 1
                trip.issue(
                    STAGE,
                    "error",
                    "No time information; add an anchor to place it",
                    recording_id=rec_id,
                )
                continue
            correction = corrections.get(cam_id, 0.0)
            if cam_id in corrections and abs(correction) > 1e-6:
                clock = clock.shifted(correction, Provenance.MANUAL, 0.95)
            elif cam_id in corrections:
                clock = ClockTransform(
                    clock.utc_at_zero, Provenance.MANUAL, 0.95, clock.zone_offset_min
                )
            if clock.provenance == Provenance.FILE_MODIFIED:
                unplaced[cam_id] = unplaced.get(cam_id, 0) + 1
            if clock.zone_offset_min is None:
                summary.unknown_zone += 1
            summary.by_provenance[clock.provenance] = (
                summary.by_provenance.get(clock.provenance, 0) + 1
            )
            db.execute(
                "INSERT INTO clocks (recording_id, utc_at_zero, drift_ppm, provenance, confidence,"
                " zone_offset_min, base_utc_at_zero, correction_s, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rec_id,
                    clock.utc_at_zero.isoformat(),
                    clock.drift_ppm,
                    clock.provenance.value,
                    clock.confidence,
                    clock.zone_offset_min,
                    base[rec_id].utc_at_zero.isoformat(),
                    correction,
                    now(),
                ),
            )
        for cam_id, n in sorted(unplaced.items(), key=lambda kv: labels.get(kv[0], "")):
            trip.issue(
                STAGE,
                "warning",
                f'{n} recording(s) from "{labels.get(cam_id, "?")}" have no recording time in'
                " the file and are placed by their modified time, which may be wrong. Add an"
                " anchor to fix it.",
            )
        if summary.unknown_zone:
            cams = sorted(
                {
                    labels[camera_of[r]]
                    for r in camera_of
                    if r in base
                    and base[r].zone_offset_min is None
                    and camera_of[r] not in corrections
                }
            )
            if cams:
                trip.issue(
                    STAGE,
                    "info",
                    "Local time zone unknown for: "
                    + ", ".join(cams)
                    + '. Set it with [cameras."<label>"] zone = "+02:00" in reelsift.toml.',
                )
    return summary


def clock(trip: Trip, recording_id: int) -> ClockTransform | None:
    row = trip.db.execute("SELECT * FROM clocks WHERE recording_id = ?", (recording_id,)).fetchone()
    if row is None:
        return None
    return ClockTransform(
        datetime.fromisoformat(row["utc_at_zero"]),
        Provenance(row["provenance"]),
        row["confidence"],
        row["zone_offset_min"],
        row["drift_ppm"],
    )
