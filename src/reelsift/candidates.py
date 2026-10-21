"""Candidate segments: the pieces selection chooses from.

Two kinds of proposals, both with a core (the part that earned the candidate) and handles (how far
a cut may move outward):
- overlapping windows over every recording, so no moment falls between two candidates;
- speech phrases, split at pauses, so conversations are offered whole.

Candidates are cheap to regenerate. Decisions made in review refer to recording intervals, not to
candidate ids, so regenerating never loses them.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .project import Trip, transaction

VERSION = 1
WINDOW = 4.0
STRIDE = 2.0
HANDLE = 3.0
CORE_MARGIN = 1.0
PHRASE_GAP = 0.6
PHRASE_PAD = 0.25
MAX_PHRASE = 15.0
MIN_PHRASE = 1.0
PHOTO_CORE = 2.0
PHOTO_MAX = 4.0
"""A selected photo shows for 2 to 4 seconds, longer in longer edits."""
BURST_SECONDS = 3.0
BURST_COS = 0.92


@dataclass
class Proposal:
    start: float
    end: float
    core_start: float
    core_end: float
    source: str


def windows(duration: float) -> list[Proposal]:
    if duration <= WINDOW:
        return [Proposal(0.0, duration, 0.0, duration, "window")]
    out = []
    t = 0.0
    while t + WINDOW <= duration + 1e-6:
        out.append(Proposal(t, t + WINDOW, t + CORE_MARGIN, t + WINDOW - CORE_MARGIN, "window"))
        t += STRIDE
    if out[-1].end < duration - 0.5:
        s = duration - WINDOW
        out.append(Proposal(s, duration, s + CORE_MARGIN, duration - CORE_MARGIN, "window"))
    return out


def phrases(words: list[tuple[float, float]], duration: float) -> list[Proposal]:
    """Group words into phrases at pauses longer than PHRASE_GAP, capped at MAX_PHRASE."""
    out: list[Proposal] = []
    group: list[tuple[float, float]] = []

    def flush() -> None:
        if not group:
            return
        start, end = group[0][0], group[-1][1]
        if end - start >= MIN_PHRASE:
            s, e = max(0.0, start - PHRASE_PAD), min(duration, end + PHRASE_PAD)
            out.append(Proposal(s, e, start, end, "speech"))
        group.clear()

    for start, end in words:
        if group and (start - group[-1][1] > PHRASE_GAP or end - group[0][0] > MAX_PHRASE):
            flush()
        group.append((start, end))
    flush()
    return out


def generate(trip: Trip) -> int:
    """Rebuild candidates for every analyzed recording; returns how many were made."""
    db = trip.db
    recordings = db.execute(
        "SELECT r.id, r.duration, r.kind FROM recordings r"
        " JOIN stage_runs v ON v.recording_id = r.id AND v.stage = 'visual'"
    ).fetchall()
    best_photos = burst_leaders(trip, [r["id"] for r in recordings if r["kind"] == "photo"])
    total = 0
    with transaction(db):
        db.execute("DELETE FROM candidates")
        for rec in recordings:
            if rec["kind"] == "photo":
                if rec["id"] in best_photos:
                    db.execute(
                        "INSERT INTO candidates (recording_id, start, end, core_start, core_end,"
                        " min_start, max_end, source, version) VALUES (?, 0, ?, 0, ?, 0, ?,"
                        " 'photo', ?)",
                        (rec["id"], PHOTO_MAX, PHOTO_CORE, PHOTO_MAX, VERSION),
                    )
                    total += 1
                continue
            duration = float(Fraction(rec["duration"]))
            words = [
                (w["start"], w["end"])
                for w in db.execute(
                    "SELECT start, end FROM words WHERE recording_id = ? ORDER BY start",
                    (rec["id"],),
                )
            ]
            proposals = windows(duration) + phrases(words, duration)
            db.executemany(
                "INSERT INTO candidates (recording_id, start, end, core_start, core_end,"
                " min_start, max_end, source, version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        rec["id"],
                        p.start,
                        p.end,
                        p.core_start,
                        p.core_end,
                        max(0.0, p.start - HANDLE),
                        min(duration, p.end + HANDLE),
                        p.source,
                        VERSION,
                    )
                    for p in proposals
                ],
            )
            total += len(proposals)
    return total


def _similar(a: tuple[str, np.ndarray] | None, b: tuple[str, np.ndarray] | None) -> bool:
    """Whether two photos look almost the same. Embeddings from different models (after
    switching presets part-way) can't be compared: such photos are never joined."""
    if a is None or b is None or a[0] != b[0] or a[1].shape != b[1].shape:
        return False
    return float(a[1] @ b[1]) >= BURST_COS


def burst_leaders(trip: Trip, photo_ids: list[int]) -> set[int]:
    """The sharpest photo of each burst. A burst is a run of photos from one camera, each within
    BURST_SECONDS of the previous and looking almost the same."""
    from .sync import clock

    db = trip.db
    info = []
    for rec_id in photo_ids:
        clk = clock(trip, rec_id)
        emb_row = db.execute(
            "SELECT path, model FROM embeddings WHERE recording_id = ?", (rec_id,)
        ).fetchone()
        try:
            vector = np.load(trip.root / emb_row["path"]).astype(np.float32)[0] if emb_row else None
            emb = (emb_row["model"], vector) if vector is not None else None
        except (OSError, ValueError, EOFError, IndexError):
            emb = None  # unknown: the photo is never merged into a burst
        sharp = db.execute(
            "SELECT blur FROM observations WHERE recording_id = ? AND second = 0", (rec_id,)
        ).fetchone()
        camera = db.execute("SELECT camera_id FROM recordings WHERE id = ?", (rec_id,)).fetchone()[
            0
        ]
        when = clk.to_utc(Fraction(0)).timestamp() if clk else None
        info.append((camera, when, rec_id, emb, (sharp["blur"] or 0.0) if sharp else 0.0))
    leaders: set[int] = set()
    info.sort(key=lambda x: (x[0], x[1] if x[1] is not None else float("inf"), x[2]))
    group: list = []

    def close() -> None:
        if group:
            leaders.add(max(group, key=lambda x: x[4])[2])
            group.clear()

    for item in info:
        if group:
            prev = group[-1]
            same = (
                item[0] == prev[0]
                and item[1] is not None
                and prev[1] is not None
                and item[1] - prev[1] <= BURST_SECONDS
                and _similar(item[3], prev[3])
            )
            if not same:
                close()
        group.append(item)
    close()
    return leaders
