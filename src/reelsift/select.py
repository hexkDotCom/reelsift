"""Choosing segments for an edit of a given length. Works only on in-memory data; no models.

Greedy with penalties: repeatedly take the best remaining candidate after penalizing ones close in
time to what is already picked, ones that look like something already picked, and ones from parts
of the trip that are already well covered. Picked clips never overlap. The edit plays in time order.
"""

import json
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from .tripdata import TripData

if TYPE_CHECKING:
    from .moments import MomentIndex

GAP = 1.0
NEAR_SECONDS = 30.0
NEAR_PENALTY = 1.0
SIMILAR_COS = 0.88
SIMILAR_PENALTY = 1.5
COVERAGE_PENALTY = 0.5
MOMENT_PENALTY = 4.0

Forbidden = dict[int, list[tuple[float, float]]]
"""Per recording, intervals no automatically sized clip may touch (dropped footage)."""


def free_bounds(
    forbidden: Forbidden | None, recording_id: int, start: float, end: float, lo: float, hi: float
) -> tuple[float, float] | None:
    """Narrow [lo, hi] to the free stretch around [start, end), or None if [start, end) itself
    touches forbidden footage."""
    for a, b in (forbidden or {}).get(recording_id, []):
        if a < end and start < b:
            return None
        if b <= start:
            lo = max(lo, b)
        elif a >= end:
            hi = min(hi, a)
    return lo, hi


@dataclass
class ClipLength:
    shortest: float
    target: float
    longest: float
    longest_speech: float


def clip_length(
    edit_seconds: float, speech: str = "normal", clip: list[float] | None = None
) -> ClipLength:
    """How long clips are. Automatic: they get longer as the edit gets longer, about 2 s at 30 s,
    4 s at 3 min, 7 s at 10 min. `clip` sets [shortest, longest] by hand.

    speech: how long a clip may run to finish a sentence: "short" (about as long as other
    clips), "normal", or "long" (up to 15 s even in a short edit)."""
    x = np.clip(np.log(edit_seconds / 30) / np.log(20), 0, 1.5)
    target = float(2 + 5 * x)
    shortest, longest = max(1.2, target * 0.6), target * 1.7
    if clip is not None:
        shortest, longest = float(clip[0]), float(clip[1])
        target = (shortest + longest) / 2
    longest_speech = {
        "short": longest,
        "normal": max(longest, float(3 + 12 * min(x, 1))),
        "long": max(longest, 15.0),
    }[speech]
    return ClipLength(shortest=shortest, target=target, longest=longest,
                      longest_speech=longest_speech)  # fmt: skip


@dataclass
class Item:
    recording_id: int
    start: float
    end: float
    candidate_id: int
    score: float
    utc: float | None
    source: str
    locked: bool = False
    crop_x: float = 0.5
    """Where a 9:16 crop sits across a landscape picture: 0 left, 0.5 centre, 1 right."""
    speed: float = 1.0
    """How much faster than real time the clip plays (a ride's moving stretches); 1 elsewhere.
    start and end are footage (source) times either way."""

    @property
    def duration(self) -> float:
        """How long the clip lasts in the edit."""
        return (self.end - self.start) / self.speed

    @property
    def source_duration(self) -> float:
        """How much footage the clip uses."""
        return self.end - self.start


@dataclass
class Selection:
    length: int
    items: list[Item]
    shortfall: float
    """Seconds missing from the target because no acceptable candidates were left."""

    @property
    def duration(self) -> float:
        return sum(i.duration for i in self.items)

    def to_json(self) -> str:
        return json.dumps([asdict(i) for i in self.items])


def items_from_json(text: str) -> list[Item]:
    return [Item(**d) for d in json.loads(text)]


def avoid_words(words: np.ndarray | None, t: float, lo: float, hi: float) -> float:
    """Move a cut point out of any word it falls in, staying within [lo, hi]."""
    if words is None or not len(words):
        return t
    inside = (words[:, 0] < t) & (t < words[:, 1])
    if not inside.any():
        return t
    w_start, w_end = words[inside][0]
    for option in sorted((w_start, w_end), key=lambda x: abs(x - t)):
        if lo <= option <= hi:
            return float(option)
    return t


def clip_interval(
    data: TripData, i: int, lengths: ClipLength, forbidden: Forbidden | None = None
) -> tuple[float, float]:
    """The clip to use for candidate i: centred on its core, sized for the edit length, and never
    reaching into forbidden footage."""
    c = data.candidates
    rec = int(c.recording[i])
    core_s, core_e = float(c.core_start[i]), float(c.core_end[i])
    lo, hi = float(c.min_start[i]), float(c.max_end[i])
    bounds = free_bounds(forbidden, rec, core_s, core_e, lo, hi)
    if bounds is None:
        return core_s, core_s  # empty: the caller skips it
    lo, hi = bounds
    if c.source[i] == "photo":
        return 0.0, float(np.clip(lengths.target, core_e, c.max_end[i]))
    if c.source[i] == "speech":
        want = min(core_e - core_s + 0.5, lengths.longest_speech)
    else:
        want = lengths.target
    centre = (core_s + core_e) / 2
    start, end = centre - want / 2, centre + want / 2
    if c.source[i] == "speech" and core_e - core_s + 0.5 <= lengths.longest_speech:
        start, end = core_s - 0.25, core_e + 0.25
    start, end = max(lo, start), min(hi, end)
    words = data.words.get(rec)
    start = avoid_words(words, start, lo, core_s)
    end = avoid_words(words, end, core_e, hi)
    return start, end


def select(
    data: TripData,
    scores: np.ndarray,
    embeddings: np.ndarray,
    length: int,
    *,
    allowed: np.ndarray | None = None,
    forced: list[Item] | None = None,
    coverage_buckets: int | None = None,
    moments: "MomentIndex | None" = None,
    forbidden: Forbidden | None = None,
    clips: ClipLength | None = None,
    coverage: float = COVERAGE_PENALTY,
) -> Selection:
    """Pick clips until the edit reaches `length` seconds.

    allowed: boolean mask of candidates that may be picked (dropped moments are False).
    forced: items that must be in the edit; they count toward the length.
    moments: finds other cameras' angles of a picked moment, which are strongly penalized.
    forbidden: footage no picked clip may include (forced items are exempt).
    clips: how long clips are (automatic for the length if not given).
    coverage: penalty for parts of the trip already covered; higher spreads the edit evenly.
    """
    c = data.candidates
    n = len(c)
    lengths = clips or clip_length(length)
    allowed = np.ones(n, bool) if allowed is None else allowed.copy()
    picked: list[Item] = list(forced or [])
    total = sum(i.duration for i in picked)

    utc = np.array(
        [data.utc(int(c.recording[i]), float(c.core_start[i])) or np.nan for i in range(n)]
    )
    known = utc[~np.isnan(utc)]
    buckets = coverage_buckets or max(4, int(length / lengths.target / 3))
    edges = np.linspace(known.min(), known.max() + 1e-6, buckets + 1) if len(known) else None
    bucket = (
        np.clip(np.searchsorted(edges, utc, side="right") - 1, 0, buckets - 1)
        if edges is not None
        else np.zeros(n, int)
    )
    bucket[np.isnan(utc)] = 0
    bucket_count = np.zeros(buckets)
    chosen_emb: list[np.ndarray] = []
    repeats = np.zeros(n, bool)

    def register(item: Item, index: int | None) -> None:
        nonlocal total
        if index is not None:
            bucket_count[bucket[index]] += 1
            if moments is not None:
                repeats[:] |= moments.same_moment(index)
            if np.any(embeddings[index]):
                chosen_emb.append(embeddings[index])
        same = (
            (c.recording == item.recording_id)
            & (c.end > item.start - GAP)
            & (c.start < item.end + GAP)
        )
        allowed[same] = False

    for item in picked:
        idx = np.flatnonzero(c.id == item.candidate_id)
        register(item, int(idx[0]) if len(idx) else None)

    # Penalties are in units of the scores' spread, so strong preferences can't swamp variety.
    pool = scores[allowed] if allowed.any() else scores
    q1, q3 = np.percentile(pool, [25, 75]) if len(pool) else (0.0, 1.0)
    spread = max((q3 - q1) / 1.349, float(np.std(pool)) * 0.5 if len(pool) else 0.0, 1e-3)
    tolerance = lengths.shortest / 2
    while total < length - tolerance:
        candidates = np.flatnonzero(allowed)
        if not len(candidates):
            break
        adjusted = scores[candidates].astype(np.float64)
        for item in picked:
            same = c.recording[candidates] == item.recording_id
            centre = (c.core_start[candidates] + c.core_end[candidates]) / 2
            distance = np.abs(centre - (item.start + item.end) / 2)
            adjusted -= same * spread * NEAR_PENALTY * np.clip(1 - distance / NEAR_SECONDS, 0, 1)
        if chosen_emb:
            sim = embeddings[candidates] @ np.stack(chosen_emb).T
            adjusted -= (
                spread
                * SIMILAR_PENALTY
                * np.clip((sim.max(axis=1) - SIMILAR_COS) / (1 - SIMILAR_COS), 0, 1)
            )
        adjusted -= spread * coverage * bucket_count[bucket[candidates]]
        adjusted -= spread * MOMENT_PENALTY * repeats[candidates]
        best = int(candidates[int(np.argmax(adjusted))])
        start, end = clip_interval(data, best, lengths, forbidden)
        allowed[best] = False
        if end - start < lengths.shortest * 0.5:
            continue
        overlaps = any(
            p.recording_id == int(c.recording[best]) and p.start < end + GAP and start < p.end + GAP
            for p in picked
        )
        if overlaps:
            continue
        remaining = length - total
        if end - start > remaining + tolerance and remaining >= lengths.shortest:
            centre = (start + end) / 2
            start, end = centre - remaining / 2, centre + remaining / 2
        item = Item(
            recording_id=int(c.recording[best]),
            start=round(start, 3),
            end=round(end, 3),
            candidate_id=int(c.id[best]),
            score=round(float(scores[best]), 4),
            utc=data.utc(int(c.recording[best]), start),
            source=str(c.source[best]),
        )
        picked.append(item)
        total += item.duration
        register(item, best)

    picked.sort(
        key=lambda i: (i.utc if i.utc is not None else float("inf"), i.recording_id, i.start)
    )
    return Selection(length, picked, max(0.0, length - tolerance - total))


def expand(
    item: Item,
    target: float,
    data: TripData,
    others: list[Item],
    forbidden: Forbidden | None = None,
) -> Item:
    """Grow a clip to `target` seconds around its centre, staying within its candidate's handles,
    the recording, clear of other clips and of forbidden footage. Never shrinks it."""
    if item.duration >= target:
        return item
    c = data.candidates
    idx = np.flatnonzero(c.id == item.candidate_id)
    duration = data.recordings[item.recording_id].duration
    lo, hi = (float(c.min_start[idx[0]]), float(c.max_end[idx[0]])) if len(idx) else (0.0, duration)
    lo, hi = max(0.0, min(lo, item.start)), min(duration, max(hi, item.end))
    bounds = free_bounds(forbidden, item.recording_id, item.start, item.end, lo, hi)
    if bounds is None:
        return item
    lo, hi = bounds
    for o in others:
        if o is item or o.recording_id != item.recording_id:
            continue
        if o.end <= item.start:
            lo = max(lo, o.end + GAP)
        elif o.start >= item.end:
            hi = min(hi, o.start - GAP)
    grow = (target - item.duration) / 2
    start, end = max(lo, item.start - grow), min(hi, item.end + grow)
    # Use what one side couldn't take on the other side.
    missing = target - (end - start)
    if missing > 0:
        start, end = max(lo, start - missing), min(hi, end + missing)
    start, end = min(start, item.start), max(end, item.end)
    return replace(item, start=round(start, 3), end=round(end, 3),
                   utc=data.utc(item.recording_id, start))  # fmt: skip


def merge(items: list[Item]) -> list[Item]:
    """Join clips on the same recording that overlap. A joined clip keeps the crop and other
    details of whichever of its clips was listed first, so list the clips that matter most first;
    its start and the time of its start come from the clip that starts first."""
    out: list[Item] = []
    rank: list[int] = []
    for k, i in sorted(enumerate(items), key=lambda p: (p[1].recording_id, p[1].start)):
        if out and out[-1].recording_id == i.recording_id and i.start < out[-1].end:
            last = out[-1]
            base = i if k < rank[-1] else last
            out[-1] = replace(base, start=last.start, utc=last.utc, end=max(last.end, i.end),
                              locked=last.locked or i.locked)  # fmt: skip
            rank[-1] = min(rank[-1], k)
        else:
            out.append(i)
            rank.append(k)
    return out


SNAP_MAX = 0.4
"""A cut moves at most this far to land on a beat."""
SNAP_DRIFT = 1.0
"""Snapping changes the edit's length by at most this much in all."""
SNAP_MIN_CLIP = 0.8


def end_limits(
    item: Item,
    data: TripData,
    neighbours: list[Item],
    forbidden: Forbidden | None = None,
) -> tuple[float, float]:
    """How far a clip's end may move: within its candidate's handles and the recording, clear of
    the next clip and of forbidden footage."""
    c = data.candidates
    idx = np.flatnonzero(c.id == item.candidate_id)
    duration = data.recordings[item.recording_id].duration
    hi = float(c.max_end[idx[0]]) if len(idx) else duration
    hi = min(max(hi, item.end), duration)
    lo = item.start + (2.0 if item.source == "photo" else SNAP_MIN_CLIP)
    bounds = free_bounds(forbidden, item.recording_id, item.start, item.end, item.start, hi)
    hi = bounds[1] if bounds else item.end
    for o in neighbours:
        if o is not item and o.recording_id == item.recording_id and o.start >= item.end - 1e-6:
            hi = min(hi, o.start - GAP)
    return lo, max(lo, hi)


def snap_to_beats(
    items: list[Item],
    beats: np.ndarray,
    data: TripData,
    forbidden: Forbidden | None = None,
) -> list[Item]:
    """Move unlocked clips' ends so each cut lands on a beat, never by more than SNAP_MAX, only
    within the clip's limits, never into a word, and changing the edit's length by at most
    SNAP_DRIFT in all. Items must be in edit order."""
    if not len(beats):
        return items
    out: list[Item] = []
    t = drift = 0.0
    for item in items:
        if not item.locked:
            cut = t + item.duration
            beat = float(beats[int(np.argmin(np.abs(beats - cut)))])
            shift = beat - cut
            if abs(shift) <= SNAP_MAX:
                lo, hi = end_limits(item, data, items, forbidden)
                new_end = round(float(np.clip(item.end + shift, lo, hi)), 3)
                moved = new_end - item.end
                if (abs(moved) > 1e-4 and abs(drift + moved) <= SNAP_DRIFT
                        and not _inside_word(data, item.recording_id, new_end)):  # fmt: skip
                    item = replace(item, end=new_end)
                    drift += moved
        out.append(item)
        t += item.duration
    return out


def _inside_word(data: TripData, recording_id: int, t: float) -> bool:
    words = data.words.get(recording_id)
    if words is None or not len(words):
        return False
    return bool(np.any((words[:, 0] < t - 0.02) & (words[:, 1] > t + 0.02)))
