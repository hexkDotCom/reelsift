"""Everything selection needs, loaded into memory once.

Selection runs many times during review, so it works on arrays rather than on the database.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction

import numpy as np

from .candidates import PHOTO_MAX
from .project import Trip
from .sync import clock

OBSERVATION_COLUMNS = (
    "motion",
    "activity",
    "shake",
    "blur",
    "brightness",
    "dark",
    "bright",
    "scene",
    "loudness",
    "speech",
)


@dataclass
class Recording:
    id: int
    camera_id: int
    camera: str
    kind: str
    duration: float
    utc_start: float | None
    """Seconds since the epoch when the recording started, or None if it isn't placed."""
    clock_confidence: float
    zone_offset_min: int | None
    width: int = 0
    height: int = 0


@dataclass
class Candidates:
    """Columns of the candidates table, as arrays aligned by index."""

    id: np.ndarray
    recording: np.ndarray
    start: np.ndarray
    end: np.ndarray
    core_start: np.ndarray
    core_end: np.ndarray
    min_start: np.ndarray
    max_end: np.ndarray
    source: np.ndarray

    def __len__(self) -> int:
        return len(self.id)


@dataclass
class VlmBlocks:
    """A recording's described blocks, as arrays; interest is NaN where the answer was invalid."""

    start: np.ndarray
    end: np.ndarray
    interest: np.ndarray
    tags: np.ndarray
    """(blocks, len(vlm.TAGS)) booleans."""
    quality: np.ndarray
    """Whether the model reported any quality problem."""
    description: list[str | None]


@dataclass
class TripData:
    recordings: dict[int, Recording]
    observations: dict[int, dict[str, np.ndarray]]
    embeddings: dict[int, np.ndarray]
    candidates: Candidates
    words: dict[int, np.ndarray] = field(default_factory=dict)
    """Per recording, an (n, 2) array of word start and end times."""
    speech: dict[int, list[tuple[float, float, str]]] = field(default_factory=dict)
    vlm: dict[int, "VlmBlocks"] = field(default_factory=dict)

    def utc(self, recording_id: int, t: float) -> float | None:
        start = self.recordings[recording_id].utc_start
        return None if start is None else start + t


def load(trip: Trip) -> TripData:
    db = trip.db
    recordings = {}
    for r in db.execute(
        "SELECT r.id, r.camera_id, r.kind, r.duration, r.width, r.height, c.label FROM recordings r"
        " JOIN cameras c ON c.id = r.camera_id"
    ):
        clk = clock(trip, r["id"])
        recordings[r["id"]] = Recording(
            id=r["id"],
            camera_id=r["camera_id"],
            camera=r["label"],
            kind=r["kind"],
            duration=PHOTO_MAX if r["kind"] == "photo" else float(Fraction(r["duration"])),
            utc_start=clk.to_utc(Fraction(0)).timestamp() if clk else None,
            clock_confidence=clk.confidence if clk else 0.0,
            zone_offset_min=clk.zone_offset_min if clk else None,
            width=r["width"] or 0,
            height=r["height"] or 0,
        )

    observations: dict[int, dict[str, np.ndarray]] = {}
    for rec_id, rec in recordings.items():
        n = max(1, int(np.ceil(rec.duration)))
        arrays = {c: np.full(n, np.nan, np.float32) for c in OBSERVATION_COLUMNS}
        observations[rec_id] = arrays
    for row in db.execute(
        f"SELECT recording_id, second, {', '.join(OBSERVATION_COLUMNS)} FROM observations"
    ):
        arrays = observations.get(row["recording_id"])
        if arrays is None:
            continue
        second = row["second"]
        n = len(arrays["motion"])
        if second >= n:
            for c in OBSERVATION_COLUMNS:
                arrays[c] = np.concatenate([arrays[c], np.full(second - n + 1, np.nan, np.float32)])
        for c in OBSERVATION_COLUMNS:
            value = row[c]
            if value is not None:
                arrays[c][second] = value

    embeddings = {}
    # Embeddings are comparable only when they come from the model that made the prompt bank;
    # after switching presets part-way, the rest stay unknown until analysis catches up.
    bank_model = _bank_model(trip)
    for row in db.execute("SELECT recording_id, path, model FROM embeddings"):
        if bank_model is not None and row["model"] != bank_model:
            continue
        path = trip.root / row["path"]
        try:
            embeddings[row["recording_id"]] = np.load(path).astype(np.float32)
        except (OSError, ValueError, EOFError):
            continue  # missing or damaged: treated as unknown until analysis runs again

    rows = db.execute(
        "SELECT id, recording_id, start, end, core_start, core_end, min_start, max_end, source"
        " FROM candidates ORDER BY recording_id, start"
    ).fetchall()
    candidates = Candidates(
        id=np.array([r["id"] for r in rows], np.int64),
        recording=np.array([r["recording_id"] for r in rows], np.int64),
        start=np.array([r["start"] for r in rows], np.float64),
        end=np.array([r["end"] for r in rows], np.float64),
        core_start=np.array([r["core_start"] for r in rows], np.float64),
        core_end=np.array([r["core_end"] for r in rows], np.float64),
        min_start=np.array([r["min_start"] for r in rows], np.float64),
        max_end=np.array([r["max_end"] for r in rows], np.float64),
        source=np.array([r["source"] for r in rows], dtype=object),
    )

    words: dict[int, list] = {}
    for w in db.execute("SELECT recording_id, start, end FROM words ORDER BY recording_id, start"):
        words.setdefault(w["recording_id"], []).append((w["start"], w["end"]))
    speech: dict[int, list] = {}
    for s in db.execute(
        "SELECT recording_id, start, end, text FROM speech_segments ORDER BY start"
    ):
        speech.setdefault(s["recording_id"], []).append((s["start"], s["end"], s["text"]))
    from .vlm import TAGS

    vlm_rows: dict[int, list] = {}
    for b in db.execute(
        "SELECT recording_id, start, end, description, tags, interest, quality FROM vlm_blocks"
        " ORDER BY recording_id, start"
    ):
        vlm_rows.setdefault(b["recording_id"], []).append(b)
    vlm = {}
    for rec_id, rows_ in vlm_rows.items():
        tag_sets = [set(json.loads(b["tags"])) for b in rows_]
        valid = [b["description"] is not None for b in rows_]
        vlm[rec_id] = VlmBlocks(
            start=np.array([b["start"] for b in rows_]),
            end=np.array([b["end"] for b in rows_]),
            interest=np.array(
                [b["interest"] if b["interest"] is not None else np.nan for b in rows_]
            ),
            tags=np.array([[t in ts for t in TAGS] for ts in tag_sets], bool).reshape(
                len(rows_), len(TAGS)
            ),
            quality=np.array([bool(json.loads(b["quality"])) for b in rows_]),
            description=[b["description"] for b in rows_],
        )
        for i, ok in enumerate(valid):
            if not ok:
                vlm[rec_id].tags[i] = False
    return TripData(
        vlm=vlm,
        recordings=recordings,
        observations=observations,
        embeddings=embeddings,
        candidates=candidates,
        words={k: np.array(v, np.float64) for k, v in words.items()},
        speech=speech,
    )


def _bank_model(trip: Trip) -> str | None:
    from .scoring import prompt_bank_model

    return prompt_bank_model(trip)


def utc_label(data: TripData, recording_id: int, t: float) -> str:
    when = data.utc(recording_id, t)
    if when is None:
        return "unknown time"
    rec = data.recordings[recording_id]
    from datetime import UTC, timedelta, timezone

    tz = (
        timezone(timedelta(minutes=rec.zone_offset_min)) if rec.zone_offset_min is not None else UTC
    )
    return datetime.fromtimestamp(when, tz).strftime("%Y-%m-%d %H:%M:%S")
