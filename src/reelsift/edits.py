"""Storing and building edits."""

import json
from dataclasses import dataclass

from .project import Trip, now
from .scoring import DEFAULT_MODE, MODES, features, load_prompt_bank, score
from .select import Item, Selection, items_from_json
from .tripdata import TripData, load


@dataclass
class StoredEdit:
    id: int
    length: int
    revision: int
    items: list[Item]
    settings: dict
    approved: bool


def save(
    trip: Trip, selection: Selection, settings: dict, soundtrack_from: int | None = None
) -> StoredEdit:
    """Store a new revision. It inherits the soundtrack of soundtrack_from (a restored
    revision), or else of the latest revision of this length."""
    from . import soundtrack

    row = trip.db.execute(
        "SELECT COALESCE(MAX(revision), 0) AS revision, (SELECT id FROM edits WHERE length = ?"
        " ORDER BY revision DESC LIMIT 1) AS latest FROM edits WHERE length = ?",
        (selection.length, selection.length),
    ).fetchone()
    revision = row["revision"] + 1
    edit_id = trip.db.execute(
        "INSERT INTO edits (length, revision, items, settings, created_at) VALUES (?, ?, ?, ?, ?)",
        (selection.length, revision, selection.to_json(), json.dumps(settings), now()),
    ).lastrowid
    source = soundtrack_from if soundtrack_from is not None else row["latest"]
    soundtrack.inherit(trip, edit_id, source, selection.items)
    return StoredEdit(edit_id, selection.length, revision, selection.items, settings, False)


def latest(trip: Trip, length: int, approved_only: bool = False) -> StoredEdit | None:
    sql = "SELECT * FROM edits WHERE length = ?"
    if approved_only:
        sql += " AND approved = 1"
    row = trip.db.execute(sql + " ORDER BY revision DESC LIMIT 1", (length,)).fetchone()
    if row is None:
        return None
    return StoredEdit(
        row["id"], row["length"], row["revision"], items_from_json(row["items"]),
        json.loads(row["settings"]), bool(row["approved"]),
    )  # fmt: skip


def build(
    trip: Trip, lengths: list[int], mode: str = DEFAULT_MODE, data: TripData | None = None
) -> list[Selection]:
    """Edits for the given lengths with a built-in scoring mode, ignoring review decisions and
    preferences (used to compare modes). Review.select_all is the full version."""
    from .moments import index as moment_index
    from .select import select

    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
    data = data or load(trip)
    if not len(data.candidates):
        raise RuntimeError("No candidates yet; run reelsift analyze first.")
    feats = features(data, load_prompt_bank(trip))
    scores, _ = score(feats, MODES[mode])
    moments = moment_index(data, feats.embedding)
    return [
        select(data, scores, feats.embedding, length, moments=moments) for length in sorted(lengths)
    ]
