"""Drafting each edit's description from the trip's own footage.

After analysis, the vision model has described every few seconds of footage. A language model reads
a summary of those descriptions and writes, for each edit length, a short request in the voice of
a person asking an editor: what this edit should show, steered by the edit's style (a dynamic
preview, highlights, or a story). Each draft is then read like any description a person types,
into sliders and searches, and can be changed in the review app. Without a language model, each
style's fixed template is used instead.
"""

from collections.abc import Callable

import numpy as np

from . import prefs as prefs_mod
from .project import Trip
from .tripdata import TripData, utc_label

SUMMARY_LINES = 90
"""Most descriptions put into the prompt: half the most interesting, half spread over time."""
DRAFT_CHARS = 800


def summary(trip: Trip, data: TripData) -> str:
    """The trip's scene descriptions in time order, shortened to fit a prompt."""
    rows = trip.db.execute(
        "SELECT recording_id, start, description, interest FROM vlm_blocks"
        " WHERE description IS NOT NULL AND description != ''"
    ).fetchall()
    rows = [r for r in rows if r["recording_id"] in data.recordings]
    if not rows:
        return ""

    def when(r):
        utc = data.utc(r["recording_id"], r["start"])
        return (utc if utc is not None else float("inf"), r["recording_id"], r["start"])

    rows.sort(key=when)
    lines, last = [], None
    for r in rows:
        text = " ".join(r["description"].split())
        if text != last:
            lines.append((r, text))
        last = text
    if len(lines) > SUMMARY_LINES:
        half = SUMMARY_LINES // 2
        by_interest = sorted(range(len(lines)), key=lambda k: -(lines[k][0]["interest"] or 0))
        spread = np.linspace(0, len(lines) - 1, half).round().astype(int).tolist()
        keep = sorted(set(by_interest[:half]) | set(spread))
        lines = [lines[k] for k in keep]
    recordings = data.recordings.values()
    photos = sum(1 for r in recordings if r.kind == "photo")
    minutes = sum(r.duration for r in recordings) / 60
    cameras = len({r.camera for r in recordings})
    head = (f"{len(data.recordings) - photos} videos ({minutes:.0f} minutes) and {photos} photos "
            f"from {cameras} camera(s).")  # fmt: skip
    body = "\n".join(
        f"{utc_label(data, r['recording_id'], r['start'])} [interest {r['interest'] or '?'}] {text}"
        for r, text in lines
    )
    return f"{head}\n{body}"


def length_label(seconds: int) -> str:
    if seconds >= 60 and seconds % 60 == 0:
        return f"{seconds // 60}-minute"
    return f"{seconds}-second"


def prompt(trip_summary: str, edits: dict[int, str]) -> str:
    wanted = "\n".join(
        f'- "{n}": a {length_label(n)} video, {prefs_mod.STYLES[style].brief}.'
        for n, style in sorted(edits.items())
    )
    return f"""You help make short videos from the footage of one outing. Here is what the footage
shows, in time order (local time, how interesting it looked from 1 to 10, description):

{trip_summary}

For each video below, write what it should show, as a request a person would give a video editor:
two to four plain English sentences that name the concrete things from this footage worth including,
and anything to leave out. Follow each video's purpose.

{wanted}

Reply with only a JSON object that maps each video's key to its request."""


def write(
    trip_summary: str, edits: dict[int, str], complete: Callable[[str], str] | None
) -> dict[int, str]:
    """A description per edit length (edits: length -> style). Uses the language model when
    there is one and footage descriptions to read; any edit it doesn't answer for gets its
    style's template."""
    texts: dict[int, str] = {}
    if complete is not None and trip_summary:
        from .ml.qwen import parse_json

        try:
            answer = parse_json(complete(prompt(trip_summary, edits))) or {}
        except Exception:  # a model failure leaves the templates
            answer = {}
        for n in edits:
            text = answer.get(str(n))
            if isinstance(text, str) and text.strip():
                texts[n] = " ".join(text.split())[:DRAFT_CHARS]
    for n, style in edits.items():
        texts.setdefault(n, prefs_mod.STYLES[style].template)
    return texts


def draft(
    trip: Trip,
    data: TripData,
    edits: dict[int, str],
    complete: Callable[[str], str] | None = None,
    keep: dict[int, "prefs_mod.Preferences"] | None = None,
) -> dict[int, "prefs_mod.Preferences"]:
    """Drafted preferences per edit length (edits: length -> style), not saved. Clip lengths
    set by hand in `keep` are carried over."""
    return to_preferences(write(summary(trip, data), edits, complete), edits, complete, keep)


def to_preferences(
    texts: dict[int, str],
    edits: dict[int, str],
    complete: Callable[[str], str] | None = None,
    keep: dict[int, "prefs_mod.Preferences"] | None = None,
) -> dict[int, "prefs_mod.Preferences"]:
    """Read drafted texts into preferences, the way every draft is read."""
    out = {}
    for n, text in texts.items():
        p = prefs_mod.compile_text(text, complete)
        # A draft only leans the edit one way or another; leaving footage out entirely is for
        # the person to decide.
        p.exclusions = []
        p.style = edits[n]
        p.clip = keep[n].clip if keep and n in keep else None
        p.drafted = complete is not None and text != prefs_mod.STYLES[edits[n]].template
        out[n] = p
    return out


def missing(trip: Trip) -> list[int]:
    """Edit lengths that have no preferences of their own yet."""
    return [n for n in sorted(trip.config.lengths) if not prefs_mod.latest(trip.db, n)[1]]


def save_drafts(
    trip: Trip,
    drafted: dict[int, "prefs_mod.Preferences"],
    text_encoder: Callable[[list[str]], np.ndarray] | None = None,
) -> None:
    """Store drafted preferences, with embeddings for their searches when an encoder is given
    (searches without one can't be used until the review app embeds them)."""
    from .project import transaction
    from .scoring import prompt_bank_model

    model = prompt_bank_model(trip) or ""
    texts = sorted({s["text"] for p in drafted.values() for s in p.searches})
    cached = {
        r["text"]
        for r in trip.db.execute("SELECT text FROM text_embeddings WHERE model = ?", (model,))
    }
    new = [t for t in texts if t not in cached]
    vectors = text_encoder(new) if new and text_encoder is not None and model else []
    # A vector of another size can't come from the analysis model: never cache it under its name.
    row = trip.db.execute("SELECT dim FROM embeddings WHERE model = ? LIMIT 1", (model,)).fetchone()
    size = row["dim"] if row else None
    with transaction(trip.db):
        for text, v in zip(new, vectors, strict=False):
            if size is not None and np.asarray(v).shape != (size,):
                continue
            trip.db.execute(
                "INSERT OR REPLACE INTO text_embeddings (text, model, embedding) VALUES (?, ?, ?)",
                (text, model, np.asarray(v, np.float32).tobytes()),
            )
        for n, p in drafted.items():
            p.clean()
            p.interpretation = prefs_mod.describe(p)
            prefs_mod.save(trip.db, n, p)
