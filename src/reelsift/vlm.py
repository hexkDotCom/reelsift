"""Describing footage with a vision-language model, in fixed blocks.

Every recording is cut into 6-second blocks (independent of candidates, so candidates can change
without re-running the model). Each block gets six frames at 1 fps and one question; the answer is
validated against a fixed vocabulary. An invalid answer is stored as unknown, never as "no".
"""

import json
import math
from collections.abc import Callable

import numpy as np

from .analysis import chapter_media, stream_frames
from .jobs import Job, Progress
from .project import Trip, now, transaction

VERSION = 1
BLOCK = 6.0
FRAMES = 6
LONG_SIDE = 448
CHUNK = 32

TAGS = (
    "scenery", "landmark", "people", "conversation", "laughter", "kids", "animal", "water",
    "food", "city", "indoors", "sky", "action", "sport", "vehicle", "walking", "riding", "stop",
    "road", "trail", "crowd", "celebration", "close_up", "boring",
)  # fmt: skip
QUALITY = ("blur", "shake", "dark", "overexposed", "obstructed")

PROMPT = f"""These {FRAMES} frames are evenly spaced over {BLOCK:g} seconds of personal footage from
a camera or phone during an outing (a trip, a walk, a day with family or friends).
Reply with only a JSON object, no other text:
{{"description": "<one short sentence about what happens>",
  "tags": [<zero or more of {json.dumps(list(TAGS))}>],
  "interest": <integer 1-10: how much someone watching a highlight video of this outing would want to see this; 1 = nothing happens, 10 = a highlight>,
  "quality": [<zero or more of {json.dumps(list(QUALITY))}>]}}"""


def _string_list(value) -> list[str] | None:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        return None
    return value


def validate(answer) -> dict | None:
    """The answer if it matches the schema, else None (stored as unknown). Unknown tags are
    dropped; anything of the wrong type, or an interest that isn't a whole number from 1 to 10,
    makes the whole answer unknown."""
    if not isinstance(answer, dict) or not isinstance(answer.get("description"), str):
        return None
    interest = answer.get("interest")
    if isinstance(interest, bool) or not isinstance(interest, int | float):
        return None
    if not math.isfinite(interest) or interest != int(interest) or not 1 <= interest <= 10:
        return None
    tags = _string_list(answer.get("tags"))
    quality = _string_list(answer.get("quality"))
    if tags is None or quality is None:
        return None
    return {
        "description": answer["description"].strip()[:300],
        "tags": sorted({t for t in tags if t in TAGS}),
        "interest": float(interest),
        "quality": sorted({q for q in quality if q in QUALITY}),
    }


PHOTO_PROMPT = PROMPT.replace(
    f"These {FRAMES} frames are evenly spaced over {BLOCK:g} seconds of personal footage from\n"
    "a camera or phone",
    "This is a photo taken with a camera or phone",
)


def frame_size(width: int, height: int) -> tuple[int, int]:
    if width >= height:
        return LONG_SIDE, max(2, round(LONG_SIDE * height / width / 2) * 2)
    return max(2, round(LONG_SIDE * width / height / 2) * 2), LONG_SIDE


def blocks(trip: Trip, recording_id: int):
    """Yield (start, end, frames) for each block of a recording; a photo is one block."""
    kind = trip.db.execute("SELECT kind FROM recordings WHERE id = ?", (recording_id,)).fetchone()[
        0
    ]
    if kind == "photo":
        for chapter in chapter_media(trip, recording_id):
            w, h = frame_size(chapter.width, chapter.height)
            for batch in stream_frames(chapter.proxy, None, w, h, gray=False, batch=1):
                yield 0.0, 0.0, [batch[0]]
                return
        return
    for chapter in chapter_media(trip, recording_id):
        w, h = frame_size(chapter.width, chapter.height)
        pending: list[np.ndarray] = []
        index = 0
        for batch in stream_frames(chapter.proxy, FRAMES / BLOCK, w, h, gray=False, batch=FRAMES):
            for frame in batch:
                pending.append(frame)
                if len(pending) == FRAMES:
                    start = chapter.start + index * BLOCK
                    yield start, start + BLOCK, pending
                    pending, index = [], index + 1
        if len(pending) >= 2:
            start = chapter.start + index * BLOCK
            yield start, start + len(pending) * BLOCK / FRAMES, pending


def version(model_name: str) -> str:
    return f"{VERSION}:{model_name}"


def run(trip: Trip, recording_id: int, describer) -> None:
    from .ml.qwen import parse_json

    rows = []
    chunk: list[tuple[float, float, list]] = []

    kind = trip.db.execute("SELECT kind FROM recordings WHERE id = ?", (recording_id,)).fetchone()[
        0
    ]
    prompt = PHOTO_PROMPT if kind == "photo" else PROMPT

    def flush() -> None:
        answers = describer.describe([c[2] for c in chunk], prompt)
        for (start, end, _), answer in zip(chunk, answers, strict=True):
            try:
                v = validate(parse_json(answer))
            except Exception:  # one bad answer must not lose the recording's other blocks
                v = None
            rows.append(
                (
                    recording_id, start, end,
                    v["description"] if v else None,
                    json.dumps(v["tags"] if v else []),
                    v["interest"] if v else None,
                    json.dumps(v["quality"] if v else []),
                    describer.name,
                )
            )  # fmt: skip
        chunk.clear()

    for block in blocks(trip, recording_id):
        chunk.append(block)
        if len(chunk) >= CHUNK:
            flush()
    if chunk:
        flush()
    with transaction(trip.db):
        trip.db.execute("DELETE FROM vlm_blocks WHERE recording_id = ?", (recording_id,))
        trip.db.executemany(
            "INSERT INTO vlm_blocks (recording_id, start, end, description, tags, interest,"
            " quality, model) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        trip.db.execute(
            "INSERT OR REPLACE INTO stage_runs (recording_id, stage, version, finished_at)"
            " VALUES (?, 'vlm', ?, ?)",
            (recording_id, version(describer.name), now()),
        )


def analyze(
    trip: Trip,
    recordings: list[int],
    model_name: str,
    describer_factory: Callable[[], object],
    progress: Progress | None = None,
) -> tuple[int, dict[str, str]]:
    """Describe every recording that isn't described with this model yet.

    The model is loaded only when there is something to describe."""
    v = version(model_name)
    done = {
        r["recording_id"]
        for r in trip.db.execute(
            "SELECT recording_id FROM stage_runs WHERE stage = 'vlm' AND version = ?", (v,)
        )
    }
    todo = [r for r in recordings if r not in done]
    if not todo:
        return 0, {}
    describer = describer_factory()
    items = list(map(str, todo))
    result = Job(trip, "vlm", {"version": v}, items, redo=items).run(
        lambda item: run(trip, int(item), describer), progress
    )
    return result.done, result.failed
