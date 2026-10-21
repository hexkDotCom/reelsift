"""Finding and preparing media for the browser: proxies, positions in them, thumbnails."""

import os
import threading
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .. import proxies as proxies_mod
from ..photos import PHOTO_EXTENSIONS
from ..project import Trip
from ..tools import run, tool


@dataclass
class Position:
    chapter: int
    time: float
    """Seconds into the chapter's proxy, as a browser's video.currentTime."""


def chapters(trip: Trip, recording_id: int) -> list[tuple[int, float, float, Path]]:
    """(index, start, duration, proxy path) for each chapter with a proxy."""
    rows = trip.db.execute(
        "SELECT c.idx, c.start, c.duration, p.path FROM chapters c JOIN proxies p"
        " ON p.recording_id = c.recording_id AND p.chapter_idx = c.idx"
        " WHERE c.recording_id = ? ORDER BY c.idx",
        (recording_id,),
    ).fetchall()
    return [
        (r["idx"], float(Fraction(r["start"])), float(Fraction(r["duration"])),
         proxies_mod.resolve(trip, r["path"]))
        for r in rows
    ]  # fmt: skip


def locate(trip: Trip, recording_id: int, t: float) -> Position | None:
    found = chapters(trip, recording_id)
    for idx, start, duration, _ in found:
        if start <= t < start + duration:
            return Position(idx, t - start)
    if found:
        idx, start, duration, _ = found[-1]
        return Position(idx, min(max(0.0, t - start), duration))
    return None


def split_spans(chapters_: list[tuple[int, float, float]], start: float, end: float) -> list[dict]:
    """The parts of [start, end) in recording time that lie in each chapter, as chapter-local
    times: [{"chapter", "start", "end"}]. chapters_ holds (index, start, duration)."""
    spans = []
    for idx, c_start, c_dur in chapters_:
        if c_dur <= 0:  # a photo: one span covering the clip
            return [{"chapter": idx, "start": 0.0, "end": round(end - start, 3)}]
        a, b = max(start, c_start), min(end, c_start + c_dur)
        if b > a:
            spans.append(
                {"chapter": idx, "start": round(a - c_start, 3), "end": round(b - c_start, 3)}
            )
    return spans


def preview_spans(trip: Trip, recording_id: int, start: float, end: float) -> list[dict]:
    return split_spans(
        [(idx, c_start, c_dur) for idx, c_start, c_dur, _ in chapters(trip, recording_id)],
        start,
        end,
    )


def proxy_path(trip: Trip, recording_id: int, chapter: int) -> Path | None:
    for idx, _, _, path in chapters(trip, recording_id):
        if idx == chapter:
            return path
    return None


def thumbnail_job(trip: Trip, recording_id: int, t: float, width: int = 320):
    """Where the thumbnail goes, and a function that makes it if missing.

    Looking up the position needs the database; making the image doesn't, so callers can do the
    slow part without holding their lock.
    """
    position = locate(trip, recording_id, t)
    if position is None:
        return None, None
    source = proxy_path(trip, recording_id, position.chapter)
    folder = trip.root / "cache" / "thumbs"
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{recording_id}_{position.chapter}_{round(position.time * 1000)}_{width}.jpg"

    def make() -> Path:
        if not out.is_file():
            partial = out.with_name(f"{out.stem}.{threading.get_ident()}.partial.jpg")
            # A photo's proxy is one picture: seeking in it makes FFmpeg 9 write nothing.
            still = source.suffix.lower() in PHOTO_EXTENSIONS
            seek = [] if still else ["-ss", f"{position.time:.3f}"]
            run([
                tool("ffmpeg"), "-v", "error", "-y", *seek, "-i", str(source),
                "-frames:v", "1", "-update", "1", "-vf", f"scale={width}:-2", "-q:v", "4",
                str(partial),
            ])  # fmt: skip
            if not partial.is_file():
                raise RuntimeError(
                    f"FFmpeg made no picture from {source.name} at {position.time:.1f} s"
                )
            os.replace(partial, out)
        return out

    return out, make


def thumbnail(trip: Trip, recording_id: int, t: float, width: int = 320) -> Path | None:
    _, make = thumbnail_job(trip, recording_id, t, width)
    return make() if make else None
