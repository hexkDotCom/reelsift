"""Rides: the whole trip from one camera in time order, with stops cut out and the rest sped up.

A ride is its own kind of edit (edit key RIDE in the edits table), next to the edits of each
length: it shares revisions, soundtracks, rendering and export with them, but is built here
rather than chosen from scored moments. What it is made from:

- the camera (by default the one with the most footage) and its recordings in time order;
- for each second of them, whether the camera was travelling (travel.py: GPS speed where the
  camera recorded it, otherwise the picture); only clear stops are cut, anything uncertain stays;
- the user's corrections: stretches dropped, and stops kept at normal speed;
- a speed: fixed, or fitted so the ride lasts a target length.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import speed as speed_mod
from . import travel
from .media import parse_probe
from .project import Trip, now, transaction
from .select import Item, Selection

RIDE = -1
"""The edit key of the trip's ride (edits of a length use the length, always positive)."""
MAX_SPEED = 64.0
SPEEDS = (1, 2, 4, 8, 16, 32, 64)
MERGE_GAP_S = 1e-6
"""Kept stretches of one recording are joined only where they touch: a gap is always something
cut on purpose."""


def _ride_mix():
    from .soundtrack import Mix

    return Mix(music=0.0, background=0.0, sound=-20.0)


RIDE_MIX = _ride_mix()
"""A ride's own default levels: the music up, the sped-up clip sound low."""


class RideError(ValueError):
    pass


@dataclass
class RideSettings:
    target: float | None = None
    """Seconds the ride should last (with speed "fit")."""
    speed: str = "8"
    """"fit" (whatever makes it last `target`) or a factor such as "8"."""
    camera: str | None = None
    """Camera label; None: the camera with the most footage."""
    keep_unknown: bool = True
    """Keep footage whose movement isn't clear (never cut what might be travel)."""
    overrides: list[dict] = field(default_factory=list)
    """{"kind": "drop" | "keep", "recording_id", "start", "end"}: stretches dropped, and stops
    kept at normal speed."""

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict | None) -> "RideSettings":
        data = data or {}
        out = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        out.validate()
        return out

    def validate(self) -> None:
        if self.speed != "fit":
            s = float(self.speed)
            if not (math.isfinite(s) and 1 <= s <= MAX_SPEED):
                raise RideError(f"speed must be fit or from 1 to {MAX_SPEED:g}")
        elif self.target is None:
            raise RideError("speed fit needs a target length")
        if self.target is not None and not (math.isfinite(self.target) and self.target > 0):
            raise RideError("the target length must be a positive number of seconds")
        for o in self.overrides:
            if o.get("kind") not in ("drop", "keep") or not o.get("end", 0) > o.get("start", 0):
                raise RideError(f"bad ride correction: {o}")


# --- travel data ----------------------------------------------------------------------------------


def analyse_travel(trip: Trip, recording_ids: list[int], say=print) -> None:
    """Speed and picture travel per second for recordings that don't have them yet (the travel
    table). GPS is read from the original files, so it needs no proxies; the picture is used for
    seconds without GPS and needs the proxies. A recording whose picture was needed but had no
    proxy is analysed again next time."""
    from fractions import Fraction

    from . import proxies as proxies_mod

    todo = [r for r in recording_ids if not trip.db.execute(
        "SELECT 1 FROM stage_runs WHERE recording_id = ? AND stage = 'travel' AND version = ?",
        (r, _version())).fetchone()]  # fmt: skip
    for n, rec in enumerate(todo):
        chapters = trip.db.execute(
            "SELECT c.start, c.duration, a.path, a.probe_json, p.path AS proxy, p.width, p.height"
            " FROM chapters c JOIN assets a ON a.id = c.asset_id LEFT JOIN proxies p"
            " ON p.recording_id = c.recording_id AND p.chapter_idx = c.idx"
            " WHERE c.recording_id = ? ORDER BY c.idx", (rec,),
        ).fetchall()  # fmt: skip
        duration = float(sum(Fraction(c["duration"]) for c in chapters))
        seconds = math.ceil(duration)
        kmh = np.full(seconds, np.nan)
        for c in chapters:
            start = float(Fraction(c["start"]))
            path = Path(c["path"])
            found = speed_mod.samples(path, parse_probe(path, json.loads(c["probe_json"])))
            if found:
                per = speed_mod.per_second([(start + t, v) for t, v in found], duration)
                kmh = np.where(np.isfinite(per), per, kmh)
        optical = np.full(seconds, np.nan)
        complete = True
        if np.isfinite(kmh).mean() < 0.95:  # seconds without GPS: the picture
            say(f"  travel from the picture: recording {rec} ({n + 1} of {len(todo)})")
            grid = np.full((seconds, travel.GRID**2), np.nan)
            for c in chapters:
                if not c["proxy"]:
                    complete = False
                    continue
                start = float(Fraction(c["start"]))
                proxy = proxies_mod.resolve(trip, c["proxy"])
                for local, tiles in travel.tile_flow(proxy, c["width"], c["height"]).items():
                    second = math.floor(start + local)
                    if 0 <= second < seconds:
                        grid[second] = tiles
            optical = travel.travel_values(grid)
        with transaction(trip.db):
            trip.db.execute("DELETE FROM travel WHERE recording_id = ?", (rec,))
            trip.db.executemany(
                "INSERT INTO travel (recording_id, second, kmh, optical) VALUES (?, ?, ?, ?)",
                [(rec, k, _num(kmh[k]), _num(optical[k])) for k in range(seconds)],
            )
            trip.db.execute("DELETE FROM stage_runs WHERE recording_id = ? AND stage = 'travel'",
                            (rec,))  # fmt: skip
            if complete:
                trip.db.execute(
                    "INSERT INTO stage_runs (recording_id, stage, version, finished_at)"
                    " VALUES (?, 'travel', ?, ?)", (rec, _version(), now()),
                )  # fmt: skip


def _version() -> str:
    return f"travel{travel.VERSION}-speed{speed_mod.VERSION}"


def _num(v: float) -> float | None:
    return float(v) if math.isfinite(v) else None


SESSION_GAP_S = 2.0
"""Recordings of one camera that follow each other within this are one session: a stop that a
dashcam splits between two files is still one stop."""


def states(trip: Trip, recording_ids: list[int]) -> dict[int, tuple[np.ndarray, str]]:
    """MOVING / STOPPED / UNKNOWN per second, and where it mostly came from ("gps" or
    "picture"), per recording. Recordings are given in time order (one camera); consecutive ones
    are classified together. GPS decides every second it covers; the picture the others."""
    from fractions import Fraction

    from .sync import clock

    raw = {}
    for rec in recording_ids:
        rows = trip.db.execute("SELECT second, kmh, optical FROM travel WHERE recording_id = ?"
                               " ORDER BY second", (rec,)).fetchall()  # fmt: skip
        kmh = np.array([np.nan if r["kmh"] is None else r["kmh"] for r in rows])
        opt = np.array([np.nan if r["optical"] is None else r["optical"] for r in rows])
        raw[rec] = (kmh, opt)
    pictures = [o[~np.isfinite(k)] for k, o in raw.values()]
    pictures = [o for o in pictures if len(o)]
    t = travel.thresholds(np.concatenate(pictures)) if pictures else None
    # Sessions of recordings that meet in time.
    sessions: list[list[int]] = []
    previous_end = None
    for rec in recording_ids:
        clk = clock(trip, rec)
        row = trip.db.execute("SELECT duration FROM recordings WHERE id = ?", (rec,)).fetchone()
        duration = float(Fraction(row["duration"])) if row else 0.0
        start = clk.to_utc(Fraction(0)).timestamp() if clk else None
        joins = (sessions and start is not None and previous_end is not None
                 and -1.0 <= start - previous_end <= SESSION_GAP_S)  # fmt: skip
        if joins:
            sessions[-1].append(rec)
        else:
            sessions.append([rec])
        previous_end = start + duration if start is not None else None
    out = {}
    for session in sessions:
        kmh = np.concatenate([raw[r][0] for r in session]) if session else np.zeros(0)
        opt = np.concatenate([raw[r][1] for r in session]) if session else np.zeros(0)
        by_gps = travel.classify_gps(kmh)
        by_picture = travel.classify(opt, t)
        combined = np.where(np.isfinite(kmh), by_gps, by_picture)
        # Where the two meet, a stop must still be long enough to count.
        for a, b in travel.runs(combined, travel.STOPPED):
            if b - a < travel.MIN_STOP_S - 2 * travel.PAD_S:
                combined[a:b] = travel.UNKNOWN
        offset = 0
        for rec in session:
            n = len(raw[rec][0])
            part = combined[offset : offset + n]
            offset += n
            source = "gps" if n and np.isfinite(raw[rec][0]).mean() >= 0.5 else "picture"
            out[rec] = (part, source)
    return out


def global_drops(trip: Trip) -> dict[int, list[tuple[float, float]]]:
    """Footage dropped from every edit, by recording."""
    out: dict[int, list[tuple[float, float]]] = {}
    for d in trip.db.execute(
        "SELECT recording_id, start, end FROM decisions WHERE kind = 'drop' AND active = 1"
        " AND length IS NULL ORDER BY id"
    ):  # fmt: skip
        out.setdefault(d["recording_id"], []).append((d["start"], d["end"]))
    return out


def subtract_items(items: list[Item], drops: dict[int, list[tuple[float, float]]]) -> list[Item]:
    """The ride without the given footage: each stretch keeps its speed and place; a dropped
    piece of it splits it in two."""
    from dataclasses import replace

    out = []
    for item in items:
        spans = [(item.start, item.end)]
        for a, b in drops.get(item.recording_id, []):
            spans = _subtract(spans, a, b)
        for a, b in spans:
            if b - a > 0.05:
                utc = item.utc + (a - item.start) if item.utc is not None else None
                out.append(replace(item, start=round(a, 3), end=round(b, 3), utc=utc))
    return out


# --- building ------------------------------------------------------------------------------------


@dataclass
class Report:
    camera: str
    recordings: int
    footage: float
    """Seconds of footage of the camera."""
    moving: float
    stopped: float
    """Seconds cut as stops."""
    unknown: float
    """Seconds of uncertain movement (kept, unless keep_unknown is off)."""
    kept_stops: float
    dropped: float
    speed: float
    duration: float
    """How long the ride lasts."""
    sources: dict = field(default_factory=dict)
    """How movement was found: {"gps": n, "picture": n} recordings."""
    notes: list[str] = field(default_factory=list)

    def options(self) -> list[tuple[float, float]]:
        """(speed, resulting seconds) for the usual speeds."""
        return [(s, self.moving / s + self.kept_stops) for s in SPEEDS]


def cameras(trip: Trip) -> list[tuple[str, float]]:
    """Video cameras and their seconds of footage, most first."""
    rows = trip.db.execute(
        "SELECT c.label, SUM(CAST(r.duration AS REAL)) AS seconds FROM recordings r"
        " JOIN cameras c ON c.id = r.camera_id WHERE r.kind = 'video' GROUP BY c.id"
    ).fetchall()
    from fractions import Fraction

    out = []
    for r in rows:
        total = sum(float(Fraction(d["duration"])) for d in trip.db.execute(
            "SELECT r.duration FROM recordings r JOIN cameras c ON c.id = r.camera_id"
            " WHERE c.label = ? AND r.kind = 'video'", (r["label"],)))  # fmt: skip
        out.append((r["label"], total))
    return sorted(out, key=lambda c: -c[1])


def recordings_of(trip: Trip, camera: str) -> list[tuple[int, float, float | None]]:
    """(recording id, duration, UTC start or None) of a camera's videos, in time order (by name
    where a time is missing)."""
    from fractions import Fraction

    from .sync import clock

    rows = trip.db.execute(
        "SELECT r.id, r.duration, (SELECT a.path FROM chapters ch JOIN assets a ON"
        " a.id = ch.asset_id WHERE ch.recording_id = r.id ORDER BY ch.idx LIMIT 1) AS path"
        " FROM recordings r JOIN cameras c ON c.id = r.camera_id"
        " WHERE c.label = ? AND r.kind = 'video'", (camera,),
    ).fetchall()  # fmt: skip
    out = []
    for r in rows:
        clk = clock(trip, r["id"])
        utc = clk.to_utc(Fraction(0)).timestamp() if clk else None
        out.append((r["id"], float(Fraction(r["duration"])), utc, r["path"]))
    out.sort(key=lambda r: (r[2] is None, r[2] or 0, r[3] or ""))
    return [(rid, dur, utc) for rid, dur, utc, _ in out]


def _subtract(spans: list[tuple[float, float]], a: float, b: float) -> list[tuple[float, float]]:
    out = []
    for s, e in spans:
        if e <= a or s >= b:
            out.append((s, e))
            continue
        if s < a:
            out.append((s, a))
        if e > b:
            out.append((b, e))
    return out


def _merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(spans):
        if out and s - out[-1][1] < MERGE_GAP_S:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out if e - s > 0.05]


def build(trip: Trip, settings: RideSettings, say=print) -> tuple[list[Item], Report]:
    """The ride's stretches in order, and what went into it."""
    from fractions import Fraction  # noqa: F401  (kept for clarity of units elsewhere)

    settings.validate()
    found = cameras(trip)
    if not found:
        raise RideError("This trip has no videos to make a ride from.")
    camera = settings.camera or found[0][0]
    if camera not in [c for c, _ in found]:
        raise RideError(f"No camera called {camera!r}; cameras: {', '.join(c for c, _ in found)}")
    recs = recordings_of(trip, camera)
    analyse_travel(trip, [r for r, _, _ in recs], say)
    by_rec = states(trip, [r for r, _, _ in recs])
    dropped_globally = global_drops(trip)
    footage = moving = stopped = unknown = kept = dropped = 0.0
    sources: dict[str, int] = {}
    stretches: list[tuple[int, float, float, bool]] = []  # recording, start, end, normal speed
    for rec, duration, _utc in recs:
        st, source = by_rec[rec]
        sources[source] = sources.get(source, 0) + 1
        footage += duration
        stopped += float((st == travel.STOPPED).sum())
        unknown += float((st == travel.UNKNOWN).sum())
        fast = [(0.0, duration)]
        for a, b in travel.runs(st, travel.STOPPED):
            fast = _subtract(fast, a, b)
        if not settings.keep_unknown:
            for a, b in travel.runs(st, travel.UNKNOWN):
                fast = _subtract(fast, a, b)
        normal: list[tuple[float, float]] = []
        # Corrections in the order they were made: the later one wins where they overlap.
        for o in settings.overrides:
            if o["recording_id"] != rec:
                continue
            a, b = max(0.0, o["start"]), min(duration, o["end"])
            if o["kind"] == "keep":
                fast = _subtract(fast, a, b)
                normal = [*_subtract(normal, a, b), (a, b)]
            else:
                before = sum(e - s for s, e in fast + normal)
                fast, normal = _subtract(fast, a, b), _subtract(normal, a, b)
                dropped += before - sum(e - s for s, e in fast + normal)
        for a, b in dropped_globally.get(rec, []):  # dropped from every edit: always out
            fast, normal = _subtract(fast, a, b), _subtract(normal, a, b)
        for a, b in _merge(fast):
            stretches.append((rec, a, b, False))
            moving += b - a
        for a, b in _merge(normal):
            stretches.append((rec, a, b, True))
            kept += b - a
    order = {rec: k for k, (rec, _, _) in enumerate(recs)}
    stretches.sort(key=lambda s: (order[s[0]], s[1]))
    if moving <= 0 and kept <= 0:
        raise RideError("Nothing is left for the ride: no travel was found in this camera's "
                        "footage (or it was all dropped).")  # fmt: skip
    if settings.speed == "fit":
        target = float(settings.target)
        if moving <= 0:
            if target < kept - 0.5:
                raise RideError(f"The stops kept at normal speed alone last {kept:.0f} s, longer "
                                f"than the {target:.0f} s target.")  # fmt: skip
        elif target <= kept:
            raise RideError(f"The stops kept at normal speed alone last {kept:.0f} s, longer "
                            f"than the {target:.0f} s target.")  # fmt: skip
        s = max(1.0, moving / (target - kept)) if moving else 1.0
        if s > MAX_SPEED:
            need = moving / MAX_SPEED + kept
            raise RideError(f"The whole route needs at least {need:.0f} s even at "
                            f"{MAX_SPEED:g}x; choose a longer length (or drop stretches).")  # fmt: skip
    else:
        s = float(settings.speed)
    utc_of = {rec: utc for rec, _, utc in recs}
    items = [
        Item(rec, round(a, 3), round(b, 3), -1, 0.0,
             (utc_of[rec] + a) if utc_of[rec] is not None else None, "ride",
             speed=1.0 if normal_speed else round(s, 6))
        for rec, a, b, normal_speed in stretches if b - a > 0.05
    ]  # fmt: skip
    report = Report(camera, len(recs), footage, moving, stopped, unknown, kept, dropped, s,
                    moving / s + kept, sources)  # fmt: skip
    others = [c for c, _ in found if c != camera]
    if others:
        report.notes.append(f"Only {camera} is used; other cameras: {', '.join(others)}.")
    if sources.get("picture"):
        report.notes.append(
            f"{sources['picture']} recording(s) without GPS: movement judged from the picture; "
            "only clear stops were cut.")  # fmt: skip
    return items, report


def save(trip: Trip, items: list[Item], settings: RideSettings, report: Report, note: str):
    """A new revision of the ride."""
    from . import edits as edits_mod
    from . import soundtrack

    first = edits_mod.latest(trip, RIDE) is None
    stored = edits_mod.save(trip, Selection(RIDE, items, 0.0),
                            {"note": note, "ride": settings.to_json(), "report": asdict(report)})  # fmt: skip
    if first:
        # The ride's own default levels (the music up, the sped-up clip sound low), under any
        # level set by hand for the ride or the trip.
        from .config import MusicSettings

        cfg = trip.config
        explicit = cfg.music_lengths.get(RIDE, MusicSettings()).over(cfg.music_settings).over(
            cfg.user_music_lengths.get(RIDE, MusicSettings())).over(cfg.user_music_settings)  # fmt: skip
        st = soundtrack.get(trip, stored.id)
        st.mix = soundtrack.Mix(**{
            k: getattr(explicit, k) if getattr(explicit, k) is not None else getattr(RIDE_MIX, k)
            for k in ("music", "background", "duck", "sound")
        })  # fmt: skip
        soundtrack.put(trip, st)
    return stored


def settings_of(trip: Trip) -> RideSettings:
    """The ride's settings: its latest revision's, else reelsift.toml's [ride]."""
    from . import edits as edits_mod

    edit = edits_mod.latest(trip, RIDE)
    if edit is not None and "ride" in edit.settings:
        return RideSettings.from_json(edit.settings["ride"])
    return RideSettings.from_json(trip.config.ride)
