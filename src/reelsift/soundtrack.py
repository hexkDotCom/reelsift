"""The music under each edit, and how it is mixed with the clips' own sound.

Every edit revision can have a soundtrack: no music, the trip's music files, or a generated
track (an immutable asset in the trip's music/ folder). A new revision inherits its
predecessor's soundtrack; restoring a revision brings back its soundtrack too. Revisions without
a soundtrack row (older trips) use the trip's music files, as before.

The mix: the clips' sound is at full while people speak and at `background` dB otherwise while
music plays; the music is at `music` dB and drops by `duck` dB while people speak. Generated
music is first brought to a common loudness, so candidates sit at the same level. With no music,
the clips' sound is left alone.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from .config import MusicSettings
from .project import Trip, now

DEFAULTS = MusicSettings(
    genre="melodic techno",
    mood="",
    bpm=None,
    candidates=2,
    # Exactly the gains Reelsift has always used: music 0.5, clips 0.35 outside speech, music
    # 0.12 under speech.
    music=20 * math.log10(0.5),
    background=20 * math.log10(0.35),
    duck=20 * math.log10(0.12 / 0.5),
    sound=0.0,
)
OFF_DB = -60.0
"""A level at or below this is silence."""
REFERENCE_LUFS = -20.0
"""Generated music is brought to this loudness before its mix level applies."""
SOURCES = ("none", "files", "generated")


def _gain(db: float) -> float:
    return 0.0 if db <= OFF_DB else 10 ** (db / 20)


@dataclass(frozen=True)
class Mix:
    music: float = DEFAULTS.music
    background: float = DEFAULTS.background
    duck: float = DEFAULTS.duck
    sound: float = 0.0
    """The clips' own sound as a whole, applied on top of the rest (OFF_DB: silent)."""

    @classmethod
    def from_settings(cls, s: MusicSettings) -> "Mix":
        return cls(s.music, s.background, s.duck, s.sound if s.sound is not None else 0.0)

    def gains(self) -> tuple[float, float, float]:
        """Linear (music, clips' sound outside speech, music during speech)."""
        music = _gain(self.music)
        return music, _gain(self.background), music * 10 ** (self.duck / 20)

    @property
    def sound_gain(self) -> float:
        return _gain(self.sound)


@dataclass
class Soundtrack:
    edit_id: int
    source: str = "files"
    asset_id: int | None = None
    files: list[str] = field(default_factory=list)
    mix: Mix = field(default_factory=Mix)
    fit: dict = field(default_factory=dict)
    """end (s, where the music stops), fade (s), timing (fingerprint of the picture it was fitted
    to), hint, fitted_cuts, unfitted (items before cuts were fitted)."""
    stored: bool = False
    """False for a soundtrack implied by the trip's music files (no row)."""


def timing(items) -> str:
    """Fingerprint of an edit's picture timing: which footage, in which order. Crop changes
    don't change it."""
    key = [(i.recording_id, round(i.start, 3), round(i.end, 3)) for i in items]
    if any(i.speed != 1 for i in items):  # older fingerprints stay as they were
        key = [(*k, round(i.speed, 6)) for k, i in zip(key, items, strict=True)]
    return hashlib.sha1(json.dumps(key).encode()).hexdigest()[:16]


def _row_to(row) -> Soundtrack:
    mix = json.loads(row["mix"])
    return Soundtrack(
        row["edit_id"], row["source"], row["asset_id"], json.loads(row["files"]),
        Mix(**{k: float(v) for k, v in mix.items()
               if k in ("music", "background", "duck", "sound")}),
        json.loads(row["fit"]), True,
    )  # fmt: skip


def get(trip: Trip, edit_id: int) -> Soundtrack:
    row = trip.db.execute("SELECT * FROM soundtracks WHERE edit_id = ?", (edit_id,)).fetchone()
    if row is not None:
        return _row_to(row)
    length = trip.db.execute("SELECT length FROM edits WHERE id = ?", (edit_id,)).fetchone()
    settings = trip.config.music_for(length["length"]) if length else DEFAULTS
    mix = Mix.from_settings(settings)
    source = settings.source or ("files" if trip.config.music else "none")
    if source == "files":
        return Soundtrack(edit_id, "files", None, list(trip.config.music), mix)
    return Soundtrack(edit_id, "none", None, [], mix)


def put(trip: Trip, st: Soundtrack) -> Soundtrack:
    if st.source not in SOURCES:
        raise ValueError(f"unknown music source {st.source!r}")
    if st.source == "generated" and st.asset_id is None:
        raise ValueError("generated music needs a track")
    trip.db.execute(
        "INSERT OR REPLACE INTO soundtracks (edit_id, source, asset_id, files, mix, fit,"
        " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (st.edit_id, st.source, st.asset_id, json.dumps(st.files), json.dumps(asdict(st.mix)),
         json.dumps(st.fit), now()),
    )  # fmt: skip
    return replace(st, stored=True)


def inherit(trip: Trip, edit_id: int, from_edit: int | None, items) -> None:
    """Give a new revision the soundtrack of the one it came from. If its timing changed, the
    music's end is fitted again (cheap) and fitted cuts no longer count as fitted."""
    if from_edit is None:
        return
    row = trip.db.execute("SELECT * FROM soundtracks WHERE edit_id = ?", (from_edit,)).fetchone()
    if row is None:
        return
    st = replace(_row_to(row), edit_id=edit_id)
    if st.fit.get("timing") not in (None, timing(items)):
        fit = {k: v for k, v in st.fit.items() if k in ("hint", "components")}
        a = asset(trip, st.asset_id) if st.source == "generated" else None
        if a is not None:
            from .musicgen import edit_duration, fit_end

            fit.update(fit_end(a.analysis, edit_duration(trip, items)))
        st.fit = {**fit, "timing": timing(items)}
    put(trip, st)


def tag(st: Soundtrack) -> str:
    """A short name for a soundtrack, for output file names: renders of one revision with
    different music or mix don't overwrite each other."""
    if not st.stored:
        return ""
    key = json.dumps([st.source, st.asset_id, st.files, asdict(st.mix)], sort_keys=True)
    return "-" + hashlib.sha1(key.encode()).hexdigest()[:6]


@dataclass
class Asset:
    id: int
    path: Path
    duration: float
    analysis: dict
    request: dict
    length: int
    created_at: str


def asset(trip: Trip, asset_id: int) -> Asset | None:
    row = trip.db.execute("SELECT * FROM music_assets WHERE id = ?", (asset_id,)).fetchone()
    if row is None:
        return None
    return Asset(row["id"], trip.root / row["path"], row["duration"], json.loads(row["analysis"]),
                 json.loads(row["request"]), row["length"], row["created_at"])  # fmt: skip


def assets(trip: Trip, length: int | None = None) -> list[Asset]:
    sql, args = "SELECT id FROM music_assets", ()
    if length is not None:
        sql, args = sql + " WHERE length = ?", (length,)
    return [asset(trip, r["id"]) for r in trip.db.execute(sql + " ORDER BY id", args)]


@dataclass
class Resolved:
    """What render plays under an edit."""

    tracks: list
    """music.Track objects, played in order; for generated music, one track played once."""
    mix: Mix | None
    generated: bool = False
    end: float | None = None
    """Generated music: where it stops (seconds into the track)."""
    fade: float = 2.0
    gain_db: float = 0.0
    """Level matching applied before the mix level."""

    @property
    def silent(self) -> bool:
        return not self.tracks


def resolve(trip: Trip, edit_id: int) -> Resolved:
    """The music to play under an edit revision. Raises if a generated track is missing: it is
    never silently replaced."""
    from . import music

    st = get(trip, edit_id)
    if st.source == "none":
        return Resolved([], st.mix)
    if st.source == "files":
        return Resolved(music.load(trip.root, st.files) if st.files else [], st.mix)
    a = asset(trip, st.asset_id)
    if a is None or not a.path.is_file():
        raise RuntimeError(
            "The generated music for this edit is missing"
            + (f" ({a.path})" if a else "")
            + "; choose or generate another track."
        )
    from .musicgen import ANALYSIS_VERSION, analyse

    if a.analysis.get("version") != ANALYSIS_VERSION:  # measured by an older Reelsift
        a.analysis = analyse(a.path, a.request.get("bpm"))
        trip.db.execute("UPDATE music_assets SET analysis = ? WHERE id = ?",
                        (json.dumps(a.analysis), a.id))  # fmt: skip
    track = music.Track(a.path, a.duration, a.analysis.get("bpm", 0.0),
                        np.array(a.analysis.get("beats", []), float),
                        a.analysis.get("confidence", 1.0))  # fmt: skip
    lufs = a.analysis.get("lufs")
    gain = REFERENCE_LUFS - lufs if lufs is not None and math.isfinite(lufs) else 0.0
    return Resolved([track], st.mix, True, st.fit.get("end"), st.fit.get("fade", 2.0),
                    max(-20.0, min(20.0, gain)))  # fmt: skip
