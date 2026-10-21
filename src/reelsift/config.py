"""User and trip configuration, both stored as TOML.

User config (defaults for every trip) lives in the platform's config folder. Trip config lives in
the trip folder as reelsift.toml and overrides the user config.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import platformdirs
import tomli_w

from .timebase import parse_zone

DEFAULT_LENGTHS = (30, 180, 600)
TRIP_CONFIG = "reelsift.toml"

CreationTime = Literal["utc", "local"]


def user_config_path() -> Path:
    return platformdirs.user_config_path("reelsift", appauthor=False) / "config.toml"


@dataclass
class CameraConfig:
    """Per-camera overrides, keyed by camera label in [cameras."<label>"]."""

    zone: str | None = None
    """Time zone where the camera was used, such as "+02:00". Fills in unknown zones."""
    creation_time: CreationTime | None = None
    """Whether the camera writes creation_time as true UTC or as local time marked UTC."""

    @property
    def zone_offset_min(self) -> int | None:
        return None if self.zone is None else parse_zone(self.zone)


@dataclass
class MusicSettings:
    """What generated music is asked for, and how music is mixed with the clips' sound. Unset
    fields (None) fall back: an edit length's settings, then the trip's, then the user's, then
    the defaults. Mix levels are in dB."""

    source: str | None = None
    """none or files (the [render] music files), for edit revisions whose music wasn't chosen
    in the app or with a command. Generated music is always chosen for a revision."""
    genre: str | None = None
    mood: str | None = None
    bpm: int | None = None
    candidates: int | None = None
    music: float | None = None
    """Music level against the clips' sound."""
    background: float | None = None
    """Clips' sound outside speech while music plays."""
    duck: float | None = None
    """How far music drops while people speak (0 = not at all)."""
    sound: float | None = None
    """The clips' own sound as a whole (-60 or lower: off)."""

    FIELDS = ("source", "genre", "mood", "bpm", "candidates", "music", "background", "duck",
              "sound")  # fmt: skip

    def over(self, base: "MusicSettings") -> "MusicSettings":
        """These settings, with unset fields taken from base."""
        return MusicSettings(**{
            k: getattr(self, k) if getattr(self, k) is not None else getattr(base, k)
            for k in self.FIELDS
        })  # fmt: skip

    def to_toml(self) -> dict:
        return {k: getattr(self, k) for k in self.FIELDS if getattr(self, k) is not None}

    @classmethod
    def from_toml(cls, data: dict) -> "MusicSettings":
        out = cls()
        for k in ("source", "genre", "mood"):
            if k in data:
                setattr(out, k, str(data[k]))
        for k in ("bpm", "candidates"):
            if k in data:
                setattr(out, k, int(data[k]))
        mix = data.get("mix", {})
        for k in ("music", "background", "duck", "sound"):
            if k in mix:
                setattr(out, k, float(mix[k]))
        return out

    def validate(self, where: str) -> None:
        import math

        if self.source not in (None, "none", "files"):
            raise ValueError(f'{where}.source must be "none" or "files" (generated music is '
                             "chosen per edit: reelsift music generate or use)")  # fmt: skip
        if self.bpm is not None and not 40 <= self.bpm <= 220:
            raise ValueError(f"{where}.bpm must be from 40 to 220")
        if self.candidates is not None and not 1 <= self.candidates <= 8:
            raise ValueError(f"{where}.candidates must be from 1 to 8")
        for k, lo, hi in (("music", -60, 12), ("background", -60, 12), ("duck", -60, 0),
                          ("sound", -60, 12)):  # fmt: skip
            v = getattr(self, k)
            if v is not None and not (math.isfinite(v) and lo <= v <= hi):
                raise ValueError(f"{where}.mix.{k} must be from {lo} to {hi} dB")


@dataclass
class Config:
    name: str = ""
    footage: list[str] = field(default_factory=list)
    lengths: list[int] = field(default_factory=lambda: list(DEFAULT_LENGTHS))
    preset: str = "auto"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    music: list[str] = field(default_factory=list)
    """Music files played in order (and repeated) under every edit; empty for no music."""
    music_settings: MusicSettings = field(default_factory=MusicSettings)
    """The trip's [music]: generated music and the mix, for every edit."""
    music_lengths: dict[int, MusicSettings] = field(default_factory=dict)
    """The trip's [music.lengths."30"]: the same, for one edit length."""
    ride: dict = field(default_factory=dict)
    """[ride]: the ride's settings until it has revisions (see ride.RideSettings)."""
    user_music_settings: MusicSettings = field(default_factory=MusicSettings)
    """The same from the user's config: defaults for the trip's unset fields, never written into
    the trip."""
    user_music_lengths: dict[int, MusicSettings] = field(default_factory=dict)

    def music_for(self, length: int) -> MusicSettings:
        from .soundtrack import DEFAULTS

        layers = [self.music_lengths.get(length), self.music_settings,
                  self.user_music_lengths.get(length), self.user_music_settings]  # fmt: skip
        out = MusicSettings()
        for layer in layers:
            if layer is not None:
                out = out.over(layer)
        return out.over(DEFAULTS)

    def camera(self, label: str) -> CameraConfig:
        return self.cameras.get(label, CameraConfig())

    def validate(self) -> None:
        if not self.lengths or any(n <= 0 for n in self.lengths):
            raise ValueError("lengths must be a non-empty list of positive numbers of seconds")
        if len(set(self.lengths)) != len(self.lengths):
            raise ValueError("lengths must not repeat")
        for label, cam in self.cameras.items():
            if cam.creation_time not in (None, "utc", "local"):
                raise ValueError(f'cameras."{label}".creation_time must be "utc" or "local"')
            if cam.zone is not None:
                parse_zone(cam.zone)
        for s in (self.music_settings, self.user_music_settings):
            s.validate("music")
        for lengths in (self.music_lengths, self.user_music_lengths):
            for n, s in lengths.items():
                s.validate(f'music.lengths."{n}"')


def _read(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _merge(base: Config, data: dict, user: bool = False) -> Config:
    trip = data.get("trip", {})
    defaults = data.get("defaults", {})
    for source in (defaults, trip):
        if "lengths" in source:
            base.lengths = [int(n) for n in source["lengths"]]
        if "preset" in source:
            base.preset = str(source["preset"])
    if "name" in trip:
        base.name = str(trip["name"])
    if "footage" in trip:
        base.footage = [str(p) for p in trip["footage"]]
    if "music" in data.get("render", {}):
        base.music = [str(m) for m in data["render"]["music"]]
    for label, cam in data.get("cameras", {}).items():
        base.cameras[label] = CameraConfig(
            zone=cam.get("zone"), creation_time=cam.get("creation_time")
        )
    if "ride" in data and not user:
        base.ride = dict(data["ride"])
    music = data.get("music", {})
    if music:
        settings = MusicSettings.from_toml(music)
        lengths = {
            int(n): MusicSettings.from_toml(own) for n, own in music.get("lengths", {}).items()
        }
        if user:
            base.user_music_settings, base.user_music_lengths = settings, lengths
        else:
            base.music_settings, base.music_lengths = settings, lengths
    return base


def load(trip_dir: Path | None = None, user_path: Path | None = None) -> Config:
    config = _merge(Config(), _read(user_path or user_config_path()), user=True)
    if trip_dir is not None:
        config = _merge(config, _read(trip_dir / TRIP_CONFIG))
    config.validate()
    return config


def _put_music(entry: dict, settings: MusicSettings) -> None:
    values = settings.to_toml()
    mix = {k: values.pop(k) for k in ("music", "background", "duck", "sound") if k in values}
    entry.update(values)
    if mix:
        entry["mix"] = {k: round(v, 2) for k, v in mix.items()}


def write_trip(trip_dir: Path, config: Config) -> None:
    """Save the trip configuration. Settings Reelsift doesn't manage, and keys it doesn't know,
    are kept as they were; the file is replaced atomically."""
    import os

    path = trip_dir / TRIP_CONFIG
    data = _read(path)
    trip = data.setdefault("trip", {})
    trip["name"] = config.name
    trip["footage"] = config.footage
    if "lengths" in trip or list(config.lengths) != list(DEFAULT_LENGTHS):
        trip["lengths"] = list(config.lengths)
    if "preset" in trip or config.preset != "auto":
        trip["preset"] = config.preset
    cameras = data.get("cameras", {})
    for label, c in config.cameras.items():
        entry = cameras.setdefault(label, {})
        for key, value in (("zone", c.zone), ("creation_time", c.creation_time)):
            if value:
                entry[key] = value
            else:
                entry.pop(key, None)
    if cameras:
        data["cameras"] = cameras
    render = data.get("render", {})
    if config.music:
        render["music"] = config.music
    else:
        render.pop("music", None)
    if render:
        data["render"] = render
    else:
        data.pop("render", None)
    music = {k: v for k, v in data.get("music", {}).items()
             if k not in MusicSettings.FIELDS and k not in ("mix", "lengths")}  # fmt: skip
    _put_music(music, config.music_settings)
    lengths = {}
    for n, s in sorted(config.music_lengths.items()):
        entry: dict = {}
        _put_music(entry, s)
        if entry:
            lengths[str(n)] = entry
    if lengths:
        music["lengths"] = lengths
    if music:
        data["music"] = music
    else:
        data.pop("music", None)
    ride = {k: v for k, v in config.ride.items() if v is not None and k != "overrides"}
    if ride:
        data["ride"] = ride
    else:
        data.pop("ride", None)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(tomli_w.dumps(data), encoding="utf-8")
    os.replace(partial, path)
