"""Generated music: installing ACE-Step 1.5, asking it for music that suits an edit, and keeping
the results.

ACE-Step runs in its own environment (its PyTorch and transformers versions differ from
Reelsift's), installed by `reelsift music install` from a pinned source archive and pinned
checkpoint revisions, and is driven through music_worker.py. Each generated track is an
immutable file in the trip's music/ folder with what was asked for and what was measured.

Measured on an RTX 5090 (see docs/design-notes.md): tempo as requested; no vocals with
"[Instrumental]"; tracks of a minute or more end on their own 1-4 s before the requested
duration; 30 s tracks end early unless asked for a final hit; 3 min take about 30 s to make.
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import platformdirs

from .project import Trip, now, transaction

ACESTEP = {
    "version": "v0.1.8",
    "commit": "dce621408bee8c31b4fcf4811682eb9359e1bc94",
    "archive": "https://github.com/ace-step/ACE-Step-1.5/archive/"
    "dce621408bee8c31b4fcf4811682eb9359e1bc94.zip",
    "sha256": "f968799d3b74032fa24b1e86cb4adfda45071daf4745d2ecd6ba24d8f0f23bc0",
    "licence": "MIT",
}
CHECKPOINTS = {
    # name: (Hugging Face repository, revision, folder under checkpoints/, licence)
    "main": ("ACE-Step/Ace-Step1.5", "19671f406d603126926c1b7e2adc169acbcade22", "", "MIT"),
    "xl-turbo": ("ACE-Step/acestep-v15-xl-turbo", "d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee",
                 "acestep-v15-xl-turbo", "MIT"),
}  # fmt: skip
WORKER_VERSION = 1


@dataclass(frozen=True)
class Profile:
    name: str
    dit: str
    lm: str | None
    checkpoints: tuple[str, ...]
    min_free_gb: float
    """GPU memory it needs free."""
    max_duration: float
    """Longest track it was tested with."""
    disk_gb: float
    measured: bool
    note: str = ""


PROFILES = {
    "xl": Profile("xl", "acestep-v15-xl-turbo", "acestep-5Hz-lm-1.7B", ("main", "xl-turbo"),
                  20, 600, 40, True, "RTX 5090: 26 s to load, 16 GB peak, 3 min of music in 28 s"),
    "standard": Profile("standard", "acestep-v15-turbo", "acestep-5Hz-lm-1.7B", ("main",),
                        12, 480, 20, False, "Not measured yet."),
}  # fmt: skip
SAMPLE_RATE = 48000


class MusicUnavailable(RuntimeError):
    pass


# --- installation --------------------------------------------------------------------------------


def home() -> Path:
    env = os.environ.get("REELSIFT_MUSIC_DIR")
    return Path(env) if env else platformdirs.user_data_path("reelsift", appauthor=False) / "music"


def _active_slot() -> Path | None:
    """The installation in use: home()/active.json names its folder. (Installations made before
    there were slots have their manifest in home()/<version>.)"""
    try:
        name = json.loads((home() / "active.json").read_text(encoding="utf-8"))["slot"]
        return home() / name
    except (OSError, ValueError, KeyError):
        legacy = home() / ACESTEP["version"]
        return legacy if (legacy / "manifest.json").is_file() else None


def install_dir(slot: Path | None = None) -> Path:
    """An installation's folder (src/, venv/, manifest.json): the given one or the one in use."""
    return slot or _active_slot() or home() / ACESTEP["version"]


def checkpoints_dir() -> Path:
    return home() / "checkpoints"


def python(slot: Path | None = None) -> Path:
    venv = install_dir(slot) / "venv"
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def manifest() -> dict | None:
    slot = _active_slot()
    if slot is None:
        return None
    try:
        return json.loads((slot / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def installed() -> Profile | None:
    m = manifest()
    if not m or m.get("commit") != ACESTEP["commit"] or not python().is_file():
        return None
    return PROFILES.get(m.get("profile"))


def recommend(vram_gb: float | None) -> Profile | None:
    for p in sorted(PROFILES.values(), key=lambda p: -p.min_free_gb):
        if vram_gb is not None and vram_gb >= p.min_free_gb + 2:
            return p
    return None


def uv() -> str:
    found = os.environ.get("REELSIFT_UV") or shutil.which("uv")
    if not found:
        raise MusicUnavailable(
            "Installing the music model needs uv (https://docs.astral.sh/uv/). Install it, or set "
            "REELSIFT_UV to its path."
        )
    return found


def clean_env(slot: Path | None = None) -> dict:
    """The environment for the music tools: without this Python's settings and Reelsift's CUDA
    library folders."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX")}  # fmt: skip
    path = env.get("PATH", "").split(os.pathsep)
    env["PATH"] = os.pathsep.join(p for p in path if f"{os.sep}nvidia{os.sep}" not in p)
    env["UV_PYTHON_INSTALL_DIR"] = str(home() / "python")
    env["UV_PROJECT_ENVIRONMENT"] = str(install_dir(slot) / "venv")
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _checkpoints_present(profile: Profile) -> int:
    total = 0
    for name in profile.checkpoints:
        folder = checkpoints_dir() / CHECKPOINTS[name][2]
        if folder.is_dir():
            total += sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    return total


def install(profile: Profile, say: Callable[[str], None] = print) -> None:
    """Install ACE-Step and the profile's checkpoints into a new folder, test it, and only then
    switch to it: a failed or interrupted installation leaves the one in use working. Finished
    model downloads are kept and reused."""
    import time

    from .gpu import _try_lock, _unlock

    home().mkdir(parents=True, exist_ok=True)
    need = profile.disk_gb * 2**30 - _checkpoints_present(profile)
    free = shutil.disk_usage(home()).free
    if need > free:
        raise MusicUnavailable(f"Not enough disk space in {home()}: about {need / 2**30:.0f} GB "
                               f"more is needed, {free / 2**30:.0f} GB is free.")  # fmt: skip
    with open(home() / "install.lock", "a+b") as lock:
        if not _try_lock(lock):
            raise MusicUnavailable("The music model is already being installed.")
        try:
            target = home() / f"{ACESTEP['version']}-{time.strftime('%Y%m%d-%H%M%S')}"
            try:
                _install_into(target, profile, say)
            except BaseException:
                shutil.rmtree(target, ignore_errors=True)
                raise
            previous = _active_slot()
            partial = home() / "active.json.partial"
            partial.write_text(json.dumps({"slot": target.name}), encoding="utf-8")
            os.replace(partial, home() / "active.json")
            if previous is not None and previous != target:
                shutil.rmtree(previous, ignore_errors=True)
        finally:
            _unlock(lock)
    say(f"Music generation is ready ({profile.name} profile)")


def _install_into(target: Path, profile: Profile, say: Callable[[str], None]) -> None:
    say(f"ACE-Step 1.5 {ACESTEP['version']} (code {ACESTEP['licence']}); models: "
        + ", ".join(f"{CHECKPOINTS[n][0]} ({CHECKPOINTS[n][3]})" for n in profile.checkpoints)
        + f". About {profile.disk_gb:.0f} GB in {home()}")  # fmt: skip
    source = target / "src"
    archive = home() / f"acestep-{ACESTEP['commit'][:12]}.zip"
    if not archive.is_file():
        say(f"Downloading ACE-Step {ACESTEP['version']}")
        _download(ACESTEP["archive"], archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ACESTEP["sha256"]:
        archive.unlink()
        raise MusicUnavailable(f"The ACE-Step download doesn't match (sha256 {digest}); "
                               "try again, or update Reelsift.")  # fmt: skip
    staging = target / "src.partial"
    with zipfile.ZipFile(archive) as z:
        z.extractall(staging)
    (inner,) = list(staging.iterdir())
    inner.rename(source)
    staging.rmdir()

    say("Installing ACE-Step's Python environment (PyTorch and its libraries)")
    env = clean_env(target)
    env.pop("HF_HUB_OFFLINE")
    result = subprocess.run([uv(), "sync", "--frozen", "--python", "3.12", "--no-dev"],
                            cwd=source, env=env, capture_output=True, text=True)  # fmt: skip
    (target / "install.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        log = home() / "install-failed.log"
        shutil.copyfile(target / "install.log", log)
        raise MusicUnavailable(f"Installing ACE-Step's environment failed; see {log}")

    from huggingface_hub import snapshot_download

    for name in profile.checkpoints:
        repo, revision, sub, _ = CHECKPOINTS[name]
        say(f"Downloading {repo} (resumes if interrupted)")
        snapshot_download(repo, revision=revision, local_dir=checkpoints_dir() / sub)

    say("Testing: generating 10 seconds of music")
    probe = target / "probe"
    events = run_worker(profile, [{"id": "probe", "caption": "house, instrumental, no vocals",
                                   "lyrics": "[Instrumental]", "bpm": 124, "duration": 10,
                                   "seed": 1}], probe, target / "probe.log", slot=target)  # fmt: skip
    made = [e for e in events if e["event"] == "candidate"]
    if not made or not Path(made[0]["path"]).is_file():
        errors = "; ".join(e["message"] for e in events if e["event"] == "error")
        log = home() / "install-failed.log"
        if (target / "probe.log").is_file():
            shutil.copyfile(target / "probe.log", log)
        raise MusicUnavailable(f"The test generation failed: {errors or 'no audio'}; see {log}")
    shutil.rmtree(probe, ignore_errors=True)
    ready = next((e for e in events if e["event"] == "ready"), {})
    record = {
        **{k: ACESTEP[k] for k in ("version", "commit", "sha256", "licence")},
        "profile": profile.name,
        "checkpoints": {
            n: dict(zip(("repo", "revision", "folder", "licence"), CHECKPOINTS[n], strict=True))
            for n in profile.checkpoints
        },
        "worker": WORKER_VERSION,
        "installed_at": now(),
        "load_seconds": ready.get("seconds"),
    }
    (target / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def _download(url: str, path: Path) -> None:
    import urllib.request

    partial = path.with_name(path.name + ".partial")
    with urllib.request.urlopen(url, timeout=60) as r, open(partial, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(partial, path)


# --- running the worker --------------------------------------------------------------------------


def run_worker(profile: Profile, candidates: list[dict], out: Path, log: Path,
               on_event: Callable[[dict], None] | None = None,
               slot: Path | None = None) -> list[dict]:  # fmt: skip
    """Generate candidates with the music environment (the one in use, or slot); returns the
    worker's events in order."""
    from .tools import stream

    request = {
        "protocol": 1,
        "job": uuid.uuid4().hex,
        "source": str(install_dir(slot) / "src"),
        "checkpoints": str(checkpoints_dir()),
        "dit": profile.dit,
        "lm": profile.lm,
        "out": str(out),
        "candidates": candidates,
    }
    events: list[dict] = []

    def line(text: str) -> None:
        try:
            event = json.loads(text)
        except ValueError:
            return  # not protocol output
        if not isinstance(event, dict) or "event" not in event:
            return
        events.append(event)
        if on_event:
            on_event(event)

    worker = Path(__file__).with_name("music_worker.py")
    code = stream([str(python(slot)), "-I", str(worker)], json.dumps(request), line, log,
                  env=clean_env(slot), cwd=install_dir(slot))  # fmt: skip
    if code != 0 and not any(e["event"] == "error" for e in events):
        events.append(
            {
                "event": "error",
                "stage": "worker",
                "id": None,
                "message": f"the music worker exited with code {code}; see {log}",
            }
        )
    return events


# --- what to ask for -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Genre:
    words: str
    bpm: tuple[int, int]
    beat: bool = True


GENRES = {
    "house": Genre("house, four-on-the-floor kick, warm chords", (120, 126)),
    "deep house": Genre("deep house, warm pads, punchy kick, smooth bassline", (118, 124)),
    "progressive house": Genre("progressive house, uplifting, evolving", (122, 128)),
    "techno": Genre("driving techno, rolling bassline, hypnotic", (125, 135)),
    "melodic techno": Genre("melodic techno, emotional synth leads, deep bass", (120, 126)),
    "trance": Genre("uplifting trance, soaring synths, arpeggios", (136, 140)),
    "drum and bass": Genre("liquid drum and bass, fast breakbeats, deep sub bass", (170, 176)),
    "dubstep": Genre("dubstep, heavy bass, half-time drums", (140, 140)),
    "electro": Genre("electro house, punchy, gritty synths", (125, 130)),
    "synthwave": Genre("synthwave, retro analog synths, gated drums", (90, 110)),
    "lo-fi": Genre("lo-fi hip hop, dusty drums, mellow keys", (70, 90)),
    "chillout": Genre("chillout, downtempo, soft percussion, airy pads", (90, 110)),
    "ambient": Genre("ambient electronic, soft evolving pads, no drums", (70, 90), beat=False),
}
STYLE_WORDS = {"preview": "energetic, punchy", "highlights": "driving, uplifting",
               "story": "evolving, atmospheric"}  # fmt: skip
STYLE_TEMPO = {"preview": 0.8, "highlights": 0.5, "story": 0.2, "ride": 0.6}
STYLE_WORDS["ride"] = "steady propulsive groove, continuous, road trip"
"""Where in the genre's tempo range each edit style sits."""


def genre(name: str) -> Genre:
    """A preset, or free text with a middle-of-the-road tempo range."""
    return GENRES.get(name.strip().lower(), Genre(name.strip(), (110, 128)))


def tempo(g: Genre, style: str, bpm: int | None) -> int:
    if bpm:
        return int(bpm)
    lo, hi = g.bpm
    return round(lo + (hi - lo) * STYLE_TEMPO.get(style, 0.5))


def arc(energy: np.ndarray, speech: float) -> str:
    """A few words about the shape of the edit, from its energy per second: a hint the model
    follows loosely."""
    if speech > 0.5:
        return "sparse and calm, leaves room for voices"
    e = energy[np.isfinite(energy)]
    if len(e) < 9 or float(np.std(e)) < 1e-6:
        return ""
    thirds = [float(np.median(p)) for p in np.array_split(e, 3)]
    peak = int(np.argmax(thirds))
    return ["starts strong, calmer later", "builds to a peak in the middle",
            "builds up gradually to a big finish"][peak]  # fmt: skip


@dataclass
class Ask:
    caption: str
    lyrics: str
    bpm: int
    duration: float
    """Requested from the model: a little more than the edit, so its ending can be placed."""
    edit_duration: float


def ask(settings, style: str, edit_duration: float, energy: np.ndarray, speech: float,
        max_duration: float) -> Ask:  # fmt: skip
    g = genre(settings.genre or "melodic techno")
    parts = [g.words, STYLE_WORDS.get(style, ""), settings.mood or "", arc(energy, speech)]
    short = edit_duration < 60
    parts.append("ends with a clear final hit" if short else "ends with an outro")
    parts.append("instrumental, no vocals")
    caption = ", ".join(p for p in parts if p)[:500]
    requested = min(max_duration, max(10.0, edit_duration + (6.0 if short else 4.0)))
    return Ask(caption, "[Instrumental]", tempo(g, style, settings.bpm), round(requested, 2),
               edit_duration)  # fmt: skip


# --- measuring a track ---------------------------------------------------------------------------

ANALYSIS_VERSION = 2
WINDOW = 2.0


def analyse(path: Path, bpm: float | None = None) -> dict:
    """Loudness, where its sound ends, its beats (near the tempo it was made at, if that is the
    clearer beat), and its loudness per 2 s window."""
    from . import music
    from .tools import run, tool

    y = music.load_audio(path)
    rate = music.RATE
    if not len(y) or not np.all(np.isfinite(y)):
        raise ValueError("the track is empty or damaged")
    half = rate // 2
    frames = y[: len(y) // half * half].reshape(-1, half)
    db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-9)
    if not len(db) or db.max() < -60:
        raise ValueError("the track is silent")
    loud = np.flatnonzero(db > db.max() - 30)
    sound_end = float((loud[-1] + 1) * 0.5)
    w = int(rate * WINDOW)
    count = max(1, len(y) // w)
    windows = 20 * np.log10(np.sqrt(np.mean(y[: count * w].reshape(count, w) ** 2, axis=1)) + 1e-9)
    track = music.analyze_file(path, bpm)
    err = run([tool("ffmpeg"), "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128",
               "-f", "null", "-"], text=True).stderr  # fmt: skip
    import re

    found = re.findall(r"I:\s+(-?[\d.]+) LUFS", err)
    return {
        "version": ANALYSIS_VERSION,
        "duration": round(len(y) / rate, 3),
        "sound_end": sound_end,
        "lufs": float(found[-1]) if found else None,
        "bpm": round(track.bpm, 2),
        "beats": [round(float(b), 3) for b in track.beats],
        "confidence": round(float(getattr(track, "confidence", 1.0)), 3),
        "windows_db": [round(float(v), 1) for v in windows],
    }


def hint(analysis: dict, energy: np.ndarray, edit_duration: float) -> tuple[float, dict]:
    """How well a track suits an edit, for ordering candidates: its loudness curve against the
    edit's energy (correlation over 2 s windows), less a penalty if it ends before the edit."""
    windows = np.array(analysis["windows_db"], float)
    n = int(edit_duration // WINDOW)
    e = energy[: n * int(WINDOW)] if len(energy) >= n * int(WINDOW) else energy
    edit_windows = np.array([np.nanmean(p) if np.isfinite(p).any() else np.nan
                             for p in np.array_split(e, max(1, n))]) if len(e) else np.array([])  # fmt: skip
    m = min(len(windows), len(edit_windows))
    a, b = windows[:m], edit_windows[:m]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() >= 4 and np.std(a[ok]) > 1e-6 and np.std(b[ok]) > 1e-6:
        follows = float(np.corrcoef(a[ok], b[ok])[0, 1])
    else:
        follows = 0.0
    early = analysis["sound_end"] < edit_duration - 0.5
    components = {"follows": round(follows, 3), "ends_early": early}
    return round(follows - (0.5 if early else 0.0), 3), components


def fit_end(analysis: dict, edit_duration: float) -> dict:
    """Where the music stops under an edit, and how long its fade is. A track that ends on its
    own at the edit's end keeps its ending; a longer one fades out over the last beats."""
    sound_end = analysis["sound_end"]
    if sound_end < edit_duration - 0.25:
        return {"end": round(sound_end, 3), "fade": 0.5, "ends_early": True}
    if sound_end <= edit_duration + 0.5:
        return {"end": round(edit_duration, 3), "fade": 0.5}
    beat = 60 / analysis["bpm"] if analysis.get("bpm") else 0.5
    return {"end": round(edit_duration, 3), "fade": round(min(3.0, max(1.0, 4 * beat)), 2)}


# --- generating for an edit ----------------------------------------------------------------------


@dataclass
class Report:
    made: list[int] = field(default_factory=list)
    """Asset ids, in the order made."""
    errors: list[str] = field(default_factory=list)
    assigned: int | None = None
    """The asset now playing under the edit, if this generation was still current."""
    note: str = ""


def _name(length: int) -> str:
    """How an edit is called in messages."""
    from .ride import RIDE

    return "ride" if length == RIDE else f"{length} s edit"


def edit_energy(data, items) -> tuple[np.ndarray, float]:
    """The edit's energy per second of edit time (motion and activity, median-smoothed over 5 s;
    NaN where unknown) and the share of its time with speech. A second of edit time covers the
    footage played in it: for a sped-up clip, several seconds of footage, never beyond the clip."""
    total = sum(i.duration for i in items)
    n = max(1, math.ceil(total - 1e-9))
    sums, counts = np.zeros(n), np.zeros(n)
    speaking = 0.0
    t0 = 0.0
    for item in items:
        obs = data.observations.get(item.recording_id, {})
        t1 = t0 + item.duration
        for k in range(int(t0), min(n, math.ceil(t1 - 1e-9))):
            lo = max(float(k), t0)
            hi = min(k + 1.0, t1)
            a = item.start + (lo - t0) * item.speed  # the footage played in edit second k
            b = min(item.end, item.start + (hi - t0) * item.speed)
            for t in range(int(a), max(int(a) + 1, math.ceil(b - 1e-9))):
                parts = [obs[c][t] for c in ("motion", "activity")
                         if c in obs and t < len(obs[c]) and np.isfinite(obs[c][t])]  # fmt: skip
                if parts:
                    # what this second of footage lasts in edit second k
                    w = (min(b, t + 1.0) - max(a, float(t))) / item.speed if b > a else hi - lo
                    sums[k] += float(np.mean(parts)) * w
                    counts[k] += w
        words = data.words.get(item.recording_id)
        if item.speed == 1 and words is not None and len(words):  # sped-up speech doesn't count
            wa = np.clip(words[:, 0], item.start, item.end)
            wb = np.clip(words[:, 1], item.start, item.end)
            speaking += float(np.sum(np.maximum(0.0, wb - wa)))
        t0 = t1
    raw = np.full(n, math.nan)
    np.divide(sums, counts, out=raw, where=counts > 0)
    smooth = raw.copy()
    for k in range(len(raw)):
        part = raw[max(0, k - 2) : k + 3]
        part = part[np.isfinite(part)]
        smooth[k] = float(np.median(part)) if len(part) else math.nan
    return smooth, (speaking / total if total else 0.0)


def edit_duration(trip: Trip, items) -> float:
    """How long the rendered edit is, from its frame counts."""
    from fractions import Fraction

    from .render import edit_plan, frames_of

    plan, fps = edit_plan(trip, items)
    return float(Fraction(sum(frames_of(p, fps) for _, p in plan)) / fps) if plan else 0.0


def style_of(trip: Trip, length: int) -> str:
    from . import prefs as prefs_mod
    from .ride import RIDE

    if length == RIDE:
        return "ride"

    p, _ = prefs_mod.latest(trip.db, length)
    return (p.style if p and p.style else None) or prefs_mod.default_style(
        length, sorted(trip.config.lengths)
    )


def generate(
    trip: Trip,
    length: int,
    *,
    candidates: int | None = None,
    seed: int | None = None,
    say: Callable[[str], None] = print,
    before_gpu: Callable[[], None] | None = None,
    data=None,
    started: tuple[str, int] | None = None,
) -> Report:
    """Generate music for the latest revision of one edit (see generate_many)."""
    return generate_many(trip, [length], candidates=candidates, seed=seed, say=say,
                         before_gpu=before_gpu, data=data,
                         started={length: started} if started else None)[length]  # fmt: skip


@dataclass
class _Job:
    length: int
    edit_id: int
    duration: float
    energy: np.ndarray
    request_id: str
    asked: list[dict]
    report: Report = field(default_factory=Report)
    wanted: dict = field(default_factory=dict)
    """The genre and mood it was asked for, kept with each track it makes."""


def generate_many(
    trip: Trip,
    lengths: list[int],
    *,
    candidates: int | None = None,
    seed: int | None = None,
    say: Callable[[str], None] = print,
    before_gpu: Callable[[], None] | None = None,
    data=None,
    started: dict[int, tuple[str, int]] | None = None,
) -> dict[int, Report]:
    """Generate music for the latest revisions of these edits, loading the model once. Each
    finished track is kept as it comes; for each edit the best-suited one is put under it, if the
    edit hasn't changed meanwhile and no newer generation was started for it."""
    import random

    from . import edits as edits_mod
    from . import gpu, soundtrack, tripdata
    from .tools import check_cancelled, is_cancelled

    profile = installed()
    if profile is None:
        raise MusicUnavailable("Music generation isn't installed yet: run  reelsift music install")
    # Accepted first, so a choice made while the edits are being looked at wins.
    started = dict(started or {})
    for length in lengths:
        if length not in started:
            started[length] = begin(trip, length)
    data = data or tripdata.load(trip)
    rng = random.Random()
    jobs: list[_Job] = []
    for length in lengths:
        request_id, edit_id = started[length]
        row = trip.db.execute("SELECT items FROM edits WHERE id = ?", (edit_id,)).fetchone()
        items = edits_mod.items_from_json(row["items"])
        edit = edits_mod.StoredEdit(edit_id, length, 0, items, {}, False)
        duration = edit_duration(trip, edit.items)
        if duration > profile.max_duration:
            raise MusicUnavailable(
                f"The {_name(length)} runs {duration:.0f} s; the music model makes at most "
                f"{profile.max_duration:.0f} s on this computer."
            )
        settings = trip.config.music_for(length)
        energy, speech = edit_energy(data, edit.items)
        a = ask(settings, style_of(trip, length), duration, energy, speech, profile.max_duration)
        count = candidates or settings.candidates or 2
        seeds = [seed + k if seed is not None else rng.randrange(2**31) for k in range(count)]
        asked = [{"id": f"{request_id[:10]}-{k}", "caption": a.caption, "lyrics": a.lyrics,
                  "bpm": a.bpm, "duration": a.duration, "seed": s, "keyscale": "",
                  "timesignature": "4"} for k, s in enumerate(seeds)]  # fmt: skip
        jobs.append(_Job(length, edit.id, duration, energy, request_id, asked,
                         wanted={"genre": settings.genre, "mood": settings.mood}))  # fmt: skip
        say(f"{_name(length)}: asking for {count} track(s): {a.caption} ({a.bpm} BPM, "
            f"{a.duration:.0f} s)")  # fmt: skip
    by_id = {c["id"]: (job, c) for job in jobs for c in job.asked}
    everything = [c for job in jobs for c in job.asked]
    folder = trip.root / "music"
    work = folder / ".partial" / jobs[0].request_id

    def on_event(e: dict) -> None:
        if e["event"] == "stage" and e["stage"] == "loading":
            say("Loading the music model")
        elif e["event"] == "stage" and e["stage"] == "generating":
            job, _ = by_id[e["id"]]
            say(f"Making track {e['index'] + 1} of {e['total']} ({_name(job.length)})")
        elif e["event"] == "error":
            if e.get("id") in by_id:
                by_id[e["id"]][0].report.errors.append(e["message"])
            else:
                for job in jobs:
                    job.report.errors.append(e["message"])
            say(f"  failed: {e['message']}")
        elif e["event"] == "candidate":
            job, spec = by_id[e["id"]]
            try:
                asset_id = _keep(trip, Path(e["path"]), {**spec, **job.wanted}, profile,
                                 job.request_id, job.length, job.edit_id, job.energy,
                                 job.duration)  # fmt: skip
            except Exception as err:  # a damaged or silent track
                job.report.errors.append(str(err))
                say(f"  unusable: {err}")
                return
            job.report.made.append(asset_id)
            say(f"  ready ({e['seconds']:.0f} s)")

    try:
        with gpu.hold(f"music generation ({trip.root.name})", cancelled=is_cancelled,
                      waiting=lambda h: say(f"Waiting for the GPU ({h or 'busy'})")):  # fmt: skip
            if before_gpu:
                before_gpu()
            run_worker(profile, everything, work,
                       folder / "logs" / f"{jobs[0].request_id}.log", on_event)  # fmt: skip
    finally:
        shutil.rmtree(work, ignore_errors=True)
    check_cancelled()
    if not any(job.report.made for job in jobs):
        errors = sorted({e for job in jobs for e in job.report.errors})
        raise RuntimeError("No usable music was made: " + ("; ".join(errors) or "unknown"))
    from .project import immediate

    for job in jobs:
        report = job.report
        if not report.made:
            report.note = "No usable track was made for this edit."
            continue
        with immediate(trip.db):  # the checks and the assignment, with no change in between
            current = edits_mod.latest(trip, job.length)
            if (current is None or current.id != job.edit_id
                    or _now_request(trip, job.length) != job.request_id):  # fmt: skip
                report.note = ("The edit or its music changed meanwhile; the new tracks are kept "
                               "but not used.")  # fmt: skip
                continue
            best = max(report.made, key=lambda i: soundtrack.asset(trip, i).request["hint"])
            use(trip, job.edit_id, best, job.duration)
            report.assigned = best
    return {job.length: job.report for job in jobs}


def _keep(trip: Trip, wav: Path, spec: dict, profile: Profile, request_id: str, length: int,
          edit_id: int, energy: np.ndarray, duration: float) -> int:  # fmt: skip
    """Store a finished track as an asset: FLAC in the trip's music/ folder."""
    from .tools import run, tool

    analysis = analyse(wav, spec.get("bpm"))
    score, components = hint(analysis, energy, duration)
    name = f"{spec['id']}.flac"
    target = trip.root / "music" / name
    partial = target.with_name(name + ".partial")
    run([tool("ffmpeg"), "-v", "error", "-y", "-i", str(wav), "-c:a", "flac", "-f", "flac",
         str(partial)])  # fmt: skip
    os.replace(partial, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    m = manifest() or {}
    request = {**spec, "request": request_id, "edit_id": edit_id, "edit_duration": duration,
               "profile": profile.name, "dit": profile.dit, "lm": profile.lm,
               "acestep": m.get("commit"), "checkpoints": m.get("checkpoints"),
               "worker": WORKER_VERSION, "hint": score, "components": components}  # fmt: skip
    with transaction(trip.db):
        asset_id = trip.db.execute(
            "INSERT INTO music_assets (path, sha256, duration, sample_rate, channels, length,"
            " analysis, request, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"music/{name}", digest, analysis["duration"], SAMPLE_RATE, 2, length,
             json.dumps(analysis), json.dumps(request), now()),
        ).lastrowid  # fmt: skip
    wav.unlink(missing_ok=True)
    return asset_id


def use(trip: Trip, edit_id: int, asset_id: int, duration: float | None = None) -> None:
    """Play a generated track under an edit revision, keeping its mix settings."""
    from . import edits as edits_mod
    from . import soundtrack

    a = soundtrack.asset(trip, asset_id)
    if a is None:
        raise KeyError(asset_id)
    row = trip.db.execute("SELECT items FROM edits WHERE id = ?", (edit_id,)).fetchone()
    items = edits_mod.items_from_json(row["items"])
    if duration is None:
        duration = edit_duration(trip, items)
    current = soundtrack.get(trip, edit_id)
    # Another track on the same picture: the edit as it was before its cuts were fitted stays
    # the starting point for fitting again.
    kept = {k: current.fit[k] for k in ("unfitted", "fitted_cuts") if k in current.fit}
    fit = {**fit_end(a.analysis, duration), "timing": soundtrack.timing(items),
           "hint": a.request.get("hint"), "components": a.request.get("components"), **kept}  # fmt: skip
    soundtrack.put(trip, soundtrack.Soundtrack(edit_id, "generated", asset_id, [], current.mix,
                                               fit))  # fmt: skip


def prune(trip: Trip) -> tuple[int, int]:
    """Delete generated tracks that no edit revision uses: (how many, bytes freed)."""
    unused = trip.db.execute(
        "SELECT id, path FROM music_assets WHERE id NOT IN"
        " (SELECT asset_id FROM soundtracks WHERE asset_id IS NOT NULL)"
    ).fetchall()
    freed = 0
    for r in unused:
        path = trip.root / r["path"]
        freed += path.stat().st_size if path.is_file() else 0
        trip.db.execute("DELETE FROM music_assets WHERE id = ?", (r["id"],))
        path.unlink(missing_ok=True)
    return len(unused), freed


def begin(trip: Trip, length: int, expected_edit_id: int | None = None) -> tuple[str, int]:
    """Accept a generation for the current revision of an edit, before any work starts: (its
    request id, the revision). A choice made by hand after this makes it stale."""
    from . import edits as edits_mod
    from .project import immediate

    with immediate(trip.db):
        edit = edits_mod.latest(trip, length)
        if edit is None or not edit.items:
            raise ValueError(f"The {_name(length)} is empty; select it first.")
        if expected_edit_id is not None and edit.id != expected_edit_id:
            raise ValueError("This edit changed in the meantime; it has been reloaded.")
        request_id = uuid.uuid4().hex
        _now_request(trip, length, request_id)
    return request_id, edit.id


def chosen(trip: Trip, length: int) -> None:
    """The music of an edit was chosen by hand: a generation still running for it keeps its
    tracks but doesn't put one under the edit."""
    _now_request(trip, length, f"chosen-{uuid.uuid4().hex}")


def _now_request(trip: Trip, length: int, request: str | None = None) -> str | None:
    key = f"music_request:{length}"
    if request is not None:
        trip.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, request))
        return request
    row = trip.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None
