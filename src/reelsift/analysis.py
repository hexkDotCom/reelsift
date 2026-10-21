"""Analysis: per-second observations, embeddings and speech for every recording.

This is the slow, GPU-heavy half of Reelsift. It reads proxies only and runs once per recording;
results carry a stage version, so changing a model or setting re-runs only that stage. Selection
never calls into this module's models.

Time here is recording time in seconds: a chapter's frames are offset by the chapter's start.
"""

import json
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import proxies as proxies_mod
from .jobs import Job, Progress
from .project import Trip, now, transaction
from .tools import run, tool

FPS = 1
MOTION_FPS = 5
MOTION_LONG_SIDE = 160
VISUAL_VERSION = 1
SPEECH_VERSION = 1


@dataclass
class ChapterMedia:
    start: float
    duration: float
    proxy: Path
    source: Path
    width: int
    height: int
    has_audio: bool


def chapter_media(trip: Trip, recording_id: int) -> list[ChapterMedia]:
    rows = trip.db.execute(
        "SELECT c.start, c.duration, a.path, a.probe_json, p.path AS proxy, p.width, p.height"
        " FROM chapters c JOIN assets a ON a.id = c.asset_id"
        " JOIN proxies p ON p.recording_id = c.recording_id AND p.chapter_idx = c.idx"
        " WHERE c.recording_id = ? ORDER BY c.idx",
        (recording_id,),
    ).fetchall()
    out = []
    for r in rows:
        streams = json.loads(r["probe_json"]).get("streams", [])
        out.append(
            ChapterMedia(
                start=float(_frac(r["start"])),
                duration=float(_frac(r["duration"])),
                proxy=proxies_mod.resolve(trip, r["proxy"]),
                source=Path(r["path"]),
                width=r["width"],
                height=r["height"],
                has_audio=any(s.get("codec_type") == "audio" for s in streams),
            )
        )
    return out


def _frac(text: str):
    from fractions import Fraction

    return Fraction(text)


def stream_frames(
    path: Path,
    fps: float | None,
    width: int,
    height: int,
    gray: bool,
    batch: int = 64,
    stream: str = "0:v:0",
) -> Iterator[np.ndarray]:
    """Decode frames at a fixed rate (every frame if fps is None, as for a photo) and size,
    yielded in batches to bound memory."""
    channels = 1 if gray else 3
    pix = "gray" if gray else "rgb24"
    cmd = [
        tool("ffmpeg"), "-v", "error", "-i", str(path), "-map", stream,
        "-vf", (f"fps={fps}," if fps else "")
        + f"scale={width}:{height}:in_color_matrix=bt709:out_color_matrix=bt709,"
        f"format={pix}",
        "-f", "rawvideo", "-",
    ]  # fmt: skip
    frame_bytes = width * height * channels
    shape = (height, width, channels) if not gray else (height, width)
    for data in stream_chunks(cmd, frame_bytes, batch, path.name):
        n = len(data) // frame_bytes
        yield np.frombuffer(data[: n * frame_bytes], np.uint8).reshape(n, *shape)


STDERR_KEEP = 64 * 1024


def stream_chunks(cmd: list[str], frame_bytes: int, batch: int, name: str) -> Iterator[bytes]:
    """Run a decoder and yield its output in chunks of `batch` frames.

    Its error output is drained by a separate thread, keeping only the last 64 KiB: a decoder
    that reports many errors would otherwise fill that pipe and block forever while this side
    waits for frames. If the consumer stops early, the decoder is stopped too.
    """
    import collections
    import threading

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    tail: collections.deque = collections.deque()
    kept = [0]

    def drain() -> None:
        while True:
            chunk = proc.stderr.read(4096)
            if not chunk:
                return
            tail.append(chunk)
            kept[0] += len(chunk)
            while kept[0] > STDERR_KEEP and len(tail) > 1:
                kept[0] -= len(tail.popleft())

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    finished = False
    try:
        while True:
            data = proc.stdout.read(frame_bytes * batch)
            if not data:
                break
            yield data
        finished = True
    finally:
        if not finished and proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        code = proc.wait()
        reader.join(timeout=5)
        proc.stderr.close()
        if finished and code != 0:
            err = b"".join(tail).decode(errors="replace")
            raise RuntimeError(f"ffmpeg failed decoding {name}: {err.strip()[-500:]}")


def decode_audio(path: Path, rate: int = 16000) -> np.ndarray:
    raw = run(
        [tool("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1",
         "-ar", str(rate), "-f", "f32le", "-"]
    ).stdout  # fmt: skip
    return np.frombuffer(raw, np.float32)


# --- visual -------------------------------------------------------------------------------


def _quality(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Blur (variance of the Laplacian), mean brightness, and fractions of crushed darks and
    clipped highlights, for each RGB frame."""
    blur, bright_mean, dark, bright = [], [], [], []
    for f in frames:
        luma = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        blur.append(cv2.Laplacian(luma, cv2.CV_32F).var())
        bright_mean.append(luma.mean() / 255.0)
        dark.append(float((luma < 16).mean()))
        bright.append(float((luma > 245).mean()))
    return np.array(blur), np.array(bright_mean), np.array(dark), np.array(bright)


def _motion_size(width: int, height: int) -> tuple[int, int]:
    if width >= height:
        w, h = MOTION_LONG_SIDE, round(MOTION_LONG_SIDE * height / width / 2) * 2
    else:
        h, w = MOTION_LONG_SIDE, round(MOTION_LONG_SIDE * width / height / 2) * 2
    return max(w, 2), max(h, 2)


def motion_signals(path: Path, width: int, height: int) -> dict[int, tuple[float, float, float]]:
    """Per second: camera motion, activity in the scene, and shake, from optical flow.

    motion: how fast the whole image moves (median flow), in frame widths per second.
    activity: movement that differs from the camera's (people, cars, wheels).
    shake: how much the camera's movement jitters within the second.
    """
    w, h = _motion_size(width, height)
    per_second: dict[int, list[tuple[float, float, float, float]]] = {}
    prev = None
    index = 0
    for batch in stream_frames(path, MOTION_FPS, w, h, gray=True, batch=256):
        for frame in batch:
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(prev, frame, None, 0.5, 2, 9, 2, 5, 1.1, 0)
                dx, dy = float(np.median(flow[..., 0])), float(np.median(flow[..., 1]))
                residual = np.hypot(flow[..., 0] - dx, flow[..., 1] - dy).mean()
                second = index // MOTION_FPS
                per_second.setdefault(second, []).append((dx, dy, float(residual), 0.0))
            prev = frame
            index += 1
    out = {}
    for second, values in per_second.items():
        arr = np.array(values)
        speed = np.hypot(arr[:, 0], arr[:, 1]) * MOTION_FPS / w
        activity = arr[:, 2] * MOTION_FPS / w
        jitter = float(np.hypot(arr[:, 0].std(), arr[:, 1].std()) * MOTION_FPS / w)
        out[second] = (float(speed.mean()), float(activity.mean()), jitter)
    return out


def recording_second(chapter_start: float, local: float) -> int:
    """The second of the recording that a moment `local` seconds into a chapter falls in."""
    return int(np.floor(chapter_start + local + 1e-9))


def run_visual(trip: Trip, recording_id: int, embedder) -> None:
    """Embeddings and picture quality at 1 fps, motion from optical flow at 5 fps.

    Every sample is placed by its own time in the recording (chapter start plus its time in the
    chapter), so chapters whose lengths aren't whole seconds can't shift later samples. Embedding
    row i is second i of the recording; a second that got no sample repeats the previous one.
    """
    size = embedder.image_size
    seconds: dict[int, dict] = {}
    by_second: dict[int, np.ndarray] = {}
    for chapter in chapter_media(trip, recording_id):
        rate = FPS if chapter.duration > 0 else None  # a photo is a single frame
        index = 0
        for batch in stream_frames(chapter.proxy, rate, size, size, gray=False):
            emb = embedder.encode_images(batch)
            blur, mean, dark, bright = _quality(batch)
            for i in range(len(batch)):
                second = recording_second(chapter.start, (index + i) / FPS)
                if second in by_second:
                    continue  # already sampled, from the end of the previous chapter
                by_second[second] = emb[i]
                seconds[second] = {
                    "blur": float(blur[i]),
                    "brightness": float(mean[i]),
                    "dark": float(dark[i]),
                    "bright": float(bright[i]),
                }
            index += len(batch)
        if chapter.duration <= 0:
            continue
        for local, (speed, activity, jitter) in motion_signals(
            chapter.proxy, chapter.width, chapter.height
        ).items():
            seconds.setdefault(recording_second(chapter.start, local), {}).update(
                motion=speed, activity=activity, shake=jitter
            )
    all_emb = fill_rows(by_second)
    scene = np.zeros(len(all_emb))
    if len(all_emb) > 1:
        scene[1:] = 1.0 - np.sum(all_emb[1:] * all_emb[:-1], axis=1)
    for i, value in enumerate(scene):
        seconds.setdefault(i, {})["scene"] = float(value)

    key = trip.db.execute("SELECT key FROM recordings WHERE id = ?", (recording_id,)).fetchone()[0]
    version = visual_version(embedder)
    path = _write_embeddings(trip, key, version, all_emb)
    old = trip.db.execute(
        "SELECT path FROM embeddings WHERE recording_id = ?", (recording_id,)
    ).fetchone()
    columns = ("motion", "activity", "shake", "blur", "brightness", "dark", "bright", "scene")
    with transaction(trip.db):
        _upsert(trip, recording_id, seconds, columns)
        trip.db.execute(
            "INSERT OR REPLACE INTO embeddings (recording_id, path, model, fps, count, dim)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                recording_id,
                path.relative_to(trip.root).as_posix(),
                embedder.name,
                FPS,
                len(all_emb),
                all_emb.shape[1],
            ),
        )
        _mark(trip, recording_id, "visual", version)
    # The previous file is removed only after the new one is published.
    if old and (trip.root / old["path"]) != path:
        (trip.root / old["path"]).unlink(missing_ok=True)


def _write_embeddings(trip: Trip, key: str, version: str, emb: np.ndarray) -> Path:
    """Write embeddings to a file named after the recording and analysis version, through a
    temporary file, so an interrupted write never damages a file the database points at."""
    import hashlib
    import os

    folder = trip.root / "cache" / "embeddings"
    folder.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha1(version.encode()).hexdigest()[:8]
    path = folder / f"{key}_{tag}.npy"
    partial = folder / f"{key}_{tag}.partial.npy"
    with partial.open("wb") as f:
        np.save(f, emb.astype(np.float16))
    os.replace(partial, path)
    return path


def embeddings_ok(trip: Trip, recording_id: int) -> bool:
    """Whether a recording's stored embeddings file exists and has the recorded shape."""
    row = trip.db.execute(
        "SELECT path, count FROM embeddings WHERE recording_id = ?", (recording_id,)
    ).fetchone()
    if row is None:
        return False
    try:
        data = np.load(trip.root / row["path"], mmap_mode="r")
    except (OSError, ValueError, EOFError):
        return False
    return data.shape[0] == row["count"]


def _upsert(trip: Trip, recording_id: int, seconds: dict[int, dict], columns) -> None:
    sets = ", ".join(f"{c} = excluded.{c}" for c in columns)
    trip.db.executemany(
        f"INSERT INTO observations (recording_id, second, {', '.join(columns)})"
        f" VALUES (?, ?, {', '.join('?' for _ in columns)})"
        f" ON CONFLICT (recording_id, second) DO UPDATE SET {sets}",
        [
            (recording_id, second, *(values.get(c) for c in columns))
            for second, values in sorted(seconds.items())
        ],
    )


def _mark(trip: Trip, recording_id: int, stage: str, version: str) -> None:
    trip.db.execute(
        "INSERT OR REPLACE INTO stage_runs (recording_id, stage, version, finished_at)"
        " VALUES (?, ?, ?, ?)",
        (recording_id, stage, version, now()),
    )


def fill_rows(by_second: dict[int, np.ndarray]) -> np.ndarray:
    """Rows 0 to the last second, each the sample of that second or, if none, the previous one."""
    if not by_second:
        return np.zeros((0, 1), np.float32)
    dim = len(next(iter(by_second.values())))
    out = np.zeros((max(by_second) + 1, dim), np.float32)
    last = np.zeros(dim, np.float32)
    for second in range(len(out)):
        if second in by_second:
            last = by_second[second]
        out[second] = last
    return out


def visual_version(embedder) -> str:
    return f"{VISUAL_VERSION}:{embedder.name}:{FPS}:{MOTION_FPS}"


def speech_version(transcriber) -> str:
    return f"{SPEECH_VERSION}:{transcriber.name if transcriber else 'none'}"


# --- speech and audio -----------------------------------------------------------------------


def run_speech(trip: Trip, recording_id: int, transcriber) -> None:
    rate = 16000
    seconds: dict[int, dict] = {}
    segments: list[dict] = []
    language = ""
    for chapter in chapter_media(trip, recording_id):
        if not chapter.has_audio:
            continue
        samples = decode_audio(chapter.proxy, rate)
        offset = chapter.start
        for i in range(0, len(samples) // rate + 1):
            chunk = samples[i * rate : (i + 1) * rate]
            if len(chunk) < rate // 10:
                continue
            rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
            seconds[int(offset) + i] = {"loudness": 20 * np.log10(rms + 1e-9), "speech": 0.0}
        if transcriber is not None and len(samples) > rate // 2:
            found, language = transcriber.transcribe(samples)
            for s in found:
                s = dict(s, start=s["start"] + offset, end=s["end"] + offset)
                s["words"] = [
                    dict(w, start=w["start"] + offset, end=w["end"] + offset) for w in s["words"]
                ]
                segments.append(s)
    for s in segments:
        for w in s["words"]:
            first, last = int(w["start"]), int(max(w["start"], w["end"] - 1e-6))
            for second in range(first, last + 1):
                overlap = min(w["end"], second + 1) - max(w["start"], second)
                if overlap > 0 and second in seconds:
                    seconds[second]["speech"] = min(1.0, seconds[second]["speech"] + overlap)
    with transaction(trip.db):
        trip.db.execute("DELETE FROM speech_segments WHERE recording_id = ?", (recording_id,))
        trip.db.execute("DELETE FROM words WHERE recording_id = ?", (recording_id,))
        for s in segments:
            trip.db.execute(
                "INSERT INTO speech_segments (recording_id, start, end, text, language,"
                " avg_logprob, no_speech_prob) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    recording_id,
                    s["start"],
                    s["end"],
                    s["text"],
                    language,
                    s["avg_logprob"],
                    s["no_speech_prob"],
                ),
            )
            trip.db.executemany(
                "INSERT INTO words (recording_id, start, end, word, probability)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (recording_id, w["start"], w["end"], w["word"], w["probability"])
                    for w in s["words"]
                ],
            )
        _upsert(trip, recording_id, seconds, ("loudness", "speech"))
        _mark(trip, recording_id, "speech", speech_version(transcriber))


# --- orchestration ------------------------------------------------------------------------


def _done(trip: Trip, stage: str, version: str) -> set[int]:
    rows = trip.db.execute(
        "SELECT recording_id FROM stage_runs WHERE stage = ? AND version = ?", (stage, version)
    )
    return {r["recording_id"] for r in rows}


def _recordings(trip: Trip) -> list[int]:
    rows = trip.db.execute(
        "SELECT r.id FROM recordings r WHERE NOT EXISTS ("
        " SELECT 1 FROM chapters c LEFT JOIN proxies p ON p.recording_id = c.recording_id"
        " AND p.chapter_idx = c.idx WHERE c.recording_id = r.id AND p.recording_id IS NULL)"
        " ORDER BY r.id"
    )
    return [r["id"] for r in rows]


@dataclass
class AnalysisSummary:
    visual: int = 0
    speech: int = 0
    described: int = 0
    failed: int = 0
    skipped_without_proxy: int = 0


def analyze(
    trip: Trip,
    embedder_factory: Callable[[], object],
    transcriber_factory: Callable[[], object | None],
    progress: Progress | None = None,
    say: Callable[[str], None] = print,
    describer: tuple[str, Callable[[], object]] | None = None,
) -> AnalysisSummary:
    """Run every stage that is missing or out of date. Models load once per stage."""
    from .ml.runtime import free_gpu

    summary = AnalysisSummary()
    proxies_mod.reconcile(trip)  # a proxy deleted since prepare must not be read
    recordings = _recordings(trip)
    total = trip.db.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]
    summary.skipped_without_proxy = total - len(recordings)
    videos = {r["id"] for r in trip.db.execute("SELECT id FROM recordings WHERE kind = 'video'")}

    speech_name = _model_name(transcriber_factory)
    transcriber = None
    if speech_name is None:
        transcriber = transcriber_factory()
        version = speech_version(transcriber)
    else:
        version = f"{SPEECH_VERSION}:{speech_name}"
    todo = [r for r in recordings if r in videos and r not in _done(trip, "speech", version)]
    if todo:
        if transcriber is None:
            transcriber = transcriber_factory()
        say(f"Speech and sound: {len(todo)} recording(s)")
        items = list(map(str, todo))
        result = Job(trip, "speech", {"version": version}, items, redo=items).run(
            lambda item: run_speech(trip, int(item), transcriber), progress
        )
        summary.speech, summary.failed = result.done, summary.failed + len(result.failed)
        _report(trip, "speech", result.failed)
    transcriber = None  # release the model before loading the next one
    free_gpu()

    from .scoring import prompt_bank_model, save_prompt_bank

    embedder = None
    if recordings:
        visual_name = _model_name(embedder_factory)
        if visual_name is None:
            embedder = embedder_factory()
            visual_name = embedder.name
        version = f"{VISUAL_VERSION}:{visual_name}:{FPS}:{MOTION_FPS}"
        done = _done(trip, "visual", version)
        todo = [r for r in recordings if r not in done or not embeddings_ok(trip, r)]
        stale_bank = prompt_bank_model(trip) != visual_name  # also when missing or damaged
        if (todo or stale_bank) and embedder is None:
            embedder = embedder_factory()
        if stale_bank:
            save_prompt_bank(trip, embedder)
        if todo:
            say(f"Pictures and motion: {len(todo)} recording(s)")
            items = list(map(str, todo))
            result = Job(trip, "visual", {"version": version}, items, redo=items).run(
                lambda item: run_visual(trip, int(item), embedder), progress
            )
            summary.visual, summary.failed = result.done, summary.failed + len(result.failed)
            _report(trip, "visual", result.failed)
    embedder = None
    free_gpu()

    if describer is not None and recordings:
        from . import vlm

        name, factory = describer
        say(f"Describing scenes with {name}")
        done, failed = vlm.analyze(trip, recordings, name, factory, progress)
        summary.described, summary.failed = done, summary.failed + len(failed)
        _report(trip, "vlm", failed)
        free_gpu()
    return summary


def _model_name(factory) -> str | None:
    """The model a factory would load, if it says so without loading it."""
    return getattr(factory, "model_name", None) or getattr(factory, "name", None)


def _report(trip: Trip, stage: str, failed: dict[str, str]) -> None:
    trip.clear_issues(stage)
    for item, error in failed.items():
        trip.issue(stage, "error", f"Analysis failed: {error}", recording_id=int(item))
