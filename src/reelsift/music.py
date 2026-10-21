"""Music: finding beats, so cuts can land on them.

A small beat tracker written with NumPy rather than a large audio library: onset strength from
spectral flux, tempo from the autocorrelation of that envelope with a preference for about 120 BPM
(refined between envelope frames), and beat times by dynamic programming (Ellis, 2007). Good
enough for music with a steady beat. Each track also gets a confidence: how much stronger onsets
are on its beats than elsewhere; music without a clear beat gets a low one, and cuts are not
snapped to it.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tools import run, tool

RATE = 22050
HOP = 512
N_FFT = 2048
BPM_RANGE = (60, 180)
PREFERRED_BPM = 120
TIGHTNESS = 100.0
LATENCY = (N_FFT // 2 + 3 * HOP // 4) / RATE
"""An onset shows up in the envelope about this much before it is heard (the window's centre plus
part of a hop); calibrated on click tracks, where it removes a constant 64 ms lead."""
VERSION = 4
WEAK_BEAT = 0.3
"""Below this confidence a track's beats are not used for snapping."""


@dataclass
class Track:
    path: Path
    duration: float
    bpm: float
    beats: np.ndarray
    """Beat times in seconds from the start of the track."""
    confidence: float = 1.0


def load_audio(path: Path, rate: int = RATE) -> np.ndarray:
    raw = run(
        [tool("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1",
         "-ar", str(rate), "-f", "f32le", "-"]
    ).stdout  # fmt: skip
    return np.frombuffer(raw, np.float32)


def onset_envelope(y: np.ndarray, rate: int = RATE) -> tuple[np.ndarray, float]:
    """Spectral flux per hop, and the envelope's sample rate in frames per second."""
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    frames = np.lib.stride_tricks.sliding_window_view(y, N_FFT)[::HOP]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(N_FFT), axis=1))
    log_mag = np.log1p(1000 * spectrum)
    flux = np.maximum(0, np.diff(log_mag, axis=0)).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    width = min(16, len(flux))
    flux -= np.convolve(flux, np.ones(width) / width, mode="same")  # remove slow loudness changes
    flux = np.maximum(flux, 0)
    peak = flux.max()
    return (flux / peak if peak > 0 else flux), rate / HOP


def estimate_period(
    env: np.ndarray, frame_rate: float, bpm_range: tuple[float, float] = BPM_RANGE,
    preferred: float = PREFERRED_BPM,
) -> float:  # fmt: skip
    """Beat period in envelope frames, refined between frames."""
    corr = _autocorrelation(env)
    lags = np.arange(len(corr), dtype=float)
    lo = int(frame_rate * 60 / bpm_range[1])
    hi = min(len(corr) - 1, int(np.ceil(frame_rate * 60 / bpm_range[0])))
    if hi <= lo:
        return frame_rate * 60 / preferred
    bpm = 60 * frame_rate / np.maximum(lags[lo : hi + 1], 1)
    prior = np.exp(-0.5 * (np.log2(bpm / preferred) / 0.9) ** 2)
    weighted = corr[lo : hi + 1] * prior
    k = int(np.argmax(weighted))
    best = float(lo + k)
    if 0 < k < len(weighted) - 1:  # parabola through the peak and its neighbours
        a, b, c = weighted[k - 1], weighted[k], weighted[k + 1]
        if a - 2 * b + c < 0:
            best += 0.5 * (a - c) / (a - 2 * b + c)
    return best


def track_beats(env: np.ndarray, period: float) -> np.ndarray:
    """Beat positions in envelope frames, by dynamic programming."""
    n = len(env)
    score = env.copy()
    back = np.full(n, -1)
    lo, hi = max(1, int(period / 2)), int(2 * period)
    for t in range(n):
        start, end = max(0, t - hi), t - lo
        if end <= start:
            continue
        prev = np.arange(start, end)
        penalty = -TIGHTNESS * np.log((t - prev) / period) ** 2
        candidates = score[prev] + penalty
        k = int(np.argmax(candidates))
        if candidates[k] > 0:
            score[t] = env[t] + candidates[k]
            back[t] = prev[k]
    tail = slice(max(0, n - int(period)), n)
    t = tail.start + int(np.argmax(score[tail]))
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    return np.array(beats[::-1], float)


def _autocorrelation(env: np.ndarray) -> np.ndarray:
    x = env - env.mean()
    size = 1 << int(np.ceil(np.log2(max(2, 2 * len(x)))))
    spectrum = np.fft.rfft(x, size)
    return np.fft.irfft(spectrum * np.conj(spectrum), size)[: len(x)]


def beat_confidence(env: np.ndarray, period: float) -> float:
    """0 to 1: how strongly the onsets repeat at the beat period (the envelope's normalised
    autocorrelation there). Steady beats give 0.6-0.9; pads, ambient music and noise near 0."""
    corr = _autocorrelation(env)
    if len(corr) < 3 or corr[0] <= 0 or period >= len(corr) - 1:
        return 0.0
    k = int(period)
    value = corr[k] + (corr[k + 1] - corr[k]) * (period - k)
    return float(np.clip(value / corr[0], 0, 1))


def analyze_file(path: Path, expected_bpm: float | None = None) -> Track:
    """Beats of a music file. With expected_bpm (generated music, made at a known tempo), a
    grid near that tempo is tried too, and whichever has the clearer beat is kept."""
    y = load_audio(path)
    env, frame_rate = onset_envelope(y)
    found = []
    tries = [(BPM_RANGE, PREFERRED_BPM)]
    if expected_bpm:
        tries.append(((expected_bpm * 0.94, expected_bpm * 1.06), expected_bpm))
    for bpm_range, preferred in tries:
        period = estimate_period(env, frame_rate, bpm_range, preferred)
        frames = track_beats(env, period)
        found.append((beat_confidence(env, period), period, frames))
    confidence, period, frames = max(found, key=lambda f: f[0])
    beats = frames / frame_rate + LATENCY
    return Track(path, len(y) / RATE, 60 * frame_rate / period, beats, confidence)


def load(trip_root: Path, paths: list[str]) -> list[Track]:
    """Beats for each music file, cached in the trip by file identity."""
    from .media import fingerprint

    folder = trip_root / "cache" / "music"
    folder.mkdir(parents=True, exist_ok=True)
    tracks = []
    for p in paths:
        path = Path(p)
        if not path.is_absolute():
            path = trip_root / path
        key = hashlib.sha1(f"{fingerprint(path)}:{VERSION}".encode()).hexdigest()[:16]
        cache = folder / f"{key}.json"
        if cache.is_file():
            data = json.loads(cache.read_text(encoding="utf-8"))
            tracks.append(Track(path, data["duration"], data["bpm"], np.array(data["beats"]),
                                data.get("confidence", 1.0)))  # fmt: skip
            continue
        track = analyze_file(path)
        cache.write_text(
            json.dumps({"duration": track.duration, "bpm": track.bpm,
                        "beats": track.beats.tolist(), "confidence": track.confidence}),
            encoding="utf-8",
        )  # fmt: skip
        tracks.append(track)
    return tracks


def beat_grid(tracks: list[Track], length: float, min_confidence: float = 0.0) -> np.ndarray:
    """Beat times over an edit of `length` seconds: the tracks played in order, looping. Tracks
    whose beat is less clear than min_confidence play their part without beats."""
    total = sum(t.duration for t in tracks)
    if not tracks or total <= 0:
        return np.zeros(0)
    out, offset = [], 0.0
    while offset < length:
        for t in tracks:
            if t.confidence >= min_confidence:
                out.append(t.beats + offset)
            offset += t.duration
            if offset >= length:
                break
    if not out:
        return np.zeros(0)
    grid = np.concatenate(out)
    return grid[grid <= length + 1]


def music_key(tracks: list[Track]) -> str:
    return ",".join(f"{t.path.name}:{t.duration:.2f}" for t in tracks)
