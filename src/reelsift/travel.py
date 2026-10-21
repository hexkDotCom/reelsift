"""Is the camera travelling? Per second of a recording: moving, stopped or unknown.

From GPS speed when the camera recorded it (see speed.py), otherwise from the picture: optical
flow on a 4x4 grid of tiles, the flow magnitude of the busiest tiles (85th percentile) among those
that move at some point in the recording (tiles that never do, such as a car's bonnet or a bike's handlebars, don't count).
Driving forward moves almost every tile of the background; waiting at a light with cars crossing
moves a few. Thresholds come from the recording's own values (they differ between dashcams,
helmets and handlebars), and only clear stops count: anything uncertain is "unknown" and is kept.
"""

import math
from dataclasses import dataclass

import numpy as np

from .analysis import MOTION_FPS, _motion_size, stream_frames

MOVING, STOPPED, UNKNOWN = 1, 0, -1
GRID = 4
BUSY_PERCENTILE = 85
VERSION = 2
MIN_TEXTURE = 6.0
"""A frame whose grey levels vary less than this (standard deviation, 0-255) shows too little to
measure movement (dark, fog, a lens against something): its seconds are unknown."""
MIN_STOP_S = 6
"""A stop is at least this long; shorter pauses are part of the ride."""
PAD_S = 1.0
"""Footage kept on each side of a stop: slowing down and setting off."""
GPS_MOVING_KMH = 3.0
GPS_STOPPED_KMH = 1.5


def tile_flow(path, width: int, height: int) -> dict[int, np.ndarray]:
    """Per second of a video: the mean over its frames of each tile's median flow magnitude, in
    frame widths per second (GRID x GRID values)."""
    import cv2

    w, h = _motion_size(width, height)
    per_second: dict[int, list[np.ndarray]] = {}
    prev = None
    index = 0
    unreadable = np.full(GRID * GRID, np.nan)
    for batch in stream_frames(path, MOTION_FPS, w, h, gray=True, batch=256):
        for frame in batch:
            if prev is not None:
                if min(float(prev.std()), float(frame.std())) < MIN_TEXTURE:
                    per_second.setdefault(index // MOTION_FPS, []).append(unreadable)
                else:
                    flow = cv2.calcOpticalFlowFarneback(prev, frame, None, 0.5, 2, 9, 2, 5, 1.1, 0)
                    magnitude = np.hypot(flow[..., 0], flow[..., 1])
                    rows = np.array_split(magnitude, GRID, axis=0)
                    tiles = [np.median(c) for r in rows for c in np.array_split(r, GRID, axis=1)]
                    per_second.setdefault(index // MOTION_FPS, []).append(np.array(tiles))
            prev = frame
            index += 1
    out = {}
    for s, v in per_second.items():
        readable = [t for t in v if np.isfinite(t).all()]
        # A second counts only if most of it could be read.
        out[s] = (np.mean(readable, axis=0) * MOTION_FPS / w if len(readable) * 2 > len(v)
                  else unreadable.copy())  # fmt: skip
    return out


def travel_values(tiles: np.ndarray) -> np.ndarray:
    """One value per second from (seconds x tiles), leaving out tiles that never move."""
    if not len(tiles):
        return np.zeros(0)
    if not np.isfinite(tiles).any():  # no picture analysed (no proxy): unknown throughout
        return np.full(len(tiles), np.nan)
    busy = np.nanpercentile(tiles, 90, axis=0)
    moving_tiles = busy >= 0.25 * np.nanmedian(busy) if np.nanmedian(busy) > 0 else busy >= 0
    if not moving_tiles.any():
        moving_tiles[:] = True
    # Not the median: on an open road most tiles (sky, the far road, smooth asphalt) barely
    # move even at speed; the busiest tiles (the roadside) tell. Checked against GPS speed on
    # a day of dashcam footage: at the 85th percentile, no driving was called stopped.
    return np.nanpercentile(tiles[:, moving_tiles], BUSY_PERCENTILE, axis=1)


@dataclass(frozen=True)
class Thresholds:
    stopped: float
    moving: float


def thresholds(values: np.ndarray) -> Thresholds | None:
    """Where stopped ends and moving starts, from the two groups the values fall into (Otsu's
    method on their logarithms). None when they don't fall into two clear groups (all driving,
    all waiting, or too little footage): then nothing is called a stop."""
    v = values[np.isfinite(values) & (values > 0)]
    if len(v) < 60:
        return None
    logs = np.log(v)
    hist, edges = np.histogram(logs, bins=64)
    centers = (edges[:-1] + edges[1:]) / 2
    best, cut = -1.0, None
    total = hist.sum()
    for k in range(1, len(hist)):
        w0, w1 = hist[:k].sum() / total, hist[k:].sum() / total
        if w0 < 0.05 or w1 < 0.05:
            continue
        m0 = (hist[:k] * centers[:k]).sum() / hist[:k].sum()
        m1 = (hist[k:] * centers[k:]).sum() / hist[k:].sum()
        between = w0 * w1 * (m1 - m0) ** 2
        if between > best:
            best, cut, low, high = between, edges[k], m0, m1
    if cut is None or high - low < math.log(3):  # the groups must be well apart
        return None
    # Clear stops sit well below the cut, clear driving above it; between is unknown.
    return Thresholds(stopped=math.exp((low + cut) / 2), moving=math.exp(cut))


def classify(values: np.ndarray, t: Thresholds | None) -> np.ndarray:
    """MOVING, STOPPED or UNKNOWN per second from optical travel values."""
    out = np.full(len(values), UNKNOWN, int)
    if t is None:
        return out
    readable = np.isfinite(values)
    values = smooth(values)
    values[~readable] = np.nan  # smoothing never makes an unreadable second look certain
    finite = np.isfinite(values)
    out[finite & (values >= t.moving)] = MOVING
    out[finite & (values <= t.stopped)] = STOPPED
    return _settle(out, unknown_short_stops=True)


SMOOTH_S = 5


def smooth(values: np.ndarray) -> np.ndarray:
    """A running median over SMOOTH_S seconds (NaN where a window has no value): one busy second
    (a car crossing) doesn't break a stop, one quiet second doesn't make one."""
    out = np.full(len(values), np.nan)
    half = SMOOTH_S // 2
    for k in range(len(values)):
        part = values[max(0, k - half) : k + half + 1]
        part = part[np.isfinite(part)]
        if len(part):
            out[k] = np.median(part)
    return out


def classify_gps(kmh: np.ndarray) -> np.ndarray:
    """MOVING, STOPPED or UNKNOWN per second from GPS speed (NaN: no fix), with hysteresis."""
    out = np.full(len(kmh), UNKNOWN, int)
    state = UNKNOWN
    for k, v in enumerate(kmh):
        if not math.isfinite(v):
            state = UNKNOWN
        elif v >= GPS_MOVING_KMH:
            state = MOVING
        elif v < GPS_STOPPED_KMH:
            state = STOPPED
        elif state == UNKNOWN:
            state = MOVING  # creeping
        out[k] = state
    return _settle(out, unknown_short_stops=False)


def _settle(states: np.ndarray, unknown_short_stops: bool) -> np.ndarray:
    """Stops shorter than MIN_STOP_S aren't stops: part of the ride (GPS, reliable) or unknown
    (the picture). Each remaining stop gives PAD_S back to the ride on both sides."""
    out = states.copy()
    for a, b in runs(out, STOPPED):
        if b - a < MIN_STOP_S:
            out[a:b] = UNKNOWN if unknown_short_stops else MOVING
    pad = round(PAD_S)
    for a, b in runs(out, STOPPED):
        out[a : a + pad] = MOVING if not unknown_short_stops else UNKNOWN
        out[max(a, b - pad) : b] = MOVING if not unknown_short_stops else UNKNOWN
    return out


def runs(states: np.ndarray, value: int) -> list[tuple[int, int]]:
    """[start, end) of each run of `value`."""
    out, start = [], None
    for k, s in enumerate(states):
        if s == value and start is None:
            start = k
        elif s != value and start is not None:
            out.append((start, k))
            start = None
    if start is not None:
        out.append((start, len(states)))
    return out
