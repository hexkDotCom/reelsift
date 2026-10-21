"""Ground speed per second from GPS that the camera recorded: GoPro telemetry (GPS5, GPS9), and
dashcams that write one text line per second into a metadata track ("mebx"), such as
`A,050426,122609,36.591196,-82.275572,-0.88,0.14,0.18,43.9,571.3,281.0,...`: fix (A valid,
V not), date, UTC time, latitude, longitude, three accelerations, speed in knots, altitude,
heading.

Speed is much more reliable than the picture for telling driving from waiting at a red light.
"""

import math
import re
from pathlib import Path

import numpy as np

from . import gpmf
from .media import Probe

KNOT_KMH = 1.852
DASHCAM_LINE = re.compile(
    rb"([AV]),(\d{6}),(\d{6}(?:\.\d+)?),(-?\d+\.\d+),(-?\d+\.\d+),"
    rb"(-?[\d.]+),(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)"
)
VERSION = 1


def samples(path: Path, probe: Probe) -> list[tuple[float, float | None]] | None:
    """(seconds after the first video frame, km/h or None without a fix), or None when the file
    has no speed data Reelsift can read."""
    streams = probe.raw.get("streams", [])
    if any(s.get("codec_tag_string") == "gpmd" for s in streams):
        found = gpmf.speeds(gpmf.packets(path, probe, count=None))
        return found if any(v is not None for _, v in found) or found else None
    for s in streams:
        if s.get("codec_tag_string") != "mebx":
            continue
        first = gpmf.packets(path, probe, count=2, stream=s["index"])
        if not any(DASHCAM_LINE.search(data) for _, data in first):
            continue
        out = []
        for t, data in gpmf.packets(path, probe, count=None, stream=s["index"]):
            m = DASHCAM_LINE.search(data)
            if m:
                valid = m.group(1) == b"A"
                out.append((t, float(m.group(9)) * KNOT_KMH if valid else None))
        return out
    return None


def per_second(found: list[tuple[float, float | None]], duration: float) -> np.ndarray:
    """km/h for each second of a file (NaN where unknown): the samples within it, averaged."""
    n = max(0, math.ceil(duration))
    total = np.zeros(n)
    count = np.zeros(n)
    for t, v in found:
        k = math.floor(t)
        if v is not None and 0 <= k < n and math.isfinite(v):
            total[k] += v
            count[k] += 1
    out = np.full(n, np.nan)
    np.divide(total, count, out=out, where=count > 0)
    return out
