"""Reading GoPro's GPMF telemetry: the camera model, and GPS time when the camera has a fix.

GPMF is a stream of KLV entries: a four-character key, a one-character type, a one-byte size of
each sample and a two-byte count, then the data padded to four bytes. Type 0 means the data is
more entries (a container). Only the first few seconds are read, or further until GPS
has a fix.
"""

import struct
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path

import numpy as np

from .media import Probe
from .tools import tool


@dataclass
class Telemetry:
    model: str | None = None
    """Device name, such as "HERO13 Black"."""
    gps_utc: datetime | None = None
    """UTC from GPS in the first packet with a fix, and that packet's time in seconds after the
    first video frame."""
    gps_offset: float = 0.0


def entries(data: bytes, start: int = 0, end: int | None = None):
    """(key, type, data) for every entry, descending into containers."""
    end = len(data) if end is None else end
    pos = start
    while pos + 8 <= end:
        key = data[pos : pos + 4]
        kind, size, repeat = (
            data[pos + 4],
            data[pos + 5],
            struct.unpack(">H", data[pos + 6 : pos + 8])[0],
        )
        length = size * repeat
        body = pos + 8
        if body + length > end or not key.isascii():
            return
        if kind == 0:
            yield from entries(data, body, body + length)
        else:
            yield key.decode("ascii", "replace"), chr(kind), data[body : body + length]
        pos = body + ((length + 3) & ~3)


NUMBERS = {"b": "b", "B": "B", "s": "h", "S": "H", "l": "i", "L": "I", "f": "f", "d": "d"}
"""GPMF number types and their struct codes (big-endian)."""
GPS_KEYS = (b"GPSF", b"GPS9")
"""Present when the camera has GPS, with or without a fix."""
EPOCH_2000 = datetime(2000, 1, 1, tzinfo=UTC)


def _numbers(kind: str, value: bytes) -> list[float]:
    code = NUMBERS.get(kind)
    if code is None:
        return []
    size = struct.calcsize(code)
    return list(struct.unpack(f">{len(value) // size}{code}", value[: len(value) // size * size]))


def _gps9(value: bytes, layout: str, scale: list[float]) -> tuple[datetime | None, int]:
    """UTC and fix of the first GPS9 sample (HERO11 and later): latitude, longitude, altitude,
    2D and 3D speed, days since 2000, seconds since midnight, DOP and fix, each divided by its
    scale."""
    codes = [NUMBERS.get(c) for c in layout]
    if len(codes) < 9 or None in codes:
        return None, 0
    sample = struct.calcsize(">" + "".join(codes))
    if len(value) < sample:
        return None, 0
    fields = struct.unpack(">" + "".join(codes), value[:sample])
    if len(scale) == 1:  # one scale for every field
        scale = scale * len(fields)
    scale = (scale + [1.0] * len(fields))[: len(fields)]
    values = [f / (s or 1) for f, s in zip(fields, scale, strict=True)]
    days, seconds, fix = values[5], values[6], int(values[8])
    if not (0 <= days < 40000 and 0 <= seconds < 86401):
        return None, fix
    return EPOCH_2000 + timedelta(days=days, seconds=seconds), fix


MAX_DOP = 10.0
"""Dilution of precision above which a GPS speed isn't trusted (GoPro records it per second)."""


def speeds(packets: list[tuple[float, bytes]]) -> list[tuple[float, float | None]]:
    """Ground speed in km/h for each packet (about a second) from GPS5 or GPS9 samples: the
    median of the packet's samples, or None without a 2D/3D fix or with a poor one (DOP over MAX_DOP)."""
    out = []
    for seconds, packet in packets:
        fix, dop, values = None, None, []
        scale: list[float] = []
        layout = ""
        for key, kind, value in entries(packet):
            if key == "SCAL":
                scale = _numbers(kind, value)
            elif key == "TYPE" and kind == "c":
                layout = value.rstrip(b"\0").decode("ascii", "replace")
            elif key == "GPSF" and len(value) >= 4:
                fix = struct.unpack(">I", value[:4])[0]
            elif key == "GPSP" and len(value) >= 2:
                dop = struct.unpack(">H", value[:2])[0] / 100
            elif key == "GPS5" and kind == "l":
                numbers = _numbers(kind, value)
                s = scale[3] if len(scale) > 3 else (scale[0] if scale else 1.0)
                values += [numbers[i + 3] / (s or 1) for i in range(0, len(numbers) - 4, 5)]
            elif key == "GPS9" and kind == "?":
                for sample in _gps9_samples(value, layout, scale):
                    values.append(sample[3])
                    fix = int(sample[8])
                    dop = sample[7]
        ok = values and (fix is None or fix >= 2) and (dop is None or dop <= MAX_DOP)
        out.append((seconds, float(np.median(values)) * 3.6 if ok else None))
    return out


def _gps9_samples(value: bytes, layout: str, scale: list[float]) -> list[list[float]]:
    codes = [NUMBERS.get(c) for c in layout]
    if len(codes) < 9 or None in codes:
        return []
    fmt = ">" + "".join(codes)
    size = struct.calcsize(fmt)
    if len(scale) == 1:
        scale = scale * len(codes)
    scale = (scale + [1.0] * len(codes))[: len(codes)]
    return [[f / (s or 1) for f, s in zip(struct.unpack(fmt, value[k : k + size]), scale,
                                          strict=True)]
            for k in range(0, len(value) - size + 1, size)]  # fmt: skip


def parse(packets: list[tuple[float, bytes]]) -> Telemetry:
    """Telemetry from (seconds after the first video frame, packet data) pairs. A packet holds
    one or more device containers (the camera, and sensors paired with it)."""
    out = Telemetry()
    for seconds, packet in packets:
        fix = None
        utc = None
        scale: list[float] = []
        layout = ""
        for key, kind, value in entries(packet):
            if key == "DVNM" and kind == "c" and out.model is None:
                out.model = value.rstrip(b"\0").decode("ascii", "replace").strip() or None
            elif key == "SCAL":  # applies to the next data in the same stream
                scale = _numbers(kind, value)
            elif key == "TYPE" and kind == "c":
                layout = value.rstrip(b"\0").decode("ascii", "replace")
            elif key == "GPSF" and len(value) >= 4:
                fix = struct.unpack(">I", value[:4])[0]
            elif key == "GPSU" and kind == "U" and len(value) >= 16:
                try:
                    utc = datetime.strptime(value[:16].decode("ascii"), "%y%m%d%H%M%S.%f")
                    utc = utc.replace(tzinfo=UTC)
                except ValueError:
                    utc = None
            elif key == "GPS9" and kind == "?" and utc is None:
                utc, fix = _gps9(value, layout, scale)
        if utc is not None and (fix is None or fix >= 2) and out.gps_utc is None:
            out.gps_utc, out.gps_offset = utc, seconds
    return out


def packets(
    path: Path, probe: Probe, count: int | None = 5, stream: int | None = None
) -> list[tuple[float, bytes]]:
    """The first telemetry packets of a file (all with count None), with their time after the
    first video frame. stream: another data stream than GoPro's "gpmd". FFmpeg joins the
    packets' data, so their sizes and times are listed separately."""
    if stream is None:
        stream = next(
            (s["index"] for s in probe.raw.get("streams", [])
             if s.get("codec_tag_string") == "gpmd"), None,
        )  # fmt: skip
    if stream is None:
        return []
    # Only the telemetry stream is read (FFmpeg skips the others), so this is quick even far
    # into a large file.
    common = [tool("ffmpeg"), "-v", "error", "-copyts", "-i", str(path), "-map", f"0:{stream}",
              "-c", "copy", *(["-frames:d", str(count)] if count else [])]  # fmt: skip
    listing = subprocess.run(
        [*common, "-f", "framecrc", "-"], capture_output=True, text=True, timeout=600
    ).stdout.splitlines()
    timebase, sizes, times = None, [], []
    for line in listing:
        if line.startswith("#tb 0:"):
            num, _, den = line.split(":", 1)[1].strip().partition("/")
            timebase = Fraction(int(num), int(den))
        elif line and not line.startswith("#") and timebase is not None:
            fields = [f.strip() for f in line.split(",")]
            try:
                times.append(float(int(fields[2]) * timebase))
                sizes.append(int(fields[4]))
            except (IndexError, ValueError):
                return []
    if not sizes:
        return []
    data = subprocess.run([*common, "-f", "data", "-"], capture_output=True, timeout=600).stdout
    video_start = float(probe.main_video.start) if probe.main_video else 0.0
    out, pos = [], 0
    for t, size in zip(times, sizes, strict=True):
        if pos + size > len(data):
            break
        out.append((t - video_start, data[pos : pos + size]))
        pos += size
    return out


SCAN_PACKETS = 1200
"""How far to look for a GPS fix when the first seconds have none: twenty minutes, longer than a
chapter. GPS can take minutes to find satellites."""


def read(path: Path, probe: Probe) -> Telemetry:
    try:
        first = packets(path, probe)
        found = parse(first)
        if found.gps_utc is None and any(k in data for _, data in first for k in GPS_KEYS):
            # The camera has GPS but no fix yet: read on until it has one.
            later = parse(packets(path, probe, SCAN_PACKETS))
            found.gps_utc, found.gps_offset = later.gps_utc, later.gps_offset
        return found
    except (OSError, subprocess.SubprocessError):
        return Telemetry()
