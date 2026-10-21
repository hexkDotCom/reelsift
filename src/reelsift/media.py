"""Probing media files with ffprobe, and identifying files by content."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

from .tools import run, tool

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".mkv", ".lrf", ".lrv", ".webm", ".avi", ".mpg", ".mpeg", ".mts",
    ".m2ts", ".ts", ".3gp", ".wmv", ".flv", ".dv", ".mxf",
}  # fmt: skip
SIDECAR_EXTENSIONS = {
    ".thm", ".srt", ".lrv", ".lrf", ".xml", ".json", ".txt", ".md", ".log", ".ini", ".db",
    ".dat", ".bin", ".zip", ".tgz", ".gz", ".gpx", ".fit", ".aae", ".xmp", ".pdf", ".csv",
    ".mat", ".h5", ".raw", ".ds_store",
}  # fmt: skip
"""Files that are often next to footage and aren't media: skipped without a mention."""
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
FINGERPRINT_SAMPLE = 1 << 20


def fingerprint(path: Path) -> str:
    """Identify a file by its size and three 1 MiB samples (start, middle, end).

    This is an identity, not an integrity check: it is fast on 100 GB of footage and survives
    moving or renaming files, which is what relinking needs.
    """
    size = path.stat().st_size
    h = hashlib.blake2b(digest_size=16)
    h.update(size.to_bytes(8, "little"))
    with path.open("rb") as f:
        for offset in (
            0,
            max(0, size // 2 - FINGERPRINT_SAMPLE // 2),
            max(0, size - FINGERPRINT_SAMPLE),
        ):
            f.seek(offset)
            h.update(f.read(FINGERPRINT_SAMPLE))
    return h.hexdigest()


def _fraction(text: str | None) -> Fraction | None:
    if not text or text in ("0/0", "N/A"):
        return None
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def _time(text: str | None) -> Fraction | None:
    """A timestamp, which (unlike a duration or a rate) can be zero or before zero."""
    if not text or text == "N/A":
        return None
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def _int(text) -> int | None:
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def parse_time(text: str | None) -> datetime | None:
    """Parse an ffprobe date tag. Returns an aware datetime only when the tag carries a zone."""
    if not text:
        return None
    # ffprobe joins repeated tags with ";" (some muxers write the time twice): use the first.
    value = text.split(";")[0].strip().replace(" ", "T")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


@dataclass
class VideoStream:
    index: int
    codec: str
    width: int
    height: int
    time_base: Fraction
    start: Fraction
    duration: Fraction | None
    frames: int | None
    avg_fps: Fraction | None
    r_fps: Fraction | None
    rotation: int
    pix_fmt: str
    color_transfer: str
    color_primaries: str
    color_space: str
    color_range: str
    field_order: str = "progressive"

    @property
    def full_range(self) -> bool:
        """Levels 0-255 instead of 16-235 (phones and JPEG-style "yuvj" video)."""
        return self.color_range == "pc" or self.pix_fmt.startswith("yuvj")

    @property
    def interlaced(self) -> bool:
        """Fields from two moments in each frame (camcorders, broadcast); needs deinterlacing."""
        return self.field_order in ("tt", "bb", "tb", "bt")

    @property
    def vfr(self) -> bool:
        """Heuristic: the nominal and average frame rates differ by more than 0.5%."""
        if not self.avg_fps or not self.r_fps:
            return False
        return abs(self.avg_fps - self.r_fps) / self.r_fps > Fraction(1, 200)

    @property
    def hdr(self) -> bool:
        return self.color_transfer in HDR_TRANSFERS

    @property
    def upright_size(self) -> tuple[int, int]:
        return (self.height, self.width) if self.rotation % 180 else (self.width, self.height)


@dataclass
class AudioStream:
    index: int
    codec: str
    sample_rate: int
    channels: int


@dataclass
class Probe:
    path: Path
    format_name: str
    duration: Fraction
    tags: dict[str, str]
    video: list[VideoStream] = field(default_factory=list)
    audio: list[AudioStream] = field(default_factory=list)
    data_handlers: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def main_video(self) -> VideoStream | None:
        return self.video[0] if self.video else None

    @property
    def start(self) -> Fraction:
        """Where the file's timestamps start (the earliest stream, possibly before zero); FFmpeg's
        -ss counts from here, while recording time counts from the first video frame."""
        return _time(self.raw.get("format", {}).get("start_time")) or Fraction(0)

    @property
    def encoder(self) -> str:
        return self.tags.get("encoder", "")

    @property
    def creation_time(self) -> datetime | None:
        return parse_time(self.tags.get("creation_time"))

    def tag(self, *names: str) -> str | None:
        for name in names:
            if name in self.tags:
                return self.tags[name]
        return None


def _rotation(stream: dict) -> int:
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            return round(-float(side["rotation"])) % 360
    rotate = stream.get("tags", {}).get("rotate")
    return int(rotate) % 360 if rotate else 0


def _is_thumbnail(stream: dict) -> bool:
    if stream.get("disposition", {}).get("attached_pic"):
        return True
    # Cameras store a cover image as an MJPEG "video" stream with a single frame. Without a
    # frame count (Matroska), a stream that lasts is a real MJPEG movie (webcams, old cameras).
    if stream.get("codec_name") not in ("mjpeg", "png"):
        return False
    frames = _int(stream.get("nb_frames"))
    if frames is not None:
        return frames <= 1
    return _stream_seconds(stream) < 0.5


def _stream_seconds(stream: dict) -> float:
    """How long a stream lasts, from its duration or Matroska's DURATION tag (0 if unknown)."""
    seconds = _fraction(stream.get("duration"))
    if seconds:
        return float(seconds)
    tag = {k.lower(): v for k, v in stream.get("tags", {}).items()}.get("duration", "")
    try:
        h, m, s = tag.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except ValueError:
        return 0.0


def parse_probe(path: Path, data: dict) -> Probe:
    fmt = data.get("format", {})
    tags = {k.lower(): str(v) for k, v in fmt.get("tags", {}).items()}
    probe = Probe(
        path=path,
        format_name=fmt.get("format_name", ""),
        duration=_fraction(fmt.get("duration")) or Fraction(0),
        tags=tags,
        raw=data,
    )
    for s in data.get("streams", []):
        kind = s.get("codec_type")
        if kind == "video" and not _is_thumbnail(s):
            time_base = _fraction(s.get("time_base")) or Fraction(1, 90000)
            duration_ts = _int(s.get("duration_ts"))
            start_pts = _int(s.get("start_pts")) or 0
            probe.video.append(
                VideoStream(
                    index=s["index"],
                    codec=s.get("codec_name", ""),
                    width=_int(s.get("width")) or 0,
                    height=_int(s.get("height")) or 0,
                    time_base=time_base,
                    start=start_pts * time_base,
                    duration=duration_ts * time_base
                    if duration_ts
                    else _fraction(s.get("duration")),
                    frames=_int(s.get("nb_frames")),
                    avg_fps=_fraction(s.get("avg_frame_rate")),
                    r_fps=_fraction(s.get("r_frame_rate")),
                    rotation=_rotation(s),
                    pix_fmt=s.get("pix_fmt", ""),
                    color_transfer=s.get("color_transfer", "unknown"),
                    color_primaries=s.get("color_primaries", "unknown"),
                    color_space=s.get("color_space", "unknown"),
                    color_range=s.get("color_range", "unknown"),
                    field_order=s.get("field_order", "progressive"),
                )
            )
        elif kind == "audio":
            probe.audio.append(
                AudioStream(
                    index=s["index"],
                    codec=s.get("codec_name", ""),
                    sample_rate=_int(s.get("sample_rate")) or 0,
                    channels=_int(s.get("channels")) or 0,
                )
            )
        elif kind == "data":
            handler = s.get("tags", {}).get("handler_name") or s.get("codec_tag_string", "")
            probe.data_handlers.append(handler)
    return probe


def probe(path: Path) -> Probe:
    out = run(
        [tool("ffprobe"), "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        text=True,
    ).stdout
    return parse_probe(path, json.loads(out))


def utc(dt: datetime) -> datetime:
    return dt.astimezone(UTC)
