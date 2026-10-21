"""Shared fixtures. Media fixtures are generated with FFmpeg at test time, so no footage is stored."""

from pathlib import Path

import pytest

from reelsift.tools import ToolNotFoundError, run, tool


def _ffmpeg() -> str | None:
    try:
        return tool("ffmpeg")
    except ToolNotFoundError:
        return None


needs_ffmpeg = pytest.mark.skipif(_ffmpeg() is None, reason="FFmpeg not installed")


def make_clip(
    path: Path,
    seconds: float = 2.0,
    *,
    size: str = "320x180",
    fps: int = 25,
    creation_time: str | None = None,
    rotation: int | None = None,
    audio: bool = True,
    timecode: str | None = None,
) -> Path:
    """Write a small H.264 test clip, optionally with creation_time and display rotation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        tool("ffmpeg"),
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={size}:rate={fps}",
    ]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    args += ["-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac"]
    if creation_time:
        args += ["-metadata", f"creation_time={creation_time}"]
    if timecode:
        args += ["-timecode", timecode]
    plain = path.with_name(path.stem + ".tmp.mp4")
    run([*args, "-f", "mp4", str(plain)])
    if rotation is None:
        plain.replace(path)
    else:
        keep = ["-metadata", f"creation_time={creation_time}"] if creation_time else []
        run(
            [
                tool("ffmpeg"),
                "-v",
                "error",
                "-y",
                "-display_rotation",
                str(rotation),
                "-i",
                str(plain),
                "-c",
                "copy",
                *keep,
                "-f",
                "mp4",
                str(path),
            ]
        )
        plain.unlink()
    return path


def write_footage(root: Path) -> Path:
    """Three DJI-named clips with .LRF proxies (one proxy wrong), and one rotated phone clip."""
    cam = root / "DJI_001"
    make_clip(cam / "DJI_20261011095133_0001_D.MP4", 2.0, creation_time="2026-10-11T07:51:34Z")
    make_clip(cam / "DJI_20261011095133_0001_D.LRF", 2.0, size="160x90")
    make_clip(cam / "DJI_20261011100000_0002_D.MP4", 3.0, creation_time="2026-10-11T08:00:01Z")
    make_clip(cam / "DJI_20261011100000_0002_D.LRF", 3.0, size="160x90")
    make_clip(cam / "DJI_20261011101500_0003_D.MP4", 2.0, creation_time="2026-10-11T08:15:00Z")
    make_clip(cam / "DJI_20261011101500_0003_D.LRF", 1.0, size="160x90")  # wrong length
    make_clip(
        root / "phone" / "PXL_20261011_081000.mp4",
        2.0,
        creation_time="2026-10-11T08:10:00Z",
        rotation=90,
    )
    return root


@pytest.fixture
def footage(tmp_path: Path) -> Path:
    return write_footage(tmp_path / "footage")


def make_photo(
    path: Path,
    taken: str | None = "2026:10:11 10:20:00",
    offset: str | None = "+02:00",
    orientation: int = 1,
    size: tuple[int, int] = (400, 300),
    color: tuple[int, int, int] = (40, 120, 200),
    make: str = "Google",
    model: str = "Pixel 10",
) -> Path:
    """A JPEG with EXIF time, zone offset, orientation and camera model."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    for x in range(0, size[0], 20):  # some detail, so sharpness isn't zero
        img.paste((255, 255, 255), (x, 0, x + 2, size[1]))
    exif = Image.Exif()
    exif[0x010F], exif[0x0110], exif[0x0112] = make, model, orientation
    sub = exif.get_ifd(0x8769)
    if taken:
        sub[0x9003] = taken
    if offset:
        sub[0x9011] = offset
    img.save(path, "JPEG", exif=exif, quality=90)
    return path


def make_music(path: Path, bpm: float = 120, seconds: float = 20.0, rate: int = 44100) -> Path:
    """A WAV click track: a short tone burst on every beat, starting at 0.25 s, over quiet noise."""
    import wave

    import numpy as np

    y = np.random.default_rng(0).normal(0, 0.01, int(rate * seconds))
    burst = np.sin(np.arange(1500) * 2 * np.pi * 880 / rate) * np.exp(-np.arange(1500) / 300)
    for t in np.arange(0.25, seconds - 0.1, 60 / bpm):
        i = int(t * rate)
        y[i : i + len(burst)] += 0.6 * burst[: len(y) - i]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return path
