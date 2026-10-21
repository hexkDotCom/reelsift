"""Photos: reading them, their time, upright proxies, and stills for rendering.

A photo becomes a recording with no duration. Its time comes from EXIF DateTimeOriginal, with the
zone from OffsetTimeOriginal when the camera wrote one (Pixel phones do). Orientation is applied
once, when proxies and render stills are made, so everything downstream sees upright pixels.
HEIC is read through pillow-heif; RAW files are not supported.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from PIL import Image, ImageOps

from .devices.base import TimeEvidence
from .timebase import Provenance, parse_zone

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".avif"}
EXIF_IFD = 0x8769
DATETIME_ORIGINAL = 0x9003
OFFSET_TIME_ORIGINAL = 0x9011
SUBSEC_ORIGINAL = 0x9291
MAKE, MODEL, ORIENTATION = 0x010F, 0x0110, 0x0112
PROXY_LONG_SIDE = 1280

_heif_registered = False


def _open(path: Path) -> Image.Image:
    global _heif_registered
    if path.suffix.lower() in (".heic", ".heif") and not _heif_registered:
        try:
            from pillow_heif import register_heif_opener
        except ImportError as e:
            raise ValueError("HEIC photos need the pillow-heif package") from e
        register_heif_opener()
        _heif_registered = True
    return Image.open(path)


def probe(path: Path) -> dict:
    """Size (upright), EXIF time fields and camera make and model."""
    with _open(path) as img:
        exif = img.getexif()
        sub = exif.get_ifd(EXIF_IFD)
        orientation = int(exif.get(ORIENTATION, 1) or 1)
        width, height = img.size
        if orientation in (5, 6, 7, 8):
            width, height = height, width
        return {
            "photo": {
                "width": width,
                "height": height,
                "orientation": orientation,
                "datetime_original": _text(sub.get(DATETIME_ORIGINAL)),
                "offset_time_original": _text(sub.get(OFFSET_TIME_ORIGINAL)),
                "subsec": _text(sub.get(SUBSEC_ORIGINAL)),
                "make": (_text(exif.get(MAKE)) or "").strip(),
                "model": (_text(exif.get(MODEL)) or "").strip(),
                "format": img.format or "",
            }
        }


def _text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return str(value).strip("\x00 ").strip() or None


def taken_at(info: dict) -> datetime | None:
    """Local wall-clock time the photo was taken, without zone."""
    raw = info.get("datetime_original")
    if not raw:
        return None
    try:
        when = datetime.strptime(raw[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    sub = info.get("subsec")
    if sub and sub.isdigit():
        when += timedelta(seconds=float(f"0.{sub}"))
    return when


def time_evidence(info: dict) -> list[TimeEvidence]:
    local = taken_at(info)
    if local is None:
        return []
    offset = info.get("offset_time_original")
    if offset:
        try:
            minutes = parse_zone(offset)
        except ValueError:
            minutes = None
        if minutes is not None:
            utc = (local - timedelta(minutes=minutes)).replace(tzinfo=UTC)
            return [TimeEvidence(Provenance.METADATA_WITH_ZONE, utc, minutes, 0.7,
                                 f"EXIF time {local:%Y-%m-%d %H:%M:%S} with offset {offset}")]  # fmt: skip
    return [
        TimeEvidence(
            Provenance.METADATA_LOCAL_UNKNOWN_ZONE, local.replace(tzinfo=UTC), None, 0.3,
            f"EXIF time {local:%Y-%m-%d %H:%M:%S} without a zone, read as UTC",
        )
    ]  # fmt: skip


def upright(path: Path) -> Image.Image:
    with _open(path) as img:
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")


def make_proxy(source: Path, out: Path) -> tuple[int, int]:
    img = upright(source)
    img.thumbnail((PROXY_LONG_SIDE, PROXY_LONG_SIDE), Image.Resampling.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.stem + ".partial.jpg")
    img.save(partial, "JPEG", quality=88)
    partial.replace(out)
    return img.size


def fitted_still(
    source: Path, out: Path, width: int, height: int, margin: float, cover_x: float | None = None
) -> Path:
    """The photo framed for a (width, height) frame enlarged by `margin`, ready for a slow zoom
    that crops back to (width, height). Fitted inside on black, or, when cover_x is given,
    filling the frame and cropped at that horizontal position (0 left, 1 right)."""
    if out.is_file():
        return out
    W, H = round(width * (1 + margin)), round(height * (1 + margin))
    img = upright(source)
    if cover_x is None:
        img = ImageOps.contain(img, (W, H), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (W, H))
        canvas.paste(img, ((W - img.width) // 2, (H - img.height) // 2))
    else:
        scale = max(W / img.width, H / img.height)
        img = img.resize(
            (round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS
        )
        x = round((img.width - W) * min(1.0, max(0.0, cover_x)))
        y = (img.height - H) // 2
        canvas = img.crop((x, y, x + W, y + H))
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.stem + ".partial.png")
    canvas.save(partial, "PNG")
    partial.replace(out)
    return out


def camera_name(info: dict) -> str:
    make, model = info.get("make") or "", info.get("model") or ""
    if model.lower().startswith(make.lower()):
        make = ""
    return " ".join(x for x in (make, model) if x) or "Unknown camera"
