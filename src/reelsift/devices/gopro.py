"""GoPro HERO, MAX and Fusion cameras.

Checked against files from the HERO4, HERO5, HERO6, HERO7, HERO8, HERO13, MAX and Fusion:
- Long recordings are split into chapters. Older cameras name them GOPRnnnn.MP4, then
  GP01nnnn.MP4, GP02nnnn.MP4...; newer ones GX01nnnn.MP4, GX02nnnn.MP4... (GH for H.264),
  where nnnn is the recording's number.
- creation_time holds the camera's local clock written as if it were UTC, at the start of each
  chapter, to the second.
- Cameras with GPS write the UTC time into their telemetry (GPMF, the "gpmd" stream) once they
  have a fix; comparing it with creation_time gives the time zone.
- Low-resolution proxies (.LRV) sit next to the files: same name for GOPR/GPnn, GL instead of
  GX/GH for newer cameras. Thumbnails (.THM) are ignored.
"""

import functools
import re
from datetime import UTC, timedelta
from pathlib import Path

from .. import gpmf
from ..media import Probe
from ..timebase import Provenance
from .base import ChapterKey, DeviceProfile, TimeEvidence, zone_from

OLD_FIRST = re.compile(r"^GOPR(\d{4})$", re.IGNORECASE)
OLD_NEXT = re.compile(r"^GP(\d{2})(\d{4})$", re.IGNORECASE)
NEW = re.compile(r"^G([XHL])(\d{2})(\d{4})$", re.IGNORECASE)
JOIN_GAP_S = 5.0
"""A chapter starts within this long of the previous one's end (creation_time is to the
second)."""
FIRMWARE_MODELS = {
    "HD3": "HERO3", "HD4": "HERO4", "HD5": "HERO5", "HD6": "HERO6", "HD7": "HERO7",
    "HD8": "HERO8", "HD9": "HERO9", "H21": "HERO10", "H22": "HERO11", "H23": "HERO12",
    "H24": "HERO13", "H19": "MAX", "FS1": "Fusion",
}  # fmt: skip


@functools.lru_cache(maxsize=256)
def _telemetry(path: str, size: int, raw_key: str) -> gpmf.Telemetry:
    import json

    from ..media import parse_probe

    return gpmf.read(Path(path), parse_probe(Path(path), json.loads(raw_key)))


class GoPro(DeviceProfile):
    name = "gopro"
    label = "GoPro"
    supported = True

    def matches(self, path: Path, probe: Probe) -> bool:
        if probe.main_video is None:
            return False
        # GoPro names its tracks ("GoPro AVC", "GoPro MET"); older cameras only write a
        # firmware tag, so for them the file name has to look like a GoPro's too.
        tracks = any(
            "gopro" in str(s.get("tags", {}).get("handler_name", "")).lower()
            for s in probe.raw.get("streams", [])
        )  # fmt: skip
        named = any(p.match(path.stem) for p in (OLD_FIRST, OLD_NEXT, NEW))
        return tracks or (named and "firmware" in probe.tags)

    def is_proxy(self, path: Path) -> bool:
        return path.suffix.lower() == ".lrv"

    def proxy_for(self, path: Path) -> Path | None:
        stems = [path.stem]
        new = NEW.match(path.stem)
        if new:
            stems.append(f"GL{new.group(2)}{new.group(3)}")
        for stem in stems:
            for suffix in (".LRV", ".lrv"):
                candidate = path.with_name(stem + suffix)
                if candidate.is_file():
                    return candidate
        return None

    def chapter_key(self, path: Path, probe: Probe) -> ChapterKey | None:
        if m := OLD_FIRST.match(path.stem):
            return ChapterKey(f"old:{m.group(1)}", 0)
        if m := OLD_NEXT.match(path.stem):
            return ChapterKey(f"old:{m.group(2)}", int(m.group(1)))
        if m := NEW.match(path.stem):
            return ChapterKey(f"{m.group(1).upper()}:{m.group(3)}", int(m.group(2)))
        return None

    def split_runs(self, members: list[tuple[int, Path, Probe]]) -> list[list[int]]:
        """Consecutive chapters, unless one doesn't start where the previous one ended (the
        files were mixed up, or the camera's number was reused)."""
        runs = []
        for run in super().split_runs(members):
            runs.append([run[0]])
            for k in run[1:]:
                before, now = members[runs[-1][-1]][2], members[k][2]
                a, b = before.creation_time, now.creation_time
                length = float(before.main_video.duration or before.duration)
                if a is None or b is None or abs((b - a).total_seconds() - length) <= JOIN_GAP_S:
                    runs[-1].append(k)
                else:
                    runs.append([k])
        return runs

    def telemetry(self, path: Path, probe: Probe) -> gpmf.Telemetry:
        import json

        size = path.stat().st_size if path.exists() else 0
        return _telemetry(str(path), size, json.dumps(probe.raw, sort_keys=True))

    def model(self, path: Path, probe: Probe) -> str:
        found = self.telemetry(path, probe).model
        if found and found.lower() not in ("camera", "gopro"):
            return found
        firmware = probe.tags.get("firmware", "")
        return FIRMWARE_MODELS.get(firmware[:3].upper(), "")

    def camera_identity(self, path: Path, probe: Probe, folder: str | None = None) -> str:
        return f"{self.model(path, probe)}|folder:{folder or path.parent.name}"

    def camera_label(self, path: Path, probe: Probe) -> str:
        model = self.model(path, probe)
        if model.lower().startswith("gopro"):
            return model  # "GoPro Max", not "GoPro GoPro Max"
        return f"GoPro {model}" if model else "GoPro"

    def time_evidence(self, path: Path, probe: Probe) -> list[TimeEvidence]:
        created = probe.creation_time
        local = created.replace(tzinfo=None) if created is not None else None
        telemetry = self.telemetry(path, probe)
        evidence = []
        if telemetry.gps_utc is not None:
            utc = telemetry.gps_utc - timedelta(seconds=telemetry.gps_offset)
            zone = zone_from(local, utc) if local is not None else None
            note = "" if zone is not None else "; the camera's clock doesn't match it"
            evidence.append(TimeEvidence(Provenance.GPS, utc, zone, 0.9,
                                         f"GPS time {telemetry.gps_utc:%Y-%m-%d %H:%M:%S} UTC "
                                         f"in the telemetry{note}"))  # fmt: skip
        if local is not None:
            evidence.append(TimeEvidence(
                Provenance.METADATA_LOCAL_UNKNOWN_ZONE, local.replace(tzinfo=UTC), None, 0.3,
                f"creation_time {local:%Y-%m-%d %H:%M:%S}, the camera's local clock",
            ))  # fmt: skip
        return evidence
