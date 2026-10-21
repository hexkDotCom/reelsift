"""DJI cameras that use DJI's file naming.

Recognized by the encoder tag ("DJI ...") or the file name pattern, which DJI action cameras and
drones share. Verified only with the Osmo Action 6 (firmware 10.00.36.61):
- Files are named DJI_<local start time YYYYMMDDhhmmss>_<counter>_D.MP4.
- A low-resolution proxy with the same name and the .LRF extension sits next to each file, with
  the same duration and frame count.
- creation_time is the start of the recording in true UTC, so comparing it with the file name
  gives the local time zone.
- The sensor and GPS track ("djmd" data stream) can't be decoded yet.
"""

import re
from datetime import UTC, datetime
from pathlib import Path

from ..media import Probe
from ..timebase import Provenance
from .base import DeviceProfile, TimeEvidence, zone_from

NAME = re.compile(r"^DJI_(\d{14})_(\d{4})_[A-Z]$", re.IGNORECASE)
ENCODER = re.compile(r"^DJI\s", re.IGNORECASE)


def filename_time(path: Path) -> datetime | None:
    match = NAME.match(path.stem)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None


class Dji(DeviceProfile):
    name = "dji"
    label = "DJI"
    supported = True
    verified_models = ("Osmo Action 6",)

    def matches(self, path: Path, probe: Probe) -> bool:
        if probe.main_video is None:
            return False
        if ENCODER.search(probe.encoder):
            return True
        return NAME.match(path.stem) is not None

    def is_proxy(self, path: Path) -> bool:
        return path.suffix.lower() == ".lrf"

    def proxy_for(self, path: Path) -> Path | None:
        for suffix in (".LRF", ".lrf"):
            candidate = path.with_suffix(suffix)
            if candidate.is_file():
                return candidate
        return None

    def time_evidence(self, path: Path, probe: Probe) -> list[TimeEvidence]:
        created = probe.creation_time
        local = filename_time(path)
        evidence = []
        if created is not None:
            created_utc = created.replace(tzinfo=UTC) if created.tzinfo is None else created
            zone = zone_from(local, created_utc) if local else None
            if zone is not None:
                evidence.append(
                    TimeEvidence(
                        Provenance.METADATA_ZONE_FROM_FILENAME,
                        created_utc,
                        zone,
                        0.8,
                        f"creation_time {probe.tags.get('creation_time')} (UTC); file name gives "
                        f"local time {local:%Y-%m-%d %H:%M:%S}",
                    )
                )
            else:
                evidence.append(
                    TimeEvidence(
                        Provenance.METADATA_UNKNOWN_ZONE,
                        created_utc,
                        None,
                        0.5,
                        f"creation_time {probe.tags.get('creation_time')} (UTC); file name gave no "
                        "usable local time",
                    )
                )
        elif local is not None:
            evidence.append(
                TimeEvidence(
                    Provenance.METADATA_LOCAL_UNKNOWN_ZONE,
                    local.replace(tzinfo=UTC),
                    None,
                    0.2,
                    f"local time {local:%Y-%m-%d %H:%M:%S} from the file name, read as UTC",
                )
            )
        return evidence
