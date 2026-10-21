"""What a device profile knows about one kind of camera or phone."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..media import Probe
from ..timebase import Provenance


@dataclass(frozen=True)
class TimeEvidence:
    """One opinion about when source time 0 of a file happened, in UTC."""

    provenance: Provenance
    utc_at_zero: datetime
    zone_offset_min: int | None
    confidence: float
    detail: str


ZONE_STEP_MIN = 15
ZONE_TOLERANCE_S = 90
LOCAL_NAME = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})(?!\d)")
"""A local date and time in a file name, as many phones and dashcams write it:
20200705_150405.mp4, VID_20200705_150405.mp4. (Pixel's PXL_20261017_165243546 has milliseconds
after the seconds, and is UTC, so it doesn't match.)"""


def local_time_in_name(path: Path) -> datetime | None:
    m = LOCAL_NAME.search(path.stem)
    if not m:
        return None
    try:
        return datetime(*(int(g) for g in m.groups()))
    except ValueError:
        return None


def zone_from(local: datetime, utc: datetime) -> int | None:
    """Time zone in minutes, if local and UTC differ by a whole number of 15-minute steps."""
    diff = (local - utc.replace(tzinfo=None)).total_seconds()
    steps = round(diff / (ZONE_STEP_MIN * 60))
    if abs(diff - steps * ZONE_STEP_MIN * 60) > ZONE_TOLERANCE_S or abs(steps) > 14 * 4:
        return None
    return steps * ZONE_STEP_MIN


@dataclass(frozen=True)
class ChapterKey:
    """Files with the same group belong to one recording, ordered by index."""

    group: str
    index: int


class DeviceProfile:
    """Base profile. Subclasses override what their device does differently.

    A device counts as supported only when sample files from it are covered by tests.
    """

    name = "generic"
    label = "Unknown camera"
    supported = False
    verified_models: tuple[str, ...] = ()
    """Models whose sample files are covered by tests."""

    def matches(self, path: Path, probe: Probe) -> bool:
        return probe.main_video is not None

    def is_proxy(self, path: Path) -> bool:
        return False

    def proxy_for(self, path: Path) -> Path | None:
        """Path of the camera-made proxy for a full-resolution file, if the device writes one."""
        return None

    def camera_identity(self, path: Path, probe: Probe, folder: str | None = None) -> str:
        """Tells apart two cameras of the same model in one trip.

        Without a serial number, the folder the files were copied into is the best guess,
        because people usually copy each camera into its own folder.
        """
        return f"folder:{folder or path.parent.name}"

    continuous = False
    """Records in numbered segments that may be deleted at will (a dashcam), so a missing
    number isn't worth a warning."""

    def split_runs(self, members: list[tuple[int, Path, Probe]]) -> list[list[int]]:
        """Split one chapter group (members as (index, path, probe), sorted by index) into
        recordings, as lists of positions. By default chapters with consecutive numbers are one
        recording: time after a missing chapter can't be known from the chapters before it."""
        runs: list[list[int]] = []
        for k, (index, _path, _probe) in enumerate(members):
            if runs and index == members[runs[-1][-1]][0] + 1:
                runs[-1].append(k)
            else:
                runs.append([k])
        return runs

    def camera_label(self, path: Path, probe: Probe) -> str:
        """The camera's name as shown to people (a folder is added to tell cameras apart)."""
        return self.label

    def chapter_key(self, path: Path, probe: Probe) -> ChapterKey | None:
        """For devices that split long recordings into several files; None means one file."""
        return None

    def time_evidence(self, path: Path, probe: Probe) -> list[TimeEvidence]:
        created = probe.creation_time
        if created is None:
            return []
        if created.tzinfo is not None and created.utcoffset().total_seconds() != 0:
            # A real zone offset in the tag (some phones): both UTC and local time are known.
            return [
                TimeEvidence(
                    Provenance.METADATA_WITH_ZONE,
                    created,
                    int(created.utcoffset().total_seconds() // 60),
                    0.7,
                    f"creation_time {probe.tags.get('creation_time')} carries a time zone",
                )
            ]
        evidence = [
            TimeEvidence(
                Provenance.METADATA_UNKNOWN_ZONE,
                created.replace(tzinfo=UTC),
                None,
                0.4,
                f"creation_time {probe.tags.get('creation_time')}, read as UTC; local zone unknown",
            )
        ]
        # A local time in the file name next to a UTC creation_time gives the zone. A zero
        # difference says nothing: the name may simply be in UTC.
        local = local_time_in_name(path)
        zone = zone_from(local, created) if local is not None else None
        if zone:
            evidence.insert(0, TimeEvidence(
                Provenance.METADATA_ZONE_FROM_FILENAME, created.replace(tzinfo=UTC), zone, 0.6,
                f"creation_time {probe.tags.get('creation_time')} (UTC); the file name gives "
                f"local time {local:%Y-%m-%d %H:%M:%S}",
            ))  # fmt: skip
        return evidence
