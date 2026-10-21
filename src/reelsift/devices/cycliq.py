"""Cycliq bike cameras (Fly12, Fly6), which record continuously in segments like a dashcam.

Checked with a Fly12 (firmware 2.0.6):
- Recording is split into segments of up to 5 minutes, numbered CYQ_0001.MP4, CYQ_0002.MP4...
  Segments the camera locked after an incident are renamed LCK_nnnn.MP4 and keep their number.
- creation_time is when a segment *ended*, on the camera's own clock (local time, set from the
  phone app), written as if it were UTC. The same clock is burned into the picture.
- Consecutive segments whose times meet form one recording; a gap starts a new one.
"""

import re
from datetime import UTC, timedelta
from pathlib import Path

from ..media import Probe
from ..timebase import Provenance
from .base import ChapterKey, DeviceProfile, TimeEvidence

NAME = re.compile(r"^(CYQ|LCK)_(\d{4})$", re.IGNORECASE)
JOIN_GAP_S = 5.0
"""Segments meet when one starts within this long of the previous one's end."""


def _start_and_end(probe: Probe):
    end = probe.creation_time
    if end is None:
        return None, None
    end = end.replace(tzinfo=None)
    seconds = float(probe.main_video.duration or probe.duration) if probe.main_video else 0.0
    return end - timedelta(seconds=seconds), end


class Cycliq(DeviceProfile):
    name = "cycliq"
    label = "Cycliq"
    supported = True
    continuous = True

    def matches(self, path: Path, probe: Probe) -> bool:
        if probe.main_video is None or not NAME.match(path.stem):
            return False
        tags = probe.tags
        return str(tags.get("comment", "")).upper().startswith("FLY") or "FW=" in str(
            tags.get("original_format", "")
        )

    def chapter_key(self, path: Path, probe: Probe) -> ChapterKey | None:
        m = NAME.match(path.stem)
        return ChapterKey("segments", int(m.group(2))) if m else None

    def split_runs(self, members: list[tuple[int, Path, Probe]]) -> list[list[int]]:
        """Consecutive segment numbers whose times meet are one recording."""
        runs: list[list[int]] = []
        previous = None
        for k, (index, _path, probe) in enumerate(sorted(members, key=lambda m: m[0])):
            start, end = _start_and_end(probe)
            joins = (
                previous is not None
                and index == previous[0] + 1
                and start is not None
                and previous[1] is not None
                and abs((start - previous[1]).total_seconds()) <= JOIN_GAP_S
            )
            if joins:
                runs[-1].append(k)
            else:
                runs.append([k])
            previous = (index, end)
        return runs

    def time_evidence(self, path: Path, probe: Probe) -> list[TimeEvidence]:
        start, end = _start_and_end(probe)
        if start is None:
            return []
        return [TimeEvidence(
            Provenance.METADATA_LOCAL_UNKNOWN_ZONE, start.replace(tzinfo=UTC), None, 0.3,
            f"creation_time {end:%Y-%m-%d %H:%M:%S} is when the segment ended, on the camera's "
            "local clock",
        )]  # fmt: skip
