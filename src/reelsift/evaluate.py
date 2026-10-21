"""Measuring selection against moments you marked by hand.

Labels live in the trip folder as labels.toml:

    [[moment]]
    file = "DJI_20261011110800_0002_D.MP4"   # any file of the recording
    from = "1:05"                            # core seconds of the moment
    to = "1:12"
    want = true                              # false marks something that should not appear

A selected clip counts for a moment only if it covers at least half of the moment's core.
"""

import tomllib
from dataclasses import dataclass, field

from .project import Trip
from .select import Selection
from .sync import find_recording, parse_offset
from .tripdata import TripData

COVER = 0.5


@dataclass
class Moment:
    recording_id: int
    start: float
    end: float
    want: bool


@dataclass
class Report:
    length: int
    wanted: int
    found: int
    unwanted_included: int
    candidate_recall: float
    missing: list[Moment] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.found / self.wanted if self.wanted else 0.0


def load_labels(trip: Trip) -> list[Moment]:
    path = trip.root / "labels.toml"
    if not path.is_file():
        return []
    with path.open("rb") as f:
        data = tomllib.load(f)
    out = []
    for m in data.get("moment", []):
        out.append(
            Moment(
                find_recording(trip, str(m["file"])),
                float(parse_offset(str(m["from"]))),
                float(parse_offset(str(m["to"]))),
                bool(m.get("want", True)),
            )
        )
    return out


def covered(m: Moment, rec: int, start: float, end: float) -> bool:
    if rec != m.recording_id:
        return False
    overlap = min(m.end, end) - max(m.start, start)
    return overlap >= COVER * max(m.end - m.start, 1e-6)


def evaluate(data: TripData, selection: Selection, moments: list[Moment]) -> Report:
    wanted = [m for m in moments if m.want]
    unwanted = [m for m in moments if not m.want]
    c = data.candidates

    def in_edit(m: Moment) -> bool:
        return any(covered(m, i.recording_id, i.start, i.end) for i in selection.items)

    def in_candidates(m: Moment) -> bool:
        return any(
            covered(m, int(c.recording[i]), float(c.min_start[i]), float(c.max_end[i]))
            for i in range(len(c))
        )

    found = [m for m in wanted if in_edit(m)]
    return Report(
        length=selection.length,
        wanted=len(wanted),
        found=len(found),
        unwanted_included=sum(in_edit(m) for m in unwanted),
        candidate_recall=(sum(in_candidates(m) for m in wanted) / len(wanted)) if wanted else 0.0,
        missing=[m for m in wanted if m not in found],
    )
