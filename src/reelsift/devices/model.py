"""Phones and cameras that name their make and model inside their videos (iPhone, Android).

Their videos and photos become one camera, so a time zone known from either applies to both.
iPhones also record the local time with its zone; Android phones record UTC only.
"""

from datetime import UTC, datetime
from pathlib import Path

from ..media import Probe
from ..timebase import Provenance
from .base import DeviceProfile, TimeEvidence

MAKE_TAGS = ("com.apple.quicktime.make", "com.android.manufacturer")
MODEL_TAGS = ("com.apple.quicktime.model", "com.android.model")


def make_model(make: str, model: str) -> str:
    """'Google' + 'Pixel 9 Pro' -> 'Google Pixel 9 Pro'; a model that already starts with its
    make keeps it once."""
    make, model = make.strip(), model.strip()
    if model.lower().startswith(make.lower()):
        make = ""
    return " ".join(x for x in (make, model) if x)


class Model(DeviceProfile):
    name = "model"
    label = "Camera"

    def _name(self, probe: Probe) -> str:
        tags = probe.tags
        make = next((tags[t] for t in MAKE_TAGS if tags.get(t)), "")
        model = next((tags[t] for t in MODEL_TAGS if tags.get(t)), "")
        return make_model(make, model) if model else ""

    def matches(self, path: Path, probe: Probe) -> bool:
        return bool(self._name(probe))

    def camera_identity(self, path: Path, probe: Probe, folder: str | None = None) -> str:
        return f"{self._name(probe)}|folder:{folder or path.parent.name}"

    def camera_label(self, path: Path, probe: Probe) -> str:
        return self._name(probe) or self.label

    def time_evidence(self, path: Path, probe: Probe) -> list[TimeEvidence]:
        local = probe.tags.get("com.apple.quicktime.creationdate")
        if local:
            try:
                when = datetime.fromisoformat(local.strip())
            except ValueError:
                when = None
            if when is not None and when.tzinfo is not None:
                minutes = int(when.utcoffset().total_seconds() // 60)
                return [
                    TimeEvidence(
                        Provenance.METADATA_WITH_ZONE,
                        when.astimezone(UTC),
                        minutes,
                        0.7,
                        f"creationdate {local} carries a time zone",
                    )
                ]
        return super().time_evidence(path, probe)
