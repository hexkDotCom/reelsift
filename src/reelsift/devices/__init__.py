"""Device profiles. The first profile that matches a file wins; generic is the fallback."""

from pathlib import Path

from ..media import Probe
from .base import ChapterKey, DeviceProfile, TimeEvidence
from .cycliq import Cycliq
from .dji import Dji
from .gopro import GoPro
from .model import Model

PROFILES: list[DeviceProfile] = [Dji(), GoPro(), Cycliq(), Model()]
GENERIC = DeviceProfile()

__all__ = [
    "GENERIC",
    "PROFILES",
    "ChapterKey",
    "DeviceProfile",
    "TimeEvidence",
    "by_name",
    "detect",
]


def detect(path: Path, probe: Probe) -> DeviceProfile:
    for profile in PROFILES:
        if profile.matches(path, probe):
            return profile
    return GENERIC


def by_name(name: str) -> DeviceProfile:
    for profile in [*PROFILES, GENERIC]:
        if profile.name == name:
            return profile
    return GENERIC


def is_proxy_name(path: Path) -> bool:
    """Whether any profile treats this file name as a camera-made proxy."""
    return any(p.is_proxy(path) for p in PROFILES)
