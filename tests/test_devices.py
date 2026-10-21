from datetime import UTC, datetime
from pathlib import Path

from reelsift import devices
from reelsift.devices.dji import filename_time, zone_from
from reelsift.media import parse_probe
from reelsift.timebase import Provenance


def probe_with(tags: dict, path: str = "x.mp4"):
    return parse_probe(
        Path(path),
        {
            "format": {"duration": "2", "tags": tags},
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 320,
                    "height": 180,
                    "time_base": "1/12800",
                    "avg_frame_rate": "25/1",
                    "r_frame_rate": "25/1",
                }
            ],
        },
    )


def test_dji_file_name_gives_local_time():
    assert filename_time(Path("DJI_20261011095133_0001_D.MP4")) == datetime(2026, 10, 11, 9, 51, 33)
    assert filename_time(Path("GX010001.MP4")) is None


def test_zone_from_local_and_utc():
    utc = datetime(2026, 10, 11, 7, 51, 34, tzinfo=UTC)
    assert zone_from(datetime(2026, 10, 11, 9, 51, 33), utc) == 120
    assert zone_from(datetime(2026, 10, 11, 13, 36, 34), utc) == 345  # Nepal, +05:45
    assert zone_from(datetime(2026, 10, 11, 9, 41, 0), utc) is None  # not a 15-minute step


def test_dji_is_detected_by_name_or_encoder():
    name = Path("DJI_20261011095133_0001_D.MP4")
    assert devices.detect(name, probe_with({})).name == "dji"
    assert (
        devices.detect(Path("clip.mp4"), probe_with({"encoder": "DJI OsmoAction6"})).name == "dji"
    )
    assert devices.detect(Path("clip.mp4"), probe_with({})).name == "generic"


def test_dji_time_evidence_uses_file_name_for_zone():
    path = Path("DJI_20261011095133_0001_D.MP4")
    [ev] = devices.detect(path, probe_with({})).time_evidence(
        path, probe_with({"creation_time": "2026-10-11T07:51:34.000000Z"})
    )
    assert ev.provenance == Provenance.METADATA_ZONE_FROM_FILENAME
    assert ev.zone_offset_min == 120
    assert ev.utc_at_zero == datetime(2026, 10, 11, 7, 51, 34, tzinfo=UTC)


def test_generic_time_evidence_reads_zone_when_present():
    p = probe_with({"creation_time": "2026-10-11T09:51:33+0200"})
    [ev] = devices.GENERIC.time_evidence(Path("a.mov"), p)
    assert ev.provenance == Provenance.METADATA_WITH_ZONE
    assert ev.zone_offset_min == 120
    p = probe_with({"creation_time": "2026-10-11T07:51:33.000000Z"})
    [ev] = devices.GENERIC.time_evidence(Path("a.mov"), p)
    assert ev.provenance == Provenance.METADATA_UNKNOWN_ZONE
    assert devices.GENERIC.time_evidence(Path("a.mov"), probe_with({})) == []


def test_lrf_is_a_proxy_name():
    assert devices.is_proxy_name(Path("DJI_20261011095133_0001_D.LRF"))
    assert not devices.is_proxy_name(Path("DJI_20261011095133_0001_D.MP4"))
