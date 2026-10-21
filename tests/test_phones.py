"""Phones: videos and photos from one phone are one camera, and a zone known anywhere applies."""

from datetime import UTC, datetime
from fractions import Fraction

from reelsift.project import Trip
from reelsift.tools import run, tool

from .conftest import make_photo, needs_ffmpeg

pytestmark = needs_ffmpeg


def phone_clip(path, tags: dict[str, str], seconds: float = 2.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = [arg for k, v in tags.items() for arg in ("-metadata", f"{k}={v}")]
    run([tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
         f"testsrc2=size=320x240:rate=30:duration={seconds}", "-c:v", "libx264", *meta,
         "-movflags", "use_metadata_tags", str(path)])  # fmt: skip
    return path


def cameras_and_clocks(trip):
    from reelsift.sync import clock, sync

    sync(trip)
    cams = [r["label"] for r in trip.db.execute("SELECT label FROM cameras ORDER BY id")]
    clocks = {}
    for r in trip.db.execute("SELECT r.id, a.path FROM recordings r JOIN chapters c ON"
                             " c.recording_id = r.id JOIN assets a ON a.id = c.asset_id"):  # fmt: skip
        clocks[r["path"].rsplit("\\", 1)[-1].rsplit("/", 1)[-1]] = clock(trip, r["id"])
    return cams, clocks


def test_an_iphones_videos_and_photos_are_one_camera_with_its_zone(tmp_path):
    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    phone_clip(footage / "IMG_0001.MOV", {
        "creation_time": "2026-09-19T16:03:47Z",
        "com.apple.quicktime.make": "Apple",
        "com.apple.quicktime.model": "iPhone 16 Pro Max",
        "com.apple.quicktime.creationdate": "2026-09-19T18:03:47+0200",
    })  # fmt: skip
    make_photo(footage / "IMG_0002.JPG", taken="2026:09:19 18:10:00", offset="+02:00",
               make="Apple", model="iPhone 16 Pro Max")  # fmt: skip
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        cams, clocks = cameras_and_clocks(trip)
    assert len(cams) == 1 and cams[0].startswith("Apple iPhone 16 Pro Max")
    video = clocks["IMG_0001.MOV"]
    assert video.zone_offset_min == 120
    assert video.to_utc(Fraction(0)) == datetime(2026, 9, 19, 16, 3, 47, tzinfo=UTC)


def test_an_android_video_takes_its_zone_from_the_phones_photos(tmp_path):
    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    phone_clip(footage / "PXL_1.mp4", {
        "creation_time": "2026-10-17T16:52:57Z",
        "com.android.manufacturer": "Google",
        "com.android.model": "Pixel 9 Pro",
    })  # fmt: skip
    make_photo(footage / "PXL_2.jpg", taken="2026:10:17 18:45:00", offset="+02:00",
               make="Google", model="Pixel 9 Pro")  # fmt: skip
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        cams, clocks = cameras_and_clocks(trip)
    assert len(cams) == 1 and cams[0].startswith("Google Pixel 9 Pro")
    video = clocks["PXL_1.mp4"]
    assert video.zone_offset_min == 120
    assert video.to_utc(Fraction(0)) == datetime(2026, 10, 17, 16, 52, 57, tzinfo=UTC), (
        "the UTC time stays; only the zone is added"
    )
    assert video.local(Fraction(0)).hour == 18


def test_a_local_time_borrows_the_zone_nearest_in_real_time():
    """Around a DST change the camera has two zones; a local time is compared with each after
    converting it with that zone, not as if it were UTC."""
    from reelsift.sync import _zones_from_camera
    from reelsift.timebase import ClockTransform, Provenance

    zoned = Provenance.METADATA_WITH_ZONE
    transforms = {
        1: ClockTransform(datetime(2026, 10, 24, 22, 30, tzinfo=UTC), zoned, 0.8, 120),
        2: ClockTransform(datetime(2026, 10, 25, 1, 30, tzinfo=UTC), zoned, 0.8, 60),
        3: ClockTransform(
            datetime(2026, 10, 25, 0, 40, tzinfo=UTC), Provenance.METADATA_LOCAL_UNKNOWN_ZONE, 0.3
        ),
    }
    out = _zones_from_camera(transforms, {1: 1, 2: 1, 3: 1})[3]
    assert out.zone_offset_min == 120
    assert out.utc_at_zero == datetime(2026, 10, 24, 22, 40, tzinfo=UTC)


def test_a_borrowed_zone_keeps_a_modified_time_guess_marked_as_one():
    from reelsift.sync import _zones_from_camera
    from reelsift.timebase import ClockTransform, Provenance

    when = datetime(2026, 7, 4, 10, 0, tzinfo=UTC)
    transforms = {
        1: ClockTransform(when, Provenance.METADATA_WITH_ZONE, 0.8, 120),
        2: ClockTransform(when, Provenance.FILE_MODIFIED, 0.1),
    }
    out = _zones_from_camera(transforms, {1: 1, 2: 1})[2]
    assert (out.provenance, out.confidence, out.zone_offset_min) == (
        Provenance.FILE_MODIFIED, 0.1, 120)  # fmt: skip


def test_a_zone_is_not_borrowed_across_days(tmp_path):
    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    phone_clip(footage / "PXL_1.mp4", {
        "creation_time": "2026-10-17T16:52:57Z",
        "com.android.manufacturer": "Google",
        "com.android.model": "Pixel 9 Pro",
    })  # fmt: skip
    make_photo(footage / "PXL_2.jpg", taken="2026:10:25 18:45:00", offset="+02:00",
               make="Google", model="Pixel 9 Pro")  # fmt: skip
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        _, clocks = cameras_and_clocks(trip)
    assert clocks["PXL_1.mp4"].zone_offset_min is None
