"""GoPro and Cycliq profiles, zones from file names, and HEIC photos."""

import struct
from datetime import UTC, datetime
from fractions import Fraction

import pytest

from reelsift import gpmf
from reelsift.project import Trip
from reelsift.tools import run, tool

from .conftest import needs_ffmpeg


def clip(path, seconds: float = 2.0, **tags: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = [arg for k, v in tags.items() for arg in ("-metadata", f"{k}={v}")]
    run([tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
         f"testsrc2=size=320x240:rate=25:duration={seconds}", "-c:v", "libx264", *meta,
         "-movflags", "use_metadata_tags", str(path)])  # fmt: skip
    return path


def recordings(trip):
    rows = trip.db.execute(
        "SELECT r.id, a.path, c.idx, r.camera_id FROM recordings r JOIN chapters c ON"
        " c.recording_id = r.id JOIN assets a ON a.id = c.asset_id ORDER BY r.id, c.idx"
    ).fetchall()
    out: dict[int, list[str]] = {}
    for r in rows:
        out.setdefault(r["id"], []).append(r["path"].replace("\\", "/").rsplit("/", 1)[-1])
    return list(out.values())


# --- GPMF ----------------------------------------------------------------------------------------


def klv(key: bytes, kind: str, data: bytes, size: int | None = None) -> bytes:
    size = size or (len(data) if kind != "\0" else 1)
    repeat = len(data) // size if kind != "\0" else len(data)
    head = key + bytes([ord(kind) if kind != "\0" else 0, size if kind != "\0" else 1])
    head += struct.pack(">H", repeat)
    return head + data + b"\0" * ((4 - len(data) % 4) % 4)


def payload(model: str, gps: str | None, fix: int = 3) -> bytes:
    inner = klv(b"DVNM", "c", model.encode())
    strm = b""
    if gps:
        strm += klv(b"GPSF", "L", struct.pack(">I", fix), 4)
        strm += klv(b"GPSU", "U", gps.encode(), 16)
    inner += klv(b"STRM", "\0", strm) if strm else b""
    return klv(b"DEVC", "\0", inner)


def test_gpmf_gives_the_model_and_gps_time():
    t = gpmf.parse([(0.0, payload("HERO8 Black", None)),
                    (1.001, payload("HERO8 Black", "191118234514.250"))])  # fmt: skip
    assert t.model == "HERO8 Black"
    assert t.gps_utc == datetime(2019, 11, 18, 23, 45, 14, 250000, tzinfo=UTC)
    assert t.gps_offset == 1.001, "the first packet without GPS is skipped"


def test_gpmf_reads_packets_with_several_devices_as_one_moment():
    """A camera paired with a sensor writes two device containers per packet (seen on a
    HERO6 with a Bluetooth sensor): they are one moment, not two."""
    sensor = payload("SENSORB6", None)
    packets = [(0.0, payload("Hero6 Black", None) + sensor),
               (1.001, payload("Hero6 Black", "180323225140.819") + sensor)]  # fmt: skip
    t = gpmf.parse(packets)
    assert t.model == "Hero6 Black" and t.gps_offset == 1.001


@needs_ffmpeg
def test_gpmf_packets_keep_their_boundaries_in_a_real_file():
    import os
    from pathlib import Path

    from reelsift.media import probe

    # Real camera files aren't in the repository; REELSIFT_TESTDATA points to a local copy.
    sample = Path(os.environ.get("REELSIFT_TESTDATA", "testdata")) / "gopro-samples/hero6+ble.mp4"
    if not sample.exists():
        pytest.skip("GoPro sample not available (set REELSIFT_TESTDATA)")
    found = gpmf.packets(sample, probe(sample))
    assert len(found) == 5
    assert all(data.count(b"DEVC") == 2 for _, data in found), "camera and sensor per packet"
    assert [round(t) for t, _ in found] == [0, 1, 2, 3, 4]


def test_gpmf_reads_on_until_gps_has_a_fix(monkeypatch):
    """GPS can take minutes to find satellites: the first seconds alone aren't enough."""
    from pathlib import Path

    calls = []

    def fake(path, probe, count=5):
        calls.append(count)
        found = [(n * 1.001, payload("HERO5 Black", "170417173108.000", fix=0))
                 for n in range(min(count, 8))]  # fmt: skip
        if count > 5:
            found[6] = (6.006, payload("HERO5 Black", "170417173114.000"))
        return found

    monkeypatch.setattr(gpmf, "packets", fake)
    t = gpmf.read(Path("GOPR0001.MP4"), None)
    assert calls == [5, gpmf.SCAN_PACKETS]
    assert t.gps_utc == datetime(2017, 4, 17, 17, 31, 14, tzinfo=UTC) and t.gps_offset == 6.006


def test_gpmf_doesnt_read_on_without_gps(monkeypatch):
    from pathlib import Path

    calls = []
    monkeypatch.setattr(gpmf, "packets", lambda p, i, count=5: calls.append(count) or [
        (0.0, payload("HERO4 Silver", None))])  # fmt: skip
    gpmf.read(Path("GOPR0001.MP4"), None)
    assert calls == [5]


def gps9_payload(model: str, when: datetime, fix: int = 3) -> bytes:
    """HERO11 and later: GPS9 samples with their scales and layout."""
    scale = [10000000, 10000000, 1000, 1000, 100, 1, 1000, 100, 1]
    days = (when.date() - datetime(2000, 1, 1).date()).days
    seconds = when.hour * 3600 + when.minute * 60 + when.second
    sample = struct.pack(">7i2H", 377749000, -1224194000, 10000, 0, 0, days, seconds * 1000,
                         150, fix)  # fmt: skip
    strm = (klv(b"SCAL", "l", struct.pack(">9i", *scale), 4) + klv(b"TYPE", "c", b"lllllllSS")
            + klv(b"GPS9", "?", sample, 32))  # fmt: skip
    return klv(b"DEVC", "\0", klv(b"DVNM", "c", model.encode()) + klv(b"STRM", "\0", strm))


def test_gpmf_reads_gps9_time():
    when = datetime(2026, 7, 19, 10, 0, 0, tzinfo=UTC)
    t = gpmf.parse([(0.0, gps9_payload("HERO13 Black", when, fix=0)),
                    (1.001, gps9_payload("HERO13 Black", when.replace(second=1)))])  # fmt: skip
    assert t.model == "HERO13 Black"
    assert t.gps_utc == when.replace(second=1) and t.gps_offset == 1.001


def test_gpmf_reads_on_until_gps9_has_a_fix(monkeypatch):
    from pathlib import Path

    when = datetime(2026, 7, 19, 10, 0, 0, tzinfo=UTC)
    calls = []

    def fake(path, probe, count=5):
        calls.append(count)
        fix = 3 if count > 5 else 0
        return [(0.0, gps9_payload("HERO13 Black", when, fix=fix))]

    monkeypatch.setattr(gpmf, "packets", fake)
    assert gpmf.read(Path("GX010001.MP4"), None).gps_utc == when
    assert calls == [5, gpmf.SCAN_PACKETS]


def test_gpmf_ignores_gps_time_without_a_fix():
    t = gpmf.parse([(0.0, payload("HERO8 Black", "191118234514.250", fix=0))])
    assert t.gps_utc is None


def test_gpmf_stops_at_damaged_data():
    data = payload("HERO8 Black", None)
    assert gpmf.parse([(0.0, data[:10]), (1.0, b"\xff" * 40)]).model is None


# --- GoPro ---------------------------------------------------------------------------------------


@needs_ffmpeg
@pytest.mark.parametrize("names", [
    ["GOPR0001.MP4", "GP010001.MP4", "GP020001.MP4"],
    ["GX010007.MP4", "GX020007.MP4"],
])  # fmt: skip
def test_gopro_chapters_are_one_recording(tmp_path, names):
    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    for n, name in enumerate(names):  # each 2 s chapter starts where the previous one ended
        clip(footage / name, creation_time=f"2026-07-19T10:10:{n * 2:02d}Z",
             firmware="H24.01.02.10.00")  # fmt: skip
    clip(footage / "GX010008.MP4", creation_time="2026-07-19T11:00:00Z", firmware="H24.01.02.10.00")
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        found = recordings(trip)
        device = {r["device"] for r in trip.db.execute("SELECT device FROM assets")}
        labels = [r["label"] for r in trip.db.execute("SELECT label FROM cameras")]
    assert names in found and ["GX010008.MP4"] in found
    assert device == {"gopro"}
    assert labels == ["GoPro HERO13 (footage)"], "named from the firmware without telemetry"


@needs_ffmpeg
def test_a_missing_gopro_chapter_splits_the_recording(tmp_path):
    """Time after a missing chapter can't be known from the chapters before it: each side is a
    recording with its own time."""
    from reelsift.ingest import ingest
    from reelsift.sync import clock, sync

    footage = tmp_path / "footage"
    clip(footage / "GOPR0001.MP4", creation_time="2026-07-19T10:00:00Z", firmware="HD4.01")
    clip(footage / "GP020001.MP4", creation_time="2026-07-19T10:00:04Z", firmware="HD4.01")
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        found = recordings(trip)
        warned = [r["message"] for r in trip.db.execute("SELECT message FROM issues")]
        sync(trip)
        second = trip.db.execute("SELECT MAX(id) AS id FROM recordings").fetchone()["id"]
        start = clock(trip, second).local(Fraction(0))
    assert found == [["GOPR0001.MP4"], ["GP020001.MP4"]]
    assert any("missing chapters" in m for m in warned)
    assert start.replace(tzinfo=None) == datetime(2026, 7, 19, 10, 0, 4)


@needs_ffmpeg
def test_gopro_chapters_that_dont_meet_in_time_are_separate_recordings(tmp_path):
    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    clip(footage / "GX010003.MP4", creation_time="2026-07-19T10:00:00Z", firmware="H24.01")
    clip(footage / "GX020003.MP4", creation_time="2026-07-20T09:00:00Z", firmware="H24.01")
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        assert recordings(trip) == [["GX010003.MP4"], ["GX020003.MP4"]]


@needs_ffmpeg
def test_a_chapter_copied_in_later_joins_its_recording(tmp_path):
    """The recording keeps its id (and so its review decisions); analysis is redone."""
    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    clip(footage / "GX010007.MP4", creation_time="2026-07-19T10:00:00Z", firmware="H24.01")
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        (rec,) = [r["id"] for r in trip.db.execute("SELECT id FROM recordings")]
        trip.db.execute("INSERT INTO stage_runs VALUES (?, 'visual', 'v', 'now')", (rec,))
        clip(footage / "GX020007.MP4", seconds=3.0, creation_time="2026-07-19T10:00:02Z",
             firmware="H24.01")  # fmt: skip
        ingest(trip)
        assert recordings(trip) == [["GX010007.MP4", "GX020007.MP4"]]
        row = trip.db.execute("SELECT id, duration FROM recordings").fetchone()
        chapters = [(c["idx"], c["start"]) for c in trip.db.execute(
            "SELECT idx, start FROM chapters ORDER BY idx")]  # fmt: skip
        redo = trip.db.execute("SELECT COUNT(*) FROM stage_runs").fetchone()[0]
    assert row["id"] == rec and Fraction(row["duration"]) == 5
    assert chapters == [(0, "0/1"), (1, "2/1")]
    assert redo == 0, "the longer recording is analysed again"


@needs_ffmpeg
def test_a_missing_chapter_found_later_joins_both_sides(tmp_path):
    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    clip(footage / "GOPR0001.MP4", creation_time="2026-07-19T10:00:00Z", firmware="HD4.01")
    clip(footage / "GP020001.MP4", creation_time="2026-07-19T10:00:04Z", firmware="HD4.01")
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        assert len(recordings(trip)) == 2
        clip(footage / "GP010001.MP4", creation_time="2026-07-19T10:00:02Z", firmware="HD4.01")
        ingest(trip)
        found = recordings(trip)
    assert found == [["GOPR0001.MP4", "GP010001.MP4", "GP020001.MP4"]]


@needs_ffmpeg
def test_cycliq_segments_copied_in_batches_are_one_ride(tmp_path):
    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    clip(footage / "CYQ_0001.MP4", seconds=4.0, creation_time="2026-08-11T10:00:04Z",
         comment="FLY")  # fmt: skip
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        clip(footage / "CYQ_0002.MP4", seconds=4.0, creation_time="2026-08-11T10:00:08Z",
             comment="FLY")  # fmt: skip
        ingest(trip)
        found = recordings(trip)
        warned = [r["message"] for r in trip.db.execute("SELECT message FROM issues")]
    assert found == [["CYQ_0001.MP4", "CYQ_0002.MP4"]]
    assert not any("missing chapters" in m for m in warned)


@needs_ffmpeg
def test_gps_in_a_later_chapter_gives_the_recordings_time(tmp_path, monkeypatch):
    from reelsift.devices import gopro
    from reelsift.ingest import ingest

    def telemetry(self, path, info):
        if path.name == "GX020007.MP4":  # GPS found its fix during the second chapter
            return gpmf.Telemetry("HERO7 Black", datetime(2018, 3, 23, 17, 25, 2, tzinfo=UTC), 0.0)
        return gpmf.Telemetry("HERO7 Black")

    monkeypatch.setattr(gopro.GoPro, "telemetry", telemetry)
    footage = tmp_path / "footage"
    clip(footage / "GX010007.MP4", creation_time="2017-12-31T12:00:00Z", firmware="HD7.01")
    clip(footage / "GX020007.MP4", creation_time="2017-12-31T12:00:02Z", firmware="HD7.01")
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        assert len(recordings(trip)) == 1
        gps = trip.db.execute(
            "SELECT utc_at_zero FROM time_evidence WHERE provenance = 'gps'").fetchone()  # fmt: skip
    assert gps is not None
    assert datetime.fromisoformat(gps["utc_at_zero"]) == datetime(2018, 3, 23, 17, 25, 0,
                                                                  tzinfo=UTC)  # fmt: skip


@needs_ffmpeg
def test_files_read_by_an_older_reelsift_are_recognized_again(tmp_path, monkeypatch):
    """A trip ingested before the GoPro profile existed has its GoPro files as generic files;
    ingesting again reads them as GoPro files and joins their chapters."""
    from reelsift import devices
    from reelsift.devices.dji import Dji
    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    clip(footage / "GX010007.MP4", creation_time="2026-07-19T10:00:00Z", firmware="H24.01")
    clip(footage / "GX020007.MP4", creation_time="2026-07-19T10:00:02Z", firmware="H24.01")
    profiles = devices.PROFILES
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        monkeypatch.setattr(devices, "PROFILES", [Dji()])
        ingest(trip)
        trip.db.execute("DELETE FROM meta WHERE key = 'detection_version'")  # an older Reelsift
        assert len(recordings(trip)) == 2
        monkeypatch.setattr(devices, "PROFILES", profiles)
        ingest(trip)
        found = recordings(trip)
        kinds = {r["device"] for r in trip.db.execute("SELECT device FROM assets")}
        cameras = [r["label"] for r in trip.db.execute("SELECT label FROM cameras")]
    assert found == [["GX010007.MP4", "GX020007.MP4"]]
    assert kinds == {"gopro"} and cameras == ["GoPro HERO13 (footage)"]


def test_camera_settings_follow_files_matched_again(tmp_path):
    """A phone's photos read by an older Reelsift were a camera of their own; matched again they
    join the phone's camera, and the zone set for them still applies."""
    from reelsift import config as config_mod
    from reelsift.ingest import PHOTO_PROFILE, _camera, ingest
    from reelsift.sync import clock, sync

    from .conftest import make_photo

    footage = tmp_path / "footage"
    footage.mkdir()
    make_photo(footage / "IMG_0001.JPG", taken="2026:07:19 10:00:00", offset=None, make="Apple",
               model="iPhone 16")  # fmt: skip
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        # As an older Reelsift left it: the photos as their own camera, with a zone set for it.
        old = _camera(trip, PHOTO_PROFILE, "x", "footage", label="Apple iPhone 16 photos")
        trip.db.execute("UPDATE recordings SET camera_id = ?", (old,))
        trip.db.execute("DELETE FROM cameras WHERE id != ?", (old,))
        trip.db.execute("DELETE FROM meta WHERE key = 'detection_version'")
        old_label = trip.db.execute("SELECT label FROM cameras").fetchone()["label"]
        trip.config.cameras[old_label] = config_mod.CameraConfig(zone="+02:00")
        ingest(trip)
        label = trip.db.execute("SELECT label FROM cameras").fetchone()["label"]
        sync(trip)
        rec = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
        utc = clock(trip, rec).to_utc(0)
        saved = config_mod.load(trip.root).cameras
    assert label == "Apple iPhone 16 (footage)"
    assert utc == datetime(2026, 7, 19, 8, 0, tzinfo=UTC)
    assert saved[label].zone == "+02:00"


@needs_ffmpeg
def test_files_matched_again_get_the_current_time_rules(tmp_path):
    """A generic file in an older trip gets the zone from the local time in its name."""
    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    clip(footage / "20200705_150405.mp4", creation_time="2020-07-05T19:04:05Z")
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        trip.db.execute(
            "DELETE FROM time_evidence WHERE provenance = 'metadata_zone_from_filename'"
        )
        trip.db.execute("DELETE FROM meta WHERE key = 'detection_version'")
        ingest(trip)
        zones = [r["zone_offset_min"] for r in trip.db.execute(
            "SELECT zone_offset_min FROM time_evidence"
            " WHERE provenance = 'metadata_zone_from_filename'")]  # fmt: skip
    assert zones == [-240]


@needs_ffmpeg
def test_gopro_time_is_the_cameras_local_clock(tmp_path):
    from reelsift import config as config_mod
    from reelsift.ingest import ingest
    from reelsift.sync import clock, sync

    footage = tmp_path / "footage"
    clip(footage / "GX010001.MP4", creation_time="2026-07-19T10:10:00Z", firmware="H24.01.02.10.00")
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        label = trip.db.execute("SELECT label FROM cameras").fetchone()["label"]
        trip.config.cameras[label] = config_mod.CameraConfig(zone="+02:00")
        sync(trip)
        rec = trip.db.execute("SELECT id FROM recordings").fetchone()["id"]
        clk = clock(trip, rec)
    assert clk.to_utc(Fraction(0)) == datetime(2026, 7, 19, 8, 10, tzinfo=UTC)
    assert clk.local(Fraction(0)).hour == 10


def test_gopro_gps_time_gives_the_zone(tmp_path, monkeypatch):
    from reelsift.devices import gopro
    from reelsift.media import parse_probe

    probe = parse_probe(tmp_path / "GX010001.MP4", {
        "format": {"duration": "10", "tags": {"creation_time": "2019-11-18T15:45:32.000000Z",
                                              "firmware": "H19.03.01.30.00"}},
        "streams": [{"index": 0, "codec_type": "video", "width": 320, "height": 240,
                     "time_base": "1/25", "avg_frame_rate": "25/1", "r_frame_rate": "25/1"}],
    })  # fmt: skip
    monkeypatch.setattr(gopro.GoPro, "telemetry", lambda self, p, i: gpmf.Telemetry(
        "GoPro Max", datetime(2019, 11, 18, 23, 45, 14, tzinfo=UTC), 0.0))  # fmt: skip
    (best, *_) = gopro.GoPro().time_evidence(tmp_path / "GX010001.MP4", probe)
    assert best.utc_at_zero == datetime(2019, 11, 18, 23, 45, 14, tzinfo=UTC)
    assert best.zone_offset_min == -8 * 60


@needs_ffmpeg
def test_gopro_low_resolution_proxies_are_paired(tmp_path):
    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    clip(footage / "GX010001.MP4", creation_time="2026-07-19T10:10:00Z", firmware="H24.01.02.10.00")
    run([tool("ffmpeg"), "-v", "error", "-y", "-i", str(footage / "GX010001.MP4"), "-vf",
         "scale=160:120", "-c:v", "libx264", "-f", "mp4", str(footage / "GL010001.LRV")])  # fmt: skip
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        summary = ingest(trip)
    assert summary.camera_proxies == 1


# --- Cycliq ---------------------------------------------------------------------------------------


@needs_ffmpeg
def test_cycliq_segments_that_meet_are_one_recording(tmp_path):
    from reelsift.ingest import ingest
    from reelsift.sync import clock, sync

    footage = tmp_path / "footage"
    # creation_time is when each segment ended: 4 s segments ending 10:00:04, :08, :12; then
    # a new ride an hour later.
    ends = {"LCK_0006.MP4": "10:00:04", "CYQ_0007.MP4": "10:00:08", "CYQ_0008.MP4": "10:00:12",
            "CYQ_0009.MP4": "11:00:04"}  # fmt: skip
    for name, end in ends.items():
        clip(footage / name, seconds=4.0, creation_time=f"2026-08-11T{end}Z", comment="FLY")
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        found = recordings(trip)
        sync(trip)
        first = next(r["id"] for r in trip.db.execute("SELECT id FROM recordings ORDER BY id"))
        start = clock(trip, first).to_utc(Fraction(0))
    assert ["LCK_0006.MP4", "CYQ_0007.MP4", "CYQ_0008.MP4"] in found
    assert ["CYQ_0009.MP4"] in found
    assert start.replace(tzinfo=None) == datetime(2026, 8, 11, 10, 0, 0), "start, not end"


# --- zone from the file name ---------------------------------------------------------------------


@needs_ffmpeg
def test_a_local_time_in_the_file_name_gives_the_zone(tmp_path):
    from reelsift.ingest import ingest
    from reelsift.sync import clock, sync

    footage = tmp_path / "footage"
    clip(footage / "20200705_150405.mp4", creation_time="2020-07-05T19:04:38Z")
    clip(
        footage / "pixel" / "PXL_20200705_190405123.mp4",
        seconds=3.0,
        creation_time="2020-07-05T19:04:38Z",
    )
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        sync(trip)
        zones = {
            r["path"].replace("\\", "/").rsplit("/", 1)[-1]: clock(trip, r["id"]).zone_offset_min
            for r in trip.db.execute("SELECT r.id, a.path FROM recordings r JOIN chapters c ON"
                                     " c.recording_id = r.id JOIN assets a ON a.id = c.asset_id")
        }  # fmt: skip
    assert zones["20200705_150405.mp4"] == -4 * 60
    assert zones["PXL_20200705_190405123.mp4"] is None, "Pixel names are UTC: no zone from them"


# --- HEIC ----------------------------------------------------------------------------------------


@needs_ffmpeg
def test_gopro_files_are_recognized_by_their_tracks_when_renamed(tmp_path):
    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    path = footage / "my ride.mp4"
    footage.mkdir()
    run([tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=320x240:rate=25:duration=2", "-c:v", "libx264",
         "-metadata:s:v:0", "handler_name=GoPro AVC", str(path)])  # fmt: skip
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        ingest(trip)
        devices = {r["device"] for r in trip.db.execute("SELECT device FROM assets")}
    assert devices == {"gopro"}


def test_avif_photos_are_read(tmp_path):
    from PIL import Image

    from reelsift.ingest import ingest

    footage = tmp_path / "footage"
    footage.mkdir()
    Image.new("RGB", (400, 300), (30, 120, 200)).save(footage / "photo.avif")
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        summary = ingest(trip)
    assert summary.photos == 1 and not summary.failed


def test_heic_photos_are_read(tmp_path):
    import pillow_heif
    from PIL import Image

    from reelsift.ingest import ingest

    pillow_heif.register_heif_opener()
    footage = tmp_path / "footage"
    footage.mkdir()
    image = Image.new("RGB", (400, 300), (30, 120, 200))
    exif = Image.Exif()
    exif[0x010F], exif[0x0110] = "Apple", "iPhone 16 Pro Max"
    exif.get_ifd(0x8769)[0x9003] = "2026:09:19 18:10:00"
    exif.get_ifd(0x8769)[0x9011] = "+02:00"
    image.save(footage / "IMG_0001.HEIC", exif=exif)
    with Trip.create(tmp_path / "trip", [footage]) as trip:
        summary = ingest(trip)
        assert not summary.failed
        row = trip.db.execute("SELECT width, height FROM recordings").fetchone()
    assert (row["width"], row["height"]) == (400, 300)
