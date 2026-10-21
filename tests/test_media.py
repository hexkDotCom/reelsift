from fractions import Fraction
from pathlib import Path

from reelsift.media import fingerprint, parse_probe, parse_time

PROBE = {
    "format": {
        "format_name": "mov,mp4",
        "duration": "22.208000",
        "tags": {"creation_time": "2026-10-11T07:51:34.000000Z", "encoder": "DJI OsmoAction6"},
    },
    "streams": [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "hevc",
            "width": 3840,
            "height": 2160,
            "time_base": "1/90000",
            "start_pts": 0,
            "duration_ts": 1998720,
            "nb_frames": "555",
            "avg_frame_rate": "25/1",
            "r_frame_rate": "25/1",
            "pix_fmt": "yuv420p10le",
            "color_transfer": "bt709",
            "color_primaries": "bt709",
            "color_space": "bt709",
            "color_range": "tv",
            "side_data_list": [{"rotation": -90}],
        },
        {
            "index": 1,
            "codec_type": "audio",
            "codec_name": "aac",
            "sample_rate": "48000",
            "channels": 2,
        },
        {
            "index": 3,
            "codec_type": "data",
            "codec_tag_string": "djmd",
            "tags": {"handler_name": "CAM meta"},
        },
        {
            "index": 6,
            "codec_type": "video",
            "codec_name": "mjpeg",
            "width": 1280,
            "height": 720,
            "time_base": "1/90000",
            "avg_frame_rate": "0/0",
            "r_frame_rate": "90000/1",
        },
    ],
}


def test_parse_probe_reads_streams_and_skips_cover_image():
    p = parse_probe(Path("x.mp4"), PROBE)
    assert len(p.video) == 1
    v = p.main_video
    assert v.duration == Fraction(1998720, 90000)
    assert v.frames == 555
    assert v.rotation == 90
    assert v.upright_size == (2160, 3840)
    assert not v.vfr and not v.hdr
    assert p.audio[0].channels == 2
    assert p.data_handlers == ["CAM meta"]
    assert p.creation_time.utcoffset().total_seconds() == 0


def test_hdr_and_vfr_are_detected():
    data = {
        "format": {},
        "streams": [
            dict(
                PROBE["streams"][0],
                color_transfer="arib-std-b67",
                avg_frame_rate="243/10",
                r_frame_rate="30/1",
            )
        ],
    }
    v = parse_probe(Path("x.mp4"), data).main_video
    assert v.hdr
    assert v.vfr


def test_parse_time_keeps_zone_only_when_given():
    assert parse_time("2026-10-11T09:51:33+0200").utcoffset().total_seconds() == 7200
    assert parse_time("2026-10-11 09:51:33").tzinfo is None
    assert parse_time("garbage") is None


def test_fingerprint_depends_on_content_not_name(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"x" * 5000)
    b = tmp_path / "b.bin"
    b.write_bytes(b"x" * 5000)
    c = tmp_path / "c.bin"
    c.write_bytes(b"x" * 4999 + b"y")
    assert fingerprint(a) == fingerprint(b) != fingerprint(c)


def test_ntsc_rate_is_not_variable():
    data = {
        "format": {},
        "streams": [dict(PROBE["streams"][0], avg_frame_rate="30000/1001", r_frame_rate="30/1")],
    }
    assert not parse_probe(Path("x.mp4"), data).main_video.vfr
