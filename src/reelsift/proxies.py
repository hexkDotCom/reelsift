"""Generate low-resolution proxies for recordings whose camera didn't make a usable one.

Proxies are what analysis and the review app read. They are SDR BT.709 H.264 with the longest
side at 1280 pixels, rotated upright, with a keyframe every second for quick seeking. HDR sources
are tone-mapped here with the same settings the final render will use. Frame timing is passed
through unchanged, and any difference in start time is stored as the proxy's offset.
"""

import json
import os
from fractions import Fraction
from pathlib import Path

from . import photos
from .jobs import Job, JobResult, Progress
from .media import Probe, parse_probe, probe
from .project import Trip
from .timebase import to_db
from .tools import filter_available, h264_args, run, tool

LONG_SIDE = 1280
QUALITY = 28
SETTINGS_VERSION = 1
TAGS = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"

SDR_CHAIN = (
    "scale=w='if(gte(iw,ih),{n},-2)':h='if(gte(iw,ih),-2,{n})'"
    ":in_color_matrix={matrix}:out_color_matrix=bt709:in_range={range}:out_range=tv"
)
HDR_CHAIN = (
    "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
    "zscale=t=bt709:m=bt709:r=tv,format=yuv420p,"
    "scale=w='if(gte(iw,ih),{n},-2)':h='if(gte(iw,ih),-2,{n})'"
)


DEINTERLACE = "bwdif=mode=send_frame:deint=interlaced"
"""One progressive frame per interlaced frame, so frame counts and timing stay the same."""


def video_filter(source: Probe) -> str:
    video = source.main_video
    if video and video.hdr:
        if not filter_available("zscale"):
            raise RuntimeError("This FFmpeg build lacks zscale, needed to tone-map HDR video")
        chain = HDR_CHAIN.format(n=LONG_SIDE)
    else:
        matrix = (
            video.color_space
            if video and video.color_space in ("bt709", "bt601", "bt470bg", "smpte170m")
            else "bt709"
        )
        color_range = "pc" if video and video.color_range == "pc" else "tv"
        chain = SDR_CHAIN.format(n=LONG_SIDE, matrix=matrix, range=color_range)
    if video and video.interlaced:
        chain = f"{DEINTERLACE},{chain}"
    # Tag every frame explicitly; encoders otherwise fall back to BT.601 tags for small frames.
    return f"{chain},format=yuv420p,{TAGS}"


def command(source: Probe, out: Path, gpu_decode: bool = False, cpu: bool = False) -> list[str]:
    video = source.main_video
    gop = round(video.avg_fps) if video and video.avg_fps else 30
    args = [tool("ffmpeg"), "-v", "error", "-y"]
    if gpu_decode:
        args += ["-hwaccel", "cuda"]
    # The probed main stream, not stream 0, which can be a cover picture.
    args += ["-i", str(source.path), "-map", f"0:{video.index}"]
    if source.audio:
        args += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "128k", "-ac", "2"]
    args += [
        "-vf",
        video_filter(source),
        *h264_args(QUALITY, cpu),
        "-g",
        str(gop),
        "-fps_mode",
        "passthrough",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(out),
    ]
    return args


def settings(source: Probe) -> dict:
    return {
        "version": SETTINGS_VERSION,
        "long_side": LONG_SIDE,
        "tonemap": bool(source.main_video and source.main_video.hdr),
        "deinterlace": bool(source.main_video and source.main_video.interlaced),
        "encoder": h264_args(QUALITY)[1],
        "quality": QUALITY,
    }


def reconcile(trip: Trip) -> int:
    """Forget proxy rows whose file is gone or was made with older settings, so the proxy is
    paired or generated again. Returns how many were forgotten."""
    removed = 0
    for row in trip.db.execute(
        "SELECT p.*, a.path AS source, a.probe_json FROM proxies p JOIN chapters c"
        " ON c.recording_id = p.recording_id AND c.idx = p.chapter_idx"
        " JOIN assets a ON a.id = c.asset_id"
    ).fetchall():
        path = resolve(trip, row["path"])
        saved = json.loads(row["settings"])
        outdated = row["origin"] == "generated" and (
            saved.get("version") not in (SETTINGS_VERSION, None)
            # Proxies made before interlaced video was deinterlaced.
            or (
                saved.get("version") is not None
                and saved.get("deinterlace", False)
                != settings(parse_probe(Path(row["source"]), json.loads(row["probe_json"])))[
                    "deinterlace"
                ]
            )
        )
        if path.is_file() and path.stat().st_size > 0 and not outdated:
            continue
        trip.db.execute(
            "DELETE FROM proxies WHERE recording_id = ? AND chapter_idx = ?",
            (row["recording_id"], row["chapter_idx"]),
        )
        if outdated and path.is_file() and row["origin"] == "generated":
            path.unlink()
        if outdated:
            # Analysis read the old proxy.
            trip.db.execute("DELETE FROM stage_runs WHERE recording_id = ?",
                            (row["recording_id"],))  # fmt: skip
        removed += 1
    return removed


def pending(trip: Trip) -> list[str]:
    rows = trip.db.execute(
        "SELECT c.recording_id, c.idx FROM chapters c"
        " LEFT JOIN proxies p ON p.recording_id = c.recording_id AND p.chapter_idx = c.idx"
        " WHERE p.recording_id IS NULL ORDER BY c.recording_id, c.idx"
    )
    return [f"{r['recording_id']}:{r['idx']}" for r in rows]


def generate(trip: Trip, progress: Progress | None = None) -> JobResult:
    db = trip.db
    reconcile(trip)

    def work(item: str) -> None:
        recording_id, idx = (int(x) for x in item.split(":"))
        row = db.execute(
            "SELECT a.path, a.kind, a.fingerprint, a.probe_json, c.source_start FROM chapters c"
            " JOIN assets a ON a.id = c.asset_id WHERE c.recording_id = ? AND c.idx = ?",
            (recording_id, idx),
        ).fetchone()
        if row["kind"] == "photo":
            relative = Path("cache") / "proxies" / f"{row['fingerprint']}.jpg"
            width, height = photos.make_proxy(Path(row["path"]), trip.root / relative)
            db.execute(
                "INSERT OR REPLACE INTO proxies (recording_id, chapter_idx, origin, path, asset_id,"
                " offset, width, height, upright, settings) VALUES (?, ?, 'generated', ?, NULL,"
                " '0/1', ?, ?, 1, ?)",
                (recording_id, idx, relative.as_posix(), width, height,
                 json.dumps({"version": SETTINGS_VERSION, "photo": True})),
            )  # fmt: skip
            return
        source = parse_probe(Path(row["path"]), json.loads(row["probe_json"]))
        relative = Path("cache") / "proxies" / f"{row['fingerprint']}.mp4"
        final = trip.root / relative
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.with_suffix(".partial.mp4")
        # GPU decode and encode; then CPU decode (codecs the GPU can't decode); then x264
        # (sizes or formats NVENC refuses).
        attempts = [
            lambda: command(source, partial, gpu_decode=True),
            lambda: command(source, partial),
            lambda: command(source, partial, cpu=True),
        ]
        for n, make in enumerate(attempts):
            try:
                run(make())
                break
            except RuntimeError:
                if n == len(attempts) - 1:
                    raise
        made = probe(partial).main_video
        if made is None:
            raise RuntimeError("proxy has no video stream")
        os.replace(partial, final)
        offset = Fraction(row["source_start"]) - made.start
        db.execute(
            "INSERT OR REPLACE INTO proxies (recording_id, chapter_idx, origin, path, asset_id,"
            " offset, width, height, upright, settings) VALUES (?, ?, 'generated', ?, NULL, ?, ?,"
            " ?, 1, ?)",
            (
                recording_id,
                idx,
                relative.as_posix(),
                to_db(offset),
                made.width,
                made.height,
                json.dumps(settings(source)),
            ),
        )

    todo = pending(trip)  # chapters without a usable proxy, whatever an earlier run recorded
    return Job(trip, "proxies", {"version": SETTINGS_VERSION}, todo, redo=todo).run(work, progress)


def resolve(trip: Trip, stored_path: str) -> Path:
    """Absolute path of a proxy: camera proxies are stored absolute, generated ones relative."""
    path = Path(stored_path)
    return path if path.is_absolute() else trip.root / path
