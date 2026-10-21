"""Rendering an edit from the original files.

Every clip is cut from the full-resolution original into an intermediate file with identical
format (size, frame rate, codecs), using frame-accurate input seeking and an exact frame count.
The intermediates are joined without re-encoding. Cutting clip by clip makes long edits resumable
(each clip is a job item) and keeps FFmpeg filter graphs small.

The joined edit then gets its final sound in one pass: the clips' own sound, quieter where nobody
speaks when there is music; the music, ducked under speech, faded in and out; and loudness
normalized in two passes to -16 LUFS with peaks under -1.5 dBTP.

Output: SDR BT.709 H.264 with AAC stereo audio at 48 kHz. In a 16:9 edit, portrait clips are
letterboxed; in a 9:16 edit, landscape clips are cropped at each clip's crop position. HDR clips
are tone-mapped.
"""

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path

from . import photos
from .jobs import Job, Progress
from .media import parse_probe
from .project import Trip
from .select import Item
from .tools import check_cancelled, encoder_works, h264_args, run, tool

TAGS = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
STANDARD_RATES = [
    Fraction(24000, 1001), 24, 25, Fraction(30000, 1001), 30, 50, Fraction(60000, 1001), 60,
]  # fmt: skip
AUDIO_FADE = 0.03
QUALITY = 19
ZOOM = 0.06
"""How much a photo zooms in over its time on screen."""

LOUDNESS = -16.0
TRUE_PEAK = -1.5
SPEECH_GAP = 1.0
"""Speech separated by less than this is treated as one stretch when mixing."""
RAMP = 0.2
SOURCE_UNDER_MUSIC = 0.35
"""Gain of the clips' own sound where nobody speaks, when music plays."""
MUSIC_GAIN = 0.5
MUSIC_UNDER_SPEECH = 0.12
MUSIC_FADE_IN = 1.0
MUSIC_FADE_OUT = 2.0


@dataclass
class Output:
    width: int = 1920
    height: int = 1080
    fps: str = "auto"
    quality: int = QUALITY

    @property
    def vertical(self) -> bool:
        return self.height > self.width


@dataclass
class Piece:
    """Part of a clip that lies in one source file."""

    source: Path
    start: Fraction
    """Seconds into the file (from its first frame)."""
    duration: Fraction
    fps: Fraction
    hdr: bool
    color_space: str
    has_audio: bool
    width: int = 0
    height: int = 0
    """Upright size of the source picture."""
    crop_x: float = 0.5
    stream: int = 0
    """Absolute index of the main video stream (stream 0 can be a cover picture)."""
    vfr: bool = False
    """Variable frame rate: frames aren't on a regular grid, so cuts aren't snapped to one."""
    interlaced: bool = False
    full_range: bool = False
    """Levels 0-255; converted to the output's 16-235."""
    indexed: bool = True
    """Whether the container can seek straight to a frame. MPEG-TS and MPEG-PS (camcorder
    .MTS, .MPG) have no index: seeking lands roughly, possibly past the keyframe a cut needs."""
    photo: bool = False
    key: str = ""
    """Fingerprint of the source, used to name cached stills."""
    origin: Fraction = Fraction(0)
    """Seconds from the start of the file's timestamps to its first video frame. Not zero when
    another stream (usually audio) starts before the video."""
    speed: Fraction = Fraction(1)
    """Played this much faster than real time (start and duration stay footage seconds)."""
    frames: int | None = None
    """Output frames, when set by plan_frames (sped-up pieces); otherwise from the duration."""


def pieces(trip: Trip, item: Item) -> list[Piece]:
    rows = trip.db.execute(
        "SELECT c.start, c.duration, a.path, a.probe_json, a.kind, a.fingerprint, a.changed"
        " FROM chapters c JOIN assets a ON a.id = c.asset_id WHERE c.recording_id = ?"
        " ORDER BY c.idx",
        (item.recording_id,),
    ).fetchall()
    for r in rows:
        if r["changed"]:
            raise RuntimeError(
                f"{r['path']} was replaced by a file that can't be read. Put the original back "
                "(or fix the file) and run reelsift ingest."
            )
    if rows and rows[0]["kind"] == "photo":
        info = json.loads(rows[0]["probe_json"])["photo"]
        duration = Fraction(item.end - item.start).limit_denominator(1000)
        return [Piece(Path(rows[0]["path"]), Fraction(0), duration, Fraction(30), False, "bt709",
                      False, info["width"], info["height"], item.crop_x, photo=True,
                      key=rows[0]["fingerprint"])]  # fmt: skip
    out = []
    start = Fraction(item.start).limit_denominator(1000)
    end = Fraction(item.end).limit_denominator(1000)
    speed = Fraction(item.speed).limit_denominator(1_000_000)  # a fitted speed, to the frame
    if speed <= 0:
        raise ValueError(f"a clip's speed must be positive, not {item.speed}")
    for r in rows:
        c_start, c_dur = Fraction(r["start"]), Fraction(r["duration"])
        s, e = max(start, c_start), min(end, c_start + c_dur)
        if e <= s:
            continue
        probe = parse_probe(Path(r["path"]), json.loads(r["probe_json"]))
        video = probe.main_video
        width, height = video.upright_size
        out.append(
            Piece(
                source=Path(r["path"]),
                start=s - c_start,
                duration=e - s,
                fps=video.avg_fps or video.r_fps or Fraction(30),
                hdr=video.hdr,
                color_space=video.color_space,
                has_audio=bool(probe.audio),
                width=width,
                height=height,
                crop_x=item.crop_x,
                stream=video.index,
                vfr=video.vfr,
                interlaced=video.interlaced,
                full_range=video.full_range,
                indexed=not {"mpegts", "mpeg"} & set(probe.format_name.split(",")),
                origin=max(Fraction(0), video.start - probe.start),
                speed=speed,
            )
        )
    return out


CHUNK_SOURCE_S = 120
"""Sped-up stretches are rendered in pieces of at most this much footage (resumable jobs)."""


def _sample_pieces(found: list[Piece], speed: Fraction, fps: Fraction) -> list[Piece]:
    """Sped-up footage samples one source frame per output frame, on one grid for the whole clip:
    output frame j shows the footage at start + j * speed / fps. Each piece (chapter, chunk) is
    moved to its first sample and given the number of samples it holds, so there is no jump where
    a chapter or chunk ends. found: the clip's pieces from pieces(), one per chapter."""
    out = []
    step = speed / fps
    offset = Fraction(0)  # recording time of each piece's start relative to the clip's start
    for p in found:
        a, b = offset, offset + p.duration
        offset = b
        first = math.ceil(a / step)
        last = math.ceil(b / step)  # exclusive
        frames = last - first
        if frames <= 0:
            continue
        shift = first * step - a
        per_chunk = max(1, math.floor(CHUNK_SOURCE_S / step))
        for k in range(first, last, per_chunk):
            n = min(per_chunk, last - k)
            out.append(replace(p, start=p.start + shift + (k - first) * step, duration=n * step,
                               frames=n))  # fmt: skip
    return out


def plan_frames(all_pieces: list[list[Piece]], fps: Fraction) -> None:
    """Fix each sped-up clip's samples on the edit's frame rate (in place). In an edit with
    sped-up clips, frames are counted out along the whole edit (each clip gets what brings the
    running total closest to the exact time so far), so many stretches don't add up to a longer
    edit than fitted; a clip never gets more frames than its footage has samples."""
    if not any(ps and ps[0].speed != 1 for ps in all_pieces):
        return  # an ordinary edit: each piece rounds its own length, as always
    exact = Fraction(0)
    done = 0
    for ps in all_pieces:
        if not ps:
            continue
        speed = ps[0].speed
        exact += sum((p.duration for p in ps), Fraction(0)) / speed
        want = max(0, round(exact * fps) - done)
        if speed == 1:
            for k, p in enumerate(ps):
                natural = math.ceil(p.duration * fps)
                p.frames = natural if k < len(ps) - 1 else 0
            have = sum(p.frames for p in ps[:-1])
            ps[-1].frames = max(0, min(math.ceil(ps[-1].duration * fps), want - have))
            ps[:] = [p for p in ps if p.frames]
        else:
            sampled = _sample_pieces(ps, speed, fps)
            extra = sum(p.frames for p in sampled) - want
            while extra > 0 and sampled:  # fewer samples: from the end
                last = sampled[-1]
                cut = min(extra, last.frames)
                if cut == last.frames:
                    sampled.pop()
                else:
                    sampled[-1] = replace(last, frames=last.frames - cut,
                                          duration=(last.frames - cut) * speed / fps)  # fmt: skip
                extra -= cut
            ps[:] = sampled
        done += sum(p.frames for p in ps)


def edit_plan(trip: Trip, items: list[Item], fps: Fraction | None = None):
    """The pieces of an edit in order, as (item index, piece), and its frame rate: the plan that
    rendering, the music and export share."""
    per_item = [pieces(trip, item) for item in items]
    videos = [p for ps in per_item for p in ps if not p.photo]
    fps = fps or choose_fps(videos)
    plan_frames(per_item, fps)
    return [(i, p) for i, ps in enumerate(per_item) for p in ps], fps


def plan_key(plan) -> list:
    """What each piece of a plan renders, for the cache key: the same edit cut into pieces
    differently is rendered again."""
    return [(str(p.source), str(p.start), str(p.duration), str(p.speed), p.frames)
            for _, p in plan]  # fmt: skip


def choose_fps(all_pieces: list[Piece]) -> Fraction:
    """The frame rate that covers most of the edit's duration, snapped to a standard rate."""
    weight: Counter = Counter()
    for p in all_pieces:
        nearest = min(STANDARD_RATES, key=lambda r: abs(Fraction(r) - p.fps))
        weight[Fraction(nearest)] += p.duration / p.speed  # time in the edit
    return weight.most_common(1)[0][0] if weight else Fraction(30)


def framing(piece: Piece, out: Output) -> str:
    """Fit the picture into the output frame: crop to fill in a vertical edit, letterbox
    otherwise."""
    W, H = out.width, out.height
    # Square pixels first: widescreen DV, DVD and camcorder footage stores 16:9 in narrower
    # pixels, and the fit below works on the picture as it is shown.
    square = "scale='trunc(iw*sar/2)*2':ih,setsar=1,"
    if out.vertical:
        x = min(1.0, max(0.0, piece.crop_x))
        return (
            f"{square}scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H}:x=(iw-{W})*{x:.4f}:y=(ih-{H})/2"
        )
    return (
        f"{square}scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black"
    )


RENDER_VERSION = 10
"""Part of the cache key of rendered clips: raise it whenever clips would come out different."""
SEEK_MARGIN = Fraction(30)
"""In a container without an index, seek at least this far before a cut, then decode up to it."""


def seek_point(piece: Piece) -> Fraction:
    """Where a piece starts, in seconds from the start of the file's timestamps (what -ss counts
    from): its first frame, rounded on the source's frame grid. Variable-frame-rate video has no
    regular grid, and a sped-up piece starts at its exact first sample (see phase), so they
    start at the exact time."""
    exact = piece.vfr or piece.speed != 1
    first = piece.start if exact else round(piece.start * piece.fps) / piece.fps
    return piece.origin + first


PREROLL = Fraction(1)
"""A sped-up piece is decoded from at least this long before its first sample, so the frame just
before it is there to be chosen, as it would be without the cut between pieces (variable-rate
footage: from that frame, however long ago it was, see preroll)."""


def preroll(piece: Piece) -> Fraction:
    """How long before its first sample a sped-up piece is decoded from."""
    seek = seek_point(piece)
    if piece.speed == 1:
        return Fraction(0)
    pre = PREROLL
    if piece.vfr:
        before = frame_before(piece.source, piece.stream, piece.start)
        if before is not None:
            pre = max(pre, piece.start - before + Fraction(1, 100))
    return min(seek, pre)


def frame_before(source: Path, stream: int, t: Fraction) -> Fraction | None:
    """The time of the last frame at or before t (seconds from the video stream's start), from the
    packets' timestamps (no decoding); None if there is none. Frames stored out of order can make
    it a frame or two earlier, never later: decoding from there still includes the frame."""

    def probe(entries: str, *more: str) -> dict:
        out = run([tool("ffprobe"), "-v", "error", "-select_streams", str(stream),
                   "-show_entries", entries, *more, "-of", "json", str(source)]).stdout  # fmt: skip
        return json.loads(out or "{}")

    start = float((probe("stream=start_time").get("streams") or [{}])[0].get("start_time") or 0)
    window = 30.0
    while True:
        lo = max(0.0, float(t) - window)
        data = probe("packet=pts_time", "-read_intervals",
                     f"{start + lo:.3f}%{start + float(t) + 0.01:.3f}")  # fmt: skip
        found = [float(p["pts_time"]) - start for p in data.get("packets", [])
                 if p.get("pts_time") not in (None, "N/A")]  # fmt: skip
        found = [x for x in found if x <= float(t) + 1e-6]
        if found:
            return Fraction(max(found)).limit_denominator(1_000_000)
        if lo <= 0:
            return None
        window *= 4


def landing(piece: Piece, at: Fraction) -> float | None:
    """When seeking to `at`, how many seconds after it the first decoded frame is (zero if
    decoding starts before it); None if nothing decodes."""
    result = run([
        tool("ffmpeg"), "-hide_banner", "-nostats", "-v", "info", "-ss", f"{float(at):.6f}",
        "-i", str(piece.source), "-map", f"0:{piece.stream}", "-vf", "showinfo",
        "-frames:v", "1", "-f", "null", "-",
    ], text=True)  # fmt: skip
    found = re.search(r"pts_time:\s*(-?[\d.]+)", result.stderr)
    return float(found.group(1)) if found else None


def seek_margin(piece: Piece) -> Fraction:
    """How far before a piece to seek in a container without an index. Seeking there can land on
    the next keyframe instead; if that is past the piece (keyframes far apart), seek further
    back, and at worst decode from the start of the file."""
    if piece.indexed:
        return Fraction(0)
    seek, margin = seek_point(piece), SEEK_MARGIN
    while margin < seek:
        landed = landing(piece, seek - margin)
        if landed is not None and landed <= margin + 1 / (2 * piece.fps):
            return margin
        margin *= 4
    return seek


TIE_S = Fraction(1, 10000)


def retime(speed: Fraction, fps: Fraction, lead: Fraction = Fraction(0)) -> str:
    """Sped-up footage: one source frame per output frame, sampled on the clip's grid from its
    exact first sample, `lead` seconds after the seek point (frames keep their timestamps from
    the seek point; a precise time base keeps the grid exact; the frames decoded before the
    first sample are there for fps to choose from). Before the costly conversion and framing, so
    they run on the frames kept only."""
    if speed == 1:
        return ""
    # A sample that falls exactly on a source frame shows that frame, wherever the piece was
    # cut: TIE_S keeps the timestamps' rounding (microseconds) from tipping it either way.
    return (f"settb=AVTB,setpts=(PTS-{float(lead + TIE_S):.9f}/TB)/{float(speed):.9g},"
            f"fps={fps}:round=up:start_time=0,")  # fmt: skip


def tempo(speed: Fraction) -> str:
    """Sound at speed x, pitch kept: atempo steps of at most 2x (better quality than one big
    step)."""
    if speed == 1:
        return ""
    steps, rest = [], float(speed)
    while rest > 2.0:
        steps.append(2.0)
        rest /= 2.0
    steps.append(rest)
    return "".join(f"atempo={s:.9g}," for s in steps)


def video_filter(
    piece: Piece, out: Output, fps: Fraction, skip: Fraction = Fraction(0),
    lead: Fraction = Fraction(0),
) -> str:  # fmt: skip
    """skip: where the cut is, in seconds after the seek point (see SEEK_MARGIN). Frames keep
    their timestamps from the seek point until then, so a seek that lands late (on a keyframe
    after the seek point) still cuts at the right frame. lead: for a sped-up piece, where its
    first sample is after the seek point (skip plus the preroll)."""
    if piece.hdr:
        convert = (
            "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
            "zscale=t=bt709:m=bt709:r=tv,format=yuv420p,"
        )
    else:
        matrix = (
            piece.color_space if piece.color_space in ("bt709", "bt470bg", "smpte170m") else "bt709"
        )
        in_range = "pc" if piece.full_range else "tv"
        # The output format is named here: FFmpeg 6 keeps a phone's full-range format
        # (yuvj420p) through the scale, levels unconverted, when the next filter allows it.
        convert = (f"scale=in_color_matrix={matrix}:out_color_matrix=bt709:in_range={in_range}"
                   ":out_range=tv,format=yuv420p,")  # fmt: skip
    deinterlace = "bwdif=mode=send_frame:deint=interlaced," if piece.interlaced else ""
    drop = f"trim=start={float(skip):.6f}," if skip else ""
    # Timestamps restart at zero, so the frame-rate conversion starts with the first frame.
    # Interlaced footage is deinterlaced before frames are dropped (bwdif uses the neighbours).
    if piece.speed != 1:
        return (f"{drop}{deinterlace}{retime(piece.speed, fps, lead)}{convert}"
                f"{framing(piece, out)},fps={fps},format=yuv420p,setsar=1,{TAGS}")  # fmt: skip
    return (f"{drop}setpts=PTS-STARTPTS,{deinterlace}{convert}"
            f"{framing(piece, out)},fps={fps},format=yuv420p,setsar=1,{TAGS}")  # fmt: skip


def frames_of(piece: Piece, fps: Fraction) -> int:
    if piece.frames is not None:
        return piece.frames
    return max(1, round(piece.duration / piece.speed * fps))


def still_command(
    piece: Piece, out: Output, fps: Fraction, path: Path, still: Path, cpu: bool = False
) -> list[str]:
    """A photo as a clip: framed and slowly zooming in, with silence."""
    frames = frames_of(piece, fps)
    seconds = Fraction(frames) / fps
    zoom = (
        f"zoompan=z='1+{ZOOM}*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d={frames}:s={out.width}x{out.height}:fps={fps}"
    )
    return [
        tool("ffmpeg"), "-v", "error", "-y", "-i", str(still),
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-map", "0:v:0", "-map", "1:a:0",
        # Pictures are full range (0-255); the output, like all video, is 16-235.
        "-vf", f"{zoom},scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setsar=1,{TAGS}",
        "-frames:v", str(frames), "-t", f"{float(seconds):.6f}",
        *h264_args(out.quality, cpu), "-g", str(round(fps)), "-bf", "0",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-video_track_timescale", "90000", "-f", "mp4", str(path),
    ]  # fmt: skip


def clip_command(
    piece: Piece,
    out: Output,
    fps: Fraction,
    path: Path,
    gpu_decode: bool,
    cpu: bool = False,
    margin: Fraction = SEEK_MARGIN,
) -> list[str]:
    """margin: how far before the piece to seek when the container has no index (seek_margin)."""
    frames = frames_of(piece, fps)
    seconds = Fraction(frames) / fps
    args = [tool("ffmpeg"), "-v", "error", "-y"]
    if gpu_decode:
        args += ["-hwaccel", "cuda"]
    seek = seek_point(piece)
    pre = preroll(piece)
    seek -= pre
    skip = Fraction(0) if piece.indexed else min(seek, margin)
    args += ["-ss", f"{float(seek - skip):.6f}", "-i", str(piece.source)]
    if not piece.has_audio:
        args += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
    audio_in = "0:a:0" if piece.has_audio else "1:a:0"
    fade_out = max(0.0, float(seconds) - AUDIO_FADE)
    args += [
        "-map", f"0:{piece.stream}", "-map", audio_in,
        # Picture and sound both count from the seek point, so both are trimmed by skip.
        "-vf", video_filter(piece, out, fps, skip, skip + pre),
        # Sound keeps its timing relative to the picture: if it starts later than the cut (a
        # file whose audio begins after its video), the gap is filled with silence.
        "-af", "aresample=48000:async=1:first_pts=0,"
        + (f"atrim=start={float(skip + pre):.6f},asetpts=PTS-STARTPTS," if skip + pre else "")
        # Only the footage the piece covers is sped up (never what comes after it).
        + (f"atrim=duration={float(piece.duration):.6f},{tempo(piece.speed)}"
           if piece.speed != 1 else "")
        + "aformat=channel_layouts=stereo,apad,"
        f"afade=t=in:d={AUDIO_FADE},afade=t=out:st={fade_out:.4f}:d={AUDIO_FADE}",
        "-frames:v", str(frames), "-t", f"{float(seconds):.6f}",
        *h264_args(out.quality, cpu), "-g", str(round(fps)), "-bf", "0",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        # 90 kHz divides every standard frame duration, including 1001/30000 s, exactly.
        "-video_track_timescale", "90000",
        "-f", "mp4", str(path),
    ]  # fmt: skip
    return args


# --- sound -------------------------------------------------------------------------------------


def speech_in_edit(
    trip: Trip, plan: list[tuple[int, Piece]], items: list[Item], fps: Fraction
) -> list[tuple[float, float]]:
    """Where people speak, in edit time, merged across short pauses. Sped-up clips don't count:
    speech there isn't intelligible, so nothing makes room for it."""
    spans: list[tuple[float, float]] = []
    offset = 0.0
    done_items: set[int] = set()
    for index, piece in plan:
        length = frames_of(piece, fps) / float(fps)
        item = items[index]
        if not piece.photo and piece.speed == 1 and index not in done_items:
            done_items.add(index)
            words = trip.db.execute(
                "SELECT start, end FROM words WHERE recording_id = ? AND end > ? AND start < ?"
                " ORDER BY start",
                (item.recording_id, item.start, item.end),
            ).fetchall()
            for w in words:
                a = offset + max(0.0, w["start"] - item.start)
                b = offset + min(item.end - item.start, w["end"] - item.start)
                if b > a:
                    spans.append((a, b))
        offset += length
    merged: list[list[float]] = []
    for a, b in sorted(spans):
        if merged and a - merged[-1][1] < SPEECH_GAP:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def gain_expression(spans: list[tuple[float, float]], inside: float, outside: float) -> str:
    """A volume expression: `inside` during the spans, `outside` elsewhere, with short ramps."""
    if not spans or inside == outside:
        return f"{outside}"
    terms = "+".join(
        f"clip((t-{a - RAMP:.3f})/{RAMP},0,1)*clip(({b + RAMP:.3f}-t)/{RAMP},0,1)" for a, b in spans
    )
    return f"{outside}+({inside - outside:.4f})*min(1,{terms})"


def _escape_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "'\\''")


def mix_graph(
    duration: float,
    spans: list[tuple[float, float]],
    music_inputs: int,
    mix=None,
    generated=None,
) -> str:
    """The audio graph: input 0 is the joined edit, inputs 1.. are music files in play order.
    mix: soundtrack.Mix levels (default: Reelsift's usual mix). generated: a soundtrack.Resolved
    for one generated track, played once from its start, stopped at its fitted end with its own
    fade and brought to the common loudness first."""
    from .soundtrack import Mix

    mix = mix or Mix()
    master = round(mix.sound_gain, 4)
    if not music_inputs:
        level = "" if master == 1 else f",volume={master}"
        return f"[0:a]aresample=48000,aformat=channel_layouts=stereo{level}[mix]"
    music_gain, background, under_speech = (round(g, 4) for g in mix.gains())
    full = list(spans)
    if generated is not None and generated.end and generated.end < duration - 0.05:
        # After the music has ended there is nothing to make room for.
        full.append((generated.end, duration + 1.0))
    source = gain_expression(full, 1.0, background)
    if master != 1:  # the clips' sound as a whole: nothing (not even speech) comes above it
        source = f"{master}*({source})"
    music = gain_expression(spans, under_speech, music_gain)
    return (
        f"[0:a]aresample=48000,aformat=channel_layouts=stereo,volume='{source}':eval=frame[src];"
        f"{music_branch(duration, music, music_inputs, generated)}[mus];"
        "[src][mus]amix=inputs=2:normalize=0:duration=first[mix]"
    )


def music_branch(duration: float, volume: str, music_inputs: int, generated=None) -> str:
    """The music part of the audio graph (inputs 1..): shaped to the edit, at its levels."""
    tracks = "".join(f"[{i}:a]" for i in range(1, music_inputs + 1))
    if generated is not None:
        end = min(duration, generated.end if generated.end else duration)
        fade = min(generated.fade, end)
        shape = (f"volume={generated.gain_db:.2f}dB,atrim=0:{end:.3f},afade=t=in:d=0.02,"
                 f"afade=t=out:st={max(0.0, end - fade):.3f}:d={fade:.3f}")  # fmt: skip
    else:
        fade_out = max(0.0, duration - MUSIC_FADE_OUT)
        shape = (f"atrim=0:{duration:.3f},afade=t=in:d={MUSIC_FADE_IN},"
                 f"afade=t=out:st={fade_out:.3f}:d={MUSIC_FADE_OUT}")  # fmt: skip
    return (
        f"{tracks}concat=n={music_inputs}:v=0:a=1,aresample=48000,aformat=channel_layouts=stereo,"
        f"{shape},volume='{volume}':eval=frame"
    )


def music_stem(trip: Trip, items: list[Item], resolved, destination: Path, fps: Fraction) -> Path:
    """The music of an edit on its own, as it plays in the render (levels, ducking under speech,
    ending), as a WAV as long as the edit: for exporting to an editor."""
    from .soundtrack import Mix

    plan, fps = edit_plan(trip, items, fps)
    duration = sum(frames_of(p, fps) for _, p in plan) / float(fps)
    spans = speech_in_edit(trip, plan, items, fps)
    if resolved.generated:
        music = [t.path for t in resolved.tracks]
    else:
        music = music_sequence(resolved.tracks, duration)
    music_gain, _, under_speech = (round(g, 4) for g in (resolved.mix or Mix()).gains())
    volume = gain_expression(spans, under_speech, music_gain)
    branch = music_branch(duration, volume, len(music), resolved if resolved.generated else None)
    # Input 0 is a placeholder so the music inputs are numbered from 1, as in the render.
    args = [tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=48000"]
    for path in music:
        args += ["-i", str(path)]
    graph = f"{branch},apad=whole_dur={duration:.3f},atrim=0:{duration:.3f}[out]"
    partial = destination.with_name(destination.stem + ".partial.wav")
    run([*args, "-filter_complex", graph, "-map", "[out]", "-c:a", "pcm_s24le", "-ar", "48000",
         str(partial)])  # fmt: skip
    os.replace(partial, destination)
    return destination


def music_sequence(tracks: list, duration: float) -> list[Path]:
    """Music files in play order, repeated until they cover the edit."""
    total = sum(t.duration for t in tracks)
    if not tracks or total <= 0:
        return []
    out, covered = [], 0.0
    while covered < duration:
        for t in tracks:
            out.append(t.path)
            covered += t.duration
            if covered >= duration:
                break
    return out


def finish_sound(
    joined: Path, destination: Path, duration: float, spans: list[tuple[float, float]], tracks
) -> None:
    """Mix the edit's sound with optional music and normalize loudness in two passes. tracks: a
    soundtrack.Resolved, or (as before) a list of music tracks played with the usual mix."""
    from .soundtrack import Resolved

    resolved = tracks if isinstance(tracks, Resolved) else Resolved(list(tracks or []), None)
    if resolved.generated:
        music = [t.path for t in resolved.tracks]  # played once, never looped
    else:
        music = music_sequence(resolved.tracks, duration)
    inputs = [tool("ffmpeg"), "-v", "error", "-y", "-i", str(joined)]
    for path in music:
        inputs += ["-i", str(path)]
    graph = mix_graph(duration, spans, len(music), resolved.mix,
                      resolved if resolved.generated else None)  # fmt: skip
    stats = measure_loudness(inputs, graph)
    if stats:
        apply = (
            f"loudnorm=I={LOUDNESS}:TP={TRUE_PEAK}:LRA=11:linear=true"
            f":measured_I={stats['input_i']}:measured_TP={stats['input_tp']}"
            f":measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}"
            f":offset={stats['target_offset']}"
        )
    else:
        # Nothing measurable (an edit of silent clips): normalizing silence would only produce
        # invalid samples, so the sound is left as it is.
        apply = "anull"
    partial = destination.with_suffix(".partial.mp4")
    run([
        *inputs, "-filter_complex", f"{graph};[mix]{apply},aresample=48000[out]",
        "-map", "0:v:0", "-map", "[out]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-f", "mp4", str(partial),
    ])  # fmt: skip
    os.replace(partial, destination)


def measure_loudness(inputs: list[str], graph: str) -> dict | None:
    """First loudnorm pass. Its statistics are printed at info level, so this pass must not run
    with the "-v error" the other commands use."""
    measure = f"loudnorm=I={LOUDNESS}:TP={TRUE_PEAK}:LRA=11:print_format=json"
    args = list(inputs)
    args[args.index("-v") + 1] = "info"
    result = run(
        [*args, "-nostats", "-hide_banner", "-filter_complex", f"{graph};[mix]{measure}[out]",
         "-map", "[out]", "-f", "null", "-"],
        text=True,
    )  # fmt: skip
    return _loudnorm_stats(result.stderr)


def _loudnorm_stats(stderr: str) -> dict | None:
    match = re.search(r"\{\s*\"input_i\".*?\}", stderr, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    keys = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    try:
        values = [float(data[k]) for k in keys]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None  # silence: nothing to measure
    return data


# --- whole edit ------------------------------------------------------------------------------


def _first_working(commands) -> None:
    """Run the first command that succeeds; raise the last error if none does."""
    error = None
    for make in commands:
        try:
            run(make())
            return
        except RuntimeError as e:
            error = e
    raise error


def settings_key(items: list[Item], out: Output, fps: Fraction, plan=None) -> str:
    data = json.dumps(
        {
            "items": [asdict(i) for i in items],
            "out": asdict(out),
            "fps": str(fps),
            "v": RENDER_VERSION,
            **({"pieces": plan_key(plan)} if plan is not None and any(
                p.speed != 1 for _, p in plan) else {}),
        },
        sort_keys=True,
    )  # fmt: skip
    return hashlib.sha1(data.encode()).hexdigest()[:12]


def render(
    trip: Trip,
    items: list[Item],
    destination: Path,
    out: Output | None = None,
    progress: Progress | None = None,
    music=None,
) -> Path:
    """Render the items in order. music: what to play under the edit, a soundtrack.Resolved
    (see soundtrack.resolve), or a list of tracks (see music.load) played with the usual mix."""
    out = out or Output()
    plan, fps = edit_plan(trip, items, Fraction(out.fps) if out.fps != "auto" else None)
    if not plan:
        raise ValueError("the edit is empty")
    key = settings_key(items, out, fps, plan)
    work = trip.root / "cache" / "render" / key
    work.mkdir(parents=True, exist_ok=True)
    gpu = encoder_works("h264_nvenc")

    def make(item: str) -> None:
        check_cancelled()
        n = int(item)
        piece = plan[n][1]
        final = work / f"{n:05d}.mp4"
        partial = work / f"{n:05d}.partial.mp4"
        if final.is_file():
            # Made by an earlier render with the same settings (the folder's key): rendering again
            # with only different sound redoes just the sound.
            return
        if piece.photo:
            cover = piece.crop_x if out.vertical else None
            name = f"{piece.key}_{2 * out.width}x{2 * out.height}_{cover}.png"
            still = photos.fitted_still(
                piece.source, trip.root / "cache" / "stills" / name,
                2 * out.width, 2 * out.height, 0.0, cover_x=cover,
            )  # fmt: skip
            _first_working([
                lambda: still_command(piece, out, fps, partial, still),
                lambda: still_command(piece, out, fps, partial, still, cpu=True),
            ])  # fmt: skip
            os.replace(partial, final)
            return
        # GPU decode and encode first; then CPU decode (codecs NVDEC lacks); then x264 (sizes or
        # settings NVENC refuses, such as frames narrower than about 145 pixels).
        margin = seek_margin(piece)
        _first_working([
            lambda: clip_command(piece, out, fps, partial, gpu_decode=gpu, margin=margin),
            lambda: clip_command(piece, out, fps, partial, gpu_decode=False, margin=margin),
            lambda: clip_command(piece, out, fps, partial, gpu_decode=False, cpu=True,
                                 margin=margin),
        ])  # fmt: skip
        os.replace(partial, final)

    # A clip a resumed job counts as done may have been deleted since (reelsift clean).
    gone = [str(n) for n in range(len(plan)) if not (work / f"{n:05d}.mp4").is_file()]
    Job(trip, "render", {"key": key}, [str(n) for n in range(len(plan))], redo=gone).run(
        make, progress
    )
    missing = [n for n in range(len(plan)) if not (work / f"{n:05d}.mp4").is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} clips failed to render; see reelsift status")
    check_cancelled()
    listing = work / "concat.txt"
    listing.write_text("".join(f"file '{n:05d}.mp4'\n" for n in range(len(plan))), encoding="utf-8")
    joined = work / "joined.mp4"
    run([
        tool("ffmpeg"), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
        "-c", "copy", "-f", "mp4", str(joined),
    ])  # fmt: skip
    duration = sum(frames_of(p, fps) for _, p in plan) / float(fps)
    destination.parent.mkdir(parents=True, exist_ok=True)
    spans = speech_in_edit(trip, plan, items, fps)
    check_cancelled()
    finish_sound(joined, destination, duration, spans, music if music is not None else [])
    return destination
