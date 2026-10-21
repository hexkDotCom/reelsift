"""Exporting an edit as a timeline for an editor: FCPXML (DaVinci Resolve, Final Cut Pro) and
OpenTimelineIO (DaVinci Resolve 18.5+, Premiere through adapters).

The contract is basic cuts: which part of which original file goes where on the timeline, with
its own sound. The edit's music, if it has any, comes as a WAV on its own audio track, at its
levels and ducked under speech as in the render; the clips' sound is not lowered under it, and
loudness, crops, photo zooms and colour conversion are not exported: they belong to Reelsift's
render or to your finishing in the editor.

Clips reference the original files at full resolution. Their positions use each file's own
timecode when it has one (DJI and GoPro cameras write one), because editors match media by it.
Timeline positions use the same frame counts as Reelsift's render, so an exported timeline and a
rendered edit cut at the same frames.
"""

import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from xml.sax.saxutils import quoteattr

from .media import parse_probe
from .project import Trip
from .render import Output, frames_of
from .select import Item

FCPXML_VERSION = "1.10"


@dataclass
class Clip:
    name: str
    path: Path
    photo: bool
    src_fps: Fraction
    tc_start: int
    """The file's first frame, in source frames of its timecode (0 without timecode)."""
    file_frames: int
    in_frame: int
    """First frame used, counted from the file's first frame."""
    src_frames: int
    seq_frames: int
    width: int
    height: int
    has_audio: bool
    speed: Fraction = Fraction(1)
    """Played this much faster; src_frames is the footage used, seq_frames its time in the
    edit."""


def timecode_frames(tc: str, fps: Fraction) -> int:
    """Frames from HH:MM:SS:FF (or HH:MM:SS;FF, drop-frame for 29.97/59.94)."""
    match = re.fullmatch(r"(\d+):(\d+):(\d+)([:;.])(\d+)", tc.strip())
    if not match:
        return 0
    h, m, s, sep, f = match.groups()
    h, m, s, f = int(h), int(m), int(s), int(f)
    nominal = round(fps)
    frames = ((h * 60 + m) * 60 + s) * nominal + f
    if sep == ";" and fps.denominator == 1001:
        drop = 2 * round(fps / 30)
        minutes = h * 60 + m
        frames -= drop * (minutes - minutes // 10)
    return frames


def _timecode(probe_json: dict) -> str | None:
    for stream in probe_json.get("streams", []):
        tc = stream.get("tags", {}).get("timecode")
        if tc:
            return tc
    return probe_json.get("format", {}).get("tags", {}).get("timecode")


def plan(trip: Trip, items: list[Item], fps: Fraction) -> list[Clip]:
    import json

    from .render import edit_plan

    clips = []
    found, _ = edit_plan(trip, items, fps)
    for _, piece in found:
        seq_frames = frames_of(piece, fps)
        if piece.photo:
            clips.append(Clip(piece.source.name, piece.source, True, fps, 0, 0, 0, seq_frames,
                              seq_frames, piece.width, piece.height, False))  # fmt: skip
            continue
        row = trip.db.execute(
            "SELECT probe_json FROM assets WHERE path = ?", (str(piece.source),)
        ).fetchone()
        data = json.loads(row["probe_json"])
        video = parse_probe(piece.source, data).main_video
        src_fps = piece.fps
        tc = _timecode(data)
        duration = video.duration or Fraction(0)
        clips.append(
            Clip(
                name=piece.source.name,
                path=piece.source,
                photo=False,
                src_fps=src_fps,
                tc_start=timecode_frames(tc, src_fps) if tc else 0,
                file_frames=max(1, round(duration * src_fps)),
                in_frame=round(piece.start * src_fps),
                src_frames=max(1, round(Fraction(seq_frames) / fps * piece.speed * src_fps)),
                seq_frames=seq_frames,
                width=piece.width,
                height=piece.height,
                has_audio=piece.has_audio,
                speed=piece.speed,
            )
        )
    return clips


def _t(frames: int, fps: Fraction) -> str:
    """An FCPXML time: a rational number of seconds."""
    value = Fraction(frames) / fps
    if value == 0:
        return "0s"
    return (
        f"{value.numerator}s"
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}s"
    )


def fcpxml(
    clips: list[Clip], name: str, out: Output, fps: Fraction, music: Path | None = None,
    sound_off: bool = False,
) -> str:  # fmt: skip
    formats: dict[tuple, str] = {}
    lines: list[str] = []

    def format_id(rate: Fraction | None, width: int, height: int) -> str:
        key = (rate, width, height)
        if key not in formats:
            fid = f"r{len(formats) + 1}"
            formats[key] = fid
            duration = f' frameDuration="{_t(1, rate)}"' if rate else ""
            lines.append(
                f'    <format id="{fid}"{duration} width="{width}" height="{height}"'
                ' colorSpace="1-1-1 (Rec. 709)"/>'
            )
        return formats[key]

    seq_format = format_id(fps, out.width, out.height)
    assets: dict[Path, str] = {}
    asset_lines: list[str] = []
    for c in clips:
        if c.path in assets:
            continue
        aid = f"a{len(assets) + 1}"
        assets[c.path] = aid
        uri = quoteattr(c.path.resolve().as_uri())
        if c.photo:
            fid = format_id(None, c.width, c.height)
            asset_lines.append(
                f'    <asset id="{aid}" name={quoteattr(c.name)} start="0s" duration="0s"'
                f' hasVideo="1" format="{fid}">\n'
                f'      <media-rep kind="original-media" src={uri}/>\n    </asset>'
            )
        else:
            fid = format_id(c.src_fps, c.width, c.height)
            audio = (' hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000"'
                     if c.has_audio else "")  # fmt: skip
            asset_lines.append(
                f'    <asset id="{aid}" name={quoteattr(c.name)}'
                f' start="{_t(c.tc_start, c.src_fps)}" duration="{_t(c.file_frames, c.src_fps)}"'
                f' hasVideo="1" format="{fid}"{audio}>\n'
                f'      <media-rep kind="original-media" src={uri}/>\n    </asset>'
            )
    total = sum(c.seq_frames for c in clips)
    connected = ""
    if music is not None:
        mid = f"a{len(assets) + 1}"
        asset_lines.append(
            f'    <asset id="{mid}" name={quoteattr(music.name)} start="0s"'
            f' duration="{_t(total, fps)}" hasAudio="1" audioSources="1" audioChannels="2"'
            f' audioRate="48000">\n'
            f'      <media-rep kind="original-media" src={quoteattr(music.resolve().as_uri())}/>\n'
            "    </asset>"
        )
    spine = []
    offset = 0
    for n, c in enumerate(clips):
        aid = assets[c.path]
        attrs = (
            f'ref="{aid}" offset="{_t(offset, fps)}" name={quoteattr(c.name)}'
            f' duration="{_t(c.seq_frames, fps)}"'
        )
        start = "0s" if c.photo else _t(c.tc_start + c.in_frame, c.src_fps)
        inside = ""
        if c.speed != 1 and not c.photo:
            # Retimed: clip-local time runs speed times slower than the media. The time map goes
            # over the whole media from its first frame, linearly; start is in retimed time.
            t0 = Fraction(c.tc_start) / c.src_fps
            length = Fraction(c.file_frames) / c.src_fps
            start = _s(t0 + Fraction(c.in_frame) / c.src_fps / c.speed)
            inside += (f'<timeMap><timept time="{_s(t0)}" value="{_s(t0)}" interp="linear"/>'
                       f'<timept time="{_s(t0 + length / c.speed)}" value="{_s(t0 + length)}"'
                       ' interp="linear"/></timeMap>')  # fmt: skip
        if sound_off and c.has_audio and not c.photo:
            inside += '<adjust-volume amount="-96dB"/>'

        if music is not None and n == 0:
            # The music hangs under the first clip, starting where the timeline starts (a
            # connected clip's offset is in its parent's own time).
            connected = (f'<asset-clip ref="{mid}" lane="-1" offset="{start}"'
                         f' name={quoteattr(music.name)} duration="{_t(total, fps)}" start="0s"/>')  # fmt: skip
        tag = "video" if c.photo else "asset-clip"
        extra = "" if c.photo else ' tcFormat="NDF"'
        if connected and n == 0:
            connected = connected.replace(f'offset="{_t(c.tc_start + c.in_frame, c.src_fps)}"',
                                          f'offset="{start}"') if c.speed != 1 else connected  # fmt: skip
            inside += connected
        if inside:
            spine.append(f'        <{tag} {attrs} start="{start}"{extra}>{inside}</{tag}>')
        else:
            spine.append(f'        <{tag} {attrs} start="{start}"{extra}/>')
        offset += c.seq_frames
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<!DOCTYPE fcpxml>",
            f'<fcpxml version="{FCPXML_VERSION}">',
            "  <resources>",
            *lines,
            *asset_lines,
            "  </resources>",
            "  <library>",
            '    <event name="Reelsift">',
            f"      <project name={quoteattr(name)}>",
            f'        <sequence format="{seq_format}" duration="{_t(offset, fps)}" tcStart="0s"'
            ' tcFormat="NDF" audioLayout="stereo" audioRate="48k">',
            "          <spine>",
            *("  " + s for s in spine),
            "          </spine>",
            "        </sequence>",
            "      </project>",
            "    </event>",
            "  </library>",
            "</fcpxml>",
            "",
        ]
    )


def _s(value: Fraction) -> str:
    """An FCPXML time from seconds."""
    value = Fraction(value).limit_denominator(1_000_000)
    if value == 0:
        return "0s"
    return (
        f"{value.numerator}s"
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}s"
    )


def otio_timeline(
    clips: list[Clip], name: str, fps: Fraction, music: Path | None = None, sound_off: bool = False
):
    import opentimelineio as otio

    rate = float(fps)
    timeline = otio.schema.Timeline(name=name)
    timeline.global_start_time = otio.opentime.RationalTime(0, rate)
    video = otio.schema.Track(name="Video", kind=otio.schema.TrackKind.Video)
    audio = otio.schema.Track(name="Audio", kind=otio.schema.TrackKind.Audio)
    for c in clips:
        uri = c.path.resolve().as_uri()
        if c.photo:
            ref = otio.schema.ExternalReference(target_url=uri)
            source = otio.opentime.TimeRange(
                otio.opentime.RationalTime(0, rate), otio.opentime.RationalTime(c.seq_frames, rate)
            )
            video.append(otio.schema.Clip(name=c.name, media_reference=ref, source_range=source))
            audio.append(otio.schema.Gap(source_range=source))
            continue
        src_rate = float(c.src_fps)
        available = otio.opentime.TimeRange(
            otio.opentime.RationalTime(c.tc_start, src_rate),
            otio.opentime.RationalTime(c.file_frames, src_rate),
        )
        # The duration is the clip's exact length on the timeline, expressed at the source's
        # rate (possibly a fractional number of source frames), so cuts stay on the timeline's
        # frame grid when source and timeline rates differ.
        source = otio.opentime.TimeRange(
            otio.opentime.RationalTime(c.tc_start + c.in_frame, src_rate),
            otio.opentime.RationalTime(c.seq_frames, rate).rescaled_to(src_rate),
        )
        for track in (video, audio):
            ref = otio.schema.ExternalReference(target_url=uri, available_range=available)
            if track is audio and not c.has_audio:
                track.append(otio.schema.Gap(source_range=source))
            else:
                clip = otio.schema.Clip(name=c.name, media_reference=ref, source_range=source)
                if c.speed != 1:
                    # The clip lasts its time in the edit; the warp says it plays speed x.
                    clip.effects.append(otio.schema.LinearTimeWarp(time_scalar=float(c.speed)))
                    clip.metadata["reelsift"] = {"speed": float(c.speed),
                                                 "source_frames": c.src_frames}  # fmt: skip
                if track is audio and sound_off:
                    clip.enabled = False
                track.append(clip)
    timeline.tracks.append(video)
    timeline.tracks.append(audio)
    if music is not None:
        total = sum(c.seq_frames for c in clips)
        span = otio.opentime.TimeRange(otio.opentime.RationalTime(0, rate),
                                       otio.opentime.RationalTime(total, rate))  # fmt: skip
        ref = otio.schema.ExternalReference(target_url=music.resolve().as_uri(),
                                            available_range=span)  # fmt: skip
        track = otio.schema.Track(name="Music", kind=otio.schema.TrackKind.Audio)
        track.append(otio.schema.Clip(name=music.name, media_reference=ref, source_range=span))
        timeline.tracks.append(track)
    return timeline


def export(
    trip: Trip, items: list[Item], name: str, folder: Path, out: Output | None = None,
    music=None,
) -> list[Path]:  # fmt: skip
    """Write <name>.fcpxml and <name>.otio into folder (and <name>-music.wav when music, a
    soundtrack.Resolved, has tracks); returns their paths."""
    import opentimelineio as otio

    from .render import edit_plan

    out = out or Output()
    found, fps = edit_plan(trip, items, Fraction(out.fps) if out.fps != "auto" else None)
    if not found:
        raise ValueError("the edit is empty")
    clips = plan(trip, items, fps)
    folder.mkdir(parents=True, exist_ok=True)
    stem = None
    if music is not None and music.tracks:
        from .render import music_stem

        stem = music_stem(trip, items, music, folder / f"{name}-music.wav", fps)
    sound_off = bool(music is not None and music.mix is not None and music.mix.sound_gain == 0)
    fcp = folder / f"{name}.fcpxml"
    fcp.write_text(fcpxml(clips, name, out, fps, stem, sound_off), encoding="utf-8")
    ot = folder / f"{name}.otio"
    otio.adapters.write_to_file(otio_timeline(clips, name, fps, stem, sound_off), str(ot))
    return [fcp, ot] + ([stem] if stem else [])
