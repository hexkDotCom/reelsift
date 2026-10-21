# Changelog

All notable changes are listed here. Versions follow [semantic versioning](https://semver.org/); before 1.0, a minor version may change behaviour or the trip format (older trip folders are upgraded automatically when opened).

## 0.1.0 — 2026-10-21, first public release

- Windows 11 (Linux is next).
- Footage from several cameras and phones: chapter grouping, camera proxies or generated proxies, one timeline across cameras, time sync from metadata, file names, per-camera time zones and manual anchors, without needing GPS.
- Devices: DJI Osmo Action 6; GoPro HERO, MAX and Fusion (chapters, `.LRV` proxies, local clock, GPS time); Cycliq Fly12 (continuous segments joined); iPhone and Android phones. Photos (JPEG, PNG, WebP, HEIC) with EXIF time and burst grouping. Other common video files (MKV, WebM, AVI, MPEG, MTS/M2TS and more) are handled generically, with the zone from a local time in its file name when there is one.
- Analysis with local models (SigLIP 2, Qwen3-VL, faster-whisper) plus optical-flow motion, sharpness, exposure and loudness; model presets for different GPUs.
- Independent edits at any list of lengths (30 s, 3 min and 10 min by default), each with its own style (dynamic preview, highlights or story), clip lengths, decisions and description.
- Descriptions drafted per edit from the footage by the local language model, steered by each edit's style, and editable in plain language; turned into sliders, free-text visual searches and exclusions; re-selection in about a second.
- A local review app: storyboard, previews, keep, drop, trim, add, suggestions, candidate browser, undo, revisions, approve, 9:16 framing, render (with a Download button for the finished video) and export.
- Music for each edit, generated locally with ACE-Step 1.5 (`reelsift music install`, then `music generate` with a genre and mood) or from your own files; levels for the music, the clips' sound under it and the drop under speech, per edit; cuts fitted to the beat on request; a Music panel in the review app with a quick preview.
- Rides: the whole trip from one camera in time order, stops cut out (from GPS speed on GoPros and dashcams, or the picture) and the rest sped up to a fixed speed or a length; corrections in the app's Ride tab; `reelsift ride`.
- A clip sound level (down to off) next to the music level, so the music can play alone.
- Frame-accurate rendering to 16:9 or 9:16 MP4 with HDR tone mapping, ducking under speech and loudness normalization.
- FCPXML 1.10 and OpenTimelineIO export with source timecode, and the edit's music as a WAV on its own track.
- Footage in WebM, AVI, MPEG-2, MPEG-TS/AVCHD (`.mts`, `.m2ts`), 3GP, WMV, FLV, DV and MXF, with interlaced video deinterlaced; files of other types are listed once instead of ignored silently.
- Phone videos and photos are one camera, with the time zone taken from either; full-range phone video and photos keep correct black and white levels.
- `reelsift init --length` chooses a trip's edit lengths; `select` warns when an edit comes out much shorter than its length.
- `reelsift make` for a fully automatic run with no review, including the ride (`--ride`, `--only-ride`), generated music (`--music`) and 9:16 versions (`--vertical`); `render --ride`.
- `AGENTS.md`: how an AI agent installs and runs Reelsift, with a review by a person in the app or fully automatically.
- `reelsift doctor`, `models pull` (shows each license first), `status`, `clean`, and resumable steps throughout.
