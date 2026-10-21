# Roadmap

Known work, roughly in order of usefulness. Each item says why it waits. Ideas and offers of help are welcome as issues.

## Footage and devices

- **More of GoPro's telemetry:** its motion sensors, GPS track and the face and scene detection newer models record could replace or add to optical flow.
- **Upside-down footage.** A camera mounted upside down without its flip setting (seen on a handlebar-mounted Cycliq) gives upside-down video with nothing in the file to say so. A per-camera rotation setting, or detecting it from the picture, would fix it.
- **GPS tracks** (`.fit`, `.gpx`) from a bike computer, watch or phone app, and GPS recorded by cameras: real time, location and speed for each moment. GPS stays optional.
- **DJI Osmo Action 6 motion and GPS data.** Existing parsers read only the header of its `djmd` stream; decoding it would give real motion and location instead of optical flow.
- **Pixel Motion Photos and iPhone Live Photos.** They are used as stills; the short video inside or beside them isn't used yet.
- **Telling cameras apart by serial number.** Cameras are told apart by folder; serial numbers would also work when files from two cameras share a folder.
- **Linux.** The code is written for it and its tests run there in CI, but nobody has used it on Linux with real footage and a GPU yet. Reports are welcome.
- **macOS.** Needs a model runtime and encoder path without CUDA.

## Review and editing

- **Rides, next steps:** stabilization (optional; vidstab on the sampled frames, to measure first), quick draft renders from the proxies, several cameras filling each other's gaps, keeping interesting stops automatically (speech, scene descriptions), slower through interesting parts, and GPS speed from more cameras.
- **Choosing a different photo from a burst.**
- **Drafts in your own language.** Drafted descriptions are in English; you can rewrite them in any language the model reads.
- **"Look closer":** run the vision model on a few candidates you choose, for footage the default pass described poorly.
- **A map** of where each clip was filmed, once GPS tracks are supported.
- **Recognizing specific people** ("more of Emma") would suit days with kids and friends. Common open face models have non-commercial licenses, so this waits for a permissively licensed one.
- **Crossfades and transitions.** Cuts are hard cuts, to keep clip-by-clip rendering and beat-matched cuts simple.

## Output

- **Checking timeline import in more editors.** The FCPXML and OTIO files pass structural tests; reports from DaVinci Resolve, Final Cut Pro and others are welcome.
- **Music:** listening tests of the default levels on more trips; a measured 12 GB profile (2B model) and smaller GPUs; edits longer than 10 minutes (joining generated parts); regenerating just the ending or a drop (ACE-Step's repaint); a live preview of the mix in the app instead of a quick render.
- **A 21 ms offset between video and audio after joining clips.** FFmpeg's concat demuxer shifts the video by the AAC encoder's priming samples. It is below what anyone can notice but should be zero.
- **Audio that starts later than the video in speech analysis and previews.** Rendering keeps each stream's own start, but speech times and proxy previews assume both streams start together. Camera files start them together; edited or remuxed files may not.

## Quality and performance

- **Labelled moments for tuning.** `reelsift eval` exists, but the default weights and prompt bank haven't been tuned against labelled footage yet.
- **Medium, small and CPU presets** need measuring on matching hardware.
- **Multi-hour footage.** Stream speech audio in chunks instead of decoding a whole chapter, compute music onsets in chunks, update similarity penalties incrementally, and save vision-model answers in smaller checkpoints. Waits for a real four-hour trip to benchmark; 85 minutes runs comfortably.
- **One shared media timeline plan.** Preview, rendering, both export formats and speech ducking each work out chapter splits, frame rates and timecode themselves. One shared plan would remove that duplication and could use real presentation timestamps for variable-frame-rate video, whose exported timelines may be off by a frame today (rendering is correct).
- **Browser tests.** The frontend's response-ordering rules are tested with Node.js against a stand-in page (`tests/js`); playback and layout in a real browser would need Playwright in CI.
- **A PyPI package.** PyTorch's CUDA builds don't install cleanly through pip, so installing from source with uv is the supported way for now.
