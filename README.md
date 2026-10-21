# Reelsift

Turn hours of footage from a trip, a ride or a day out into short videos worth sharing, on your own computer, with you choosing what stays.

Bring together the videos and photos from your GoPro, phone, dashcam or other cameras. Reelsift watches all of it once and suggests clips for videos at several lengths. Tell it in plain words what you want more or less of, check its choices in your browser, then save the videos. Your original files are never changed.

- **Private.** Your videos and photos are processed on your computer and never uploaded. No account, no subscription.
- **The whole outing at once.** Footage from several folders and cameras goes on one timeline, including long recordings a camera split into many files.
- **Several videos from one outing.** By default a 30-second teaser, a 3-minute highlight and a 10-minute video of the whole day, or any lengths you choose. Landscape 16:9, or vertical 9:16 for phones.
- **Say what matters.** "More of the kids playing and the lake, less road." Each video has its own description, which Reelsift drafts from your footage and you can rewrite. Changing it never means scanning the footage again.
- **You have the final say.** Keep, drop, trim or add clips in a review app in your browser, undo anything, and try again before a single video is made.
- **Music.** Use your own songs, or have music generated on your computer for each video's length, genre and mood. It's mixed with the original sound at levels you choose, and quieter when people talk.
- **Rides, without the waiting.** A whole ride or drive from one camera, in order, with red lights and stops cut out and the rest sped up to a length you choose. GoPro and dashcam GPS helps; without GPS, the picture shows when you stopped.
- **Finish it in your editor.** Export a timeline for DaVinci Resolve, Final Cut Pro and other editors. Free and open source (Apache-2.0).

**Alpha (0.1.0).** Reelsift works end to end on real footage: the largest test so far was an 85-minute, 23-file 4K ride on one action camera. Only one camera model is verified yet, and clip choices can miss moments that matter to you; that's what the review is for. Reports from your cameras and computers are very welcome.

## What you need

- **Windows 11.** Linux and macOS aren't supported yet. Setup uses a few terminal commands; reviewing and saving videos happen in your browser.
- **An NVIDIA graphics card** (RTX 20-series or newer, driver 580 or newer) to analyze footage in reasonable time. Cards with 7 GB of memory or more get scene descriptions. Without one, analysis runs on the processor, much more slowly and without scene descriptions or drafted descriptions.
- **[Git](https://git-scm.com/downloads), [uv](https://docs.astral.sh/uv/getting-started/installation/) and [FFmpeg](https://ffmpeg.org/download.html)** 5.1 or newer. uv installs Python by itself.
- **Disk space.**
  - About 5 GB for the software.
  - 2 to 23 GB for the analysis models, depending on your card.
  - Room for each trip's preview copies. 50 GB free is comfortable.
  - **Generated music (optional):** another 20 GB, or 40 GB for the larger model, and a card with at least 14 GB of memory.
- **Internet**, for setup only. After that, Reelsift works offline.

## Quick start

Install Git, uv and FFmpeg first:

```bash
winget install --id Git.Git -e --accept-source-agreements --accept-package-agreements
winget install --id astral-sh.uv -e --accept-source-agreements --accept-package-agreements
winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
```

Open a new terminal (so it finds what you just installed), then set up Reelsift:

```bash
git clone https://github.com/hexkDotCom/reelsift.git
cd reelsift
uv sync --extra ml
uv run reelsift doctor
uv run reelsift models pull
```

- `doctor` checks your computer and says what to fix; a line marked `FAIL` needs fixing before you go on.
- `models pull` shows each model's license and size and asks before downloading.
- Run every command below from this `reelsift` folder.

Now make your first video. Replace `D:/videos/my-trip` with the folder that holds your footage. `D:/reelsift-trips/my-trip` becomes a new folder where Reelsift keeps its work (a "trip"). Put quotes around paths with spaces.

```bash
uv run reelsift init D:/reelsift-trips/my-trip --footage D:/videos/my-trip --length 30
uv run reelsift prepare D:/reelsift-trips/my-trip
uv run reelsift analyze D:/reelsift-trips/my-trip
uv run reelsift select D:/reelsift-trips/my-trip
uv run reelsift review D:/reelsift-trips/my-trip
```

1. `init` makes the trip, here with one 30-second video. Leave out `--length 30` to get the three default lengths, and repeat `--footage` for footage in several folders. Keep each camera's files in their own folder.
2. `prepare` finds your footage and makes small preview copies.
3. `analyze` is the slow step: on an RTX 5090, 85 minutes of 4K footage takes about 11 minutes. It only happens once per trip.
4. `select` chooses the clips.
5. `review` opens the review app in your browser:
   - Look through the clips, keep or drop them, trim them, or add moments it missed.
   - Rewrite the description and choose music.
   - Click **Render 16:9** or **Render 9:16**, then **Download** when it's done.
   - The video is also saved in the trip's `output` folder.
   - Press Ctrl+C in the terminal when you're finished.

If there isn't enough good footage for the length you asked for, the video comes out shorter, and `select` tells you so.

### Other ways to use it

- **No review, just the videos:**

  ```bash
  uv run reelsift make D:/reelsift-trips/my-trip
  ```

  This does every step and saves every video to the trip's `output` folder. `--vertical` adds 9:16 versions, and `--music synthwave --mood "night drive"` adds generated music.
- **A ride or a drive:**

  ```bash
  uv run reelsift make D:/reelsift-trips/my-ride --only-ride --ride-length 3:00
  ```

  This turns dashcam or bike footage into a 3-minute ride. It skips the slow analysis, so `uv sync` without `--extra ml` and no `models pull` are enough. Or build the ride in the review app's Ride tab, where you can see which stops were cut and put any back.
- **Let an AI agent do it:** Claude Code, Codex and others can install Reelsift and run it for you, with or without your review. [AGENTS.md](AGENTS.md) tells them how.

Every command, option and setting is in [docs/usage.md](docs/usage.md).

## Current limits

- **Where it has been tested.** Only the DJI Osmo Action 6 is verified with sample files, and only one computer has been measured: an RTX 5090 on Windows 11. Smaller cards should work but are unmeasured. Linux is next: the code is written for it and its tests run there, but nobody has used it on Linux yet.
- **Clip choices.**
  - Reelsift doesn't yet know your best moments: the review is there to fix its choices.
  - It understands "more people", but not "more of Emma".
- **Cuts and framing.**
  - Videos use hard cuts, with no transitions.
  - 9:16 videos are cropped from the landscape picture.
  - Rides aren't stabilized.
- **Rides** use one camera, from 1x to 64x. Stops it isn't sure about are kept; you can drop them in the review app.
- **Generated music** is at most 10 minutes long, or 8 minutes with the smaller model.
- **Exported timelines** carry the cuts, speed changes and music. Crops, photo zooms and the finished sound mix are not exported, and importing has only been checked for file structure, not yet in each editor.
- **Long trips.** Trips of many hours should work but haven't been benchmarked yet.

More in [docs/roadmap.md](docs/roadmap.md).

## Supported devices

| Device | Status | What Reelsift uses |
|---|---|---|
| DJI Osmo Action 6 | Verified | Camera `.LRF` proxies; UTC start time plus local time from the file name, which gives the time zone |
| Other DJI cameras and drones with `DJI_<time>_<n>_D` file names | Recognized, not verified | Same as above |
| GoPro HERO, MAX, Fusion | Supported | Chapters joined into one recording (GOPR/GPnn and GX/GHnn naming); `.LRV` proxies; the camera's local clock; UTC, time zone and speed from GPS on models that have it. 360° footage is used as the camera wrote it; it isn't reframed. |
| Dashcams that record GPS in the video | Supported for rides | Speed from the GPS lines in the file, to find stops |
| Cycliq Fly12 (and Fly6) | Supported | 5-minute segments joined into continuous rides, including locked `LCK_` segments; segment times corrected (the camera writes when each segment ended) |
| iPhone and Android phones | Supported | Videos and photos from one phone are one camera; iPhone videos and phone photos give the time zone |
| Photos (JPEG, PNG, WebP, HEIC, AVIF) | Supported | EXIF time and zone offset, orientation, camera model; bursts count once |
| Other video files (MP4, MOV, MKV, WebM, AVI, MPEG, MTS/M2TS, 3GP, WMV, FLV, DV, MXF) | Generic | `creation_time` read as UTC; a local time in the file name (such as `20200705_150405.mp4`) gives the zone, or you set it per camera |

A device counts as verified once short sample files from it are in the test suite. If you have a camera or phone that isn't listed, a few seconds of footage from it is one of the most useful contributions you can make.

## How it works

Reelsift separates **analysis** from **selection**.
- **Analysis** looks at the pictures and sound once and stores what it finds.
- **Selection** only reads those results. Changing a description, a length or a decision chooses the clips again without analyzing the footage again, in about a second; a new description or search may first load a local model.

Models score and describe short stretches of footage; ordinary code decides where the cuts go.

Each video has a **style**:
- The shortest is a dynamic teaser: the strongest, most active moments, with short cuts.
- The longest covers the day from start to finish, with people and conversations.
- Any in between are highlights.

After analysis, the language model reads its own descriptions of the footage and drafts a description for each video in its style. That text becomes sliders you can see and change.

| Job | Model (large preset) | License |
|---|---|---|
| Picture embeddings and visual search | SigLIP 2 so400m | Apache-2.0 |
| Scene descriptions and reading preferences | Qwen3-VL-8B-Instruct | Apache-2.0 |
| Speech | Whisper large-v3-turbo via faster-whisper | MIT |
| Music (optional) | ACE-Step 1.5 | MIT |

More documentation:
- [docs/usage.md](docs/usage.md): commands, settings and model presets.
- [docs/design.md](docs/design.md): the design.
- [docs/design-notes.md](docs/design-notes.md): the decisions behind the details.
- [docs/benchmarks.md](docs/benchmarks.md): speed and memory.
- [docs/research.md](docs/research.md): how existing tools compare.

## Contributing

Bug reports, sample footage from devices that aren't verified yet, and pull requests are welcome. Please don't attach private footage to public issues. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, tests and how to add a device.

## License

Apache-2.0; see [LICENSE](LICENSE). The models Reelsift downloads and the libraries it uses have their own licenses, listed in [NOTICE](NOTICE). FFmpeg isn't bundled; you install your own build.
