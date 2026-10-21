# Using Reelsift

The full reference: every step, command and setting. For a first video, start with the [README](../README.md).

Every command runs from the `reelsift` folder as `uv run reelsift …`. `TRIP` is a trip folder made with `reelsift init`.

## Setup details

`uv sync --extra ml` installs Reelsift with PyTorch and the analysis models' libraries. On Windows the `ml` extra installs PyTorch built for CUDA 13.0. Leave out `--extra ml` to install without them. Reading footage, the timeline, rides, rendering and export work without them; analysis doesn't.

Run setup from a normal terminal. Some sandboxed or packaged apps on Windows redirect where uv installs Python, which leaves an environment other terminals can't start ("No Python at ..."). If that happens, delete the `.venv` folder inside `reelsift` and run `uv sync --extra ml` again from a normal terminal.

`doctor` checks Python, FFmpeg, the graphics card, the models, music generation and free disk space, and says what to fix. FFmpeg is found on `PATH`, in winget's install folder, or in the folder `REELSIFT_FFMPEG_DIR` points to.

### Model presets

Analysis picks a model preset from your graphics card's memory:

| Preset | GPU memory | Models | Download | Status |
|---|---|---|---|---|
| large | 24 GB or more | Qwen3-VL-8B, SigLIP 2 so400m, Whisper large-v3-turbo | 23 GB | Measured: 85 min of 4K footage analyzed in about 11 min, 20 GB peak |
| medium | 12 GB or more | Qwen3-VL-4B, SigLIP 2 so400m, Whisper large-v3-turbo | 15 GB | Not measured yet |
| small | 7 GB or more | Qwen3-VL-2B, SigLIP 2 base, Whisper large-v3-turbo | 7 GB | Not measured yet |
| cpu | none | No scene descriptions; SigLIP 2 base, Whisper small | 2 GB | Not measured yet; slow |

`reelsift models list` shows the presets, the one recommended for your card, and what is downloaded. Rendering, export and the review app need no graphics card.

### Music generation

Generating music is optional and needs its own install: `reelsift music install`. It installs ACE-Step 1.5 in a separate environment, in your local app data folder (`%LOCALAPPDATA%\reelsift\music` on Windows). To install it elsewhere, set `REELSIFT_MUSIC_DIR` to that folder; it must stay set whenever music is used.

| Profile | GPU memory | Disk | Longest track | Status |
|---|---|---|---|---|
| xl | 22 GB or more | about 40 GB | 10 minutes | Measured on an RTX 5090: 3 minutes of music in about 30 s |
| standard (`--profile standard`) | 14 GB or more | about 20 GB | 8 minutes | Not measured yet |

## The steps

`reelsift make TRIP` runs every step. They can also be run one at a time:

```bash
uv run reelsift prepare TRIP --open       # find footage, make proxies, build the timeline
uv run reelsift analyze TRIP              # the slow GPU part; runs once
uv run reelsift select TRIP               # choose clips for every length
uv run reelsift review TRIP               # check them in the browser, render and download
uv run reelsift render TRIP --length 180  # or render from the command line
```

Every step can be interrupted and run again. It continues where it stopped and reuses work already done for unchanged footage and settings. A proxy is a small preview copy of a recording; cameras that make their own (`.LRF`, `.LRV`) are used as they are.

`make` options:

| Option | What it adds |
|---|---|
| `--length 30` | Only some of the trip's lengths |
| `--vertical` | A 9:16 version of every video as well |
| `--ride`, `--ride-length 10:00`, `--ride-speed 8` | Also the ride |
| `--only-ride` | Only the ride: no analysis and no highlight videos |
| `--music GENRE --mood MOOD` | Generated music under every video (needs `music install`) |

## Commands

| Command | What it does |
|---|---|
| `reelsift init TRIP --footage FOLDER --length 30 --length 180` | Make a trip that reads footage from these folders, with these video lengths in seconds (default 30, 180 and 600) |
| `reelsift status TRIP` | Recordings, proxies, unfinished steps and problems |
| `reelsift draft TRIP` | Draft every video's description from the footage again (`--length` for one) |
| `reelsift render TRIP --vertical` | Render 9:16, cropped at each clip's crop position |
| `reelsift render TRIP --ride` | Render the ride again (after changing its music or levels) |
| `reelsift export TRIP --length 180` | Write the edit as `.fcpxml` and `.otio` timelines that reference the original files |
| `reelsift ride TRIP --length 10:00` or `--length 600` | Make the ride: stops cut out, the rest sped up to last 10 minutes (`--speed 8` for a fixed speed instead, from 1 to 64; `--render` to render it). The app's Ride tab shows the speeds and lengths, lets you drop stretches and keep stops |
| `reelsift music install` | Install music generation (once; optional) |
| `reelsift music generate TRIP --all --genre synthwave --mood "night drive"` | Generate music for every video (or `--length 30`, or `--ride`); the best-suited track plays, the others stay available |
| `reelsift music list TRIP` / `music use TRIP 180 4` | List the generated tracks; play another one under a video |
| `reelsift music fit TRIP` | Move the cuts onto the music's beats (a new revision; never inside a word) |
| `reelsift music mix TRIP --music -9 --background -12 --duck -15` | How loud the music is, how loud the clips' sound is under it, and how far the music drops when people speak (in dB) |
| `reelsift music set TRIP song.mp3` | Play your own music under the videos instead; cuts land on its beats after the next selection |
| `reelsift music clear TRIP` | No music |
| `reelsift anchor add TRIP FILE 1:23 --utc 2026-10-11T10:05:00+02:00` | Pin a moment in a recording to a real time |
| `reelsift anchor add TRIP FILE_A 1:23 --same-as FILE_B --other-at 0:45` | Say two cameras show the same instant |
| `reelsift timeline TRIP --open` | Show every recording on the timeline, to check the cameras' clocks |
| `reelsift select TRIP --mode motion` | Choose clips with another scoring mode (`combined`, `motion`, `embedding`) |
| `reelsift eval TRIP` | Compare scoring modes against moments you marked in `labels.toml` |
| `reelsift clean TRIP` | Free disk space by deleting caches that can be rebuilt (`--proxies`, `--analysis` for more) |
| `reelsift models list` | Show the model presets and what each needs |

## Exported timelines

`reelsift export` writes FCPXML 1.10 (Final Cut Pro, DaVinci Resolve) and OpenTimelineIO files that reference your original files, with their source timecode. They carry:

- the cuts;
- the speed changes of a ride;
- the music, as a WAV on its own track.

Crops, photo zooms, colour conversion and the finished sound mix are not exported. Importing has been checked for file structure only, not yet in each editor; reports are welcome.

## Configuration

Each trip has a `reelsift.toml`. The settings you are most likely to change:

```toml
[trip]
lengths = [30, 180, 600]   # video lengths in seconds
preset = "auto"            # auto, large, medium, small or cpu

[cameras."Unknown camera (phone)"]
zone = "+02:00"            # time zone, for a camera that doesn't record one
creation_time = "local"    # or "utc": how the camera writes its times, if Reelsift guesses wrong

[music]
genre = "melodic techno"   # or techno, house, drum and bass, synthwave, lo-fi, ambient... or your own words
mood = "sunny, uplifting"

[music.mix]                # dB; these are the defaults
music = -6.0               # music against the clips' sound
background = -9.1          # the clips' sound outside speech, under the music
duck = -12.4               # how far the music drops while people speak (0 = not at all)

[music.lengths."30"]       # any of the above for one video length
genre = "drum and bass"
```

Defaults for every trip go under `[defaults]` (`lengths`, `preset`) in your user config file: `%LOCALAPPDATA%\reelsift\config.toml` on Windows, `~/.config/reelsift/config.toml` on Linux.
