# Reelsift for AI agents

This file is for an AI agent (Claude Code, Codex, or another) asked to turn someone's footage into videos with Reelsift. Everything runs from the command line with no prompts. People review the result in the browser app; no agent ever needs it. The last section is for agents changing Reelsift's code.

Two ways to work:

- **With a review (the default).** You install everything, prepare the edits, then open the review app for the person. They check the clips, render and download the videos themselves.
- **Fully automatic.** One `reelsift make` command does every step and writes the finished videos. Use it when the person says they don't want to review, or when nobody is there to do it.

## Before you start, ask

- Where the footage is (one or more folders), and where the trip folder should go. A trip folder holds Reelsift's work, such as `D:/reelsift-trips/<trip-name>`. The footage itself is only ever read.
- What they want:
  - **Highlight edits** at chosen lengths. The defaults are 30 s, 3 min and 10 min.
  - **A ride:** the whole trip from one camera, with the stops (red lights, waiting) cut out and the rest sped up. It's the natural choice for dashcam or commute footage. They can give a target length (`10:00`) or a fixed speed (`8`).
  - Both.
- Whether they want generated music, and in what genre and mood. Music needs a separate install of about 40 GB.
- Whether they want 9:16 (vertical) versions as well.
- Whether they want to review before the videos are made. Without an answer, prepare everything and open the review app.

Get the person's OK before large downloads, and quote the sizes:
- **Python packages** (`uv sync --extra ml`): about 5 GB, mostly PyTorch with CUDA.
- **The analysis models** (`models pull`). `uv run reelsift models list` marks the preset "recommended here" for this GPU and says what is already downloaded. Download sizes:

  | Preset | Download |
  |---|---|
  | large | 23 GB |
  | medium | 15 GB |
  | small | 7 GB |
  | cpu | 2 GB |

- **The music model** (`music install`): about 40 GB.

## Install

The requirements:

- Windows 11. Linux and macOS aren't supported yet.
- `git`, `uv`, and FFmpeg 5.1 or newer.
- An NVIDIA GPU (RTX 20-series or newer, driver 580 or newer) for fast analysis. Without one, analysis runs on the CPU, much more slowly.

Install only what's missing; check first with `git --version`, `uv --version` and `ffmpeg -version`.

Install with winget, with the agreement flags. Without them, winget can stop and wait for an answer on a new profile:

```bash
winget install --id Git.Git -e --accept-source-agreements --accept-package-agreements
winget install --id astral-sh.uv -e --accept-source-agreements --accept-package-agreements
winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
```

A program installed by winget is only on `PATH` in shells started after the install. Open a new shell, or call it by its full path. Reelsift finds a winget FFmpeg by itself. `REELSIFT_FFMPEG_DIR` can also point to a folder that contains `ffmpeg` and `ffprobe`.

Then:

```bash
git clone https://github.com/hexkDotCom/reelsift.git
cd reelsift
uv sync --extra ml
uv run reelsift doctor
uv run reelsift models pull --yes
```

- **Running commands.** Run every `reelsift` command from the `reelsift` folder, as `uv run reelsift …`. Reelsift's own hints, such as "Next: reelsift prepare …", leave out `uv run`; add it.
- **`doctor`** exits with 1 when something must be fixed, and says how to fix it.
- **`models pull --yes`** downloads the analysis models for this GPU without asking. Ask the person first.
- **Rides only.** If the setup is only for rides (dashcam, commute), `--extra ml` and `models pull` can be skipped. Rides, rendering and export need no models.
- **"No Python at …".** If `uv` fails with this, delete `.venv` and run `uv sync --extra ml` again from a normal terminal.

### Music (optional)

Music needs at least 14 GB of GPU memory. First check `doctor`: its "Music generation" line says whether music is already installed, and where. If it is, skip the install.

```bash
uv run reelsift music install
```

- It installs into the user's local app data: `%LOCALAPPDATA%\reelsift\music` on Windows.
- To put the 40 GB on another drive, set `REELSIFT_MUSIC_DIR` to a folder there. It must then be set for every command that uses music: `music …`, `make --music`, and `review`.

## With a review

```bash
uv run reelsift init TRIP --footage FOOTAGE_FOLDER
uv run reelsift prepare TRIP
uv run reelsift analyze TRIP
uv run reelsift select TRIP
```

- `init` takes one `--footage` per folder.
- **Lengths.** The edit lengths are chosen at `init`, in seconds: `--length 30 --length 60`. The default is 30, 180 and 600. To change them later, edit `lengths` under `[trip]` in `TRIP/reelsift.toml`, then run `select`.
- **Analysis** is the slow, GPU-bound step. On an RTX 5090, 85 minutes of 4K footage takes about 11 minutes. Its log includes download progress bars and model warnings; only the exit code and Reelsift's own lines matter.
- **Short edits.** `select` prints each edit's length. If an edit is much shorter than asked, it says so: there isn't enough suitable footage. Tell the person, rather than promising a 3-minute video.

Optional extras, before opening the app:

```bash
uv run reelsift ride TRIP --length 10:00
uv run reelsift music generate TRIP --all --genre synthwave --mood "night drive"
uv run reelsift music generate TRIP --ride --genre synthwave --mood "night drive"
```

- **The ride.** The first line makes it: `--length` fits it to a length, `--speed 8` uses a fixed speed instead. It needs only `prepare`, not `analyze`.
- **Music.** The other two put generated music under every highlight edit, and under the ride.
- **Quoting.** Put quotes around a genre or mood with spaces.

Then open the app for the person. It is a server that keeps running until it is stopped, so start it in a way that outlives your own session. Background jobs that end with your session would close the person's app with it. Start it in its own window:

```bash
Start-Process uv -ArgumentList 'run','reelsift','review','TRIP'
```

- **When it's ready.** It prints `Review app: http://127.0.0.1:8765/`, and `GET http://127.0.0.1:8765/` then returns 200. It opens the person's browser itself. Add `--no-browser` to leave that to them.
- **Port.** If port 8765 is taken, add `--port 8766`.
- **Ride-only trips.** The app opens on the Ride tab.
- **Tell the person:**
  - they can keep or drop clips, add others, describe what they want more or less of, and choose the music;
  - **Render 16:9** or **Render 9:16** makes the video, and **Download** saves it;
  - the videos are also in `TRIP/output/`;
  - Ctrl+C in the app's window stops it.
- **Stopping it.** Stop the app only when they say they're done. `uv run` starts a chain of processes, and ending only the first one leaves the server running. End the whole tree with `taskkill /PID <pid> /T /F`, using the PID of the process listening on the port (`netstat -ano | findstr :8765`).

## Fully automatic

```bash
uv run reelsift init TRIP --footage FOOTAGE_FOLDER --length 30 --length 180
uv run reelsift make TRIP
```

`make` runs `prepare`, `analyze` and `select`, then renders every edit length. Add options to `make` as needed:

| Option | What it adds |
|---|---|
| `--length 30` | Only some of the trip's lengths (those chosen at `init`) |
| `--ride` | Also the ride, at the trip's setting (8x by default) |
| `--ride-length 10:00` / `--ride-speed 8` | The ride at this length or speed (implies `--ride`) |
| `--only-ride` | Only the ride: no analysis and no highlight edits (dashcam, commute) |
| `--music synthwave --mood "night drive"` | Generated music under every video; needs `music install` |
| `--vertical` | A 9:16 version of every video as well |

For example:
- `uv run reelsift make TRIP --only-ride --ride-length 3:00` turns dashcam footage into a 3-minute ride. Use `--only-ride` rather than `--ride` for footage that is all driving or riding: `--ride` also runs the slow analysis and makes highlight edits.
- `uv run reelsift make TRIP --length 180 --ride --music lo-fi --vertical` makes a 3-minute highlight and a ride, with music, in both formats.

At the end, `make` lists the paths of the videos it wrote, all in `TRIP/output/`. Give those paths to the person, and mention any edit `select` warned was short.

To render again later, without running every step again:

```bash
uv run reelsift render TRIP --length 180
uv run reelsift render TRIP --ride --vertical
```

The second line renders the ride alone, in 9:16.

## Running commands

- **Exit codes.**
  - 0 means the command did its work.
  - 1 means it failed. The reason goes to stderr in plain words, often with the command that fixes it. Read it, act on it, and run the command again.
  - 2 means a mistyped command or option. Check it against `uv run reelsift COMMAND --help`.
- **Warnings.** A step can also succeed with warnings, such as a file it couldn't read or an edit that came out short. Read the whole output, not only the exit code. `uv run reelsift status TRIP` lists open problems.
- **Interrupted steps.** Every step can be stopped and run again: it continues where it stopped and skips finished work. If a long command is cut off by a timeout, run the same command again. `make` run again reuses:
  - the analysis;
  - a ride built from the same settings;
  - music already generated with the same genre and mood;
  - the clips it already cut for rendering.

  To make new music instead, use `music generate`.
- **Long commands.** `analyze`, `make`, `render`, `music install` and `music generate` can take from minutes to an hour or more. Run them in the background if your environment allows it.
- **Output.** Output is UTF-8, also when saved to a file.
- **Reading the trip.** `uv run reelsift status TRIP` sums up the trip: recordings, proxies, unfinished steps and problems.
- **Wrong times.** If the cameras' clocks were wrong, the timeline is wrong too. `uv run reelsift timeline TRIP --open` shows it; `uv run reelsift anchor add` fixes it (see docs/usage.md).

Rules:
- Never move, change or delete the footage; Reelsift only reads it.
- Don't delete a trip folder or run `reelsift clean` unless the person asks. A trip folder holds their decisions and edits.
- Don't commit footage or trip folders to git.

## Working on Reelsift's code

The README says how Reelsift works, and `CONTRIBUTING.md` covers the rest. Before handing in a change, run:

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format src tests
```

The tests need FFmpeg. Tests for the review app's JavaScript run with Node through `tests/test_frontend.py`.
