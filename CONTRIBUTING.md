# Contributing to Reelsift

Thanks for helping. Bug reports, sample footage, measurements from other hardware and pull requests are all welcome.

## Reporting a problem

Open an issue with:

- what you ran and what happened (the full error, or a screenshot of the review app);
- the output of `uv run reelsift doctor`;
- the output of `uv run reelsift status TRIP` if it's about a trip;
- the camera or phone the footage came from.

Please don't attach personal footage to a public issue. If a bug only shows with a particular file, say so and we'll find a way to reproduce it, for example with a few seconds you're happy to share or a file made with FFmpeg.

## Sample footage

A device is verified only once short sample files from it are in the test suite. If you have a camera or phone that isn't verified yet (GoPro HERO and Google Pixel 10 are the most wanted), a few seconds of footage helps a lot: ideally one short clip as it comes off the card, untouched, with any files the camera writes next to it (`.LRV`, `.THM`, `.LRF`). Only share footage you shot yourself and are happy to publish under the project's license, without people who haven't agreed to it. Open an issue first to agree on how to send it.

## Setting up

You need [uv](https://docs.astral.sh/uv/), FFmpeg, and Node.js for the frontend tests (optional; those tests are skipped without it).

```bash
git clone https://github.com/hexkDotCom/reelsift.git
cd reelsift
uv sync              # add --extra ml to run analysis with the real models
uv run pytest
uv run ruff check src tests
uv run ruff format src tests
```

The tests make their own small clips and photos with FFmpeg and Pillow and use stand-in models, so they need neither a GPU nor downloaded models. CI runs lint and tests on Windows, and on Linux as an early warning (Linux isn't supported yet), without the `ml` extra.

## Pull requests

- Keep each pull request to one change, and describe what it changes and how you checked it.
- Add a test for new behaviour, and a regression test for every bug fix, named after the behaviour it checks.
- Match the code around you; `ruff` handles formatting and imports.
- For a visible change in the review app, include a screenshot.
- A change to the trip database needs a new migration at the end of `MIGRATIONS` in `src/reelsift/project.py`; never edit a released one.

## Adding a device

Each camera or phone is a small profile in `src/reelsift/devices/`, a subclass of `DeviceProfile` in `base.py`. `dji.py` is a complete example. A profile can say:

- `matches`: whether a file comes from this device (encoder tag, file name pattern, metadata);
- `is_proxy` and `proxy_for`: which files are camera-made low-resolution copies, and which proxy belongs to a file;
- `chapter_key`: how the device splits long recordings into several files, so they become one recording;
- `camera_identity`: what tells two cameras of the same model apart;
- `time_evidence`: when each recording started, and how sure that is.

Register the profile in `PROFILES` in `devices/__init__.py`, add tests that use generated files with the device's names and metadata (see `tests/conftest.py`), and, once real samples exist, add those and update the device table in the README.

## Code of conduct

Be kind and assume good intent. Harassment or personal attacks aren't welcome in issues, pull requests or anywhere else in the project.
