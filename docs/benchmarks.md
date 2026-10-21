# Benchmarks

Measured in October 2026 on Windows 11 with an RTX 5090 (32 GB), 128 GB of RAM and FFmpeg 9.0.2, using the large model preset. Footage: one bike ride filmed on a DJI Osmo Action 6, 23 files, 85 minutes, 49 GB, 3840×2160 at 25 fps, 10-bit HEVC (BT.709, not a log profile), with the camera's 1280×720 `.LRF` proxies.

Other GPUs, Linux, and the medium, small and CPU presets haven't been measured yet. Results from other machines are welcome as issues or pull requests.

## The whole ride

| Step | Time |
|---|---|
| `prepare` (identify 23 files, pair proxies, timeline) | 23 s; 0.6 s when run again |
| Speech, sound, embeddings and motion | 4 min 12 s |
| Scene descriptions (850 blocks of 6 s) | 7 min |
| Selecting all three lengths | 0.4 s |
| Applying new preferences, models already loaded | 0.7 s |
| Rendering the 3-minute edit from the 4K originals | 61 s |

## Individual models and stages

| Stage | Measured | Projected for 4 h of footage |
|---|---|---|
| Decoding a proxy at 1 fps, 384 px (CPU) | 752 frames in 10.5 s | ~3.5 min |
| SigLIP 2 so400m, fp16, batch 64 | 38 frames/s, 3.6 GB peak GPU memory | ~6.5 min |
| Qwen3-VL-8B, bf16, 6 frames × 448 px per block, batch 1 | 4.2 s per block, 16.8 GB peak | — |
| Qwen3-VL-8B, same, batch 8 | 2.4 s per block, 19.8 GB peak | ~1.6 h for every block |
| faster-whisper large-v3-turbo, fp16, with VAD | 138–171× real time | ~2 min |
| Generated proxy, 4K HEVC → 720p | 6.3× real time with GPU decode, 1.9× with CPU decode | — |
| Render 4K HEVC → 1080p H.264, NVDEC + NVENC | 8 s of output in 2.2 s (6.0 s with CPU decode) | 10-minute edit ≈ 3 min |
| Model weights on disk | Qwen 17 GB, SigLIP 4.3 GB, Whisper 1.6 GB | 23 GB once |

Model load times from the local cache: Qwen3-VL 12 s, SigLIP 2 a few seconds, Whisper 2 s. The first run also downloads the weights. Each block sent to the vision model costs 1,354 input tokens.

## Frame accuracy

Two segments of 100 frames each were cut from the 4K originals and joined. The output has exactly 200 frames. For each segment's first and last output frame, the mean squared difference against the intended source frame and its neighbours:

| Output frame | Source frame | Previous | Intended | Next |
|---|---|---|---|---|
| 0 | clip 1, 125 | 263.9 | **1.5** | 262.9 |
| 99 | clip 1, 224 | 410.5 | **6.1** | 448.2 |
| 100 | clip 2, 7500 | 2290.8 | **4.6** | 2336.4 |
| 199 | clip 2, 7599 | 154.4 | **12.3** | 155.6 |

Every cut lands on its intended frame. The test suite checks the same for variable-frame-rate video and for files whose audio and video start at different times.

## Observations

- **Scene descriptions** were accurate (riders stopping to talk, a descent on a rocky trail, someone helping with a fallen bike), but **interest scores bunch together** and "shake" is reported for almost every block, so scores are only useful after per-trip normalization.
- **SigLIP 2 prompt matches** made sense ("blurry or shaky" found motion-blurred frames; "riding downhill" found a gravel descent), and also need per-trip normalization.
- **Voice activity segments are useless as cut points** on windy outdoor footage (one segment ran 136 s), so cuts use word timestamps.
- **The DJI Osmo Action 6 writes two nearly identical stereo tracks;** one is used.
- **Color:** converting with an explicit BT.709 matrix differs from FFmpeg's default by about 2 levels out of 255, so Reelsift always sets the matrix and range.
- **The Osmo Action 6's motion and GPS data can't be read** with existing open tools, so its motion comes from optical flow.

## Setup notes

- FFmpeg from `winget install Gyan.FFmpeg` works. winget's alias isn't on PATH until a new terminal starts, so Reelsift also looks in winget's package folder.
- On Windows without Developer Mode, the Hugging Face cache stores copies instead of symlinks; it works, at some extra disk use.
- faster-whisper needs NVIDIA's CUDA 12 library wheels next to PyTorch's CUDA 13; they are dependencies of the `ml` extra and their folders are added at startup on Windows.
