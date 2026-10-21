"""Check that this machine can run Reelsift, and say what to fix."""

import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import models
from .tools import ToolNotFoundError, encoder_works, filter_available, run, tool

OK, WARN, FAIL = "ok", "warn", "fail"
MIN_FREE_GB = 50


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


@dataclass
class Gpu:
    name: str
    vram_gb: float
    source: str


def gpu() -> Gpu | None:
    """The first NVIDIA GPU, from PyTorch when installed, otherwise from nvidia-smi."""
    if importlib.util.find_spec("torch"):
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return Gpu(props.name, round(props.total_memory / 2**30, 1), "torch")
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = (
                subprocess.run(
                    [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                .stdout.strip()
                .splitlines()
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if out:
            name, mem = (x.strip() for x in out[0].split(",", 1))
            return Gpu(name, round(float(mem) / 1024, 1), "nvidia-smi")
    return None


def checks(target: Path | None = None, preset: str = "auto") -> list[Check]:
    results = []
    ok_python = sys.version_info[:2] == (3, 12)
    results.append(
        Check(
            "Python",
            OK if ok_python else FAIL,
            sys.version.split()[0],
            "" if ok_python else "Reelsift needs Python 3.12.",
        )
    )

    try:
        first_line = run([tool("ffmpeg"), "-version"], text=True).stdout.splitlines()[0]
        version = first_line.split(" Copyright")[0]
        results.append(Check("FFmpeg", OK, f"{version} ({tool('ffmpeg')})"))
        tool("ffprobe")
        if encoder_works("h264_nvenc"):
            results.append(Check("Video encoding", OK, "NVENC (GPU) works"))
        elif encoder_works("libx264"):
            results.append(
                Check(
                    "Video encoding",
                    WARN,
                    "x264 on the CPU; NVENC unavailable",
                    "Rendering will be slower. NVENC needs an NVIDIA GPU and driver.",
                )
            )
        else:
            results.append(
                Check(
                    "Video encoding",
                    FAIL,
                    "no H.264 encoder",
                    "Install an FFmpeg build with libx264 or NVENC.",
                )
            )
        if not filter_available("zscale"):
            results.append(
                Check(
                    "HDR support",
                    WARN,
                    "FFmpeg has no zscale filter",
                    "HDR footage can't be converted; use a full FFmpeg build.",
                )
            )
    except (ToolNotFoundError, RuntimeError) as e:
        results.append(
            Check(
                "FFmpeg",
                FAIL,
                str(e).split(".")[0],
                "Windows: winget install Gyan.FFmpeg. Linux: install ffmpeg from "
                "your package manager. Or set REELSIFT_FFMPEG_DIR.",
            )
        )

    card = gpu()
    has_ml = importlib.util.find_spec("torch") is not None
    if card:
        results.append(Check("GPU", OK, f"{card.name}, {card.vram_gb} GB"))
    else:
        results.append(
            Check(
                "GPU",
                WARN,
                "no NVIDIA GPU found",
                "Analysis will run on the CPU with the cpu preset, which is slow.",
            )
        )
    if has_ml:
        import torch

        cuda = torch.cuda.is_available()
        results.append(
            Check(
                "PyTorch",
                OK if cuda or not card else FAIL,
                f"{torch.__version__}, CUDA {'available' if cuda else 'not available'}",
                ""
                if cuda or not card
                else "Install the PyTorch build for your GPU (see docs/usage.md).",
            )
        )
    else:
        results.append(
            Check(
                "Analysis models",
                WARN,
                "the ml extra is not installed",
                "Install with: uv sync --extra ml (or pip install 'reelsift[ml]').",
            )
        )

    chosen = models.resolve(preset, card.vram_gb if card else None)
    missing = [r for r in models.repos(chosen) if not models.is_cached(r)]
    results.append(
        Check(
            "Model preset",
            OK if not missing else WARN,
            f"{chosen.name}"
            + ("" if chosen.measured else " (not measured yet)")
            + (f"; not downloaded: {', '.join(missing)}" if missing else "; all downloaded"),
            "" if not missing else f"Run: reelsift models pull --preset {chosen.name}",
        )
    )

    from . import musicgen

    music_profile = musicgen.installed()
    if music_profile is not None:
        m = musicgen.manifest() or {}
        licences = ", ".join(sorted({c["licence"] for c in m.get("checkpoints", {}).values()}))
        results.append(Check(
            "Music generation", OK,
            f"ACE-Step {m.get('version')} ({music_profile.name} profile, up to "
            f"{music_profile.max_duration:.0f} s; code {m.get('licence')}, models {licences}) "
            f"in {musicgen.install_dir()}",
        ))  # fmt: skip
    else:
        suggested = musicgen.recommend(card.vram_gb if card else None)
        results.append(Check(
            "Music generation", WARN, "not installed",
            "Optional. Install with: reelsift music install (about "
            f"{suggested.disk_gb:.0f} GB)" if suggested else
            "Optional; needs an NVIDIA GPU with at least 14 GB of memory.",
        ))  # fmt: skip

    where = target or Path.cwd()
    while not where.exists():
        where = where.parent
    free_gb = shutil.disk_usage(where).free / 2**30
    results.append(
        Check(
            "Disk space",
            OK if free_gb >= MIN_FREE_GB else WARN,
            f"{free_gb:.0f} GB free at {where}",
            ""
            if free_gb >= MIN_FREE_GB
            else f"Proxies and caches can take tens of GB; {MIN_FREE_GB} GB free is a comfortable minimum.",
        )
    )
    return results
