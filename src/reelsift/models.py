"""Model presets, picked from GPU memory, and downloading their weights.

Medium and small presets are provisional until they are measured on matching hardware.
"""

from dataclasses import dataclass
from pathlib import Path

LARGE_MIN_GB = 24
MEDIUM_MIN_GB = 12
SMALL_MIN_GB = 7


@dataclass(frozen=True)
class Preset:
    name: str
    min_vram_gb: float
    vision: str | None
    embedding: str
    speech: str
    measured: bool
    note: str
    vlm_batch: int = 8


PRESETS = {
    "large": Preset(
        "large",
        LARGE_MIN_GB,
        "Qwen/Qwen3-VL-8B-Instruct",
        "google/siglip2-so400m-patch14-384",
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        True,
        "Measured on an RTX 5090: 19.8 GB peak at batch 8.",
    ),
    "medium": Preset(
        "medium",
        MEDIUM_MIN_GB,
        "Qwen/Qwen3-VL-4B-Instruct",
        "google/siglip2-so400m-patch14-384",
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        False,
        "Not measured yet.",
        4,
    ),
    "small": Preset(
        "small",
        SMALL_MIN_GB,
        "Qwen/Qwen3-VL-2B-Instruct",
        "google/siglip2-base-patch16-224",
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        False,
        "Not measured yet.",
        2,
    ),
    "cpu": Preset(
        "cpu",
        0,
        None,
        "google/siglip2-base-patch16-224",
        "Systran/faster-whisper-small",
        False,
        "No vision model; slow. For testing and short outings.",
    ),
}


def recommend(vram_gb: float | None) -> Preset:
    if vram_gb is None:
        return PRESETS["cpu"]
    for name in ("large", "medium", "small"):
        if vram_gb >= PRESETS[name].min_vram_gb:
            return PRESETS[name]
    return PRESETS["cpu"]


def resolve(name: str, vram_gb: float | None) -> Preset:
    if name == "auto":
        return recommend(vram_gb)
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from auto, {', '.join(PRESETS)}")
    return PRESETS[name]


def repos(preset: Preset) -> list[str]:
    return [r for r in (preset.vision, preset.embedding, preset.speech) if r]


@dataclass
class RepoInfo:
    repo: str
    license: str
    size_bytes: int
    cached: bool


def describe(repo: str) -> RepoInfo:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, files_metadata=True)
    card = info.card_data.to_dict() if info.card_data else {}
    size = sum(s.size or 0 for s in info.siblings or [])
    files = [s.rfilename for s in info.siblings or []]
    return RepoInfo(repo, str(card.get("license", "unknown")), size, is_cached(repo, files))


WEIGHTS = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx")


def is_cached(repo: str, files: list[str] | None = None, cache_dir: Path | None = None) -> bool:
    """Whether the whole model is downloaded. With the repository's file list (online), every
    file must be there; without it (offline), its weights: at least one weights file, every
    file its index names, and no download left half done. An interrupted download then counts
    as not downloaded, and models pull resumes it."""
    import json

    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    try:
        cache = scan_cache_dir(cache_dir)
    except CacheNotFound:
        return False
    for r in cache.repos:
        if r.repo_id != repo or r.repo_type != "model":
            continue
        if any((r.repo_path / "blobs").glob("*.incomplete")):
            return False
        for rev in r.revisions:
            have = {f.file_path.relative_to(rev.snapshot_path).as_posix() for f in rev.files}
            if files is not None:
                if set(files) <= have:
                    return True
                continue
            if not any(f.endswith(WEIGHTS) for f in have):
                continue
            needed: set[str] = set()
            for index in (f for f in have if f.endswith(".index.json")):
                try:
                    data = json.loads((rev.snapshot_path / index).read_text(encoding="utf-8"))
                    needed |= set(data.get("weight_map", {}).values())
                except (OSError, ValueError):
                    needed.add(index + " (unreadable)")
            if needed <= have:
                return True
    return False


def download(repo: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo)
