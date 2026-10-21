"""Turning observations into one score per candidate.

Every feature is normalized across the whole trip (robust z-score: median and interquartile range),
so weights mean the same thing for a dim forest walk and a sunny beach day. Unknown values count
as average, never as "no".
"""

import json
import os
import warnings
import zipfile
from dataclasses import dataclass

import numpy as np

from .project import Trip
from .tripdata import TripData

PROMPT_BANK_VERSION = 1
PROMPTS = {
    "scenery": "a beautiful scenic view of nature",
    "landmark": "an impressive building or landmark",
    "people": "people smiling, laughing or having fun together",
    "faces": "a close-up of a person's face",
    "kids": "a child playing",
    "action": "an exciting fast action moment",
    "water": "a lake, a river or the sea",
    "food": "food and drinks on a table",
    "animal": "an animal",
    "sky": "a colorful sunset or sunrise sky",
    "city": "a busy city street",
    "road": "a plain road seen from a moving vehicle or bike",
    "ground": "a view of the ground or the floor",
    "obstructed": "a blurry, dark or blocked camera view",
}

SIGNALS = (
    "motion",
    "activity",
    "shake",
    "sharpness",
    "exposure_problem",
    "speech",
    "loudness",
    "scene",
)


def vlm_feature_names() -> list[str]:
    from .vlm import TAGS

    return ["interest", "vlm_quality", *(f"tag:{t}" for t in TAGS)]


MODES = {
    "motion": {"motion": 1.0, "activity": 0.7, "shake": -0.3},
    "embedding": {
        "scenery": 1.0,
        "people": 0.8,
        "action": 0.8,
        "water": 0.5,
        "landmark": 0.5,
        "sky": 0.5,
        "road": -0.6,
        "ground": -0.8,
        "obstructed": -0.8,
    },
    "combined": {
        "scenery": 0.8,
        "people": 0.7,
        "action": 0.6,
        "landmark": 0.4,
        "water": 0.3,
        "sky": 0.4,
        "kids": 0.3,
        "animal": 0.3,
        "food": 0.2,
        "motion": 0.4,
        "activity": 0.4,
        "speech": 0.3,
        "scene": 0.2,
        "sharpness": 0.3,
        "exposure_problem": -0.6,
        "obstructed": -0.6,
        "ground": -0.6,
        "road": -0.3,
        "shake": -0.2,
        "interest": 1.0,
        "vlm_quality": -0.4,
        "tag:boring": -0.6,
        "tag:laughter": 0.4,
        "tag:celebration": 0.3,
        "tag:conversation": 0.2,
    },
}
DEFAULT_MODE = "combined"


def prompt_bank_path(trip: Trip):
    return trip.root / "cache" / "prompt_bank.npz"


def save_prompt_bank(trip: Trip, embedder) -> None:
    names = list(PROMPTS)
    emb = embedder.encode_text([PROMPTS[n] for n in names])
    path = prompt_bank_path(trip)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"model": embedder.name, "version": PROMPT_BANK_VERSION}
    partial = path.with_suffix(".partial.npz")
    with partial.open("wb") as f:  # a file object, so numpy doesn't change the name
        np.savez(f, names=np.array(names), embeddings=emb, meta=json.dumps(meta))
    os.replace(partial, path)


def read_prompt_bank(trip: Trip) -> tuple[list[str], np.ndarray, dict] | None:
    """(names, embeddings, meta), or None if the bank is missing or damaged (an interrupted
    write); analysis then makes it again."""
    path = prompt_bank_path(trip)
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            names = [str(n) for n in data["names"]]
            embeddings = data["embeddings"].astype(np.float32)
            meta = json.loads(str(data["meta"]))
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
        return None
    if len(names) != len(embeddings) or not isinstance(meta, dict):
        return None
    return names, embeddings, meta


def prompt_bank_model(trip: Trip) -> str | None:
    """The model that made the prompt bank, or None if there is no usable bank."""
    bank = read_prompt_bank(trip)
    return bank[2].get("model") if bank else None


def load_prompt_bank(trip: Trip) -> tuple[list[str], np.ndarray]:
    bank = read_prompt_bank(trip)
    if bank is None:
        return [], np.zeros((0, 1), np.float32)
    return bank[0], bank[1]


@dataclass
class Features:
    names: list[str]
    raw: np.ndarray
    """(candidates, features) values before normalization; NaN where unknown."""
    z: np.ndarray
    """Normalized values, clipped to [-3, 3], with unknown set to 0."""
    embedding: np.ndarray
    """Mean embedding of each candidate's core, L2-normalized (zeros if unknown)."""


def _seconds(start: float, end: float, n: int) -> slice:
    a = max(0, min(n - 1, int(np.floor(start))))
    b = max(a + 1, min(n, int(np.ceil(end))))
    return slice(a, b)


def per_second_similarity(data: TripData, text_emb: np.ndarray) -> dict[int, np.ndarray]:
    """For each recording, (seconds, texts) cosine similarities between frames and texts."""
    return {
        rec: emb @ text_emb.T
        for rec, emb in data.embeddings.items()
        if len(emb) and emb.shape[1] == text_emb.shape[1]
    }


def features(
    data: TripData,
    bank: tuple[list[str], np.ndarray],
    searches: dict[str, np.ndarray] | None = None,
) -> Features:
    """Feature matrix for all candidates. searches maps a feature name to a text embedding."""
    prompt_names, prompt_emb = bank
    search_names = list(searches or {})
    texts = (
        np.concatenate(
            [prompt_emb] + ([np.stack([searches[n] for n in search_names])] if search_names else [])
        )
        if len(prompt_names) or search_names
        else np.zeros((0, 1), np.float32)
    )
    text_names = prompt_names + search_names
    sims = per_second_similarity(data, texts) if len(text_names) else {}

    c = data.candidates
    vlm_names = vlm_feature_names()
    names = list(SIGNALS) + vlm_names + text_names
    text_offset = len(SIGNALS) + len(vlm_names)
    raw = np.full((len(c), len(names)), np.nan, np.float64)
    dim = next((e.shape[1] for e in data.embeddings.values() if len(e)), 1)
    mean_emb = np.zeros((len(c), dim), np.float32)
    for i in range(len(c)):
        rec = int(c.recording[i])
        obs = data.observations.get(rec)
        if obs is not None:
            s = _seconds(c.core_start[i], c.core_end[i], len(obs["motion"]))
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN means unknown
                values = {
                    "motion": np.nanmean(obs["motion"][s]),
                    "activity": np.nanmean(obs["activity"][s]),
                    "shake": np.nanmean(obs["shake"][s]),
                    "sharpness": np.log1p(np.nanmean(obs["blur"][s])),
                    "exposure_problem": np.nanmean(obs["dark"][s] + obs["bright"][s]),
                    "speech": np.nanmean(obs["speech"][s]),
                    "loudness": np.nanmean(obs["loudness"][s]),
                    "scene": np.nanmax(obs["scene"][s])
                    if np.any(~np.isnan(obs["scene"][s]))
                    else np.nan,
                }
            for j, name in enumerate(SIGNALS):
                raw[i, j] = values[name]
        blocks = data.vlm.get(rec)
        if blocks is not None and len(blocks.start):
            point = blocks.end <= blocks.start  # a photo's block has no duration
            hit = np.where(
                point,
                (blocks.start >= c.core_start[i]) & (blocks.start < c.core_end[i]),
                (blocks.start < c.core_end[i]) & (blocks.end > c.core_start[i]),
            )
            hit &= ~np.isnan(blocks.interest)
            if hit.any():
                j = len(SIGNALS)
                raw[i, j] = blocks.interest[hit].mean()
                raw[i, j + 1] = blocks.quality[hit].mean()
                raw[i, j + 2 : j + len(vlm_names)] = blocks.tags[hit].mean(axis=0)
        if rec in sims:
            sim = sims[rec]
            s = _seconds(c.core_start[i], c.core_end[i], len(sim))
            raw[i, text_offset:] = sim[s].mean(axis=0)
            emb = data.embeddings[rec][s].mean(axis=0)
            norm = np.linalg.norm(emb)
            if norm > 0:
                mean_emb[i] = emb / norm
    return Features(names, raw, normalize(raw), mean_emb)


def normalize(raw: np.ndarray) -> np.ndarray:
    z = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        known = ~np.isnan(col)
        if known.sum() < 2:
            continue
        median = np.median(col[known])
        q1, q3 = np.percentile(col[known], [25, 75])
        scale = (q3 - q1) / 1.349
        if scale <= 1e-12:
            scale = np.std(col[known]) or 1.0
        z[known, j] = np.clip((col[known] - median) / scale, -3, 3)
    return z


def score(feats: Features, weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Return (score per candidate, contribution matrix aligned with feats.names)."""
    w = np.array([weights.get(n, 0.0) for n in feats.names])
    contributions = feats.z * w
    return contributions.sum(axis=1), contributions
