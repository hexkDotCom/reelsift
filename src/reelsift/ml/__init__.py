"""Model wrappers. Everything here needs the ml extra; callers import lazily.

Each wrapper has a small interface (see the Protocols) so tests and CPU-only installs can swap in
simpler implementations.
"""

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    image_size: int

    def encode_images(self, frames: np.ndarray) -> np.ndarray:
        """uint8 frames (N, size, size, 3) -> L2-normalized float32 embeddings (N, dim)."""
        ...

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Texts -> L2-normalized float32 embeddings (N, dim), comparable with images."""
        ...


class Transcriber(Protocol):
    name: str

    def transcribe(self, samples: np.ndarray) -> tuple[list[dict], str]:
        """16 kHz mono float32 audio -> (segments with words, language).

        Each segment: {"start", "end", "text", "avg_logprob", "no_speech_prob",
        "words": [{"start", "end", "word", "probability"}]}.
        """
        ...


def ml_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("torch") is not None
