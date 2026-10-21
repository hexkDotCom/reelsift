"""Stand-in models for tests: deterministic, instant, no GPU or downloads."""

import hashlib

import numpy as np


class FakeEmbedder:
    """Embeds a frame as its normalized color histogram; text as a seeded random vector."""

    name = "fake-embedder"
    image_size = 32
    dim = 24

    def encode_images(self, frames: np.ndarray) -> np.ndarray:
        out = []
        for f in frames:
            hist = [np.histogram(f[..., ch], bins=8, range=(0, 256))[0] for ch in range(3)]
            v = np.concatenate(hist).astype(np.float32) + 1e-3
            out.append(v / np.linalg.norm(v))
        return np.array(out, np.float32).reshape(-1, self.dim)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).random(self.dim).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.array(out)


class FakeTranscriber:
    """Hears two words in every clip that has sound."""

    name = "fake-transcriber"

    def transcribe(self, samples: np.ndarray) -> tuple[list[dict], str]:
        seconds = len(samples) / 16000
        if seconds < 1.5:
            return [], "en"
        words = [
            {"start": 0.3, "end": 0.7, "word": "hello", "probability": 0.9},
            {"start": 0.8, "end": 1.3, "word": "there", "probability": 0.9},
        ]
        return [
            {
                "start": 0.3,
                "end": 1.3,
                "text": "hello there",
                "avg_logprob": -0.2,
                "no_speech_prob": 0.01,
                "words": words,
            }
        ], "en"


class FakeDescriber:
    """Calls bright blocks interesting scenery and dark ones boring."""

    name = "fake-describer"

    def describe(self, blocks, prompt):
        import json

        out = []
        for frames in blocks:
            bright = float(np.mean([f.mean() for f in frames]))
            if bright > 100:
                answer = {"description": "a bright view", "tags": ["scenery", "sky", "nonsense"],
                          "interest": 8, "quality": []}  # fmt: skip
            else:
                answer = {"description": "a dark view", "tags": ["boring"], "interest": 3,
                          "quality": ["dark"]}  # fmt: skip
            out.append("Sure! " + json.dumps(answer))
        return out
