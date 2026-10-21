"""SigLIP 2 image and text embeddings.

Frames arrive already resized to the model's square input size (FFmpeg does the resize while
decoding), so preprocessing is just scaling to [-1, 1] on the GPU, which is far faster than the
Hugging Face image processor on the CPU.
"""

import numpy as np

from .runtime import device

BATCH = 64


class SiglipEmbedder:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.name = model_id
        self.device = device()
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.image_size = int(self.model.config.vision_config.image_size)
        self.dtype = dtype

    @staticmethod
    def _pooled(output):
        return output.pooler_output if hasattr(output, "pooler_output") else output

    def encode_images(self, frames: np.ndarray) -> np.ndarray:
        import torch

        out = []
        with torch.inference_mode():
            for i in range(0, len(frames), BATCH):
                batch = torch.from_numpy(np.array(frames[i : i + BATCH])).to(self.device)
                pixels = batch.permute(0, 3, 1, 2).to(self.dtype) / 127.5 - 1.0
                feats = self._pooled(self.model.get_image_features(pixel_values=pixels))
                out.append(torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy())
        dim = self.model.config.vision_config.hidden_size
        return np.concatenate(out) if out else np.zeros((0, dim), np.float32)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        import torch

        with torch.inference_mode():
            tokens = self.processor(
                text=texts, padding="max_length", max_length=64, return_tensors="pt"
            ).to(self.device)
            feats = self._pooled(self.model.get_text_features(**tokens))
            return torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy()
