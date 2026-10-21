"""Qwen3-VL: describing short blocks of footage, and reading preference text.

Runs in-process with transformers (measured: 2.4 s per 6-frame block at
batch 8 on an RTX 5090, 19.8 GB peak).
"""

import json
import re

import numpy as np

from .runtime import device


class QwenVL:
    def __init__(self, model_id: str, batch: int = 8):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.name = model_id
        self.batch = batch
        on_gpu = device() == "cuda"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if on_gpu else torch.float32,
            attn_implementation="sdpa",
            device_map="cuda" if on_gpu else "cpu",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"

    def _generate(self, messages: list[list[dict]], images: list, max_new_tokens: int) -> list[str]:
        import torch

        texts = [
            self.processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            for m in messages
        ]
        inputs = self.processor(
            text=texts, images=images or None, padding=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def describe(self, blocks: list[list[np.ndarray]], prompt: str) -> list[str]:
        """One answer per block; each block is a list of RGB frames."""
        from PIL import Image

        answers = []
        for i in range(0, len(blocks), self.batch):
            chunk = blocks[i : i + self.batch]
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            *({"type": "image"} for _ in frames),
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                for frames in chunk
            ]
            images = [Image.fromarray(f) for frames in chunk for f in frames]
            answers.extend(self._generate(messages, images, 200))
        return answers

    def complete(self, prompt: str, max_new_tokens: int = 600) -> str:
        messages = [[{"role": "user", "content": [{"type": "text", "text": prompt}]}]]
        return self._generate(messages, [], max_new_tokens)[0]


def parse_json(text: str) -> dict | None:
    """The first JSON object in a model answer, or None."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
