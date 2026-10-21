"""Speech transcription with faster-whisper and its built-in Silero voice activity detection."""

import numpy as np

from .runtime import add_cuda12_dlls, device


class WhisperTranscriber:
    def __init__(self, model_id: str):
        add_cuda12_dlls()
        from faster_whisper import WhisperModel

        self.name = model_id
        on_gpu = device() == "cuda"
        self.model = WhisperModel(
            model_id,
            device="cuda" if on_gpu else "cpu",
            compute_type="float16" if on_gpu else "int8",
        )

    def transcribe(self, samples: np.ndarray) -> tuple[list[dict], str]:
        segments, info = self.model.transcribe(
            samples, vad_filter=True, word_timestamps=True, condition_on_previous_text=False
        )
        out = []
        for s in segments:
            out.append(
                {
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": s.text.strip(),
                    "avg_logprob": float(s.avg_logprob),
                    "no_speech_prob": float(s.no_speech_prob),
                    "words": [
                        {
                            "start": float(w.start),
                            "end": float(w.end),
                            "word": w.word.strip(),
                            "probability": float(w.probability),
                        }
                        for w in (s.words or [])
                    ],
                }
            )
        return out, info.language
