"""Generate music with ACE-Step for Reelsift, in ACE-Step's own environment.

This script never imports reelsift: it runs with the music environment's Python (see
musicgen.py), whose PyTorch and transformers differ from Reelsift's.

Protocol 1. Standard input: one JSON object
    {"protocol": 1, "job": str, "source": ACE-Step folder, "checkpoints": folder,
     "dit": checkpoint name, "lm": checkpoint name or null, "out": folder,
     "candidates": [{"id": str, "caption": str, "lyrics": str, "bpm": int, "duration": float,
                     "seed": int, "keyscale": str, "timesignature": str}, ...]}
Standard output: one JSON object per line, flushed:
    {"event": "stage", "stage": "loading" | "generating", ...}
    {"event": "ready", "seconds": float, "vram_gb": float}
    {"event": "candidate", "id": str, "path": str, "seconds": float, "seed": int}
    {"event": "error", "stage": str, "id": str | null, "message": str}
    {"event": "done"}
Everything else (ACE-Step's own printing, warnings) goes to standard error. Candidates are made
one at a time; each file is complete before it is reported. The process exits when done, which
frees the GPU.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

PROTOCOL = 1


def main() -> int:
    # The protocol gets a private copy of standard output; anything else written to standard
    # output, by Python or by native libraries, lands on standard error.
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(**message) -> None:
        protocol.write(json.dumps(message) + "\n")
        protocol.flush()

    try:
        request = json.loads(sys.stdin.read())
        if request.get("protocol") != PROTOCOL:
            raise ValueError(f"protocol {request.get('protocol')!r}, expected {PROTOCOL}")
    except Exception as e:
        send(event="error", stage="request", id=None, message=str(e))
        return 2

    os.environ["ACESTEP_CHECKPOINTS_DIR"] = request["checkpoints"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, request["source"])
    out = Path(request["out"])
    out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    send(event="stage", stage="loading")
    try:
        import torch
        from acestep.handler import AceStepHandler
        from acestep.inference import GenerationConfig, GenerationParams, generate_music
        from acestep.llm_inference import LLMHandler

        dit = AceStepHandler()
        status, ok = dit.initialize_service(
            project_root=request["source"], config_path=request["dit"], device="cuda"
        )
        if not ok:
            raise RuntimeError(f"loading {request['dit']}: {status}")
        llm = LLMHandler()
        if request.get("lm"):
            status, ok = llm.initialize(
                checkpoint_dir=request["checkpoints"], lm_model_path=request["lm"], backend="pt",
                device="cuda",
            )  # fmt: skip
            if not ok:
                raise RuntimeError(f"loading {request['lm']}: {status}")
    except Exception as e:
        traceback.print_exc()
        send(event="error", stage="loading", id=None, message=str(e))
        return 1
    send(event="ready", seconds=round(time.time() - started, 1),
         vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))  # fmt: skip

    candidates = request["candidates"]
    for n, c in enumerate(candidates):
        send(event="stage", stage="generating", index=n, total=len(candidates), id=c["id"])
        t = time.time()
        try:
            params = GenerationParams(
                task_type="text2music", caption=c["caption"], lyrics=c.get("lyrics", ""),
                instrumental=True, bpm=int(c["bpm"]) if c.get("bpm") else None,
                keyscale=c.get("keyscale", ""), timesignature=c.get("timesignature", "4"),
                duration=float(c["duration"]), thinking=bool(request.get("lm")),
                inference_steps=8, shift=3.0, seed=int(c["seed"]),
            )  # fmt: skip
            config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[int(c["seed"])],
                                      audio_format="wav")  # fmt: skip
            work = out / f".{c['id']}"
            result = generate_music(dit, llm, params, config, save_dir=str(work))
            if not result.success or not result.audios:
                raise RuntimeError(result.error or result.status_message or "no audio")
            target = out / f"{c['id']}.wav"
            os.replace(result.audios[0]["path"], target)
            for leftover in work.glob("*"):
                leftover.unlink()
            work.rmdir()
        except Exception as e:
            traceback.print_exc()
            send(event="error", stage="generating", id=c["id"], message=str(e))
            continue
        send(event="candidate", id=c["id"], path=str(target), seconds=round(time.time() - t, 1),
             seed=int(c["seed"]))  # fmt: skip
    send(event="done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
