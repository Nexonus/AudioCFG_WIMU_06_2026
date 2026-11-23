from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from tqdm import tqdm

from audiocfg.data.audio import load_window
from audiocfg.repro.seeds import derive_seed


def encode_files(
    sao,
    paths: Sequence[str],
    ids: Sequence[str],
    master_seed: int,
    max_seconds: float = 30.0,
    batch_size: int = 4,
) -> dict:
    latents, seconds, kept, failed = [], [], [], []
    for start in tqdm(range(0, len(paths), batch_size), desc="encode"):
        chunk = list(
            zip(paths[start : start + batch_size], ids[start : start + batch_size], strict=True)
        )
        audio, secs, chunk_ids = [], [], []
        for path, item_id in chunk:
            try:
                wave, sec = load_window(path, sao.window_samples, max_seconds)
            except Exception as err:
                failed.append({"id": str(item_id), "error": repr(err)[:200]})
                continue
            audio.append(wave)
            secs.append(sec)
            chunk_ids.append(str(item_id))
        for wave, item_id in zip(audio, chunk_ids, strict=True):
            gen = torch.Generator(sao.device).manual_seed(derive_seed(master_seed, "vae", item_id))
            latents.append(sao.encode_audio(wave.unsqueeze(0), gen).cpu().half())
        seconds += secs
        kept += chunk_ids
    return {
        "latents": torch.cat(latents),
        "seconds_total": torch.tensor(seconds, dtype=torch.float32),
        "ids": kept,
        "failed": failed,
    }


def save_latents(path: str | Path, cache: dict) -> Path:
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"latents": cache["latents"].contiguous(), "seconds_total": cache["seconds_total"]},
        str(path),
        metadata={"ids": json.dumps(cache["ids"]), "failed": json.dumps(cache["failed"])},
    )
    return path


def load_latents(path: str | Path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
        return {
            "latents": f.get_tensor("latents"),
            "seconds_total": f.get_tensor("seconds_total"),
            "ids": json.loads(meta["ids"]),
            "failed": json.loads(meta.get("failed", "[]")),
        }
