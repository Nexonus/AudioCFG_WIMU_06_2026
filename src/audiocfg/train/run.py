from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from audiocfg.analysis.bootstrap import paired_ci
from audiocfg.data.latents import encode_files, load_latents, save_latents
from audiocfg.nulls.variants import save_null, t5_empty_null, zero_null
from audiocfg.repro.config import NullTrainConfig
from audiocfg.repro.manifest import build_manifest, write_manifest
from audiocfg.train.loop import train_null
from audiocfg.train.objective import draws
from audiocfg.train.probe import eval_losses, null_states, text_states


def _cached(path: Path, make) -> dict:
    if path.exists():
        return load_latents(path)
    cache = make()
    save_latents(path, cache)
    return cache


def run_null_experiment(
    sao, cfg: NullTrainConfig, fma_root: Path, song_describer_audio: Path, out: Path
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    tracks = pd.read_csv(cfg.null_tracks)
    sanity = pd.read_csv(cfg.sanity_tracks, dtype={"track_id": str})
    report: dict = {"config_hash": cfg.config_hash()}

    def encode(rows, root, col, ids):
        paths = [str(root / p) for p in rows[col]]
        return lambda: encode_files(sao, paths, ids, cfg.master_seed, cfg.max_seconds)

    caches = {}
    for split in ("train", "val"):
        rows = tracks[tracks.split == split]
        ids = rows["id"].astype(str).tolist()
        caches[split] = _cached(
            out / f"latents_{split}.safetensors", encode(rows, fma_root, "fma_large_path", ids)
        )
    caches["sanity"] = _cached(
        out / "latents_sanity.safetensors",
        encode(sanity, song_describer_audio, "ref_audio", sanity["track_id"].tolist()),
    )
    report["latents"] = {
        k: {"n": len(v["ids"]), "failed": len(v["failed"])} for k, v in caches.items()
    }
    shape = tuple(caches["train"]["latents"].shape[1:])

    def losses(cache, k, stream, states, needs_seconds=False, items=None):
        ids = cache["ids"] if items is None else cache["ids"][:items]
        n = len(ids)
        t, noise = draws(cfg.master_seed, stream, ids, k, shape)
        return eval_losses(
            sao,
            cache["latents"][:n],
            cache["seconds_total"][:n],
            t,
            noise,
            states,
            cfg.eval_batch,
            needs_seconds,
        ).mean(1)

    s = caches["sanity"]
    by_id = sanity.set_index("track_id")
    own = [by_id.loc[i, "caption"] for i in s["ids"]]
    wrong = [by_id.loc[i, "wrong_caption"] for i in s["ids"]]
    zero = zero_null(sao.text_shape)
    sanity_loss = {
        "own_caption": losses(s, cfg.sanity_draws, "sanity", text_states(sao, own), True),
        "zero_null": losses(s, cfg.sanity_draws, "sanity", null_states(zero)),
        "wrong_caption": losses(s, cfg.sanity_draws, "sanity", text_states(sao, wrong), True),
    }
    pd.DataFrame({"id": s["ids"], **{k: v.numpy() for k, v in sanity_loss.items()}}).to_csv(
        out / "sanity_losses.csv", index=False
    )
    report["sanity"] = {
        "own_vs_zero": paired_ci(sanity_loss["zero_null"] - sanity_loss["own_caption"]),
        "wrong_vs_zero": paired_ci(sanity_loss["wrong_caption"] - sanity_loss["zero_null"]),
    }
    print(json.dumps(report["sanity"], indent=1), flush=True)

    val = caches["val"]
    nulls = {"zero": zero, "t5_empty": t5_empty_null(sao)}
    val_loss = {
        name: losses(val, cfg.val_draws, "val", null_states(n)) for name, n in nulls.items()
    }

    def quick_val(null: torch.Tensor) -> float:
        return float(losses(val, 2, "val_quick", null_states(null), items=cfg.eval_items).mean())
    for init in cfg.inits:
        name = f"learned_{init}"
        null, history = train_null(
            sao,
            caches["train"]["latents"],
            caches["train"]["seconds_total"],

            nulls[init],
            steps=cfg.steps,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            master_seed=cfg.master_seed,
            warmup=cfg.warmup,
            weight_decay=cfg.weight_decay,
            grad_clip=cfg.grad_clip,
            eval_every=cfg.eval_every,
            eval_fn=quick_val,
            log=lambda row: print(row, flush=True),
        )
        save_null(out / f"{cfg.name}_{name}.safetensors", null, {"init": init, **report})
        pd.DataFrame(history).to_csv(out / f"history_{name}.csv", index=False)
        val_loss[name] = losses(val, cfg.val_draws, "val", null_states(null))
        report[f"p0_{name}"] = paired_ci(val_loss["zero"] - val_loss[name])
        print(name, report[f"p0_{name}"], flush=True)

    report["val_mean_loss"] = {k: float(v.mean()) for k, v in val_loss.items()}
    report["t5_empty_vs_zero"] = paired_ci(val_loss["zero"] - val_loss["t5_empty"])
    pd.DataFrame({"id": val["ids"], **{k: v.numpy() for k, v in val_loss.items()}}).to_csv(
        out / "val_losses.csv", index=False
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    outputs = {p.name: p for p in sorted(out.glob("*")) if p.suffix in {".csv", ".json"}}
    outputs |= {p.name: p for p in out.glob(f"{cfg.name}_*.safetensors")}
    write_manifest(
        out / "manifest.json",
        build_manifest(
            cfg,
            inputs={"null_tracks": cfg.null_tracks, "sanity_tracks": cfg.sanity_tracks},
            outputs=outputs,
        ),
    )
    return report
