"""E1 / E3: resumable CFG sweep, prompts x seeds x w for one model and one null (§4.3, §5.1, §5.3).

Only ``w`` (and, across runs, the null variant) varies; prompt, seed, sampler, steps and
length are fixed per model. Each clip is written to ``outputs_dir / spec.relpath()`` and
described by one row in ``outputs_dir/<model>/meta.parquet`` (``CLIP_COLUMNS``).

Stored wavs are the raw model output: loudness normalisation to -23 LUFS is NOT applied here;
metrics and the listening test apply it on load (§4.3). Existing wavs are skipped, so an
interrupted sweep continues where it stopped.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import pandas as pd

from audiocfg.schema import CLIP_COLUMNS, ClipSpec

FLUSH_EVERY = 20  # clips between meta.parquet writes


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="model key from experiment.yaml (sao, audioldm2)")
    p.add_argument("--null", default="default", help="null variant: default | learned | t5_empty | mean_caption")
    p.add_argument("--null-path", default=None, help="tensor file for --null learned (output of train-null)")
    p.add_argument("--prompts", default=None, help="prompts CSV (default: config prompts_csv)")
    p.add_argument("--limit-prompts", type=int, default=None, help="first N prompts, balanced over domains (M1 pilot: 20)")
    p.add_argument("--scales", type=float, nargs="+", default=None, help="override guidance_scales")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="override seeds")
    p.add_argument("--backend", default=None, choices=["fake", "sao", "audioldm2"],
                   help="default: same as --model; 'fake' writes under '<model>_fake' (no GPU)")
    p.add_argument("--device", default=None)


def limit_prompts(prompts: pd.DataFrame, n: int | None) -> pd.DataFrame:
    """Keep ``n`` prompts split evenly over domains (file order within a domain)."""
    if n is None or n >= len(prompts):
        return prompts
    per = math.ceil(n / prompts["domain"].nunique())
    return prompts.groupby("domain", sort=False).head(per).head(n)


def meta_path(outputs_dir: Path, model: str) -> Path:
    return Path(outputs_dir) / model / "meta.parquet"


def write_meta(path: Path, rows: list[dict]) -> pd.DataFrame:
    """Merge ``rows`` into ``path``, deduplicated by ``clip_id`` (newest wins)."""
    new = pd.DataFrame(rows, columns=list(CLIP_COLUMNS))
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    df = new.drop_duplicates("clip_id", keep="last").reset_index(drop=True)[list(CLIP_COLUMNS)]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return df


def run_sweep(cfg, model: str, null: str = "default", null_path: str | None = None,
              prompts: pd.DataFrame | None = None, scales=None, seeds=None,
              backend_name: str | None = None, device: str | None = None) -> dict[str, int]:
    """Run the sweep; returns ``{"generated": n, "skipped": m}``."""
    from audiocfg.audio_io import save_audio
    from audiocfg.generate.backends import get_backend

    mcfg = cfg.models[model]
    backend_name = backend_name or model
    out_model = f"{model}_fake" if backend_name == "fake" else model
    if prompts is None:
        prompts = pd.read_csv(cfg.prompts_csv)
    scales = tuple(scales or cfg.guidance_scales)
    seeds = tuple(seeds if seeds is not None else cfg.seeds)

    backend = get_backend(backend_name, mcfg, **({"device": device} if device else {}))
    captions = prompts["text"].tolist() if null == "mean_caption" else None
    null_cond = backend.null_condition(null, path=null_path, captions=captions)

    mpath = meta_path(cfg.outputs_dir, out_model)
    pending: list[dict] = []
    counts = {"generated": 0, "skipped": 0}
    known = set(pd.read_parquet(mpath, columns=["clip_id"])["clip_id"]) if mpath.exists() else set()
    try:
        for p in prompts.itertuples(index=False):
            for seed in seeds:
                for w in scales:  # innermost: same prompt + seed, only w changes (§4.3)
                    spec = ClipSpec(model=out_model, prompt_id=p.prompt_id, domain=p.domain, text=p.text,
                                    w=float(w), null=null, seed=int(seed), sampler=mcfg.sampler,
                                    steps=mcfg.steps, seconds=cfg.clip_seconds)
                    wav = Path(cfg.outputs_dir) / spec.relpath()
                    if wav.exists():
                        counts["skipped"] += 1
                        if spec.clip_id not in known:  # wav without a meta row: recover it
                            pending.append(spec.to_row(spec.relpath(), backend.sample_rate,
                                                       float("nan")))
                        continue
                    t0 = time.perf_counter()
                    audio, sr = backend.generate(spec, null_cond)
                    gen_time = time.perf_counter() - t0
                    tmp = wav.with_name(wav.stem + ".partial.wav")
                    save_audio(tmp, audio, sr)
                    os.replace(tmp, wav)  # a wav exists only once fully written
                    # path relative to outputs_dir, so meta.parquet is portable between machines
                    pending.append(spec.to_row(spec.relpath(), sr, gen_time))
                    counts["generated"] += 1
                    if len(pending) >= FLUSH_EVERY:
                        write_meta(mpath, pending)
                        pending.clear()
    finally:
        if pending:
            write_meta(mpath, pending)
    return counts


def run(args: argparse.Namespace) -> None:
    from audiocfg.config import DEFAULT_CONFIG, load_config

    cfg = load_config(args.config or DEFAULT_CONFIG)
    prompts = pd.read_csv(args.prompts or cfg.prompts_csv)
    prompts = limit_prompts(prompts, args.limit_prompts)
    counts = run_sweep(cfg, args.model, null=args.null, null_path=args.null_path, prompts=prompts,
                       scales=args.scales, seeds=args.seeds, backend_name=args.backend, device=args.device)
    print(f"sweep {args.model}/{args.null}: {counts['generated']} generated, {counts['skipped']} skipped")
