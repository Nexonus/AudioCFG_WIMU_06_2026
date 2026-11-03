"""E1 metrics (§5.1, §6). CLI: ``audiocfg metrics``.

* ``audiocfg metrics --calibrate-clap`` -> ``results/clap_calibration.csv`` from the real
  reference recordings listed in ``data/prompts.csv`` (``reference_audio``), per domain and
  CLAP model (§6.1). Run once before the per-model pass.
* ``audiocfg metrics --model sao`` -> ``results/e1_per_prompt.parquet`` (CCDS with CLAP and the
  aux extractor PANNs/MERT, CLAP_norm, KL_passt) and ``results/e1_fd.parquet`` (FD per setting
  with bootstrap CI over prompts, equal N, fixed reference set per domain).
* ``--embedder-override random`` swaps every model for a deterministic numpy projection
  (dry run of the full pipeline without torch).

Clips are loudness-normalised to the config LUFS before any embedding (§4.3).
"""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np

AUX_EMBEDDER = {"sfx": "panns", "music": "mert"}  # second extractor for CCDS (§5.1)
PRIMARY_CLAP = "laion_clap"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None, help="model name from the config (e.g. sao)")
    p.add_argument("--calibrate-clap", action="store_true",
                   help="only compute clap_calibration.csv from real reference audio")
    p.add_argument("--clap-models", nargs="*", default=[PRIMARY_CLAP, "ms_clap"])
    p.add_argument("--fd-embedders", nargs="*", default=["openl3"], help="e.g. openl3 vggish")
    p.add_argument("--embedder-override", default=None, choices=["random"])
    p.add_argument("--n-boot", type=int, default=None, help="default: config stats.bootstrap_reps")
    p.add_argument("--seed", type=int, default=0)


# ---------------------------------------------------------------- pure computation

def calibration_table(prompts, ref_audios: list[np.ndarray], sr: int, clap_models: list,
                      seed: int = 0):
    """One row per (domain, clap_model): C_real, C_mis, n. ``ref_audios`` align with ``prompts``."""
    import pandas as pd

    from audiocfg.metrics.clap_score import calibrate

    prompts = prompts.reset_index(drop=True)
    rows = []
    for clap in clap_models:
        for domain in sorted(prompts["domain"].unique()):
            idx = np.flatnonzero(prompts["domain"].to_numpy() == domain)
            if len(idx) < 2:
                continue
            texts = list(prompts["text"].iloc[idx])
            a = clap.embed([ref_audios[i] for i in idx], sr)
            t = clap.embed_text(texts)
            cal = calibrate(a, t, t, np.random.default_rng(seed))
            rows.append((domain, clap.name, cal.c_real, cal.c_mis, cal.n))
    return pd.DataFrame(rows, columns=["domain", "clap_model", "c_real", "c_mis", "n"])


def compute_e1(meta, audios: list[np.ndarray], sr: int, get: Callable[[str], object],
               calibration=None, prompt_refs: dict[str, np.ndarray] | None = None,
               ref_sets: dict[str, list[np.ndarray]] | None = None,
               fd_embedders: tuple[str, ...] = ("openl3",), n_boot: int = 1000, seed: int = 0):
    """Compute ``(e1_per_prompt, e1_fd)`` DataFrames from in-memory clips.

    ``get(name)`` returns an embedder. ``calibration`` is a clap_calibration DataFrame (rows for
    ``PRIMARY_CLAP``); ``prompt_refs`` maps prompt_id -> its real recording (for KL);
    ``ref_sets`` maps domain -> real reference clips (for FD). Missing inputs give NaN.
    """
    import pandas as pd

    from audiocfg.metrics.clap_score import Calibration, clap_norm, clap_scores
    from audiocfg.metrics.diversity import GROUP_KEYS, ccds
    from audiocfg.metrics.fad import equal_n_indices, fd_with_ci
    from audiocfg.metrics.kl import kl_per_prompt

    meta = meta.reset_index(drop=True)
    prompt_refs = prompt_refs or {}
    ref_sets = ref_sets or {}
    clap = get(PRIMARY_CLAP)
    clap_a = clap.embed(audios, sr)
    clap_s = clap_scores(clap.embed_text(list(meta["text"])), clap_a)
    passt = get("passt")
    gen_p = passt.label_probs(audios, sr)
    ref_ids = [p for p in meta["prompt_id"].unique() if p in prompt_refs]
    ref_p = dict(zip(ref_ids, passt.label_probs([prompt_refs[p] for p in ref_ids], sr))) if ref_ids else {}

    per_prompt = []
    fd_rows = []
    for domain, dmeta in meta.groupby("domain", sort=True):
        di = dmeta.index.to_numpy()
        aux = get(AUX_EMBEDDER.get(domain, "panns"))
        aux_e = aux.embed([audios[i] for i in di], sr)
        aux_of = dict(zip(di, aux_e))
        cal = None
        if calibration is not None:
            c = calibration[(calibration["domain"] == domain) & (calibration["clap_model"] == clap.name)]
            if len(c):
                cal = Calibration(float(c["c_real"].iloc[0]), float(c["c_mis"].iloc[0]), int(c["n"].iloc[0]))
        for key, idx in dmeta.groupby(list(GROUP_KEYS), sort=True).indices.items():
            rows = di[idx]
            pid = key[GROUP_KEYS.index("prompt_id")]
            per_prompt.append((*key, ccds(clap_a[rows]), ccds(np.stack([aux_of[r] for r in rows])),
                               float(clap_norm(clap_s[rows].mean(), cal)) if cal else np.nan,
                               kl_per_prompt(gen_p[rows], ref_p[pid]) if pid in ref_p else np.nan))
        # FD: one reference set per (model, domain), equal N per (null, w) setting.
        refs = ref_sets.get(domain, [])
        settings = {k: di[v] for k, v in dmeta.groupby(["model", "null", "w"], sort=True).indices.items()}
        settings = equal_n_indices(settings, seed=seed)
        for name in fd_embedders:
            e = get(name)
            if len(refs) < 2:
                for (model, null, w), rows in settings.items():
                    fd_rows.append((model, domain, null, w, name, len(rows), np.nan, np.nan, np.nan))
                continue
            ref_e = e.embed(refs, sr)
            emb_of = dict(zip(di, e.embed([audios[i] for i in di], sr)))
            for (model, null, w), rows in settings.items():
                g = np.stack([emb_of[r] for r in rows])
                v, lo, hi = fd_with_ci(g, ref_e, meta["prompt_id"].to_numpy()[rows], n_boot, seed=seed)
                fd_rows.append((model, domain, null, w, name, len(rows), v, lo, hi))

    from audiocfg.schema import RESULT_FILES

    return (pd.DataFrame(per_prompt, columns=list(RESULT_FILES["e1_per_prompt.parquet"])),
            pd.DataFrame(fd_rows, columns=list(RESULT_FILES["e1_fd.parquet"])))


# ---------------------------------------------------------------- CLI

def _cfg(args):
    from audiocfg.config import load_config

    return load_config(args.config) if getattr(args, "config", None) else load_config()


def _references(cfg, sr: int):
    """prompts.csv rows and their (loudness-normalised) reference audio; missing files -> None."""
    import pandas as pd

    from audiocfg.metrics._io import existing, load_clips, warn

    prompts = pd.read_csv(cfg.prompts_csv)
    paths = prompts["reference_audio"].fillna("").astype(str).tolist()
    ok = existing(paths)
    if not ok.all():
        warn(f"{(~ok).sum()} of {len(ok)} reference_audio files missing (TODO(M1): download)")
    loaded, _ = load_clips([p for p, k in zip(paths, ok) if k], cfg.loudness_lufs, sr)
    it = iter(loaded)
    return prompts, [next(it) if k else None for k in ok]


def run(args: argparse.Namespace) -> None:
    import pandas as pd

    from audiocfg.metrics._io import load_clips, warn, write_result
    from audiocfg.metrics.embedders import get_embedder

    cfg = _cfg(args)
    cache: dict[str, object] = {}

    def get(name: str):
        if name not in cache:
            cache[name] = get_embedder(name, args.embedder_override)
        return cache[name]

    if args.calibrate_clap:
        sr = 48000
        prompts, refs = _references(cfg, sr)
        have = np.array([r is not None for r in refs])
        if have.sum() < 2:
            raise SystemExit("not enough reference audio to calibrate CLAP")
        df = calibration_table(prompts[have], [r for r in refs if r is not None], sr,
                               [get(n) for n in args.clap_models], seed=args.seed)
        print(write_result(df, cfg.results_dir, "clap_calibration.csv",
                           replace_on={"clap_model": [get(n).name for n in args.clap_models]}))
        return

    if not args.model:
        raise SystemExit("--model is required (or use --calibrate-clap)")
    # `sweep --backend fake` writes under "<model>_fake"; the sample rate is the real model's.
    sr = cfg.models[args.model.removesuffix("_fake")].sample_rate
    meta = pd.read_parquet(cfg.outputs_dir / args.model / "meta.parquet")
    # meta.path is ClipSpec.relpath() (relative to outputs_dir) or absolute.
    audios, _ = load_clips(list(meta["path"]), cfg.loudness_lufs, sr, base=cfg.outputs_dir)
    prompts, refs = _references(cfg, sr)
    prompt_refs = {p: r for p, r in zip(prompts["prompt_id"], refs) if r is not None}
    ref_sets = {d: [r for r, dd in zip(refs, prompts["domain"]) if r is not None and dd == d]
                for d in prompts["domain"].unique()}
    # TODO(M1): add further AudioCaps-test clips to the SFX reference set (§6.2).
    cal_path = cfg.results_dir / "clap_calibration.csv"
    calibration = pd.read_csv(cal_path) if cal_path.exists() else None
    if calibration is None:
        warn("clap_calibration.csv missing; clap_norm will be NaN (run --calibrate-clap)")
    n_boot = args.n_boot if args.n_boot is not None else int(cfg.section("stats")["bootstrap_reps"])
    per_prompt, fd_df = compute_e1(meta, audios, sr, get, calibration, prompt_refs, ref_sets,
                                   tuple(args.fd_embedders), n_boot, args.seed)
    print(write_result(per_prompt, cfg.results_dir, "e1_per_prompt.parquet", {"model": args.model}))
    print(write_result(fd_df, cfg.results_dir, "e1_fd.parquet", {"model": args.model}))
