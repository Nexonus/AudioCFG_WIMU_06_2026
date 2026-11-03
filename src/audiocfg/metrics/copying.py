"""E2: training-data copying, two stages with a held-out control (§5.2). CLI: ``audiocfg copies``.

* **T** (training set): clips downloaded from SAO's published attribution lists (Freesound IDs,
  FMA tracks). TODO(M1): verify where the lists live in the HF model repo
  (stabilityai/stable-audio-open-1.0) and build ``data/e2_train.csv``.
* **H** (held-out control): Freesound/FMA clips **not** on the lists, especially uploaded after
  SAO's data cutoff, matched to T in category and length -> ``data/e2_held.csv``.
  Manifests have columns ``id, domain, path``.

Stage 1 (candidates): nearest neighbour of each generation in T by cosine similarity of an
embedding (CLAP, PANNs) -> ``s_gen``; the same for every H clip -> ``s_held``. Candidate iff
``s_gen`` > ``candidate_percentile`` of ``s_held`` (per domain and embedder).

Stage 2 (verification): signal-level match (max normalised log-mel cross-correlation over time
lags; chroma as well for music) of each candidate with its NN. Confirmed copy iff the score
exceeds ``verify_percentile`` of the H→NN-in-T match scores ("same kind, different recording").

T is only a subset of the training data, so the detected copy rate is a **lower bound**; the
main claim is the **trend over w**, since the same T applies to every scale.
"""

from __future__ import annotations

import argparse

import numpy as np

MIN_OVERLAP_FRAMES = 20


# ---------------------------------------------------------------- stage 1: embedding NN

def nearest_neighbors(query_emb: np.ndarray, train_emb: np.ndarray,
                      chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Cosine NN of each query row in ``train_emb``: ``(idx (n,), sim (n,))``. FAISS if available."""
    q = np.ascontiguousarray(query_emb, dtype=np.float32)
    t = np.ascontiguousarray(train_emb, dtype=np.float32)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    try:
        import faiss  # TODO(M1): untested here (no faiss in dev venv)

        index = faiss.IndexFlatIP(t.shape[1])
        index.add(t)
        sim, idx = index.search(q, 1)
        return idx[:, 0].astype(np.int64), sim[:, 0].astype(np.float64)
    except ImportError:
        idx = np.empty(len(q), dtype=np.int64)
        sim = np.empty(len(q), dtype=np.float64)
        for i in range(0, len(q), chunk):
            s = q[i:i + chunk] @ t.T
            idx[i:i + chunk] = s.argmax(axis=1)
            sim[i:i + chunk] = s[np.arange(len(s)), idx[i:i + chunk]]
        return idx, sim


def candidate_threshold(s_held: np.ndarray, percentile: float) -> float:
    """Stage-1 threshold: ``percentile`` of the held-out NN similarities."""
    return float(np.percentile(np.asarray(s_held, dtype=np.float64), percentile))


# ---------------------------------------------------------------- stage 2: signal match

def _features(x: np.ndarray, sr: int, kind: str) -> np.ndarray:
    from audiocfg.metrics.dsp import chroma, log_mel

    if kind == "logmel":
        return log_mel(x, sr, n_fft=2048, hop=512, n_mels=64)
    if kind == "chroma":
        return chroma(x, sr)
    raise ValueError(f"unknown feature {kind!r}")


def max_normalized_xcorr(fa: np.ndarray, fb: np.ndarray, min_overlap: int = MIN_OVERLAP_FRAMES) -> float:
    """Max over time lags of the normalised correlation of two feature maps ``(frames, bins)``.

    For each lag the overlapping frames are centred **per bin** (removing the shared spectral
    envelope, so "same kind of sound" does not score high) and the cosine of the flattened
    residuals is taken. Lags with fewer than ``min_overlap`` frames are skipped.
    """
    na, nb = len(fa), len(fb)
    min_overlap = min(min_overlap, na, nb)
    best = -1.0
    for lag in range(-(nb - min_overlap), na - min_overlap + 1):
        a0, b0 = max(lag, 0), max(-lag, 0)
        L = min(na - a0, nb - b0)
        if L < min_overlap:
            continue
        x = fa[a0:a0 + L]
        y = fb[b0:b0 + L]
        x = x - x.mean(axis=0)
        y = y - y.mean(axis=0)
        den = np.sqrt(np.sum(x * x) * np.sum(y * y))
        if den > 0:
            best = max(best, float(np.sum(x * y) / den))
    return best


def signal_match(a: np.ndarray, b: np.ndarray, sr: int, kind: str = "logmel") -> float:
    """Stage-2 score in [-1, 1]: max normalised cross-correlation of log-mel (or chroma) over lags.

    TODO(M1): optional Chromaprint fingerprint match for music (pyacoustid).
    """
    return max_normalized_xcorr(_features(a, sr, kind), _features(b, sr, kind))


def domain_match(a: np.ndarray, b: np.ndarray, sr: int, domain: str) -> float:
    """log-mel score; for music the max of log-mel and chroma scores (§5.2)."""
    s = signal_match(a, b, sr, "logmel")
    return max(s, signal_match(a, b, sr, "chroma")) if domain == "music" else s


# ---------------------------------------------------------------- CLI

def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="sao", help="E2 runs on SAO only (§5.2)")
    p.add_argument("--train-manifest", default="data/e2_train.csv", help="T: id,domain,path")
    p.add_argument("--held-manifest", default="data/e2_held.csv", help="H: id,domain,path")
    p.add_argument("--embedders", nargs="*", default=None, help="default: config e2_copying.embedders")
    p.add_argument("--embedder-override", default=None, choices=["random"],
                   help="replace every embedder with a deterministic random projection (dry run)")
    p.add_argument("--sr", type=int, default=22050, help="common rate for the signal match")


def run(args: argparse.Namespace) -> None:
    import pandas as pd

    from audiocfg.config import load_config
    from audiocfg.metrics._io import load_clips, resolve, write_result
    from audiocfg.metrics.embedders import get_embedder

    cfg = load_config(args.config) if getattr(args, "config", None) else load_config()
    sec = cfg.section("e2_copying")
    names = args.embedders or list(sec["embedders"])
    meta = pd.read_parquet(cfg.outputs_dir / args.model / "meta.parquet")
    train = pd.read_csv(resolve(args.train_manifest))
    held = pd.read_csv(resolve(args.held_manifest))

    copies, helds = [], []
    for domain in sorted(meta["domain"].unique()):
        g = meta[meta["domain"] == domain].reset_index(drop=True)
        t = train[train["domain"] == domain].reset_index(drop=True)
        h = held[held["domain"] == domain].reset_index(drop=True)
        if len(t) == 0 or len(h) == 0:
            print(f"[copies] skip {domain}: empty T or H")
            continue
        sr = args.sr
        # TODO(M2): stream embeddings in batches; ~10k clips per model do not fit in RAM at once
        ga, _ = load_clips(list(g["path"]), cfg.loudness_lufs, sr, base=cfg.outputs_dir)
        ta, _ = load_clips(list(t["path"]), cfg.loudness_lufs, sr)
        ha, _ = load_clips(list(h["path"]), cfg.loudness_lufs, sr)
        for name in names:
            emb = get_embedder(name, args.embedder_override)
            te = emb.embed(ta, sr)
            g_idx, s_gen = nearest_neighbors(emb.embed(ga, sr), te)
            h_idx, s_held = nearest_neighbors(emb.embed(ha, sr), te)
            thr = candidate_threshold(s_held, float(sec["candidate_percentile"]))
            h_match = np.array([domain_match(ha[i], ta[j], sr, domain) for i, j in enumerate(h_idx)])
            v_thr = candidate_threshold(h_match, float(sec["verify_percentile"]))
            is_cand = s_gen > thr
            match = np.full(len(g), np.nan)
            for i in np.flatnonzero(is_cand):
                match[i] = domain_match(ga[i], ta[g_idx[i]], sr, domain)
            copies.append(pd.DataFrame({
                "clip_id": g["clip_id"], "model": g["model"], "null": g["null"], "w": g["w"],
                "prompt_id": g["prompt_id"], "seed": g["seed"], "domain": domain,
                "embedder": name,
                "nn_train_id": t["id"].to_numpy()[g_idx].astype(str), "s_gen": s_gen,
                "is_candidate": is_cand, "match_score": match,
                "is_copy": is_cand & (np.nan_to_num(match, nan=-np.inf) > v_thr),
            }))
            helds.append(pd.DataFrame({
                "held_id": h["id"].astype(str), "domain": domain, "embedder": name,
                "nn_train_id": t["id"].to_numpy()[h_idx].astype(str), "s_held": s_held,
                "match_score": h_match,
            }))
            print(f"[copies] {domain}/{name}: cand thr={thr:.4f} ({is_cand.mean():.2%}), "
                  f"verify thr={v_thr:.4f}")
    if copies:
        write_result(pd.concat(copies, ignore_index=True), cfg.results_dir, "e2_copies.parquet",
                     replace_on={"model": args.model})
        write_result(pd.concat(helds, ignore_index=True), cfg.results_dir, "e2_held.parquet",
                     replace_on={"embedder": names})
