from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audiocfg.metrics.clap import CLAP_SR, Clap, load_mono
from audiocfg.metrics.distribution import ccds, frechet_distance
from audiocfg.repro.config import ScoreConfig

GROUP = ["null", "domain", "prompt_id", "guidance"]


def split_ref(spec: str) -> tuple[str, str | None]:
    repo, _, revision = spec.partition("@")
    return repo, revision or None


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def read_clip(path: Path) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    wave, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(wave.T.copy()), sr


def to_clap(wave: torch.Tensor, sr: int) -> np.ndarray:
    import librosa

    return librosa.resample(wave.mean(0).numpy(), orig_sr=sr, target_sr=CLAP_SR)


def fit_length(wave: np.ndarray, n: int) -> np.ndarray:
    wave = wave[:n]
    return np.pad(wave, (0, n - len(wave)))


def reference_files(root: Path, n: int, seed: int) -> list[Path]:
    files = sorted(root.rglob("*.mp3"))
    pick = np.random.default_rng(seed).choice(len(files), size=min(n, len(files)), replace=False)
    return [files[i] for i in sorted(pick)]


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(1)


def diversity(clips: pd.DataFrame, emb: dict[str, np.ndarray], index: dict[str, int]):
    rows = []
    for key, group in clips.groupby(GROUP):
        idx = [index[c] for c in group["clip"]]
        row = dict(zip(GROUP, key, strict=True)) | {"n": len(idx)}
        row |= {f"ccds_{m}": ccds(e[idx]) for m, e in emb.items()}
        rows.append(row)
    return pd.DataFrame(rows)


def fd_table(clips, emb, index, ref_emb, domain="music"):
    rows = []
    sub = clips[clips["domain"] == domain]
    for (null, g), group in sub.groupby(["null", "guidance"]):
        idx = [index[c] for c in group["clip"]]
        row = {"domain": domain, "null": null, "guidance": g, "n": len(idx)}
        row |= {f"fd_{m}": frechet_distance(e[idx], ref_emb[m]) for m, e in emb.items()}
        rows.append(row)
    return pd.DataFrame(rows)


def score_sweep(
    cfg: ScoreConfig, clips_path: Path, audio_root: Path, reference_root: Path | None, out: Path
) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    clips = pd.read_parquet(clips_path)
    unique = clips.drop_duplicates("clip").reset_index(drop=True)
    index = {c: i for i, c in enumerate(unique["clip"])}
    print(f"{len(clips)} rows, {len(unique)} unique clips", flush=True)

    from audiocfg.metrics.aesthetics import Aesthetics

    aes = Aesthetics(*split_ref(cfg.aesthetics))
    claps = {name: Clap(*split_ref(spec)) for name, spec in cfg.clap_models.items()}
    texts = unique["text"].tolist()
    text_emb = {
        name: np.concatenate([m.embed_text(b) for b in batches(texts, cfg.batch_size)])
        for name, m in claps.items()
    }
    parts, emb = [], {name: [] for name in claps}
    clip_names = unique["clip"].tolist()
    for i, names in enumerate(batches(clip_names, cfg.batch_size)):
        loaded = [read_clip(audio_root / c) for c in names]
        sr = loaded[0][1]
        waves = [w for w, _ in loaded]
        parts.append(aes.score(waves, sr))
        mono = [to_clap(w, sr) for w in waves]
        for name, m in claps.items():
            emb[name].append(m.embed_audio(mono))
        if i % 20 == 0:
            print(f"scored {i * cfg.batch_size + len(names)}/{len(clip_names)}", flush=True)
    emb = {name: np.concatenate(e) for name, e in emb.items()}
    scores = pd.concat(parts, ignore_index=True)
    scores.insert(0, "clip", clip_names)
    for name in claps:
        scores[f"clap_{name}"] = cosine(emb[name], text_emb[name])

    ref_emb = {}
    if reference_root is not None:
        files = reference_files(reference_root, cfg.reference_tracks, cfg.master_seed)
        ref = [load_mono(str(f), offset=cfg.reference_offset, seconds=cfg.seconds) for f in files]
        n = int(cfg.seconds * CLAP_SR)
        ref = [fit_length(w, n) for w in ref if len(w) >= 0.9 * n]
        print(f"reference: {len(ref)} of {len(files)} tracks", flush=True)
        for name, m in claps.items():
            ref_emb[name] = np.concatenate([m.embed_audio(b) for b in batches(ref, cfg.batch_size)])

    primary = unique["domain"].map(cfg.primary_clap)
    scores["clap"] = [scores.at[i, f"clap_{m}"] for i, m in enumerate(primary)]
    rows = clips.drop(columns=["guidance_rms"], errors="ignore").merge(scores, on="clip")

    paths = {"scores": out / "scores.parquet", "diversity": out / "diversity.parquet"}
    rows.to_parquet(paths["scores"], index=False)
    div = diversity(rows, emb, index)
    div["ccds"] = [r[f"ccds_{cfg.primary_clap[r['domain']]}"] for _, r in div.iterrows()]
    div.to_parquet(paths["diversity"], index=False)
    np.savez_compressed(out / "embeddings.npz", clips=unique["clip"].to_numpy(), **emb)
    if ref_emb:
        paths["fd"] = out / "fd_music.csv"
        fd_table(rows, emb, index, ref_emb).to_csv(paths["fd"], index=False, lineterminator="\n")
        np.savez_compressed(out / "reference_embeddings.npz", **ref_emb)
    return paths
