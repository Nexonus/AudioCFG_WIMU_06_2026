from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audiocfg.analysis.bootstrap import paired_ci

CONDITIONS = {"matched_clap": (7.0, 2.0), "same_g": (7.0, 7.0)}
QUESTIONS = ("quality", "text")


def make_pairs(clips: pd.DataFrame, domain="music", conditions=CONDITIONS, seed=2026):
    rng = np.random.default_rng(seed)
    sub = clips[clips["domain"] == domain]
    prompts = sub.drop_duplicates("prompt_id").sort_values("prompt_id")
    draws = rng.integers(0, sub["sample"].max() + 1, len(prompts))
    samples = dict(zip(prompts["prompt_id"], draws, strict=True))
    rows = []
    for cond, (g_learned, g_zero) in conditions.items():
        for _, p in prompts.iterrows():
            same = sub["prompt_id"] == p["prompt_id"]
            pick = sub[same & (sub["sample"] == samples[p["prompt_id"]])]
            pick = pick.set_index(["null", "guidance"])["clip"]
            learned, zero = pick[("learned", g_learned)], pick[("zero", g_zero)]
            a_learned = bool(rng.integers(2))
            rows.append(
                {
                    "condition": cond,
                    "prompt_id": p["prompt_id"],
                    "text": p["text"],
                    "sample": samples[p["prompt_id"]],
                    "g_learned": g_learned,
                    "g_zero": g_zero,
                    "clip_a": learned if a_learned else zero,
                    "clip_b": zero if a_learned else learned,
                    "a_is": "learned" if a_learned else "zero",
                }
            )
    pairs = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    pairs.insert(0, "pair_id", [f"p{i:02d}" for i in range(len(pairs))])
    return pairs


def tally(answers: pd.DataFrame, pairs: pd.DataFrame, seed=0) -> pd.DataFrame:
    df = answers.merge(pairs[["pair_id", "condition", "a_is"]], on="pair_id")
    rows = []
    for cond, part in df.groupby("condition"):
        for q in QUESTIONS:
            learned = ((part[q] == "A") == (part["a_is"] == "learned")).astype(float)
            per_pair = learned.groupby(part["pair_id"]).mean()
            ci = paired_ci(per_pair.to_numpy(), seed=seed)
            rows.append(
                {"condition": cond, "question": q, "listeners": part["listener"].nunique()}
                | {"learned_pref": ci["mean"], "lo": ci["lo"], "hi": ci["hi"], "pairs": ci["n"]}
                | {"answers": len(part)}
            )
    return pd.DataFrame(rows)


def level_match(waves, rms_db=-20.0, peak_db=-1.0):
    waves = [w * 10 ** (rms_db / 20) / (np.sqrt(np.mean(w**2)) + 1e-12) for w in waves]
    peak = max(np.abs(w).max() for w in waves)
    return [w * min(1.0, 10 ** (peak_db / 20) / peak) for w in waves]


def export_audio(pairs: pd.DataFrame, audio_root: Path, out: Path) -> Path:
    import soundfile as sf

    (out / "audio").mkdir(parents=True, exist_ok=True)
    items = []
    for _, r in pairs.iterrows():
        loaded = [sf.read(audio_root / r[c], dtype="float32") for c in ("clip_a", "clip_b")]
        files = {}
        for side, wave in zip("ab", level_match([w for w, _ in loaded]), strict=True):
            files[side] = f"audio/{r['pair_id']}{side}.mp3"
            sf.write(out / files[side], wave, loaded[0][1], format="MP3")
        items.append({"id": r["pair_id"], "text": r["text"]} | files)
    page = (Path(__file__).with_name("page.html")).read_text(encoding="utf-8")
    path = out / "index.html"
    path.write_text(
        page.replace("__PAIRS__", json.dumps(items, ensure_ascii=False)), encoding="utf-8"
    )
    return path


def load_answers(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
