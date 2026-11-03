"""Synthetic ``results/`` set with plausible trends, for dry-running ``analyze --synthetic``.

NOT research data. Shapes follow ``schema.RESULT_FILES`` exactly; trends follow the hypotheses
(H1: CCDS falls / CLAP saturates with w; H2: candidates rise with w; H3: learned null lowers FD
at high w; H4: judge prefers a higher w than listeners) so every table and figure has content.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from audiocfg.schema import DOMAINS, QUESTIONS, RESULT_FILES


def _write(df: pd.DataFrame, results_dir: Path, name: str) -> None:
    df = df[list(RESULT_FILES[name])]
    path = results_dir / name
    if name.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def write_synthetic(results_dir: Path, scales=(1, 3, 5, 7, 9), seeds=tuple(range(10)),
                    n_prompts: int = 16, n_e4_prompts: int = 12, n_listeners: int = 12,
                    pairs_per_listener: int = 60, seed: int = 0) -> Path:
    """Write every file of ``RESULT_FILES`` into ``results_dir``; returns the directory."""
    rng = np.random.default_rng(seed)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    scales = [float(w) for w in scales]
    prompts = {d: [f"{d}_{i:03d}" for i in range(n_prompts)] for d in DOMAINS}
    nulls = {"sao": ["default", "learned"], "audioldm2": ["default"]}

    # ---- E1 per prompt + FD
    rows, fd_rows = [], []
    for model, nl in nulls.items():
        for dom in DOMAINS:
            peff = rng.normal(0, 0.03, n_prompts)
            for null in nl:
                learned = null == "learned"
                for w in scales:
                    for i, pid in enumerate(prompts[dom]):
                        ccds = 0.55 - 0.025 * w + (0.008 * w if learned else 0) + peff[i]
                        rows.append({
                            "model": model, "domain": dom, "null": null, "w": w,
                            "prompt_id": pid,
                            "ccds_clap": ccds + rng.normal(0, 0.02),
                            "ccds_aux": 0.9 * ccds + rng.normal(0, 0.02),
                            "clap_norm": 0.85 * (1 - np.exp(-w / 2.2)) + rng.normal(0, 0.06),
                            "kl_passt": 2.0 - 0.12 * w + 0.012 * w ** 2 + rng.normal(0, 0.1)})
                    for emb in ("openl3", "vggish"):
                        curv = 0.05 if learned else 0.09
                        fd = (60 if emb == "openl3" else 4) * (1 + curv * (w - 6) ** 2 / 4)
                        h = 0.04 * fd
                        fd_rows.append({"model": model, "domain": dom, "null": null, "w": w,
                                        "embedder": emb, "n": n_prompts * len(seeds), "fd": fd,
                                        "ci_low": fd - h, "ci_high": fd + h})
    _write(pd.DataFrame(rows), results_dir, "e1_per_prompt.parquet")
    _write(pd.DataFrame(fd_rows), results_dir, "e1_fd.parquet")
    _write(pd.DataFrame([{"domain": d, "clap_model": c, "c_real": 0.45, "c_mis": 0.08, "n": 500}
                         for d in DOMAINS for c in ("laion_clap", "ms_clap")]),
           results_dir, "clap_calibration.csv")

    # ---- E2 copies (SAO only) + held-out control
    held, cop = [], []
    for emb in ("clap", "panns"):
        for dom in DOMAINS:
            s = np.clip(rng.normal(0.55, 0.08, 400), -1, 1)
            m = rng.normal(0.3, 0.08, 400)
            held += [{"held_id": f"h_{dom}_{emb}_{i}", "domain": dom, "embedder": emb,
                      "nn_train_id": f"t{rng.integers(1e5)}", "s_held": a, "match_score": b}
                     for i, (a, b) in enumerate(zip(s, m))]
    held = pd.DataFrame(held)
    for emb in ("clap", "panns"):
        he = held[held["embedder"] == emb]
        thr_c, thr_m = np.percentile(he["s_held"], 99), np.percentile(he["match_score"], 99.9)
        for null in nulls["sao"]:
            for w in scales:
                for dom in DOMAINS:
                    for pid in prompts[dom]:
                        for sd in seeds:
                            s = float(np.clip(rng.normal(0.6 + 0.022 * w, 0.08), -1, 1))
                            cand = s > thr_c
                            ms = (rng.normal(0.85, 0.05) if cand and rng.random() < 0.06 * w
                                  else rng.normal(0.3, 0.08))
                            cop.append({"clip_id": f"sao__{null}__w{w:g}__{pid}__s{sd}",
                                        "model": "sao", "null": null, "w": w, "prompt_id": pid,
                                        "seed": sd, "domain": dom, "embedder": emb,
                                        "nn_train_id": f"t{rng.integers(1e5)}", "s_gen": s,
                                        "is_candidate": bool(cand), "match_score": ms,
                                        "is_copy": bool(cand and ms > thr_m)})
    _write(pd.DataFrame(cop), results_dir, "e2_copies.parquet")
    _write(held, results_dir, "e2_held.parquet")

    # ---- E3 training log
    steps = np.arange(0, 5001, 250)
    _write(pd.DataFrame([{"init": init, "step": s, "train_loss": 0.30 * np.exp(-s / 2000) + 0.5,
                          "val_loss": 0.31 * np.exp(-s / 2000) + 0.52, "val_loss_zero_null": 0.83}
                         for init in ("zeros", "normal") for s in steps]),
           results_dir, "e3_null_training.csv")

    # ---- E4: judge strengths peak higher than listeners' (H4)
    true = {"qwen2-audio": {w: -0.10 * (w - 7) ** 2 for w in scales},
            "audiobox-aesthetics": {w: 0.15 * w for w in scales},
            "listeners": {w: -0.10 * (w - 5) ** 2 for w in scales}}

    def choose(rater: str, wa: float, wb: float) -> str:
        if rng.random() < 0.15:
            return "tie"
        pa = 1 / (1 + np.exp(true[rater][wb] - true[rater][wa]))
        return "a" if rng.random() < pa else "b"

    e4_prompts = [(d, p, int(rng.choice(seeds))) for d in DOMAINS for p in prompts[d][:n_e4_prompts]]
    pairs = [{"pair_id": f"{p}__s{sd}__w{a:g}-w{b:g}", "prompt_id": p, "domain": d, "seed": sd,
              "w_a": a, "w_b": b} for d, p, sd in e4_prompts for a, b in combinations(scales, 2)]
    judge = []
    for pr in pairs:
        for q in QUESTIONS:
            for jname, orders in (("qwen2-audio", ("ab", "ba")), ("audiobox-aesthetics", ("ab",))):
                for o in orders:
                    judge.append({"judge": jname, **pr, "question": q, "order": o,
                                  "choice": choose(jname, pr["w_a"], pr["w_b"])})
    _write(pd.DataFrame(judge), results_dir, "e4_judge.parquet")

    # Balanced plan: each pair rated by 3 listeners (cyclic chunk assignment), random sides.
    order = rng.permutation(len(pairs))
    chunk = pairs_per_listener // 3  # listener li gets chunks li, li+n/3, li+2n/3 (mod n)
    chunks = [order[i * chunk:(i + 1) * chunk] for i in range(len(pairs) // chunk)]
    plan, answers = [], []
    for li in range(n_listeners):
        lid = f"L{li:02d}"
        mine = np.concatenate([chunks[(li + r * len(chunks) // 3) % len(chunks)] for r in range(3)])
        items = [(pairs[i], False) for i in mine] + [
            ({"pair_id": f"check_{lid}_{k}", "prompt_id": pairs[k]["prompt_id"],
              "domain": pairs[k]["domain"], "seed": pairs[k]["seed"], "w_a": 7.0, "w_b": 7.0},
             True) for k in range(4)]
        for pos, j in enumerate(rng.permutation(len(items))):
            pr, chk = items[j]
            swap = rng.random() < 0.5
            wl, wr = (pr["w_b"], pr["w_a"]) if swap else (pr["w_a"], pr["w_b"])
            plan.append({"listener_id": lid, "position": pos, **{k: pr[k] for k in
                         ("pair_id", "prompt_id", "domain", "seed")},
                         "w_left": wl, "w_right": wr,
                         "path_left": f"sao/default/w{wl:g}/{pr['prompt_id']}/s{pr['seed']}.wav",
                         "path_right": f"sao/default/w{wr:g}/{pr['prompt_id']}/s{pr['seed']}.wav",
                         "is_attention_check": chk})
            for q in QUESTIONS:
                ch = ("tie" if rng.random() < 0.9 else "a") if chk else choose(
                    "listeners", pr["w_a"], pr["w_b"])
                answers.append({"listener_id": lid, "pair_id": pr["pair_id"],
                                "prompt_id": pr["prompt_id"], "domain": pr["domain"],
                                "w_a": pr["w_a"], "w_b": pr["w_b"], "question": q, "choice": ch,
                                "is_attention_check": chk})
    _write(pd.DataFrame(plan), results_dir, "e4_listening_plan.csv")
    _write(pd.DataFrame(answers), results_dir, "e4_listeners.csv")
    return results_dir
