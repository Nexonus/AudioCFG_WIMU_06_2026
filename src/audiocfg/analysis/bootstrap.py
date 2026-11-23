from __future__ import annotations

import numpy as np


def paired_ci(diff, n_boot: int = 10000, seed: int = 0, level: float = 0.95) -> dict:
    diff = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(diff))}
