"""Within-prompt diversity: CCDS (§5.1, arXiv:2505.08175).

Diversity compares generated clips **with each other**: the mean pairwise cosine distance of
audio embeddings over the 10 seeds of one (prompt, w). Text–audio CLAP similarity measures
prompt adherence, not diversity, and is never used here (§5.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_KEYS = ("model", "domain", "null", "w", "prompt_id")


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def ccds(emb: np.ndarray) -> float:
    """Mean cosine distance ``1 - cos`` over all k(k-1)/2 pairs of rows of ``emb (k, d)``."""
    k = len(emb)
    if k < 2:
        return float("nan")
    u = _unit(emb)
    sim = u @ u.T
    iu = np.triu_indices(k, 1)
    return float(np.mean(1.0 - sim[iu]))


def ccds_table(meta: pd.DataFrame, emb: np.ndarray, column: str = "ccds",
               keys: tuple[str, ...] = GROUP_KEYS) -> pd.DataFrame:
    """CCDS per (model, domain, null, w, prompt_id); ``emb`` rows align with ``meta`` rows."""
    meta = meta.reset_index(drop=True)
    if len(meta) != len(emb):
        raise ValueError("meta and emb must have the same number of rows")
    rows = [(*k, ccds(emb[idx])) for k, idx in meta.groupby(list(keys), sort=True).indices.items()]
    return pd.DataFrame(rows, columns=[*keys, column])
