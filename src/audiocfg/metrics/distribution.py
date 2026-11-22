from __future__ import annotations

import numpy as np


def ccds(embeddings: np.ndarray) -> float:
    e = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    n = len(e)
    if n < 2:
        return float("nan")
    sim = e @ e.T
    return float(1 - (sim.sum() - n) / (n * (n - 1)))


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import linalg

    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(cov_a @ cov_b)
    covmean = covmean.real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a + cov_b - 2 * covmean))
