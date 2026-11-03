"""Fréchet distance between embedding sets (§6.2).

Rules: same reference set for every setting of a model x domain; identical N per setting
(FD is biased by N); 95% CI by bootstrap over prompts (resample prompts, keep all their
clips); split-half real-vs-real FD as the attainable floor.
"""

from __future__ import annotations

from typing import Hashable

import numpy as np
from scipy import linalg


def gaussian_stats(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    emb = np.asarray(emb, dtype=np.float64)
    return emb.mean(axis=0), np.cov(emb, rowvar=False)


def frechet_distance(mu1: np.ndarray, s1: np.ndarray, mu2: np.ndarray, s2: np.ndarray,
                     eps: float = 1e-6) -> float:
    """``|mu1-mu2|^2 + Tr(S1 + S2 - 2 sqrt(S1 S2))``.

    ``Tr sqrt(S1 S2)`` is computed as ``sum sqrt(eig(A S2 A))`` with ``A = S1^{1/2}`` (symmetric
    PSD, so no complex parts and stable for rank-deficient covariances). Falls back to
    ``scipy.linalg.sqrtm`` (real part, small imaginary residue tolerated) if eigh fails.
    """
    diff = mu1 - mu2
    try:
        w1, v1 = np.linalg.eigh((s1 + s1.T) / 2)
        a = (v1 * np.sqrt(np.clip(w1, 0, None))) @ v1.T
        ev = np.linalg.eigvalsh(a @ s2 @ a)
        tr_covmean = float(np.sum(np.sqrt(np.clip(ev, 0, None))))
    except np.linalg.LinAlgError:
        off = np.eye(len(s1)) * eps
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                raise ValueError(f"sqrtm imaginary component {np.abs(covmean.imag).max():.3g}")
            covmean = covmean.real
        tr_covmean = float(np.trace(covmean))
    return float(max(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * tr_covmean, 0.0))


def fd(gen_emb: np.ndarray, ref_emb: np.ndarray) -> float:
    return frechet_distance(*gaussian_stats(gen_emb), *gaussian_stats(ref_emb))


def fd_with_ci(gen_emb: np.ndarray, ref_emb: np.ndarray, prompt_ids: np.ndarray,
               n_boot: int = 1000, ci: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """FD with a percentile bootstrap CI over prompts: ``(fd, ci_low, ci_high)``.

    The reference statistics stay fixed; only the generated set is resampled (by prompt).
    """
    prompt_ids = np.asarray(prompt_ids)
    point = fd(gen_emb, ref_emb)
    if n_boot <= 0:
        return point, float("nan"), float("nan")
    mu_r, s_r = gaussian_stats(ref_emb)
    uniq, inv = np.unique(prompt_ids, return_inverse=True)
    rows_of = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([rows_of[i] for i in pick])
        boots[b] = frechet_distance(*gaussian_stats(gen_emb[idx]), mu_r, s_r)
    a = (1 - ci) / 2
    return point, float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def equal_n_indices(groups: dict[Hashable, np.ndarray], seed: int = 0,
                    n: int | None = None) -> dict[Hashable, np.ndarray]:
    """Subsample each group's row indices to a common N (min size, or ``n``) with a fixed seed."""
    n_common = min(len(v) for v in groups.values()) if n is None else n
    out = {}
    for key, idx in groups.items():
        idx = np.asarray(idx)
        if len(idx) < n_common:
            raise ValueError(f"group {key!r} has {len(idx)} < N={n_common} clips")
        rng = np.random.default_rng(seed)  # same draw pattern per group
        out[key] = np.sort(rng.choice(idx, size=n_common, replace=False)) if len(idx) > n_common else idx
    return out


def check_equal_n(sizes: dict[Hashable, int]) -> None:
    """Raise if settings that will be compared do not have identical N (§6.2)."""
    if len(set(sizes.values())) > 1:
        raise ValueError(f"FD settings must have equal N, got {sizes}")


def split_half_floor(ref_emb: np.ndarray, n_splits: int = 10, seed: int = 0) -> float:
    """Mean FD between two disjoint random halves of the reference set (real-vs-real floor).

    Note: each half has N/2 rows, so the floor is at a smaller N than the settings.
    """
    rng = np.random.default_rng(seed)
    n = len(ref_emb)
    vals = []
    for _ in range(n_splits):
        p = rng.permutation(n)
        vals.append(fd(ref_emb[p[: n // 2]], ref_emb[p[n // 2: 2 * (n // 2)]]))
    return float(np.mean(vals))
