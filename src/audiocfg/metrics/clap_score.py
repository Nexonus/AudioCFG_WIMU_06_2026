"""Prompt adherence as calibrated CLAP_norm (§6.1). No absolute thresholds.

* ``C_real`` = mean CLAP(real clip, its own caption)
* ``C_mis``  = mean CLAP(real clip, a random caption of the same domain, never its own)
* ``CLAP_norm = (s - C_mis) / (C_real - C_mis)``: 0 = no better than a random caption,
  1 = as matched as a real recording. Compare only across w within a model and domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audiocfg.metrics.diversity import _unit


def clap_scores(text_emb: np.ndarray, audio_emb: np.ndarray) -> np.ndarray:
    """Row-wise cosine between paired text and audio embeddings ``(n, d)`` -> ``(n,)``."""
    return np.sum(_unit(text_emb) * _unit(audio_emb), axis=-1)


@dataclass(frozen=True)
class Calibration:
    c_real: float
    c_mis: float
    n: int


def calibrate(real_audio_emb: np.ndarray, own_text_emb: np.ndarray, all_text_emb: np.ndarray,
              rng: np.random.Generator, own_idx: np.ndarray | None = None) -> Calibration:
    """C_real / C_mis from real clips of one domain.

    ``all_text_emb (m, d)`` is the pool of captions of the same domain; ``own_idx[i]`` is the
    row of clip i's own caption in that pool (default ``arange(n)``, i.e. pool starts with the
    own captions). Each clip gets one random mismatched caption drawn uniformly, never its own.
    """
    n, m = len(real_audio_emb), len(all_text_emb)
    own_idx = np.arange(n) if own_idx is None else np.asarray(own_idx)
    if m < 2:
        raise ValueError("need at least 2 captions in the pool for a mismatched draw")
    # Draw from m-1 options and skip the own index -> uniform over "not own".
    j = rng.integers(0, m - 1, size=n)
    j = j + (j >= own_idx)
    assert not np.any(j == own_idx)
    c_real = float(np.mean(clap_scores(own_text_emb, real_audio_emb)))
    c_mis = float(np.mean(clap_scores(all_text_emb[j], real_audio_emb)))
    return Calibration(c_real, c_mis, n)


def clap_norm(s: np.ndarray | float, cal: Calibration) -> np.ndarray | float:
    """``(s - C_mis) / (C_real - C_mis)``."""
    denom = cal.c_real - cal.c_mis
    if abs(denom) < 1e-12:
        raise ValueError("degenerate calibration: C_real == C_mis")
    return (np.asarray(s) - cal.c_mis) / denom
