"""KL_passt: divergence of PaSST label distributions, generation vs its real reference (§6.2).

Per prompt, the reference is the real recording the prompt caption came from. We report
``KL(p_ref || p_gen)`` averaged over the prompt's clips (direction as in AudioLDM-eval).
"""

from __future__ import annotations

import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Row-wise ``KL(p || q)``; inputs are renormalised after clipping by ``eps``."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, None)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, None)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def kl_per_prompt(gen_probs: np.ndarray, ref_probs: np.ndarray) -> float:
    """Mean over the prompt's clips of ``KL(ref || gen)``; ``ref_probs`` is ``(C,)`` or ``(1, C)``."""
    ref = np.atleast_2d(ref_probs)
    return float(np.mean(kl_divergence(np.broadcast_to(ref, gen_probs.shape), gen_probs)))
