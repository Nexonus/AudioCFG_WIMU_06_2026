"""Bradley-Terry ranking of guidance scales and judge-vs-listener comparison (PROJECT_DESIGN.md §5.4).

- BT strengths from pairwise outcomes, ties count as half a win for each side, fitted with the
  MM algorithm (Hunter 2004); identified by ``sum(log strength) = 0``.
- Bootstrap over prompts (and over listeners for listener data), vectorised over replicates.
- Preferred scale = argmax strength, with the bootstrap distribution of the argmax.
- H4: difference of preferred scales judge - listeners with a CI from shared bootstrap draws.
- Pairwise agreement with the listener majority, Cohen's kappa, inter-listener ceiling
  (each listener vs the majority of the others) and the judge's order-flip rate.

All choices are relative to ``(w_a, w_b)`` with ``w_a < w_b``: "a", "b" or "tie".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from audiocfg.analysis.stats import percentile_ci
from audiocfg.schema import CHOICES

_SCORE_A = {"a": 1.0, "b": 0.0, "tie": 0.5}


# --------------------------------------------------------------------------- fitting
def contributions(df: pd.DataFrame, levels: Sequence[float]) -> np.ndarray:
    """Per-row win contributions, shape ``(rows, K*K)``; entry ``i*K+j`` = wins of i over j."""
    levels = [float(v) for v in levels]
    K = len(levels)
    index = {v: i for i, v in enumerate(levels)}
    ia = df["w_a"].astype(float).map(index).to_numpy()
    ib = df["w_b"].astype(float).map(index).to_numpy()
    sa = df["choice"].map(_SCORE_A).to_numpy(dtype=float)
    E = np.zeros((len(df), K * K))
    ok = np.isfinite(ia.astype(float)) & np.isfinite(ib.astype(float)) & np.isfinite(sa)
    ok &= np.where(ok, ia, -1) != np.where(ok, ib, -1)
    r = np.nonzero(ok)[0]
    a, b = ia[ok].astype(int), ib[ok].astype(int)
    E[r, a * K + b] = sa[ok]
    E[r, b * K + a] = 1.0 - sa[ok]
    return E


def wins_matrix(df: pd.DataFrame, levels: Sequence[float]) -> np.ndarray:
    """``W[i, j]`` = (fractional) wins of scale i over scale j."""
    K = len(levels)
    return contributions(df, levels).sum(axis=0).reshape(K, K)


def fit_bt(wins: np.ndarray, prior: float = 0.1, max_iter: int = 5000,
           tol: float = 1e-10) -> np.ndarray:
    """Log-strengths (centred: sum = 0) from wins ``(..., K, K)``; vectorised over leading axes.

    ``prior`` adds that many virtual wins in each direction for every pair, which keeps the MLE
    finite when a scale never wins (and connects the comparison graph in sparse bootstraps).
    """
    W = np.asarray(wins, dtype=float)
    K = W.shape[-1]
    W = W + prior * (1.0 - np.eye(K))
    N = W + np.swapaxes(W, -1, -2)
    won = W.sum(axis=-1)
    p = np.ones(W.shape[:-1])
    for _ in range(max_iter):
        denom = (N / (p[..., :, None] + p[..., None, :])).sum(axis=-1)
        new = won / denom
        new = new / np.exp(np.log(new).mean(axis=-1, keepdims=True))
        done = np.max(np.abs(np.log(new) - np.log(p))) < tol
        p = new
        if done:
            break
    return np.log(p)


@dataclass
class BTFit:
    """Point estimate and bootstrap of BT log-strengths for one rater group."""

    levels: np.ndarray
    log_strength: np.ndarray
    boot: np.ndarray  # (B, K)
    ci: float
    n_comparisons: int

    @property
    def ci_low(self) -> np.ndarray:
        return percentile_ci(self.boot, self.ci)[0]

    @property
    def ci_high(self) -> np.ndarray:
        return percentile_ci(self.boot, self.ci)[1]

    @property
    def preferred(self) -> float:
        return float(self.levels[int(np.argmax(self.log_strength))])

    @property
    def argmax_boot(self) -> np.ndarray:
        return self.levels[np.argmax(self.boot, axis=1)]

    @property
    def preferred_ci(self) -> tuple[float, float]:
        lo, hi = percentile_ci(self.argmax_boot, self.ci)
        return float(lo), float(hi)

    @property
    def p_preferred(self) -> float:
        """Bootstrap probability that the argmax equals the point-estimate argmax."""
        return float(np.mean(self.argmax_boot == self.preferred))

    def to_frame(self) -> pd.DataFrame:
        am = self.argmax_boot
        return pd.DataFrame({"w": self.levels, "log_strength": self.log_strength,
                             "ci_low": self.ci_low, "ci_high": self.ci_high,
                             "p_argmax": [float(np.mean(am == v)) for v in self.levels]})


def _cells(df: pd.DataFrame, levels: Sequence[float], cell_cols: list[str]):
    """Aggregate contributions per bootstrap cell (prompt, or prompt x listener)."""
    E = contributions(df, levels)
    grp = df.groupby(cell_cols, sort=True)
    codes = grp.ngroup().to_numpy()
    keys = grp.size().reset_index()[cell_cols]
    M = np.zeros((len(keys), E.shape[1]))
    np.add.at(M, codes, E)
    return keys, M


def _fit_from_counts(df: pd.DataFrame, levels: np.ndarray, prompt_index: dict, pc: np.ndarray,
                     rater_col: str | None, rater_index: dict | None, rc: np.ndarray | None,
                     ci: float, prior: float, cluster: str) -> BTFit:
    K = len(levels)
    cols = [cluster] + ([rater_col] if rater_col else [])
    keys, M = _cells(df, levels, cols)
    wt = pc[:, keys[cluster].map(prompt_index).to_numpy()]
    if rater_col:
        wt = wt * rc[:, keys[rater_col].map(rater_index).to_numpy()]
    point = fit_bt(M.sum(axis=0).reshape(K, K), prior)
    boot = fit_bt((wt @ M).reshape(-1, K, K), prior)
    return BTFit(levels, point, boot, ci, int(len(df)))


def _draw(ids: Sequence, n_boot: int, rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    ids = sorted(set(ids))
    n = len(ids)
    return {v: i for i, v in enumerate(ids)}, rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)


def bt_bootstrap(df: pd.DataFrame, levels: Sequence[float], cluster: str = "prompt_id",
                 rater_col: str | None = None, n_boot: int = 10_000, ci: float = 0.95,
                 seed: int = 0, prior: float = 0.1) -> BTFit:
    """BT fit with a bootstrap over prompts (and over ``rater_col``, e.g. listeners)."""
    rng = np.random.default_rng(seed)
    levels = np.asarray(sorted(float(v) for v in levels))
    pidx, pc = _draw(df[cluster], n_boot, rng)
    ridx, rc = _draw(df[rater_col], n_boot, rng) if rater_col else (None, None)
    return _fit_from_counts(df, levels, pidx, pc, rater_col, ridx, rc, ci, prior, cluster)


def bt_compare(judge: pd.DataFrame, listeners: pd.DataFrame, levels: Sequence[float],
               n_boot: int = 10_000, ci: float = 0.95, seed: int = 0, prior: float = 0.1,
               cluster: str = "prompt_id", rater_col: str = "listener_id"
               ) -> tuple[BTFit, BTFit, dict]:
    """Judge vs listeners on shared prompt draws; returns the H4 difference of preferred scales.

    In each replicate the same prompts are resampled for both raters, listeners are resampled
    too; ``diff = argmax_judge - argmax_listeners``. H4 holds if ``ci_low > 0``.
    """
    rng = np.random.default_rng(seed)
    levels = np.asarray(sorted(float(v) for v in levels))
    pidx, pc = _draw(list(judge[cluster]) + list(listeners[cluster]), n_boot, rng)
    ridx, rc = _draw(listeners[rater_col], n_boot, rng)
    fj = _fit_from_counts(judge, levels, pidx, pc, None, None, None, ci, prior, cluster)
    fl = _fit_from_counts(listeners, levels, pidx, pc, rater_col, ridx, rc, ci, prior, cluster)
    d = fj.argmax_boot - fl.argmax_boot
    lo, hi = percentile_ci(d, ci)
    diff = {"diff": fj.preferred - fl.preferred, "ci_low": float(lo), "ci_high": float(hi),
            "p_diff_gt_0": float(np.mean(d > 0)), "h4_supported": bool(lo > 0)}
    return fj, fl, diff


# --------------------------------------------------------------------------- agreement
def as_bool(s: pd.Series) -> pd.Series:
    """Parse booleans that may come back from CSV as strings."""
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "t"])


def combine_orders(judge: pd.DataFrame) -> pd.DataFrame:
    """One decision per (judge, pair, question): inconsistent decisions across orders -> tie."""
    def combine(ch: pd.Series) -> str:
        u = ch.unique()
        return u[0] if len(u) == 1 else "tie"

    return (judge.groupby(["judge", "pair_id", "question"], as_index=False, sort=True)
            .agg(prompt_id=("prompt_id", "first"), domain=("domain", "first"),
                 w_a=("w_a", "first"), w_b=("w_b", "first"),
                 choice=("choice", combine), n_orders=("order", "nunique")))


def flip_rate(judge: pd.DataFrame) -> float:
    """Share of pairs (shown in both orders) whose decision changes after swapping the order."""
    per = judge.groupby(["judge", "pair_id", "question"]).agg(
        n=("order", "nunique"), k=("choice", "nunique"))
    per = per[per["n"] >= 2]
    return float((per["k"] > 1).mean()) if len(per) else float("nan")


def _majority(choices: Sequence[str]) -> str | None:
    s = pd.Series(list(choices))
    if s.empty:
        return None
    vc = s.value_counts()
    return vc.index[0] if vc.iloc[0] > len(s) / 2 else None


def listener_majority(listeners: pd.DataFrame) -> pd.DataFrame:
    """Strict-majority listener choice per (pair_id, question); pairs without one are dropped."""
    m = listeners.groupby(["pair_id", "question"])["choice"].agg(_majority).rename("majority")
    return m.dropna().reset_index()


def cohen_kappa(a: Sequence[str], b: Sequence[str], categories: Sequence[str] = CHOICES) -> float:
    a, b = list(a), list(b)
    if not a:
        return float("nan")
    idx = {c: i for i, c in enumerate(categories)}
    C = np.zeros((len(categories), len(categories)))
    for x, y in zip(a, b):
        C[idx[x], idx[y]] += 1
    n = C.sum()
    po = np.trace(C) / n
    pe = float((C.sum(0) / n) @ (C.sum(1) / n))
    if pe >= 1.0:
        return 1.0 if po == 1.0 else float("nan")
    return float((po - pe) / (1 - pe))


def agreement_with_majority(decisions: pd.DataFrame, majority: pd.DataFrame) -> dict:
    """Agreement and kappa between one rater's decisions and the listener majority."""
    m = decisions.merge(majority, on=["pair_id", "question"])
    if m.empty:
        return {"agreement": float("nan"), "kappa": float("nan"), "n_agree": 0}
    return {"agreement": float((m["choice"] == m["majority"]).mean()),
            "kappa": cohen_kappa(m["choice"], m["majority"]), "n_agree": int(len(m))}


def inter_listener_ceiling(listeners: pd.DataFrame) -> dict:
    """Each listener vs the strict majority of the other listeners on the same pair."""
    own, other = [], []
    for _, g in listeners.groupby(["pair_id", "question"]):
        ch = g["choice"].tolist()
        if len(ch) < 2:
            continue
        for i, c in enumerate(ch):
            m = _majority(ch[:i] + ch[i + 1:])
            if m is not None:
                own.append(c)
                other.append(m)
    if not own:
        return {"agreement": float("nan"), "kappa": float("nan"), "n_agree": 0}
    return {"agreement": float(np.mean(np.array(own) == np.array(other))),
            "kappa": cohen_kappa(own, other), "n_agree": len(own)}


def clean_listeners(listeners: pd.DataFrame, min_check_acc: float = 0.5
                    ) -> tuple[pd.DataFrame, dict]:
    """Drop attention-check rows and listeners who fail them (correct answer: "tie")."""
    chk = as_bool(listeners["is_attention_check"])
    checks = listeners[chk]
    acc = (checks["choice"] == "tie").groupby(checks["listener_id"]).mean()
    bad = sorted(acc[acc < min_check_acc].index.tolist())
    keep = (~chk) & (~listeners["listener_id"].isin(bad)) & (
        listeners["w_a"].astype(float) != listeners["w_b"].astype(float))
    return listeners[keep].copy(), {"excluded_listeners": bad,
                                    "attention_accuracy": acc.to_dict()}
