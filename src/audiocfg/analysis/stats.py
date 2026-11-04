"""Shared statistics (PROJECT_DESIGN.md §6.4).

- 95% CIs from a bootstrap over prompts (clusters), 10 000 replicates by default;
- paired differences thanks to shared seeds (metric(w) - metric(w=1), learned - default null);
- Wilcoxon signed-rank on per-prompt differences with Holm correction;
- the operational "changes with w" criterion: CIs at w=1 and at the extreme w do not overlap;
- E2 trend test: cluster-bootstrapped slope of a plain logistic fit ``copy ~ w``.
  TODO: the design (§5.2) names a GLMM with a prompt random effect; statsmodels is not a
  dependency, so we report the population-averaged slope with a prompt-cluster bootstrap instead.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.special import expit, logit

_MAX_CELLS = 4_000_000  # bootstrap index matrix is chunked to stay below this many entries


def percentile_ci(samples: np.ndarray, ci: float = 0.95, axis: int = 0) -> tuple:
    """Percentile interval of bootstrap replicates (NaN replicates ignored)."""
    a = (1.0 - ci) / 2.0
    s = np.asarray(samples, dtype=float)
    return np.nanquantile(s, a, axis=axis), np.nanquantile(s, 1.0 - a, axis=axis)


def bootstrap_mean(values: Sequence[float], n_boot: int = 10_000, ci: float = 0.95,
                   rng: np.random.Generator | int | None = None) -> tuple[float, float, float]:
    """(mean, ci_low, ci_high) of the mean of i.i.d. cluster-level values."""
    rng = np.random.default_rng(rng)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    k = v.size
    if k == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(v.mean())
    if k == 1:
        return m, m, m
    means = np.empty(n_boot)
    chunk = max(1, _MAX_CELLS // k)
    for s in range(0, n_boot, chunk):
        e = min(n_boot, s + chunk)
        means[s:e] = v[rng.integers(0, k, size=(e - s, k))].mean(axis=1)
    lo, hi = percentile_ci(means, ci)
    return m, float(lo), float(hi)


def iter_groups(df: pd.DataFrame, group_cols: Sequence[str]) -> Iterator[tuple[tuple, pd.DataFrame]]:
    """Yield ``(key_tuple, sub_frame)``; a single empty-key group when ``group_cols`` is empty."""
    group_cols = list(group_cols)
    if not group_cols:
        yield (), df
        return
    for key, g in df.groupby(group_cols, sort=True, dropna=False):
        yield (key if isinstance(key, tuple) else (key,)), g


def bootstrap_ci(df: pd.DataFrame, value: str, group_cols: Sequence[str] = (),
                 cluster: str = "prompt_id", n_boot: int = 10_000, ci: float = 0.95,
                 seed: int = 0) -> pd.DataFrame:
    """Mean of ``value`` per group with a cluster bootstrap CI (§6.4).

    Rows are first averaged within each cluster (e.g. over seeds of one prompt), then clusters
    are resampled with replacement, so every prompt has equal weight.
    """
    group_cols = list(group_cols)
    rng = np.random.default_rng(seed)
    cm = df.groupby(group_cols + [cluster], dropna=False)[value].mean().reset_index()
    rows = []
    for key, g in iter_groups(cm, group_cols):
        m, lo, hi = bootstrap_mean(g[value].to_numpy(), n_boot, ci, rng)
        rows.append({**dict(zip(group_cols, key)), "mean": m, "ci_low": lo, "ci_high": hi,
                     "n_clusters": int(g[value].notna().sum())})
    return pd.DataFrame(rows, columns=group_cols + ["mean", "ci_low", "ci_high", "n_clusters"])


def wilcoxon_p(diffs: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value of paired differences (1.0 if all are zero)."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    if d.size < 2 or np.all(d == 0):
        return 1.0
    try:
        return float(sps.wilcoxon(d).pvalue)
    except ValueError:
        return float("nan")


def paired_diff_ci(df: pd.DataFrame, value: str, by: str, ref, group_cols: Sequence[str] = (),
                   cluster: str = "prompt_id", pair_keys: Sequence[str] = (),
                   levels: Sequence | None = None, n_boot: int = 10_000, ci: float = 0.95,
                   seed: int = 0) -> pd.DataFrame:
    """Paired difference ``value[by=level] - value[by=ref]`` per group (§5.1 step 4, §5.3).

    Differences are formed on matching ``(cluster, *pair_keys)`` rows (shared prompts/seeds),
    averaged per cluster, and then cluster-bootstrapped. Also returns the Wilcoxon p-value of the
    per-cluster differences (apply :func:`holm` afterwards).
    """
    group_cols = [c for c in group_cols if c != by]
    keys = [cluster] + [k for k in pair_keys if k != cluster]
    out_cols = group_cols + [by, "ref", "mean_diff", "ci_low", "ci_high", "n_clusters",
                             "p_wilcoxon"]
    wide = df.groupby(group_cols + keys + [by], dropna=False)[value].mean().unstack(by)
    if ref not in wide.columns:
        return pd.DataFrame(columns=out_cols)
    todo = [lv for lv in (levels if levels is not None else wide.columns)
            if lv != ref and lv in wide.columns]
    rng = np.random.default_rng(seed)
    rows = []
    for level in todo:
        d = (wide[level] - wide[ref]).dropna()
        if d.empty:
            continue
        d = d.groupby(level=group_cols + [cluster]).mean().reset_index(name="diff")
        for key, g in iter_groups(d, group_cols):
            x = g["diff"].to_numpy()
            m, lo, hi = bootstrap_mean(x, n_boot, ci, rng)
            rows.append({**dict(zip(group_cols, key)), by: level, "ref": ref, "mean_diff": m,
                         "ci_low": lo, "ci_high": hi, "n_clusters": int(x.size),
                         "p_wilcoxon": wilcoxon_p(x)})
    return pd.DataFrame(rows, columns=out_cols)


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (NaNs are left out of the family and stay NaN)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    m = q.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(q, kind="stable")):
        running = max(running, (m - rank) * q[i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out


def add_holm(df: pd.DataFrame, p_col: str = "p_wilcoxon", family: Sequence[str] = (),
             out_col: str = "p_holm") -> pd.DataFrame:
    """Add Holm-adjusted p-values, one family per combination of ``family`` columns."""
    df = df.copy()
    df[out_col] = np.nan
    if df.empty:
        return df
    if not family:
        df[out_col] = holm(df[p_col].to_numpy())
        return df
    for _, g in df.groupby(list(family), dropna=False):
        df.loc[g.index, out_col] = holm(g[p_col].to_numpy())
    return df


def ci_overlap(lo1: float, hi1: float, lo2: float, hi2: float) -> bool:
    return max(lo1, lo2) <= min(hi1, hi2)


def changes_with_w(ci_table: pd.DataFrame, group_cols: Sequence[str] = (), w_col: str = "w",
                   w_ref: float = 1.0, w_ext: float | None = None) -> pd.DataFrame:
    """Operational criterion of §6.4/§2: does the CI at ``w_ref`` overlap the CI at ``w_ext``?

    ``ci_table`` has columns ``mean, ci_low, ci_high`` per (group, w); ``w_ext`` defaults to the
    largest w of each group. ``changes`` is True when the two CIs do not overlap.
    """
    group_cols = list(group_cols)
    rows = []
    for key, g in iter_groups(ci_table, group_cols):
        g = g.set_index(w_col)
        ext = g.index.max() if w_ext is None else w_ext
        if w_ref not in g.index or ext not in g.index:
            continue
        r, e = g.loc[w_ref], g.loc[ext]
        rows.append({**dict(zip(group_cols, key)), "w_ref": w_ref, "w_ext": ext,
                     "mean_ref": r["mean"], "ci_ref": f"[{r['ci_low']:.4g}, {r['ci_high']:.4g}]",
                     "mean_ext": e["mean"], "ci_ext": f"[{e['ci_low']:.4g}, {e['ci_high']:.4g}]",
                     "changes": not ci_overlap(r["ci_low"], r["ci_high"],
                                               e["ci_low"], e["ci_high"])})
    return pd.DataFrame(rows)


def diff_ci_from_cis(m1: float, lo1: float, hi1: float, m2: float, lo2: float, hi2: float,
                     ci: float = 0.95) -> tuple[float, float, float]:
    """Approximate CI of ``m1 - m2`` from two independent CIs (normal approximation).

    Used for FD, which ``e1_fd.parquet`` gives only as per-setting CIs. Treating the settings as
    independent ignores the positive correlation from shared seeds, so it is conservative.
    """
    z = sps.norm.ppf(0.5 + ci / 2)
    se1, se2 = (hi1 - lo1) / (2 * z), (hi2 - lo2) / (2 * z)
    d = m1 - m2
    h = z * float(np.hypot(se1, se2))
    return d, d - h, d + h


def fit_logistic(x: np.ndarray, k: np.ndarray, n: np.ndarray, ridge: float = 1e-6,
                 max_iter: int = 100) -> np.ndarray:
    """MLE of ``logit p = b0 + b1 x`` on grouped binomial data (Newton-Raphson, tiny ridge).

    Returns ``[b0, b1]``; NaNs when there are no events or only events (slope not identified).
    """
    x, k, n = (np.asarray(a, dtype=float) for a in (x, k, n))
    tot, ev = n.sum(), k.sum()
    if tot == 0 or ev == 0 or ev == tot:
        return np.array([np.nan, np.nan])
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([logit(ev / tot), 0.0])
    reg = ridge * np.eye(2)
    for _ in range(max_iter):
        p = expit(X @ beta)
        g = X.T @ (k - n * p) - reg @ beta
        H = (X * (n * p * (1 - p))[:, None]).T @ X + reg
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def logistic_trend(df: pd.DataFrame, outcome: str, x: str = "w", cluster: str = "prompt_id",
                   group_cols: Sequence[str] = (), n_boot: int = 10_000, ci: float = 0.95,
                   seed: int = 0) -> pd.DataFrame:
    """Slope of ``outcome ~ x`` (log-odds per unit x) with a cluster-bootstrap CI (§5.2).

    TODO: GLMM with a random intercept per prompt (needs statsmodels / custom Laplace fit).
    """
    group_cols = list(group_cols)
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in iter_groups(df, group_cols):
        agg = g.assign(_y=g[outcome].astype(float)).groupby([cluster, x])["_y"].agg(["sum", "count"])
        xs = np.sort(g[x].unique()).astype(float)
        K = agg["sum"].unstack(x, fill_value=0).reindex(columns=xs, fill_value=0).to_numpy()
        N = agg["count"].unstack(x, fill_value=0).reindex(columns=xs, fill_value=0).to_numpy()
        b = fit_logistic(xs, K.sum(0), N.sum(0))
        C = K.shape[0]
        counts = rng.multinomial(C, np.full(C, 1.0 / C), size=n_boot)
        kb, nb = counts @ K, counts @ N
        slopes = np.array([fit_logistic(xs, kb[i], nb[i])[1] for i in range(n_boot)])
        fin = slopes[np.isfinite(slopes)]
        lo, hi = percentile_ci(fin, ci) if fin.size else (np.nan, np.nan)
        rows.append({**dict(zip(group_cols, key)), "outcome": outcome, "slope": b[1],
                     "ci_low": float(lo), "ci_high": float(hi),
                     "p_boot_slope_le_0": float(np.mean(fin <= 0)) if fin.size else np.nan,
                     "n_events": int(K.sum()), "n": int(N.sum()), "n_clusters": int(C),
                     "method": "plain logistic + prompt-cluster bootstrap (GLMM: TODO)"})
    return pd.DataFrame(rows)
