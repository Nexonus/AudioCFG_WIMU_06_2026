from __future__ import annotations

import numpy as np
import pandas as pd

from audiocfg.analysis.bootstrap import paired_ci

KEY = ["domain", "null", "prompt_id", "guidance"]
METRICS = ("clap", "PQ", "CE", "CU", "PC", "ccds")


def per_prompt(scores: pd.DataFrame, diversity: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds for each (domain, null, prompt, g), joined with that group's CCDS."""
    cols = [m for m in METRICS if m in scores]
    table = scores.groupby(KEY, as_index=False)[cols].mean()
    return table.merge(diversity[KEY + ["ccds"]], on=KEY)


def curves(table: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in METRICS if m in table]
    return table.groupby(["domain", "null", "guidance"], as_index=False)[metrics].mean()


def same_g_differences(table: pd.DataFrame, a="learned", b="zero", seed=0) -> pd.DataFrame:
    rows = []
    wide = table.pivot_table(index=["domain", "prompt_id", "guidance"], columns="null")
    for (domain, g), part in wide.groupby(level=["domain", "guidance"]):
        for m in METRICS:
            if (m, a) not in part:
                continue
            ci = paired_ci((part[(m, a)] - part[(m, b)]).to_numpy(), seed=seed)
            rows.append({"domain": domain, "guidance": g, "metric": m} | ci)
    return pd.DataFrame(rows)


def match_on_curve(x: np.ndarray, y: np.ndarray, g: np.ndarray, target: float):
    """Lowest g at which the piecewise-linear curve x(g) reaches target; y and g there."""
    for i in range(len(x) - 1):
        lo, hi = sorted((x[i], x[i + 1]))
        if lo <= target <= hi:
            t = 0.0 if x[i + 1] == x[i] else (target - x[i]) / (x[i + 1] - x[i])
            return g[i] + t * (g[i + 1] - g[i]), y[i] + t * (y[i + 1] - y[i])
    return np.nan, np.nan


def _matched(arr: dict, g: np.ndarray, gi: int, metric: str, ref: str, cand: str):
    """arr[null][metric] are (prompts, g) means; returns (g*, delta) for cand at g[gi]."""
    x_ref = arr[ref]["clap"].mean(0)
    target = arr[cand]["clap"].mean(0)[gi]
    g_star, y_star = match_on_curve(x_ref, arr[ref][metric].mean(0), g, target)
    return g_star, arr[cand][metric].mean(0)[gi] - y_star


def matched_clap_test(
    table: pd.DataFrame,
    metrics=("PQ", "ccds", "CE"),
    ref="zero",
    cand="learned",
    n_boot=2000,
    seed=0,
    level=0.95,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for domain, part in table.groupby("domain"):
        g = np.sort(part["guidance"].unique())
        prompts = np.sort(part["prompt_id"].unique())
        arr = {
            null: {
                m: part[part["null"] == null]
                .pivot(index="prompt_id", columns="guidance", values=m)
                .loc[prompts, g]
                .to_numpy()
                for m in ("clap", *metrics)
            }
            for null in (ref, cand)
        }
        boots = rng.integers(0, len(prompts), size=(n_boot, len(prompts)))
        for gi, g_cand in enumerate(g):
            if g_cand == 1:
                continue
            for m in metrics:
                g_star, delta = _matched(arr, g, gi, m, ref, cand)
                draws = np.array(
                    [
                        _matched(
                            {n: {k: v[b] for k, v in d.items()} for n, d in arr.items()},
                            g,
                            gi,
                            m,
                            ref,
                            cand,
                        )[1]
                        for b in boots
                    ]
                )
                ok = ~np.isnan(draws)
                lo, hi = (
                    np.quantile(draws[ok], [(1 - level) / 2, (1 + level) / 2])
                    if ok.any()
                    else (np.nan, np.nan)
                )
                rows.append(
                    {
                        "domain": domain,
                        "metric": m,
                        "g_learned": g_cand,
                        "g_star_zero": g_star,
                        "delta": delta,
                        "lo": lo,
                        "hi": hi,
                        "matched_frac": ok.mean(),
                    }
                )
    return pd.DataFrame(rows)
