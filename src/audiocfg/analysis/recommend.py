"""Recommended guidance scale ``w*_auto`` (PROJECT_DESIGN.md §5.1, fixed before the experiment).

``w*_auto`` = the smallest w such that
  (1) mean CLAP_norm(w) lies within the 95% CI of the w with the highest mean CLAP_norm, and
  (2) FD(w) is not significantly worse than FD at the best-FD w: the CI of FD(w) - FD(best)
      contains 0 or lies below it.
At ``w*_auto`` we report the diversity loss CCDS(w*) - CCDS(w=1) with a paired CI.

FD comes as per-setting CIs (``e1_fd.parquet``), so the difference CI uses the normal
approximation of :func:`audiocfg.analysis.stats.diff_ci_from_cis` (conservative: ignores the
positive correlation from shared seeds).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from audiocfg.analysis.stats import bootstrap_ci, diff_ci_from_cis, iter_groups, paired_diff_ci


def select_fd(fd: pd.DataFrame | None, embedder: str = "openl3") -> pd.DataFrame | None:
    """Rows of ``e1_fd`` for the requested embedder (substring match, case-insensitive)."""
    if fd is None or fd.empty:
        return None
    sel = fd[fd["embedder"].astype(str).str.lower().str.contains(embedder.lower())]
    return sel if not sel.empty else None


def w_star_auto(pp: pd.DataFrame, fd: pd.DataFrame | None = None, n_boot: int = 10_000,
                ci: float = 0.95, seed: int = 0, w_ref: float | None = None) -> dict:
    """``w*_auto`` for one (model, domain, null): ``pp`` are its E1 per-prompt rows, ``fd`` its
    FD rows (one embedder) or None (then criterion (2) is skipped and flagged)."""
    clap = bootstrap_ci(pp, "clap_norm", ["w"], n_boot=n_boot, ci=ci, seed=seed).set_index("w")
    w_max = float(clap["mean"].idxmax())
    lo_max, hi_max = clap.loc[w_max, "ci_low"], clap.loc[w_max, "ci_high"]
    clap_ok = {float(w) for w, r in clap.iterrows() if lo_max <= r["mean"] <= hi_max}

    fd_ok: set[float] | None = None
    w_fd_best = np.nan
    if fd is not None and not fd.empty:
        f = fd.groupby("w")[["fd", "ci_low", "ci_high"]].mean()
        w_fd_best = float(f["fd"].idxmin())
        b = f.loc[w_fd_best]
        fd_ok = set()
        for w, r in f.iterrows():
            _, d_lo, _ = diff_ci_from_cis(r["fd"], r["ci_low"], r["ci_high"],
                                          b["fd"], b["ci_low"], b["ci_high"], ci)
            if d_lo <= 0:  # CI of the difference contains 0 or is negative
                fd_ok.add(float(w))

    cands = sorted(w for w in clap_ok if fd_ok is None or w in fd_ok)
    w_star = cands[0] if cands else np.nan
    w_ref = float(min(clap.index)) if w_ref is None else w_ref
    out = {"w_star_auto": w_star, "w_clap_max": w_max, "w_fd_best": w_fd_best,
           "clap_ok": ",".join(f"{w:g}" for w in sorted(clap_ok)),
           "fd_ok": "n/a" if fd_ok is None else ",".join(f"{w:g}" for w in sorted(fd_ok)),
           "fd_criterion": "skipped (no FD)" if fd_ok is None else "approx. diff CI",
           "w_ref": w_ref, "ccds_loss": np.nan, "ccds_loss_ci_low": np.nan,
           "ccds_loss_ci_high": np.nan, "ccds_loss_pct": np.nan,
           "note": "" if cands else "no w satisfies both criteria"}
    if np.isfinite(w_star) and "ccds_clap" in pp:
        base = pp.loc[pp["w"] == w_ref, "ccds_clap"].mean()
        if w_star == w_ref:
            out.update(ccds_loss=0.0, ccds_loss_ci_low=0.0, ccds_loss_ci_high=0.0,
                       ccds_loss_pct=0.0)
        else:
            d = paired_diff_ci(pp, "ccds_clap", by="w", ref=w_ref, levels=[w_star],
                               n_boot=n_boot, ci=ci, seed=seed)
            if not d.empty:
                r = d.iloc[0]
                out.update(ccds_loss=r["mean_diff"], ccds_loss_ci_low=r["ci_low"],
                           ccds_loss_ci_high=r["ci_high"],
                           ccds_loss_pct=100 * r["mean_diff"] / base if base else np.nan)
    return out


def recommend_table(pp: pd.DataFrame, fd: pd.DataFrame | None = None, fd_embedder: str = "openl3",
                    n_boot: int = 10_000, ci: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """``w*_auto`` per (model, domain, null)."""
    fd_sel = select_fd(fd, fd_embedder)
    keys = ["model", "domain", "null"]
    rows = []
    for key, g in iter_groups(pp, keys):
        f = None
        if fd_sel is not None:
            m = np.logical_and.reduce([fd_sel[k] == v for k, v in zip(keys, key)])
            f = fd_sel[m]
        rows.append({**dict(zip(keys, key)), **w_star_auto(g, f, n_boot, ci, seed)})
    return pd.DataFrame(rows)
