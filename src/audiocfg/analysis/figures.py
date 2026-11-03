"""Figures F1-F6 (PROJECT_DESIGN.md §2), saved as PNG + PDF into ``figures/``.

Colours: Okabe-Ito (colourblind-safe), fixed per series so a model x domain keeps its colour in
every figure. Each function takes already-computed tables (see :mod:`tables`) and returns paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from audiocfg.schema import DOMAINS, QUESTIONS

OKABE_ITO = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442",
             "#000000")
SERIES = [(m, d) for m in ("sao", "audioldm2") for d in DOMAINS]  # fixed colour order
DOMAIN_COLOR = {"sfx": OKABE_ITO[0], "music": OKABE_ITO[1]}
NULL_STYLE = {"default": "-", "learned": "--", "t5_empty": ":", "mean_caption": "-."}
RATER_COLOR = {"listeners": "#000000"}
LABELS = {"ccds_clap": "CCDS (CLAP)", "ccds_aux": "CCDS (PANNs/MERT)", "clap_norm": "CLAP_norm",
          "fd_openl3": "FD_openl3", "kl_passt": "KL_passt"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                         "font.size": 9})
    return plt


def series_color(model: str, domain: str) -> str:
    key = (model, domain)
    i = SERIES.index(key) if key in SERIES else 4 + sum(map(ord, model + domain)) % 3
    return OKABE_ITO[i]


def _save(fig, out_dir: Path, stem: str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    fig.tight_layout()
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
    _plt().close(fig)
    return paths


def _line_ci(ax, g: pd.DataFrame, color: str, label: str, ls: str = "-") -> None:
    g = g.sort_values("w")
    ax.plot(g["w"], g["mean"], marker="o", ms=3, color=color, ls=ls, label=label)
    ax.fill_between(g["w"], g["ci_low"], g["ci_high"], color=color, alpha=0.18, lw=0)


def f1_ccds(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    """F1: CCDS vs w, one line per model x domain with 95% CI band (both embedders)."""
    plt = _plt()
    s = summary[summary["null"] == "default"]
    metrics = [m for m in ("ccds_clap", "ccds_aux") if (s["metric"] == m).any()]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.2), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for (model, dom), g in s[s["metric"] == m].groupby(["model", "domain"]):
            _line_ci(ax, g, series_color(model, dom), f"{model} / {dom}")
        ax.set(xlabel="guidance scale w", ylabel=LABELS[m], title=f"F1: {LABELS[m]} vs w")
    axes[0][0].legend(frameon=False)
    return _save(fig, out_dir, "f1_ccds_vs_w")


def f2_tradeoff(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    """F2: CLAP_norm vs CCDS (CLAP) with w as labelled points: the trade-off curve (H1)."""
    plt = _plt()
    s = summary[summary["null"] == "default"]
    x = s[s["metric"] == "ccds_clap"]
    y = s[s["metric"] == "clap_norm"]
    j = x.merge(y, on=["model", "domain", "w"], suffixes=("_x", "_y"))
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for (model, dom), g in j.groupby(["model", "domain"]):
        g = g.sort_values("w")
        c = series_color(model, dom)
        ax.errorbar(g["mean_x"], g["mean_y"],
                    xerr=[g["mean_x"] - g["ci_low_x"], g["ci_high_x"] - g["mean_x"]],
                    yerr=[g["mean_y"] - g["ci_low_y"], g["ci_high_y"] - g["mean_y"]],
                    color=c, marker="o", ms=4, capsize=2, lw=1, label=f"{model} / {dom}")
        for r in g.itertuples():
            ax.annotate(f"w={r.w:g}", (r.mean_x, r.mean_y), textcoords="offset points",
                        xytext=(4, 3), fontsize=7, color=c)
    ax.set(xlabel="diversity: CCDS (CLAP)", ylabel="prompt adherence: CLAP_norm",
           title="F2: adherence vs diversity")
    ax.legend(frameon=False, fontsize=7)
    return _save(fig, out_dir, "f2_clap_vs_ccds")


def f3_quality(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    """F3: CLAP_norm, FD_openl3 and KL_passt vs w (subplots)."""
    plt = _plt()
    s = summary[summary["null"] == "default"]
    metrics = ["clap_norm", "fd_openl3", "kl_passt"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, m in zip(axes, metrics):
        sub = s[s["metric"] == m]
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
        for (model, dom), g in sub.groupby(["model", "domain"]):
            _line_ci(ax, g, series_color(model, dom), f"{model} / {dom}")
        ax.set(xlabel="guidance scale w", ylabel=LABELS[m], title=f"F3: {LABELS[m]}")
    axes[0].legend(frameon=False, fontsize=7)
    return _save(fig, out_dir, "f3_quality_vs_w")


def f4_nn_similarity(copies: pd.DataFrame, held: pd.DataFrame | None, out_dir: Path,
                     candidate_percentile: float = 99.0, null: str = "default") -> list[Path]:
    """F4: nearest-neighbour similarity s_gen per w vs the held-out control s_held (§5.2)."""
    plt = _plt()
    c = copies[copies["null"] == null]
    embs = sorted(c["embedder"].astype(str).unique())
    fig, axes = plt.subplots(1, len(embs), figsize=(4.6 * len(embs), 3.3), squeeze=False)
    ws = sorted(c["w"].unique())
    cmap = plt.get_cmap("viridis")
    for ax, emb in zip(axes[0], embs):
        ce = c[c["embedder"].astype(str) == emb]
        he = held[held["embedder"].astype(str) == emb] if held is not None else None
        allv = np.concatenate([ce["s_gen"].to_numpy()] +
                              ([he["s_held"].to_numpy()] if he is not None else []))
        bins = np.linspace(np.nanmin(allv), np.nanmax(allv), 40)
        for i, w in enumerate(ws):
            ax.hist(ce.loc[ce["w"] == w, "s_gen"], bins=bins, density=True, histtype="step",
                    lw=1.3, color=cmap(i / max(1, len(ws) - 1) * 0.9), label=f"w={w:g}")
        if he is not None and not he.empty:
            ax.hist(he["s_held"], bins=bins, density=True, histtype="stepfilled", alpha=0.25,
                    color="#555555", label="held-out control")
            thr = np.percentile(he["s_held"], candidate_percentile)
            ax.axvline(thr, color="#D55E00", ls="--", lw=1,
                       label=f"candidate thr. (p{candidate_percentile:g})")
        ax.set(xlabel="NN cosine similarity to training set T", ylabel="density",
               title=f"F4: {emb}")
    axes[0][0].legend(frameon=False, fontsize=7)
    return _save(fig, out_dir, "f4_nn_similarity")


def f5_null_overlay(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    """F5: F1/F3 curves for the default vs learned (and control) nulls, overlaid (§5.3)."""
    plt = _plt()
    multi = summary.groupby("model")["null"].nunique()
    models = multi[multi > 1].index.tolist() or summary["model"].unique().tolist()
    s = summary[summary["model"].isin(models)]
    metrics = [m for m in ("ccds_clap", "clap_norm", "fd_openl3", "kl_passt")
               if (s["metric"] == m).any()]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.2), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for (model, dom, null), g in s[s["metric"] == m].groupby(["model", "domain", "null"]):
            _line_ci(ax, g, DOMAIN_COLOR.get(dom, OKABE_ITO[2]), f"{model}/{dom}/{null}",
                     NULL_STYLE.get(null, "-"))
        ax.set(xlabel="guidance scale w", ylabel=LABELS.get(m, m), title=f"F5: {LABELS.get(m, m)}")
    axes[0][0].legend(frameon=False, fontsize=6)
    return _save(fig, out_dir, "f5_null_overlay")


def f6_bt_strength(strengths: pd.DataFrame, out_dir: Path, domain: str = "all") -> list[Path]:
    """F6: Bradley-Terry log-strength vs w, judges and listeners on one plot per question."""
    plt = _plt()
    s = strengths[strengths["domain"] == domain]
    qs = [q for q in QUESTIONS if (s["question"] == q).any()]
    raters = sorted(r for r in s["rater"].unique() if r != "listeners")
    colors = {**{r: OKABE_ITO[(i + 2) % 7] for i, r in enumerate(raters)}, **RATER_COLOR}
    fig, axes = plt.subplots(1, len(qs), figsize=(3.8 * len(qs), 3.2), squeeze=False)
    for ax, q in zip(axes[0], qs):
        for r, g in s[s["question"] == q].groupby("rater"):
            g = g.rename(columns={"log_strength": "mean"})
            _line_ci(ax, g, colors.get(r, "#777777"), r, "-" if r == "listeners" else "--")
        ax.axhline(0, color="#999999", lw=0.6)
        ax.set(xlabel="guidance scale w", ylabel="BT log-strength", title=f"F6: {q}")
    axes[0][0].legend(frameon=False, fontsize=7)
    return _save(fig, out_dir, "f6_bt_strength")
