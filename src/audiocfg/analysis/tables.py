"""Tables T1-T4 (PROJECT_DESIGN.md §5.1-§5.4), written as CSV (numbers) + Markdown (formatted).

Each ``build_*`` returns numeric DataFrames; :func:`write_table` writes them. No value is typed by
hand: everything comes from the result files in ``schema.RESULT_FILES``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from audiocfg.analysis import bradley_terry as bt
from audiocfg.analysis.recommend import select_fd
from audiocfg.analysis.stats import (add_holm, bootstrap_ci, changes_with_w, diff_ci_from_cis,
                                     paired_diff_ci)
from audiocfg.schema import DOMAINS, QUESTIONS

PER_PROMPT_METRICS = ("ccds_clap", "ccds_aux", "clap_norm", "kl_passt")
METRIC_LABELS = {
    "ccds_clap": "CCDS (CLAP)", "ccds_aux": "CCDS (PANNs/MERT)", "clap_norm": "CLAP_norm",
    "fd_openl3": "FD_openl3", "kl_passt": "KL_passt", "cand_pct": "candidates (%)",
    "copy_pct": "confirmed copies (%)",
}
T1_METRICS = ("ccds_clap", "ccds_aux", "clap_norm", "fd_openl3", "kl_passt")


# --------------------------------------------------------------------------- formatting
def fmt_ci(m: float, lo: float, hi: float, digits: int = 3) -> str:
    if m is None or not np.isfinite(m):
        return "—"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _cell(v, digits: int) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "—"
        return f"{v:g}" if float(v).is_integer() else f"{v:.{digits}f}"
    return str(v)


def to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    """Minimal Markdown table (no ``tabulate`` dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v, digits) for v in row) + " |")
    return "\n".join(lines)


def write_table(df: pd.DataFrame, out_dir: Path, stem: str, markdown: str | None = None,
                title: str | None = None) -> list[Path]:
    """Write ``<stem>.csv`` (numeric) and ``<stem>.md`` (formatted, or a plain dump)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv, md = out_dir / f"{stem}.csv", out_dir / f"{stem}.md"
    df.to_csv(csv, index=False)
    body = markdown if markdown is not None else to_markdown(df)
    md.write_text((f"# {title}\n\n" if title else "") + body + "\n", encoding="utf-8")
    return [csv, md]


# --------------------------------------------------------------------------- E1 / T1
def summary_e1(pp: pd.DataFrame, fd: pd.DataFrame | None, n_boot: int, ci: float,
               seed: int = 0) -> pd.DataFrame:
    """Long table (model, domain, null, w, metric, mean, ci_low, ci_high, n) for E1/E3 curves."""
    keys = ["model", "domain", "null", "w"]
    parts = []
    for i, metric in enumerate(m for m in PER_PROMPT_METRICS if m in pp):
        t = bootstrap_ci(pp, metric, keys, n_boot=n_boot, ci=ci, seed=seed + i)
        parts.append(t.rename(columns={"n_clusters": "n"}).assign(metric=metric))
    if fd is not None and not fd.empty:
        f = fd.assign(metric="fd_" + fd["embedder"].astype(str).str.lower())
        parts.append(f[keys + ["metric", "fd", "ci_low", "ci_high", "n"]]
                     .rename(columns={"fd": "mean"}))
    out = pd.concat(parts, ignore_index=True)
    return out[keys + ["metric", "mean", "ci_low", "ci_high", "n"]]


def build_t1(summary: pd.DataFrame, rec: pd.DataFrame | None, null: str = "default"
             ) -> tuple[pd.DataFrame, str]:
    """T1 per model x domain (null = default): w, CCDS(CLAP), CCDS(aux), CLAP_norm, FD, KL."""
    s = summary[summary["null"] == null]
    wide = s.pivot_table(index=["model", "domain", "w"], columns="metric",
                         values=["mean", "ci_low", "ci_high"]).sort_index()
    num = pd.DataFrame(index=wide.index)
    for m in T1_METRICS:
        for stat, suf in (("mean", ""), ("ci_low", "_lo"), ("ci_high", "_hi")):
            num[m + suf] = wide[(stat, m)] if (stat, m) in wide.columns else np.nan
    num = num.reset_index()

    md = []
    for (model, domain), g in num.groupby(["model", "domain"], sort=True):
        md.append(f"## {model} / {domain}\n")
        disp = pd.DataFrame({"w": g["w"]})
        for m in T1_METRICS:
            disp[METRIC_LABELS[m] + " ± CI"] = [fmt_ci(a, b, c) for a, b, c in
                                                 zip(g[m], g[m + "_lo"], g[m + "_hi"])]
        md.append(to_markdown(disp))
        if rec is not None and not rec.empty:
            r = rec[(rec["model"] == model) & (rec["domain"] == domain) & (rec["null"] == null)]
            if not r.empty:
                r = r.iloc[0]
                md.append(f"\n**w\\*_auto = {_cell(r['w_star_auto'], 3)}** (CLAP_norm max at "
                          f"w={_cell(r['w_clap_max'], 3)}, FD best at w={_cell(r['w_fd_best'], 3)}"
                          f"; FD criterion: {r['fd_criterion']}). CCDS(CLAP) change vs "
                          f"w={_cell(r['w_ref'], 3)}: "
                          f"{fmt_ci(r['ccds_loss'], r['ccds_loss_ci_low'], r['ccds_loss_ci_high'])}"
                          f" ({_cell(r['ccds_loss_pct'], 1)}%). {r.get('note', '')}")
        md.append("")
    md.append("CI: 95% bootstrap over prompts (§6.4). FD CI as provided by `e1_fd.parquet`.")
    return num, "\n".join(md)


def e1_paired_vs_ref(pp: pd.DataFrame, w_ref: float, n_boot: int, ci: float,
                     seed: int = 0) -> pd.DataFrame:
    """metric(w) - metric(w_ref) per prompt, Wilcoxon + Holm per (model, domain, null) (§5.1)."""
    parts = []
    for i, m in enumerate(x for x in PER_PROMPT_METRICS if x in pp):
        d = paired_diff_ci(pp, m, by="w", ref=w_ref, group_cols=["model", "domain", "null"],
                           n_boot=n_boot, ci=ci, seed=seed + i)
        parts.append(d.assign(metric=m))
    out = pd.concat(parts, ignore_index=True)
    return add_holm(out, family=["model", "domain", "null"])


# --------------------------------------------------------------------------- E2 / T2
def build_t2(copies: pd.DataFrame, held: pd.DataFrame | None, n_boot: int, ci: float,
             seed: int = 0) -> pd.DataFrame:
    """T2 per (model, null, embedder, w): N, candidates %, confirmed copies %, medians."""
    keys = ["model", "null", "embedder", "w"]
    c = copies.assign(cand_pct=100 * bt.as_bool(copies["is_candidate"]).astype(float),
                      copy_pct=100 * bt.as_bool(copies["is_copy"]).astype(float))
    out = c.groupby(keys).agg(n_clips=("clip_id", "size"), median_s_gen=("s_gen", "median"))
    for i, m in enumerate(("cand_pct", "copy_pct")):
        t = bootstrap_ci(c, m, keys, n_boot=n_boot, ci=ci, seed=seed + i).set_index(keys)
        out[m], out[m + "_lo"], out[m + "_hi"] = t["mean"], t["ci_low"], t["ci_high"]
    out = out.reset_index()
    if held is not None and not held.empty:
        med = held.groupby("embedder")["s_held"].median()
        out["median_s_held"] = out["embedder"].map(med)
    else:
        out["median_s_held"] = np.nan
    return out


def t2_markdown(t2: pd.DataFrame, trend: pd.DataFrame | None) -> str:
    md = []
    for (model, null, emb), g in t2.groupby(["model", "null", "embedder"], sort=True):
        md.append(f"## {model} / null={null} / embedder={emb}\n")
        disp = pd.DataFrame({
            "w": g["w"], "N": g["n_clips"],
            "candidates (%) ± CI": [fmt_ci(*x, 2) for x in
                                    zip(g["cand_pct"], g["cand_pct_lo"], g["cand_pct_hi"])],
            "confirmed copies (%) ± CI": [fmt_ci(*x, 2) for x in
                                          zip(g["copy_pct"], g["copy_pct_lo"], g["copy_pct_hi"])],
            "median s_gen": g["median_s_gen"], "median s_held": g["median_s_held"]})
        md.append(to_markdown(disp))
        if trend is not None and not trend.empty:
            tr = trend[(trend["model"] == model) & (trend["null"] == null)
                       & (trend["embedder"] == emb)]
            for r in tr.itertuples():
                md.append(f"\nTrend `{r.outcome} ~ w`: slope {fmt_ci(r.slope, r.ci_low, r.ci_high)}"
                          f" log-odds per unit w ({r.n_events}/{r.n} events; {r.method}).")
        md.append("")
    md.append("Confirmed copies are a lower bound: T is a subset of the SAO training data (§5.2).")
    return "\n".join(md)


# --------------------------------------------------------------------------- E3 / T3
def build_t3(pp: pd.DataFrame, fd: pd.DataFrame | None, copies: pd.DataFrame | None,
             n_boot: int, ci: float, seed: int = 0, ref_null: str = "default",
             fd_embedder: str = "openl3", copy_embedder: str | None = None) -> pd.DataFrame:
    """T3: Δ = null variant - default per (model, domain, w, metric), paired by prompt (§5.3).

    Per-prompt metrics: paired bootstrap + Wilcoxon, Holm across metrics per (model, domain,
    null, w). FD: approximate CI from per-setting CIs (no p-value). Copy candidates: paired by
    prompt on the per-prompt candidate rate.
    """
    keys = ["model", "domain", "w"]
    parts = []
    for i, m in enumerate(x for x in PER_PROMPT_METRICS if x in pp):
        d = paired_diff_ci(pp, m, by="null", ref=ref_null, group_cols=keys, n_boot=n_boot,
                           ci=ci, seed=seed + i)
        parts.append(d.assign(metric=m, method="paired bootstrap over prompts"))
    if copies is not None and not copies.empty:
        c = copies
        emb = copy_embedder or sorted(c["embedder"].astype(str).unique())[0]
        c = c[c["embedder"].astype(str) == emb]
        c = c.assign(cand_pct=100 * bt.as_bool(c["is_candidate"]).astype(float))
        d = paired_diff_ci(c, "cand_pct", by="null", ref=ref_null, group_cols=keys,
                           n_boot=n_boot, ci=ci, seed=seed + 50)
        parts.append(d.assign(metric="cand_pct", method=f"paired bootstrap ({emb})"))
    fsel = select_fd(fd, fd_embedder)
    if fsel is not None:
        rows = []
        f = fsel.groupby(keys + ["null"])[["fd", "ci_low", "ci_high"]].mean()
        for (model, domain, w, null), r in f.iterrows():
            if null == ref_null or (model, domain, w, ref_null) not in f.index:
                continue
            b = f.loc[(model, domain, w, ref_null)]
            dm, lo, hi = diff_ci_from_cis(r["fd"], r["ci_low"], r["ci_high"],
                                          b["fd"], b["ci_low"], b["ci_high"], ci)
            rows.append({"model": model, "domain": domain, "w": w, "null": null, "ref": ref_null,
                         "mean_diff": dm, "ci_low": lo, "ci_high": hi, "n_clusters": np.nan,
                         "p_wilcoxon": np.nan, "metric": f"fd_{fd_embedder}",
                         "method": "approx. from per-setting CIs (independent)"})
        if rows:
            parts.append(pd.DataFrame(rows))
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = add_holm(out, family=["model", "domain", "null", "w"])
    out["significant"] = np.where(out["p_holm"].notna(), out["p_holm"] < 1 - ci,
                                  (out["ci_low"] > 0) | (out["ci_high"] < 0))
    return out


def t3_markdown(t3: pd.DataFrame) -> str:
    md = []
    for (model, domain, null), g in t3.groupby(["model", "domain", "null"], sort=True):
        md.append(f"## {model} / {domain}: Δ = {null} − {g['ref'].iloc[0]}\n")
        disp = g.assign(val=[fmt_ci(a, b, c) + (" *" if s else "") for a, b, c, s in
                             zip(g["mean_diff"], g["ci_low"], g["ci_high"], g["significant"])])
        disp = disp.pivot_table(index="w", columns="metric", values="val", aggfunc="first")
        order = list(PER_PROMPT_METRICS) + sorted(c for c in disp.columns if c.startswith("fd_"))
        disp = disp[[c for c in order if c in disp.columns]
                    + [c for c in disp.columns if c not in order]]
        disp.columns = [METRIC_LABELS.get(c, c) for c in disp.columns]
        md.append(to_markdown(disp.reset_index()))
        md.append("")
    md.append("`*` = significant after Holm (per-prompt metrics) or CI excludes 0 (FD). "
              "H3 supported if Δ FD < 0 significantly at w ≥ 7.")
    return "\n".join(md)


# --------------------------------------------------------------------------- E4 / T4
def build_t4(judge: pd.DataFrame | None, listeners: pd.DataFrame | None, n_boot: int, ci: float,
             seed: int = 0, prior: float = 0.1
             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """T4 (per question x domain): preferred scale ± CI, agreement, kappa, flip rate; plus
    BT strengths (for F6), H4 differences judge − listeners and listener-cleaning info."""
    dec = bt.combine_orders(judge) if judge is not None and not judge.empty else None
    lis, info = (bt.clean_listeners(listeners) if listeners is not None and not listeners.empty
                 else (None, {}))
    frames = [x for x in (dec, lis) if x is not None and not x.empty]
    levels = sorted(set(pd.concat([f["w_a"] for f in frames]).astype(float))
                    | set(pd.concat([f["w_b"] for f in frames]).astype(float)))
    domains = ["all"] + [d for d in DOMAINS if any((f["domain"] == d).any() for f in frames)]
    t4, strengths, h4 = [], [], []
    k = 0
    for q in QUESTIONS:
        for dom in domains:
            def sub(df):
                if df is None:
                    return None
                s = df[df["question"] == q]
                return s if dom == "all" else s[s["domain"] == dom]
            L, J = sub(lis), sub(dec)
            Jraw = sub(judge) if judge is not None else None
            if L is not None and L.empty:
                L = None
            base = {"question": q, "domain": dom}
            fl = None
            maj = bt.listener_majority(L) if L is not None else None
            for name in (sorted(J["judge"].unique()) if J is not None else []):
                Jn = J[J["judge"] == name]
                if L is not None and Jn["pair_id"].isin(L["pair_id"]).any():
                    Jn = Jn[Jn["pair_id"].isin(L["pair_id"])]  # exactly the listeners' pairs
                    fj, fl_, diff = bt.bt_compare(Jn, L, levels, n_boot, ci, seed + k, prior)
                    if fl is None:
                        fl = fl_
                    h4.append({**base, "judge": name, "preferred_judge": fj.preferred,
                               "preferred_listeners": fl_.preferred, **diff})
                    agr = bt.agreement_with_majority(Jn, maj)
                else:
                    fj = bt.bt_bootstrap(Jn, levels, n_boot=n_boot, ci=ci, seed=seed + k,
                                         prior=prior)
                    agr = {"agreement": np.nan, "kappa": np.nan, "n_agree": 0}
                k += 1
                raw = Jraw[(Jraw["judge"] == name) & Jraw["pair_id"].isin(Jn["pair_id"])]
                t4.append({**base, "rater": name, **_pref(fj), **agr,
                           "flip_rate": bt.flip_rate(raw), "agreement_ref": "listener majority"})
                strengths.append(fj.to_frame().assign(**base, rater=name))
            if L is not None:
                if fl is None:
                    fl = bt.bt_bootstrap(L, levels, rater_col="listener_id", n_boot=n_boot,
                                         ci=ci, seed=seed + k, prior=prior)
                    k += 1
                t4.append({**base, "rater": "listeners", **_pref(fl),
                           **bt.inter_listener_ceiling(L), "flip_rate": np.nan,
                           "agreement_ref": "ceiling: each listener vs majority of others"})
                strengths.append(fl.to_frame().assign(**base, rater="listeners"))
    t4 = pd.DataFrame(t4)
    strengths = pd.concat(strengths, ignore_index=True) if strengths else pd.DataFrame()
    return t4, strengths, pd.DataFrame(h4), info


def _pref(f: bt.BTFit) -> dict:
    lo, hi = f.preferred_ci
    return {"n_comparisons": f.n_comparisons, "preferred_w": f.preferred, "pref_ci_low": lo,
            "pref_ci_high": hi, "p_preferred": f.p_preferred}


def t4_markdown(t4: pd.DataFrame, h4: pd.DataFrame) -> str:
    md = []
    for (q, dom), g in t4.groupby(["question", "domain"], sort=False):
        md.append(f"## question = {q}, domain = {dom}\n")
        disp = pd.DataFrame({
            "": g["rater"],
            "preferred scale (BT) [CI]": [f"{p:g} [{lo:g}, {hi:g}] (p={pp:.2f})" for p, lo, hi, pp
                                         in zip(g["preferred_w"], g["pref_ci_low"],
                                                g["pref_ci_high"], g["p_preferred"])],
            "agreement with listener majority": [
                ("ceiling " if r == "listeners" else "") + _cell(a, 3)
                for r, a in zip(g["rater"], g["agreement"])],
            "kappa": g["kappa"], "flip rate (A/B)": g["flip_rate"]})
        md.append(to_markdown(disp))
        if not h4.empty:
            hh = h4[(h4["question"] == q) & (h4["domain"] == dom)]
            for r in hh.itertuples():
                md.append(f"\nH4 ({r.judge} − listeners): Δw* = "
                          f"{fmt_ci(r.diff, r.ci_low, r.ci_high, 1)}, P(Δ>0)={r.p_diff_gt_0:.2f}"
                          f" → {'supported' if r.h4_supported else 'not supported'}.")
        md.append("")
    md.append("Preferred scale = argmax BT strength; CI = percentile interval of the bootstrap "
              "argmax (prompts, and listeners for listener data).")
    return "\n".join(md)


# --------------------------------------------------------------------------- §2 falsification
def falsification_check(summary: pd.DataFrame | None, t2: pd.DataFrame | None,
                        strengths: pd.DataFrame | None, w_ref: float = 1.0) -> pd.DataFrame:
    """CI at w=1 vs extreme w for diversity, copy rate and judge preference (§2, §6.4)."""
    parts = []
    if summary is not None:
        s = summary[(summary["null"] == "default") & summary["metric"].isin(
            ["ccds_clap", "ccds_aux", "clap_norm"])]
        c = changes_with_w(s, ["metric", "model", "domain"], w_ref=w_ref)
        if not c.empty:
            c["group"] = c["model"] + "/" + c["domain"]
            parts.append(c.assign(experiment="E1"))
    if t2 is not None and not t2.empty:
        t = t2[t2["null"] == "default"].rename(columns={"cand_pct": "mean", "cand_pct_lo": "ci_low",
                                                        "cand_pct_hi": "ci_high"})
        c = changes_with_w(t, ["model", "embedder"], w_ref=w_ref)
        if not c.empty:
            c["group"] = c["model"] + "/" + c["embedder"]
            parts.append(c.assign(experiment="E2", metric="cand_pct"))
    if strengths is not None and not strengths.empty:
        s = strengths[(strengths["question"] == "overall") & (strengths["domain"] == "all")]
        s = s.rename(columns={"log_strength": "mean"})
        c = changes_with_w(s, ["rater"], w_ref=w_ref)
        if not c.empty:
            c["group"] = c["rater"]
            parts.append(c.assign(experiment="E4", metric="bt_log_strength"))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out[["experiment", "metric", "group", "w_ref", "w_ext", "mean_ref", "ci_ref",
                "mean_ext", "ci_ext", "changes"]]
