"""``audiocfg analyze``: tables T1-T4 and figures F1-F6 from ``results/`` (PROJECT_DESIGN.md §2).

Builds whatever has its inputs present and logs what it skips; writes ``results/MANIFEST.md``
mapping each output to its research question (PB) and input files. ``--synthetic`` writes a fake
results set into a temporary directory first, to dry-run the whole pipeline.

Top-level imports are kept light because ``audiocfg.cli`` imports every command at startup.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger("audiocfg.analysis")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--results-dir", type=Path, default=None, help="default: config results_dir")
    p.add_argument("--figures-dir", type=Path, default=None, help="default: config figures_dir")
    p.add_argument("--n-boot", type=int, default=None, help="default: config stats.bootstrap_reps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fd-embedder", default="openl3")
    p.add_argument("--synthetic", action="store_true",
                   help="generate a fake results/ set into a temp dir and analyse that")
    p.add_argument("--synthetic-dir", type=Path, default=None,
                   help="where --synthetic writes results/ and figures/ (default: new temp dir)")


def _read(results_dir: Path, name: str):
    import pandas as pd

    from audiocfg.schema import RESULT_FILES

    path = results_dir / name
    if not path.exists():
        return None
    df = pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path)
    missing = set(RESULT_FILES[name]) - set(df.columns)
    if missing:
        log.warning("%s is missing columns %s; skipping it", name, sorted(missing))
        return None
    if "w" in df:
        df["w"] = df["w"].astype(float)
    return df


def run(args: argparse.Namespace) -> dict:
    """Run the analysis; returns ``{output_name: [paths]}`` of what was produced."""
    import tempfile

    from audiocfg.analysis import figures as F
    from audiocfg.analysis import tables as T
    from audiocfg.analysis.recommend import recommend_table
    from audiocfg.analysis.stats import logistic_trend
    from audiocfg.config import DEFAULT_CONFIG, load_config
    from audiocfg.schema import RESULT_FILES

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(getattr(args, "config", None) or DEFAULT_CONFIG)
    stats_cfg = cfg.raw.get("stats", {})
    n_boot = args.n_boot or int(stats_cfg.get("bootstrap_reps", 10_000))
    ci = float(stats_cfg.get("ci", 0.95))
    seed = args.seed

    if args.synthetic:
        from audiocfg.analysis.synthetic import write_synthetic

        root = Path(args.synthetic_dir or tempfile.mkdtemp(prefix="audiocfg_synthetic_"))
        results_dir, figures_dir = root / "results", root / "figures"
        write_synthetic(results_dir, scales=cfg.guidance_scales, seeds=cfg.seeds, seed=seed)
        log.info("synthetic results written to %s (NOT research data)", results_dir)
    else:
        results_dir = Path(args.results_dir or cfg.results_dir)
        figures_dir = Path(args.figures_dir or cfg.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    data = {name: _read(results_dir, name) for name in RESULT_FILES}
    present = {k for k, v in data.items() if v is not None}
    log.info("inputs present: %s", ", ".join(sorted(present)) or "none")
    w_ref = float(min(cfg.guidance_scales))
    e2_pct = float(cfg.raw.get("e2_copying", {}).get("candidate_percentile", 99.0))

    produced: dict[str, list[Path]] = {}
    manifest: list[tuple[str, str, list[str], str]] = []

    def step(out: str, pb: str, needs: list[str], fn, optional: list[str] = ()) -> None:
        missing = [n for n in needs if n not in present]
        inputs = needs + [o for o in optional if o in present]
        if missing:
            log.info("skip %s: missing %s", out, ", ".join(missing))
            manifest.append((out, pb, needs + list(optional), "skipped: missing " + ", ".join(missing)))
            return
        try:
            paths = fn()
        except Exception as e:  # keep going: one broken input should not block the others
            log.exception("failed %s", out)
            manifest.append((out, pb, inputs, f"FAILED: {type(e).__name__}: {e}"))
            return
        produced[out] = paths
        manifest.append((out, pb, inputs, "built: " + ", ".join(p.name for p in paths)))
        log.info("built %s", out)

    pp, fd = data["e1_per_prompt.parquet"], data["e1_fd.parquet"]
    copies, held = data["e2_copies.parquet"], data["e2_held.parquet"]
    cache: dict = {}

    def summary():
        if "summary" not in cache:
            cache["summary"] = T.summary_e1(pp, fd, n_boot, ci, seed)
        return cache["summary"]

    def t1():
        s = summary()
        rec = recommend_table(pp, fd, args.fd_embedder, n_boot, ci, seed)
        num, md = T.build_t1(s, rec)
        paired = T.e1_paired_vs_ref(pp, w_ref, n_boot, ci, seed)
        return (T.write_table(num, results_dir, "t1_e1_scales", md,
                              "T1 — E1 guidance-scale sweep (PB1a, PB1b)")
                + [_csv(s, results_dir, "e1_summary"), _csv(rec, results_dir, "t1_recommended_scale"),
                   _csv(paired, results_dir, "e1_paired_vs_w1")])

    def t2():
        t = T.build_t2(copies, held, n_boot, ci, seed)
        cache["t2"] = t
        c = copies.assign(is_candidate=copies["is_candidate"].astype(bool),
                          is_copy=copies["is_copy"].astype(bool))
        trend = [logistic_trend(c, o, group_cols=["model", "null", "embedder"], n_boot=n_boot,
                                ci=ci, seed=seed) for o in ("is_candidate", "is_copy")]
        import pandas as pd
        trend = pd.concat(trend, ignore_index=True)
        return (T.write_table(t, results_dir, "t2_copies", T.t2_markdown(t, trend),
                              "T2 — E2 training-data copies vs w (PB2)")
                + [_csv(trend, results_dir, "t2_trend")])

    def t3():
        t = T.build_t3(pp, fd, copies, n_boot, ci, seed, fd_embedder=args.fd_embedder)
        if t.empty:
            raise ValueError("no null variant other than 'default' in e1_per_prompt")
        return T.write_table(t, results_dir, "t3_null_delta", T.t3_markdown(t),
                             "T3 — E3 learned vs default null (PB3)")

    def t4():
        t, strengths, h4, info = T.build_t4(data["e4_judge.parquet"], data["e4_listeners.csv"],
                                            n_boot, ci, seed)
        cache["strengths"] = strengths
        if info.get("excluded_listeners"):
            log.info("excluded listeners (attention checks): %s", info["excluded_listeners"])
        out = T.write_table(t, results_dir, "t4_judge_vs_listeners", T.t4_markdown(t, h4),
                            "T4 — E4 judge vs listeners (PB4)")
        out.append(_csv(strengths, results_dir, "t4_bt_strengths"))
        if not h4.empty:
            out.append(_csv(h4, results_dir, "t4_h4_difference"))
        return out

    E1, FD = "e1_per_prompt.parquet", "e1_fd.parquet"
    step("T1", "PB1a, PB1b", [E1], t1, optional=[FD])
    step("T2", "PB2", ["e2_copies.parquet"], t2, optional=["e2_held.parquet"])
    step("T3", "PB3", [E1], t3, optional=[FD, "e2_copies.parquet"])
    if "e4_judge.parquet" in present or "e4_listeners.csv" in present:
        step("T4", "PB4", [], t4, optional=["e4_judge.parquet", "e4_listeners.csv"])
    else:
        step("T4", "PB4", ["e4_judge.parquet"], t4, optional=["e4_listeners.csv"])

    step("F1", "PB1a", [E1], lambda: F.f1_ccds(summary(), figures_dir))
    step("F2", "PB1a", [E1], lambda: F.f2_tradeoff(summary(), figures_dir))
    step("F3", "PB1b", [E1], lambda: F.f3_quality(summary(), figures_dir), optional=[FD])
    step("F4", "PB2", ["e2_copies.parquet"],
         lambda: F.f4_nn_similarity(copies, held, figures_dir, e2_pct), optional=["e2_held.parquet"])
    step("F5", "PB3", [E1], lambda: F.f5_null_overlay(summary(), figures_dir), optional=[FD])

    def f6():
        if cache.get("strengths") is None or cache["strengths"].empty:
            raise ValueError("T4 produced no Bradley-Terry strengths")
        return F.f6_bt_strength(cache["strengths"], figures_dir)

    step("F6", "PB4", [] if "T4" in produced else ["T4 (Bradley-Terry strengths)"], f6,
         optional=["e4_judge.parquet", "e4_listeners.csv"])

    def falsification():
        f = T.falsification_check(cache.get("summary"), cache.get("t2"), cache.get("strengths"),
                                  w_ref)
        if f.empty:
            raise ValueError("nothing to check")
        return [_csv(f, results_dir, "falsification_check")]

    step("§2 falsification check", "H1, H2, H4", [], falsification,
         optional=[E1, "e2_copies.parquet", "e4_judge.parquet", "e4_listeners.csv"])

    _write_manifest(results_dir, manifest, synthetic=args.synthetic)
    produced["MANIFEST"] = [results_dir / "MANIFEST.md"]
    log.info("manifest: %s", results_dir / "MANIFEST.md")
    return produced


def _csv(df, out_dir: Path, stem: str) -> Path:
    path = Path(out_dir) / f"{stem}.csv"
    df.to_csv(path, index=False)
    return path


def _write_manifest(results_dir: Path, rows: list, synthetic: bool) -> None:
    lines = ["# Results manifest", ""]
    if synthetic:
        lines += ["> **SYNTHETIC DRY RUN — not research data.**", ""]
    lines += ["Generated by `audiocfg analyze`. Each output, the research question it answers "
              "(PROJECT_DESIGN.md §2) and the result files it was built from.", "",
              "| Output | PB | Inputs | Status |", "|---|---|---|---|"]
    for out, pb, inputs, status in rows:
        lines.append(f"| {out} | {pb} | {', '.join(f'`{i}`' for i in inputs) or '—'} | {status} |")
    (results_dir / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
