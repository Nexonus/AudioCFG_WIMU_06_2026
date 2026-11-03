"""Analysis tests on synthetic data (CPU only): BT, bootstrap, Holm, w*_auto, analyze, listening form."""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from audiocfg.analysis import bradley_terry as bt
from audiocfg.analysis.recommend import w_star_auto
from audiocfg.analysis.stats import bootstrap_ci, changes_with_w, holm, logistic_trend, paired_diff_ci

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the app package
from app import listening  # noqa: E402

LEVELS = [1.0, 3.0, 5.0, 7.0, 9.0]


def _pairs(log_s: dict, n_prompts: int, reps: int, rng, tie_p: float = 0.0) -> pd.DataFrame:
    rows = []
    for p in range(n_prompts):
        for a, b in combinations(LEVELS, 2):
            for _ in range(reps):
                if rng.random() < tie_p:
                    ch = "tie"
                else:
                    pa = 1 / (1 + np.exp(log_s[b] - log_s[a]))
                    ch = "a" if rng.random() < pa else "b"
                rows.append({"prompt_id": f"p{p}", "w_a": a, "w_b": b, "choice": ch})
    return pd.DataFrame(rows)


def test_bt_recovers_ordering():
    rng = np.random.default_rng(1)
    true = dict(zip(LEVELS, [-1.0, -0.3, 0.4, 0.9, 0.0]))
    df = _pairs(true, 40, 3, rng)
    est = bt.fit_bt(bt.wins_matrix(df, LEVELS), prior=0.0)
    assert abs(est.sum()) < 1e-8  # identifiability
    assert list(np.argsort(est)) == list(np.argsort(list(true.values())))
    fit = bt.bt_bootstrap(df, LEVELS, n_boot=200, seed=0)
    assert fit.preferred == 7.0 and fit.p_preferred > 0.8


def test_bt_ties_count_half():
    df = pd.DataFrame({"prompt_id": ["p"] * 4, "w_a": [1.0] * 4, "w_b": [3.0] * 4,
                       "choice": ["tie", "tie", "a", "b"]})
    W = bt.wins_matrix(df, [1.0, 3.0])
    assert W[0, 1] == pytest.approx(2.0) and W[1, 0] == pytest.approx(2.0)
    assert np.allclose(bt.fit_bt(W, prior=0.0), 0.0)
    only_ties = df.assign(choice="tie")
    assert np.allclose(bt.fit_bt(bt.wins_matrix(only_ties, [1.0, 3.0])), 0.0)


def test_bootstrap_ci_covers_true_mean():
    rng = np.random.default_rng(0)
    hits = 0
    for rep in range(40):
        df = pd.DataFrame({"prompt_id": np.repeat(np.arange(60), 5),
                           "x": rng.normal(2.0, 1.0, 300)})
        r = bootstrap_ci(df, "x", n_boot=1000, seed=rep).iloc[0]
        hits += r["ci_low"] <= 2.0 <= r["ci_high"]
    assert hits >= 34  # nominal 95%


def test_holm_known_example():
    adj = holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert np.isnan(holm([np.nan, 0.01])[0])


def test_paired_diff_and_changes_with_w():
    rng = np.random.default_rng(0)
    prompts = np.arange(50)
    eff = rng.normal(0, 1, 50)  # large prompt effect, removed by pairing
    df = pd.DataFrame([{"prompt_id": p, "w": w, "v": eff[p] - 0.05 * w + rng.normal(0, 0.02)}
                       for p in prompts for w in LEVELS])
    d = paired_diff_ci(df, "v", by="w", ref=1.0, n_boot=500).set_index("w")
    assert d.loc[9.0, "ci_high"] < 0 and d.loc[9.0, "p_wilcoxon"] < 1e-6
    ci = bootstrap_ci(df, "v", ["w"], n_boot=500)
    assert not changes_with_w(ci).iloc[0]["changes"]  # prompt spread hides it unpaired


def test_logistic_trend_positive():
    rng = np.random.default_rng(0)
    df = pd.DataFrame([{"prompt_id": p, "w": w, "y": rng.random() < 1 / (1 + np.exp(3 - 0.4 * w))}
                       for p in range(40) for w in LEVELS for _ in range(5)])
    r = logistic_trend(df, "y", n_boot=300).iloc[0]
    assert r["ci_low"] > 0 and abs(r["slope"] - 0.4) < 0.15


def _crafted(fd_means):
    rng = np.random.default_rng(0)
    clap = dict(zip(LEVELS, [0.2, 0.5, 0.7, 0.83, 0.83]))
    pp = pd.DataFrame([{"prompt_id": f"p{p}", "w": w, "clap_norm": clap[w] + rng.normal(0, 0.03),
                        "ccds_clap": 0.5 - 0.02 * w + rng.normal(0, 0.01)}
                       for p in range(100) for w in LEVELS])
    fd = pd.DataFrame({"w": LEVELS, "fd": fd_means, "ci_low": np.array(fd_means) - 0.3,
                       "ci_high": np.array(fd_means) + 0.3})
    return pp, fd


def test_w_star_auto_crafted():
    pp, fd = _crafted([5.0, 3.0, 2.0, 2.1, 4.0])
    r = w_star_auto(pp, fd, n_boot=1000)
    assert r["w_star_auto"] == 7.0  # w=5 fails CLAP, w=7 passes CLAP and FD
    assert r["ccds_loss"] == pytest.approx(-0.12, abs=0.01) and r["ccds_loss_ci_high"] < 0
    pp, fd = _crafted([5.0, 3.0, 2.0, 4.0, 2.1])
    assert w_star_auto(pp, fd, n_boot=1000)["w_star_auto"] == 9.0  # w=7 FD significantly worse
    assert w_star_auto(pp, None, n_boot=1000)["w_star_auto"] == 7.0


def test_judge_agreement_kappa_and_flips():
    judge = pd.DataFrame({"judge": "j", "pair_id": ["x", "x", "y", "y"], "prompt_id": "p",
                          "domain": "sfx", "w_a": 1.0, "w_b": 3.0, "question": "overall",
                          "order": ["ab", "ba", "ab", "ba"], "choice": ["a", "a", "a", "b"]})
    assert bt.flip_rate(judge) == 0.5
    dec = bt.combine_orders(judge).set_index("pair_id")["choice"]
    assert dec["x"] == "a" and dec["y"] == "tie"
    assert bt.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == pytest.approx(1.0)
    lis = pd.DataFrame({"listener_id": ["l1", "l2", "l3"], "pair_id": "x", "question": "overall",
                        "choice": ["a", "a", "b"]})
    assert bt.listener_majority(lis)["majority"].tolist() == ["a"]
    assert bt.inter_listener_ceiling(lis)["n_agree"] == 1  # only l3's others (a, a) agree


def test_analyze_synthetic_end_to_end(tmp_path):
    from audiocfg.analysis import report

    p = argparse.ArgumentParser()
    report.add_args(p)
    args = p.parse_args(["--synthetic", "--synthetic-dir", str(tmp_path), "--n-boot", "200"])
    args.config = None
    report.run(args)
    res, fig = tmp_path / "results", tmp_path / "figures"
    for stem in ("t1_e1_scales", "t2_copies", "t3_null_delta", "t4_judge_vs_listeners"):
        assert (res / f"{stem}.csv").exists() and (res / f"{stem}.md").exists(), stem
    for stem in ("f1_ccds_vs_w", "f2_clap_vs_ccds", "f3_quality_vs_w", "f4_nn_similarity",
                 "f5_null_overlay", "f6_bt_strength"):
        assert (fig / f"{stem}.png").exists() and (fig / f"{stem}.pdf").exists(), stem
    manifest = (res / "MANIFEST.md").read_text(encoding="utf-8")
    assert "FAILED" not in manifest and "skipped" not in manifest
    t4 = pd.read_csv(res / "t4_judge_vs_listeners.csv")
    overall = t4[(t4["question"] == "overall") & (t4["domain"] == "all")].set_index("rater")
    assert overall.loc["listeners", "preferred_w"] < overall.loc["qwen2-audio", "preferred_w"]


def test_analyze_skips_missing_inputs(tmp_path):
    from audiocfg.analysis import report

    p = argparse.ArgumentParser()
    report.add_args(p)
    args = p.parse_args(["--results-dir", str(tmp_path / "r"), "--figures-dir",
                         str(tmp_path / "f"), "--n-boot", "50"])
    args.config = None
    assert set(report.run(args)) == {"MANIFEST"}
    assert "skipped" in (tmp_path / "r" / "MANIFEST.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("w_left,w_right,answer,expected", [
    (3, 7, "left", ("a", 3.0, 7.0)),
    (3, 7, "right", ("b", 3.0, 7.0)),
    (7, 3, "left", ("b", 3.0, 7.0)),
    (7, 3, "right", ("a", 3.0, 7.0)),
    (7, 3, "tie", ("tie", 3.0, 7.0)),
])
def test_left_right_to_ab(w_left, w_right, answer, expected):
    assert listening.lr_to_ab(w_left, w_right, answer) == expected


def test_listening_record_and_resume(tmp_path):
    plan_row = {"listener_id": "L01", "position": "0", "pair_id": "p1__s0__w3-w9",
                "prompt_id": "p1", "domain": "sfx", "seed": "0", "w_left": "9", "w_right": "3",
                "path_left": "a.wav", "path_right": "b.wav", "is_attention_check": "False"}
    out = tmp_path / "e4_listeners.csv"
    listening.record(out, plan_row, {"match": "left", "realism": "right", "overall": "tie"})
    listening.record(out, plan_row, {"match": "left", "realism": "left", "overall": "tie"})
    df = pd.read_csv(out)
    assert list(df.columns) == list(listening.LISTENER_COLUMNS) and len(df) == 3  # upserted
    got = df.set_index("question")
    assert got.loc["match", "choice"] == "b" and got.loc["realism", "choice"] == "b"
    assert (got["w_a"] == 3.0).all() and (got["w_b"] == 9.0).all()
    assert listening.remaining([plan_row], out, "L01") == []
