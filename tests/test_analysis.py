import numpy as np
import pandas as pd

from audiocfg.analysis.pareto import match_on_curve, matched_clap_test, same_g_differences

G = (1.0, 2.0, 3.0, 4.5, 6.0, 7.0, 9.0, 12.0)


def test_match_on_curve():
    x, g = np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0, 3.0])
    g_star, y = match_on_curve(x, 10 * x, g, 0.25)
    assert abs(g_star - 2.5) < 1e-9 and abs(y - 2.5) < 1e-9
    assert np.isnan(match_on_curve(x, x, g, 0.5)[0])
    g_star, _ = match_on_curve(np.array([0.1, 0.3, 0.2]), x, g, 0.25)
    assert abs(g_star - 1.75) < 1e-9


def synthetic(rescale: float, pq_gain: float, n=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n):
        offset = rng.normal(0, 0.02)
        for null in ("zero", "learned"):
            for g in G:
                ge = g if null == "zero" or g == 1 else 1 + (g - 1) * rescale
                rows.append(
                    {
                        "domain": "music",
                        "null": null,
                        "prompt_id": f"p{p}",
                        "guidance": g,
                        "clap": 0.2 + 0.05 * np.log(ge) + offset,
                        "PQ": 8 - 0.1 * ge + (pq_gain if null == "learned" and g > 1 else 0),
                        "CE": 6.0,
                        "ccds": 0.5 / ge,
                    }
                )
    return pd.DataFrame(rows)


def test_pure_guidance_rescale_shows_no_advantage():
    res = matched_clap_test(synthetic(0.5, 0.0), n_boot=200)
    pq = res[(res.metric == "PQ") & (res.g_learned == 7)].iloc[0]
    assert abs(pq.g_star_zero - 4.0) < 0.3
    assert abs(pq.delta) < 0.03 and pq.lo <= 0.03 and pq.hi >= -0.03


def test_real_gain_is_detected():
    res = matched_clap_test(synthetic(0.5, 0.5), n_boot=200)
    pq = res[(res.metric == "PQ") & (res.g_learned == 7)].iloc[0]
    assert abs(pq.delta - 0.5) < 0.05 and pq.lo > 0.3


def test_same_g_differences():
    d = same_g_differences(synthetic(1.0, 0.5))
    pq = d[(d.metric == "PQ") & (d.guidance == 7)].iloc[0]
    assert abs(pq["mean"] - 0.5) < 1e-9 and pq.n == 30


def _toy_clips():
    rows = []
    for p in ("a", "b", "c"):
        for s in range(3):
            for null, gs in (("learned", (7.0,)), ("zero", (2.0, 7.0))):
                for g in gs:
                    rows.append(
                        {"prompt_id": p, "domain": "music", "text": p, "sample": s, "guidance": g}
                        | {"null": null, "clip": f"{null}/{p}_s{s}_g{g:g}.flac"}
                    )
    return pd.DataFrame(rows)


def test_make_pairs_blind_and_deterministic():
    from audiocfg.listen.abx import make_pairs

    pairs = make_pairs(_toy_clips())
    assert pairs.equals(make_pairs(_toy_clips())) and len(pairs) == 6
    for _, r in pairs.iterrows():
        learned = r["clip_a"] if r["a_is"] == "learned" else r["clip_b"]
        zero = r["clip_b"] if r["a_is"] == "learned" else r["clip_a"]
        assert learned.startswith("learned/") and zero.startswith("zero/")
        assert zero.endswith("_g2.flac") == (r["condition"] == "matched_clap")
        assert f"_s{r['sample']}_" in learned and f"_s{r['sample']}_" in zero


def test_tally_maps_answers_to_learned_preference():
    from audiocfg.listen.abx import make_pairs, tally

    pairs = make_pairs(_toy_clips())
    pick = pairs["a_is"].map({"learned": "A", "zero": "B"})
    answers = pd.DataFrame(
        {"listener": "x", "pair_id": pairs["pair_id"], "quality": pick}
        | {"text": pick.map({"A": "B", "B": "A"})}
    )
    out = tally(answers, pairs).set_index(["condition", "question"])
    assert (out.xs("quality", level="question")["learned_pref"] == 1).all()
    assert (out.xs("text", level="question")["learned_pref"] == 0).all()


def test_level_match_equal_rms_and_peak_cap():
    from audiocfg.listen.abx import level_match

    rng = np.random.default_rng(0)
    quiet, loud = 0.01 * rng.standard_normal((1000, 2)), 0.5 * rng.standard_normal((1000, 2))
    a, b = level_match([quiet, loud])
    assert abs(np.sqrt(np.mean(a**2)) / np.sqrt(np.mean(b**2)) - 1) < 1e-6
    assert max(np.abs(a).max(), np.abs(b).max()) <= 10 ** (-1 / 20) + 1e-6


def test_load_answers_from_folder(tmp_path):
    from audiocfg.listen.abx import load_answers

    for who in ("JS", "JK"):
        rows = ["listener,pair_id,quality,text,at", f"{who},p00,A,B,t", f"{who},p01,B,B,t"]
        (tmp_path / f"{who}.csv").write_text("\n".join(rows) + "\n")
    out = load_answers(tmp_path)
    assert len(out) == 4 and set(out["listener"]) == {"JS", "JK"}


def test_verdict_non_inferiority_and_fd_direction():
    from audiocfg.analysis.criterion import verdict
    from audiocfg.repro.config import CriterionConfig

    cfg = CriterionConfig(name="c", guidance=(7.0,), ccds_margin={"music": 0.05})
    rows = [
        {"domain": "music", "metric": "PQ", "g_learned": 7.0, "g_star_zero": 2.0, "lo": 0.1},
        {"domain": "music", "metric": "ccds", "g_learned": 7.0, "g_star_zero": 2.0, "lo": -0.04},
    ]
    fd = pd.DataFrame({"null": ["zero"] * 2 + ["learned"] * 2, "guidance": [1.0, 7.0] * 2})
    fd["fd_music"] = [0.6, 0.4, 0.6, 0.5]
    ok = verdict(pd.DataFrame(rows), cfg, fd).iloc[0]
    assert ok.pq_better and ok.ccds_not_worse and ok.fd_agrees and ok.advantage
    rows[1]["lo"] = -0.06
    assert not verdict(pd.DataFrame(rows), cfg, fd).iloc[0].advantage
