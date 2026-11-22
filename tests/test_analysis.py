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
