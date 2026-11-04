import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from audiocfg.config import load_config
from audiocfg.judge import listening_plan, run
from audiocfg.judge.listening_plan import build_plan, prompt_clap_at_w, stratified_prompts
from audiocfg.judge.pairs import (build_pairs, flip_rate, pair_id, resolve_orders, resolve_table,
                                  scale_pairs, side_to_relative, to_relative)
from audiocfg.judge.qwen_judge import (AudioboxAestheticsJudge, FakeJudge, load_template,
                                       parse_answer, render_prompt)
from audiocfg.judge.run import ClipLocator, judge_pairs, merge_results, pairs_from_plan
from audiocfg.schema import CHOICES, QUESTIONS, RESULT_FILES

SCALES = (1, 3, 5, 7, 9)


def _prompts(n_per_domain=12):
    return pd.DataFrame(
        [{"prompt_id": f"{d}_{i:03d}", "domain": d, "text": f"{d} {i}"}
         for d in ("sfx", "music") for i in range(n_per_domain)])


def _plan(seed=2026, tmp=Path("out")):
    return build_plan(_prompts(), range(10), SCALES, ClipLocator(tmp), seed=seed)


# ---------------------------------------------------------------- pairs and orders
def test_scale_pairs_are_ordered_and_complete():
    p = scale_pairs(SCALES)
    assert len(p) == 10 and all(a < b for a, b in p) and len(set(p)) == 10


def test_to_relative_maps_ba_back():
    assert to_relative("A", "ab") == "a" and to_relative("B", "ab") == "b"
    assert to_relative("A", "ba") == "b" and to_relative("B", "ba") == "a"
    assert to_relative("tie", "ba") == "tie"
    with pytest.raises(ValueError):
        to_relative("C", "ab")


@pytest.mark.parametrize("ab,ba,expected", [
    ("A", "B", "a"),      # consistent: w_a chosen in both orders
    ("B", "A", "b"),      # consistent: w_b chosen in both orders
    ("A", "A", "tie"),    # position bias (always first) -> flip -> tie
    ("B", "B", "tie"),
    ("A", "tie", "tie"),
    ("tie", "tie", "tie"),
    ("a", "a", "a"),      # already relative
])
def test_resolve_orders(ab, ba, expected):
    assert resolve_orders(ab, ba) == expected


def test_side_to_relative():
    assert side_to_relative("left", 3, 7) == "a"
    assert side_to_relative("left", 7, 3) == "b"
    assert side_to_relative("tie", 7, 3) == "tie"


def _locate_fake(prompt_id, seed, w):
    return Path(f"{prompt_id}/s{seed}/w{w:g}.wav")


def _w_of(path: Path) -> float:
    return float(path.stem[1:])


def test_judge_rows_columns_and_flip_rate():
    pairs = build_pairs(_prompts(2), [0], SCALES)
    good = judge_pairs(FakeJudge(_w_of), pairs, _locate_fake)
    assert tuple(good.columns) == RESULT_FILES["e4_judge.parquet"]
    assert len(good) == 4 * 10 * 2 * 3
    assert set(good["choice"]) <= set(CHOICES)
    res = resolve_table(good)
    assert (res["choice"] == "b").all()  # prefers higher w consistently
    assert (flip_rate(good)["flip_rate"] == 0).all()

    biased = judge_pairs(FakeJudge(_w_of, position_bias="A"), pairs, _locate_fake)
    assert (resolve_table(biased)["choice"] == "tie").all()
    assert (flip_rate(biased)["flip_rate"] == 1).all()
    assert (flip_rate(biased, strict=True)["flip_rate"] == 1).all()


def test_audiobox_judge_skips_match_and_uses_scores():
    scores = {}
    judge = AudioboxAestheticsJudge(scorer=lambda p: {"PQ": _w_of(p), "CE": -_w_of(p)})
    pairs = build_pairs(_prompts(1), [0], SCALES)
    rows = judge_pairs(judge, pairs, _locate_fake)
    assert set(rows["question"]) == {"realism", "overall"}
    r = resolve_table(rows)
    assert (r[r.question == "realism"]["choice"] == "b").all()
    assert (r[r.question == "overall"]["choice"] == "a").all()
    assert scores == {}


def test_merge_results_dedupes(tmp_path):
    pairs = build_pairs(_prompts(1), [0], SCALES)
    rows = judge_pairs(FakeJudge(_w_of), pairs, _locate_fake)
    merge_results(tmp_path / "j.parquet", rows)
    both = merge_results(tmp_path / "j.parquet", rows)
    assert len(both) == len(rows)


# ---------------------------------------------------------------- template / parsing
def test_template_questions_match_config():
    t = load_template()
    q = load_config().section("e4_judge")["questions"]
    for name in QUESTIONS:
        assert t[f"question.{name}.pl"] == q[name]
        assert t[f"question.{name}.en"]
    msg = render_prompt("dog barking", "overall", t)
    assert "dog barking" in msg and q["overall"] in msg and "{" not in msg


@pytest.mark.parametrize("text,expected", [
    ("A", "A"), ("B", "B"), ("b", "B"), ("**A**", "A"), ("A.", "A"), ("B\n", "B"),
    ("TIE", "tie"), ("tie", "tie"), ("Remis", "tie"),
    ("Recording A sounds more natural.", "A"),
    ("Answer: B", "B"), ("Odpowiedź: A", "A"), ("The answer is B.", "B"),
    ("A is better because it matches the description.", "A"),
    ("A clear preference: B", "B"), ("The answer is a tie.", "tie"),
    ("A good match is recording B", "B"),
    ("There is no difference between them.", "tie"),
    ("Both recordings sound the same.", "tie"),
    ("Bez różnicy", "tie"),
    ("I prefer the second recording.", "B"),
    ("The first one.", "A"),
    ("", "tie"), ("I cannot answer that.", "tie"),
])
def test_parse_answer(text, expected):
    assert parse_answer(text) == expected


# ---------------------------------------------------------------- listening plan
def test_plan_balance():
    plan = _plan()
    assert tuple(plan.columns) == RESULT_FILES["e4_listening_plan.csv"]
    real = plan[~plan["is_attention_check"]]
    checks = plan[plan["is_attention_check"]]
    assert real["pair_id"].nunique() == 240
    assert (real["pair_id"].value_counts() == 3).all()
    per_listener = plan.groupby("listener_id")
    assert plan["listener_id"].nunique() == 12
    assert (real.groupby("listener_id").size() == 60).all()
    assert (checks.groupby("listener_id").size() == 4).all()
    assert (per_listener["position"].apply(lambda s: sorted(s) == list(range(1, 65)))).all()
    # balanced domains and each scale pair 3x per domain per listener
    assert (real.groupby(["listener_id", "domain"]).size() == 30).all()
    real = real.assign(w_a=real[["w_left", "w_right"]].min(axis=1),
                       w_b=real[["w_left", "w_right"]].max(axis=1))
    assert (real.groupby(["listener_id", "domain", "w_a", "w_b"]).size() == 3).all()
    # no listener sees the same pair twice
    assert not real.duplicated(["listener_id", "pair_id"]).any()
    # one seed per prompt, attention checks identical on both sides
    assert (plan.groupby("prompt_id")["seed"].nunique() == 1).all()
    assert (checks["path_left"] == checks["path_right"]).all()
    assert (checks["w_left"] == checks["w_right"]).all()
    # left/right randomised: w_a appears on the left roughly half the time
    left_low = (real["w_left"] < real["w_right"]).mean()
    assert 0.35 < left_low < 0.65


def test_plan_reproducible_with_seed():
    pd.testing.assert_frame_equal(_plan(1), _plan(1))
    assert not _plan(1).equals(_plan(2))


def test_stratified_prompt_selection_and_plan_pairs():
    rng = np.random.default_rng(0)
    rows = [{"model": "sao", "null": "default", "w": w, "prompt_id": f"{d}_{i:03d}",
             "domain": d, "clap_norm": i / 100 + w}
            for d in ("sfx", "music") for i in range(100) for w in SCALES]
    clap = prompt_clap_at_w(pd.DataFrame(rows), 7)
    assert len(clap) == 200
    chosen = stratified_prompts(clap, 12, rng)
    assert (chosen.groupby("domain").size() == 12).all()
    assert (chosen.groupby(["domain", "tercile"]).size() == 4).all()
    pairs = pairs_from_plan(_plan())
    assert len(pairs) == 240 and (pairs["w_a"] < pairs["w_b"]).all()


def test_pair_id_is_order_free():
    assert pair_id("p", 1, 7, 3) == pair_id("p", 1, 3, 7) == "p__s1__w3-w7"


def test_cli_modules_expose_args():
    for mod in (run, listening_plan):
        p = argparse.ArgumentParser()
        mod.add_args(p)
        assert callable(mod.run)
