"""Synthetic CPU-only tests for audiocfg.metrics (§5.1, §5.2, §6)."""

import argparse

import numpy as np
import pandas as pd
import pytest

from audiocfg.metrics import copying
from audiocfg.metrics.clap_score import Calibration, calibrate, clap_norm, clap_scores
from audiocfg.metrics.diversity import ccds, ccds_table
from audiocfg.metrics.embedders import RandomProjectionEmbedder, get_embedder
from audiocfg.metrics.fad import (check_equal_n, equal_n_indices, fd, fd_with_ci,
                                  split_half_floor)
from audiocfg.metrics.kl import kl_divergence, kl_per_prompt
from audiocfg.metrics.run import calibration_table, compute_e1
from audiocfg.schema import CLIP_COLUMNS, RESULT_FILES

SR = 16000
RNG = np.random.default_rng(0)


def tone(f, sec=1.0, sr=SR):
    return (0.3 * np.sin(2 * np.pi * f * np.arange(int(sec * sr)) / sr)).astype(np.float32)


# ---------------------------------------------------------------- diversity

def test_ccds_identical_is_zero_and_random_is_larger():
    v = RNG.standard_normal(32)
    assert ccds(np.tile(v, (10, 1))) == pytest.approx(0.0, abs=1e-9)
    assert ccds(RNG.standard_normal((10, 32))) > 0.5


def test_ccds_on_audio_via_random_embedder():
    emb = RandomProjectionEmbedder()
    same = emb.embed([tone(440)] * 5, SR)
    diff = emb.embed([tone(f) for f in (200, 450, 900, 1800, 3500)], SR)
    assert ccds(same) == pytest.approx(0.0, abs=1e-9)
    assert ccds(diff) > ccds(same) + 0.05


def test_ccds_table_groups():
    meta = pd.DataFrame({"model": "sao", "domain": "sfx", "null": "default",
                         "w": [1.0] * 4 + [7.0] * 4, "prompt_id": "p1"})
    emb = np.vstack([RNG.standard_normal((4, 8)), np.ones((4, 8))])
    t = ccds_table(meta, emb)
    assert len(t) == 2 and t.set_index("w").loc[7.0, "ccds"] == pytest.approx(0, abs=1e-9)


# ---------------------------------------------------------------- CLAP_norm

def test_clap_norm_maps_floor_and_ceiling():
    cal = Calibration(c_real=0.4, c_mis=0.1, n=10)
    assert clap_norm(0.1, cal) == pytest.approx(0.0)
    assert clap_norm(0.4, cal) == pytest.approx(1.0)


def test_calibrate_never_uses_own_caption():
    n, d = 50, 16
    text = RNG.standard_normal((n, d))
    audio = text + 0.1 * RNG.standard_normal((n, d))  # matched pairs are close
    cal = calibrate(audio, text, text, np.random.default_rng(1))
    assert cal.n == n and cal.c_real > 0.9 and abs(cal.c_mis) < 0.3
    s = clap_scores(text, audio)
    assert np.mean(clap_norm(s, cal)) == pytest.approx(1.0)


# ---------------------------------------------------------------- FD

def test_fd_zero_for_same_set_and_grows_with_shift():
    x = RNG.standard_normal((500, 8))
    assert fd(x, x) == pytest.approx(0.0, abs=1e-6)
    d1, d2 = fd(x + 0.5, x), fd(x + 2.0, x)
    assert d1 == pytest.approx(8 * 0.25, rel=1e-3) and d2 > d1


def test_fd_with_ci_brackets_point_and_equal_n():
    ref = RNG.standard_normal((400, 4))
    gen = RNG.standard_normal((200, 4)) + 1.0
    pids = np.repeat(np.arange(20), 10)
    v, lo, hi = fd_with_ci(gen, ref, pids, n_boot=200)
    assert lo <= v <= hi or abs(v - lo) < 0.5
    groups = equal_n_indices({"a": np.arange(10), "b": np.arange(10, 25)}, seed=0)
    check_equal_n({k: len(v) for k, v in groups.items()})
    with pytest.raises(ValueError):
        check_equal_n({"a": 10, "b": 12})
    assert split_half_floor(ref) < fd(gen, ref)


# ---------------------------------------------------------------- KL

def test_kl():
    p = np.array([0.7, 0.2, 0.1])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)
    assert kl_per_prompt(np.tile([0.1, 0.2, 0.7], (3, 1)), p) > 0.5


# ---------------------------------------------------------------- copying

def test_signal_match_copy_vs_same_kind():
    sr, n = 16000, 2 * 16000
    rng = np.random.default_rng(3)
    a = rng.standard_normal(n).astype(np.float32)
    shifted = np.concatenate([rng.standard_normal(4000), a[:-4000]]).astype(np.float32)
    other = rng.standard_normal(n).astype(np.float32)  # same spectrum class (white noise)
    s_copy = copying.signal_match(a, shifted, sr)
    s_other = copying.signal_match(a, other, sr)
    assert s_copy > 0.8 and s_other < 0.3 and s_copy > s_other + 0.5
    # chroma variant runs and also separates
    assert copying.signal_match(a, shifted, sr, "chroma") > copying.signal_match(a, other, sr, "chroma")


def test_nearest_neighbors_and_candidate_threshold():
    train = RNG.standard_normal((200, 16))
    held = RNG.standard_normal((300, 16))
    idx, sim = copying.nearest_neighbors(train[[5, 17]] * 3.0, train)
    assert list(idx) == [5, 17] and np.allclose(sim, 1.0, atol=1e-5)
    _, s_held = copying.nearest_neighbors(held, train)
    thr = copying.candidate_threshold(s_held, 99.0)
    assert np.mean(s_held > thr) <= 0.011
    assert sim.min() > thr  # exact copies are candidates


# ---------------------------------------------------------------- end-to-end shapes

def _synthetic_meta(tmp_path=None):
    rows, audios = [], []
    for domain in ("sfx", "music"):
        for p in range(3):
            for w in (1.0, 7.0):
                for s in range(4):
                    f = 200 + 150 * p + (40 * s if w == 1.0 else 5 * s)
                    rows.append(dict(clip_id=f"sao__default__w{w:g}__{domain}_{p}__s{s}", model="sao",
                                     prompt_id=f"{domain}_{p}", domain=domain, text=f"prompt {p}",
                                     w=w, null="default", seed=s, sampler="x", steps=1,
                                     sample_rate=SR, seconds=0.5, path="", gen_time_s=0.0))
                    audios.append(tone(f, 0.5))
    return pd.DataFrame(rows, columns=list(CLIP_COLUMNS)), audios


def test_e1_outputs_match_result_files():
    meta, audios = _synthetic_meta()
    get = lambda name: get_embedder(name, "random")  # noqa: E731
    prompts = meta.drop_duplicates("prompt_id")[["prompt_id", "domain", "text"]]
    refs = {p: tone(200 + 150 * int(p[-1]), 0.5) for p in prompts["prompt_id"]}
    cal = calibration_table(prompts, [refs[p] for p in prompts["prompt_id"]], SR,
                            [get("laion_clap"), get("ms_clap")])
    assert tuple(cal.columns) == RESULT_FILES["clap_calibration.csv"] and len(cal) == 4
    ref_sets = {d: [tone(f, 0.5) for f in range(180, 900, 60)] for d in ("sfx", "music")}
    per_prompt, fd_df = compute_e1(meta, audios, SR, get, cal, refs, ref_sets, n_boot=20)
    assert tuple(per_prompt.columns) == RESULT_FILES["e1_per_prompt.parquet"]
    assert tuple(fd_df.columns) == RESULT_FILES["e1_fd.parquet"]
    assert len(per_prompt) == 12 and len(fd_df) == 4
    assert per_prompt[["ccds_clap", "ccds_aux", "clap_norm", "kl_passt"]].notna().all().all()
    assert fd_df["n"].nunique() == 1
    # w=7 clips are closer together than w=1 clips by construction
    m = per_prompt.groupby("w")["ccds_clap"].mean()
    assert m[7.0] < m[1.0]


def test_copies_cli_writes_result_files(tmp_path, monkeypatch):
    import yaml

    from audiocfg.audio_io import save_audio
    from audiocfg.config import DEFAULT_CONFIG

    rng = np.random.default_rng(7)
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw.update(outputs_dir=str(tmp_path / "outputs"), results_dir=str(tmp_path / "results"))
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    def clip(name, x):
        p = tmp_path / "wav" / f"{name}.wav"
        save_audio(p, x[None], SR)
        return str(p)

    train = [rng.standard_normal(SR).astype(np.float32) * 0.1 for _ in range(6)]
    t_rows = [dict(id=f"t{i}", domain="sfx", path=clip(f"t{i}", x)) for i, x in enumerate(train)]
    h_rows = [dict(id=f"h{i}", domain="sfx", path=clip(f"h{i}", rng.standard_normal(SR).astype(np.float32) * 0.1))
              for i in range(5)]
    pd.DataFrame(t_rows).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(h_rows).to_csv(tmp_path / "held.csv", index=False)
    gen = [train[0], rng.standard_normal(SR).astype(np.float32) * 0.1]
    meta = pd.DataFrame([dict(clip_id=f"c{i}", model="sao", prompt_id="sfx_0", domain="sfx", text="x",
                              w=7.0, null="default", seed=i, sampler="x", steps=1, sample_rate=SR,
                              seconds=1.0, path=clip(f"g{i}", x), gen_time_s=0.0)
                         for i, x in enumerate(gen)], columns=list(CLIP_COLUMNS))
    (tmp_path / "outputs" / "sao").mkdir(parents=True)
    meta.to_parquet(tmp_path / "outputs" / "sao" / "meta.parquet")

    p = argparse.ArgumentParser()
    p.add_argument("--config")
    copying.add_args(p)
    args = p.parse_args(["--config", str(cfg_path), "--train-manifest", str(tmp_path / "train.csv"),
                         "--held-manifest", str(tmp_path / "held.csv"), "--embedder-override", "random",
                         "--sr", str(SR)])
    copying.run(args)
    c = pd.read_parquet(tmp_path / "results" / "e2_copies.parquet")
    h = pd.read_parquet(tmp_path / "results" / "e2_held.parquet")
    assert tuple(c.columns) == RESULT_FILES["e2_copies.parquet"]
    assert tuple(h.columns) == RESULT_FILES["e2_held.parquet"]
    exact = c[(c["clip_id"] == "c0")]
    assert exact["is_candidate"].all() and exact["is_copy"].all()
    assert (exact["nn_train_id"] == "t0").all()
