"""Generation tests: fake backend only, CPU, no downloads."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest
import yaml

from audiocfg.config import DEFAULT_CONFIG, load_config
from audiocfg.generate import prompts as prompts_mod
from audiocfg.generate.backends import FakeBackend, cfg_combine, get_backend
from audiocfg.generate.sweep import limit_prompts, meta_path, run_sweep
from audiocfg.schema import CLIP_COLUMNS, PROMPT_COLUMNS, ClipSpec


@pytest.fixture
def cfg(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw.update(clip_seconds=0.05, outputs_dir=str(tmp_path / "outputs"),
               prompts_csv=str(tmp_path / "prompts.csv"))
    raw["models"]["sao"]["sample_rate"] = 8000
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


@pytest.fixture
def fake_prompts():
    return pd.DataFrame({
        "prompt_id": ["sfx_000", "sfx_001", "music_000", "music_001"],
        "domain": ["sfx", "sfx", "music", "music"],
        "text": ["rain on a roof", "dog barking", "calm piano", "fast drums"],
    })


def _spec(w: float, seed: int = 3, null: str = "default") -> ClipSpec:
    return ClipSpec("sao", "sfx_000", "sfx", "rain", w, null, seed, "dpmpp-3m-sde", 100, 0.05)


def test_cfg_combine():
    u, c = np.array([1.0, -2.0]), np.array([3.0, 5.0])
    assert np.allclose(cfg_combine(u, c, 1.0), c)
    assert np.allclose(cfg_combine(u, c, 0.0), u)
    assert np.allclose(cfg_combine(u, c, 3.0), u + 3 * (c - u))


def test_fake_noise_paired_across_w(cfg):
    be = get_backend("fake", cfg.models["sao"])
    assert isinstance(be, FakeBackend)
    assert np.array_equal(be.initial_noise(3, (2, 100)), be.initial_noise(3, (2, 100)))
    assert not np.array_equal(be.initial_noise(3, (2, 100)), be.initial_noise(4, (2, 100)))
    null = be.null_condition("default")
    outs = {w: be.generate(_spec(w), null)[0] for w in (1.0, 3.0, 7.0)}
    assert all(o.shape == (2, 400) for o in outs.values())
    assert not np.allclose(outs[1.0], outs[7.0])  # w matters
    assert np.array_equal(be.generate(_spec(7.0), null)[0], outs[7.0])  # deterministic


def test_fake_null_variants(cfg, tmp_path):
    be = get_backend("fake", cfg.models["sao"])
    np.save(tmp_path / "n.npy", np.array([220.0]))
    a = be.generate(_spec(5.0), be.null_condition("default"))[0]
    b = be.generate(_spec(5.0), be.null_condition("learned", path=tmp_path / "n.npy"))[0]
    assert not np.allclose(a, b)
    with pytest.raises(ValueError):
        be.null_condition("negative_prompt")


def test_sweep_resumable_and_meta(cfg, fake_prompts):
    kw = dict(prompts=fake_prompts, scales=[1, 7], seeds=[0, 1], backend_name="fake")
    first = run_sweep(cfg, "sao", **kw)
    assert first == {"generated": 16, "skipped": 0}
    second = run_sweep(cfg, "sao", **kw)
    assert second == {"generated": 0, "skipped": 16}

    meta = pd.read_parquet(meta_path(cfg.outputs_dir, "sao_fake"))
    assert tuple(meta.columns) == CLIP_COLUMNS
    assert len(meta) == meta["clip_id"].nunique() == 16
    assert (meta["gen_time_s"] >= 0).all()
    assert set(meta["w"]) == {1.0, 7.0} and set(meta["sampler"]) == {"dpmpp-3m-sde"}


def test_limit_prompts_balanced(fake_prompts):
    sub = limit_prompts(fake_prompts, 2)
    assert sorted(sub["domain"]) == ["music", "sfx"]


def _fake_sources(n: int):
    audiocaps = pd.DataFrame({
        "audiocap_id": range(3 * n),
        "youtube_id": [f"yt{i // 3}" for i in range(3 * n)],  # 3 captions per clip
        "caption": [f"sound {i}" for i in range(3 * n)],
    })
    sd = pd.DataFrame({
        "caption_id": range(2 * n),
        "track_id": [i // 2 for i in range(2 * n)],
        "caption": [f"music {i}" for i in range(2 * n)],
        "path": [f"{i // 2}.mp3" for i in range(2 * n)],
        "no_vocals": [(i // 2) % 2 == 0 for i in range(2 * n)],
    })
    return audiocaps, sd


def test_prompt_sampling_reproducible():
    audiocaps, sd = _fake_sources(40)
    sd = prompts_mod.nonvocal_filter(sd, "no_vocals", None)
    a = prompts_mod.sample_prompts(audiocaps, sd, n_sfx=10, n_music=10, seed=2026)
    b = prompts_mod.sample_prompts(audiocaps.sample(frac=1, random_state=1), sd, 10, 10, seed=2026)
    assert tuple(a.columns) == PROMPT_COLUMNS
    pd.testing.assert_frame_equal(a, b)  # independent of seed reuse and input row order
    assert (a.domain == "sfx").sum() == 10 and (a.domain == "music").sum() == 10
    assert a["prompt_id"].is_unique and a.groupby("domain")["source_id"].nunique().eq(10).all()
    assert a.loc[a.domain == "music", "source_id"].astype(int).mod(2).eq(0).all()  # non-vocal only
    c = prompts_mod.sample_prompts(audiocaps, sd, 10, 10, seed=7)
    assert not a["text"].equals(c["text"])


def test_prompts_cli(tmp_path):
    audiocaps, sd = _fake_sources(30)
    audiocaps.to_csv(tmp_path / "ac.csv", index=False)
    sd.to_csv(tmp_path / "sd.csv", index=False)
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    prompts_mod.add_args(p)
    out = tmp_path / "prompts.csv"
    args = p.parse_args(["--audiocaps", str(tmp_path / "ac.csv"), "--song-describer", str(tmp_path / "sd.csv"),
                         "--sd-nonvocal-column", "no_vocals", "--n-sfx", "5", "--n-music", "5",
                         "--out", str(out)])
    prompts_mod.run(args)
    df = pd.read_csv(out)
    assert len(df) == 10 and tuple(df.columns) == PROMPT_COLUMNS
