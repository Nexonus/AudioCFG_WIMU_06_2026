import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from audiocfg.generate.sweep import SHARED, build_jobs, collect, run_sweep, select_prompts, shard
from audiocfg.repro.config import SweepConfig

pytest.importorskip("soundfile")


def prompts_csv(tmp_path, n=3):
    rows = [
        {"prompt_id": f"{d}_{i}", "domain": d, "rank": i, "text": f"{d} prompt {i}"}
        for d in ("music", "sfx")
        for i in range(n)
    ]
    path = tmp_path / "prompts.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def make_cfg(tmp_path, **kw):
    base = dict(
        name="t",
        prompts=prompts_csv(tmp_path),
        prompts_per_domain=2,
        samples_per_prompt=2,
        guidance=(1, 3, 7),
        nulls={"zero": "zero", "other": "zero"},
        steps=4,
        seconds=0.01,
        batch_size=3,
    )
    return SweepConfig(**(base | kw))


class FakeSAO:
    text_shape = (4, 3)

    def __init__(self):
        self.calls = []

    def sample(self, prompts, seeds, guidance, null_text, num_steps, seconds):
        self.calls.append(list(zip(prompts, seeds, guidance, strict=True)))
        b = len(prompts)
        audio = torch.stack([torch.full((2, 441), s % 7 / 10) for s in seeds])
        return SimpleNamespace(
            audio=audio, guidance_rms=torch.ones(b, num_steps), sample_rate=44100
        )


def test_jobs_share_seed_across_guidance_and_null(tmp_path):
    cfg = make_cfg(tmp_path)
    prompts = select_prompts(pd.read_csv(cfg.prompts), cfg.prompts_per_domain)
    jobs = build_jobs(prompts, cfg)
    assert len(prompts) == 4
    assert len(jobs) == 4 * 2 * (1 + 2 * 2)
    assert {j.null for j in jobs if j.guidance == 1} == {SHARED}
    seeds = {}
    for j in jobs:
        seeds.setdefault((j.prompt_id, j.sample), set()).add(j.seed)
    assert all(len(s) == 1 for s in seeds.values())
    assert len({next(iter(s)) for s in seeds.values()}) == len(seeds)
    assert build_jobs(prompts, cfg) == jobs
    other = build_jobs(prompts, make_cfg(tmp_path, seed_set="B"))
    assert {j.seed for j in other}.isdisjoint({j.seed for j in jobs})


def test_shards_partition_jobs(tmp_path):
    cfg = make_cfg(tmp_path)
    jobs = build_jobs(select_prompts(pd.read_csv(cfg.prompts), 2), cfg)
    parts = [shard(jobs, i, 2) for i in range(2)]
    assert sorted(parts[0] + parts[1], key=jobs.index) == jobs
    assert not set(parts[0]) & set(parts[1])


def test_sweep_resumes_and_collects(tmp_path):
    cfg = make_cfg(tmp_path)
    out = tmp_path / "run"
    out.mkdir()
    for i in range(2):
        run_sweep(FakeSAO(), cfg, out, i, 2)
    total = len(build_jobs(select_prompts(pd.read_csv(cfg.prompts), 2), cfg))
    assert len(list((out / "audio").rglob("*.flac"))) == total

    log = out / "clips_shard0.jsonl"
    lines = log.read_text("utf-8").splitlines()
    log.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    sao = FakeSAO()
    run_sweep(sao, cfg, out, 0, 2)
    assert sum(len(c) for c in sao.calls) == 1
    redone = json.loads(log.read_text("utf-8").splitlines()[-1])
    assert redone["clip"] == json.loads(lines[-1])["clip"]

    clips = collect(out, tuple(cfg.nulls))
    assert SHARED not in set(clips["null"])
    assert len(clips) == 4 * 2 * 3 * 2
    g1 = clips[clips.guidance == 1].groupby(["prompt_id", "sample"])["clip"].nunique()
    assert (g1 == 1).all()


def test_select_prompts_window():
    prompts = pd.DataFrame(
        {"prompt_id": [f"p{i}" for i in range(6)], "domain": "music", "rank": range(6)}
    )
    assert select_prompts(prompts, 2)["prompt_id"].tolist() == ["p0", "p1"]
    assert select_prompts(prompts, 2, first=3)["prompt_id"].tolist() == ["p3", "p4"]
