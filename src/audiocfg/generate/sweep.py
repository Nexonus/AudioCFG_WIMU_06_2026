from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

from audiocfg.nulls.variants import resolve_null
from audiocfg.repro.config import SweepConfig
from audiocfg.repro.seeds import derive_seed

SHARED = "shared"


@dataclass(frozen=True)
class Job:
    prompt_id: str
    domain: str
    text: str
    sample: int
    seed: int
    guidance: float
    null: str

    @property
    def clip(self) -> str:
        return f"{self.null}/{self.prompt_id}_s{self.sample}_g{self.guidance:g}.flac"


def select_prompts(prompts: pd.DataFrame, per_domain: int) -> pd.DataFrame:
    chosen = prompts[prompts["rank"] < per_domain]
    return chosen.sort_values(["domain", "rank"]).reset_index(drop=True)


def build_jobs(prompts: pd.DataFrame, cfg: SweepConfig) -> list[Job]:
    jobs = []
    stream = f"eval_{cfg.seed_set}"
    for k in range(cfg.samples_per_prompt):
        for row in prompts.itertuples():
            seed = derive_seed(cfg.master_seed, stream, row.prompt_id, k)
            for g in cfg.guidance:
                nulls = [SHARED] if g == 1 else list(cfg.nulls)
                for null in nulls:
                    jobs.append(Job(row.prompt_id, row.domain, row.text, k, seed, float(g), null))
    return jobs


def shard(jobs: list[Job], index: int, count: int) -> list[Job]:
    return jobs[index::count]


def expand_shared(rows: pd.DataFrame, nulls: tuple[str, ...]) -> pd.DataFrame:
    shared = rows[rows["null"] == SHARED]
    copies = [shared.assign(null=n) for n in nulls]
    return pd.concat([rows[rows["null"] != SHARED], *copies], ignore_index=True)


def run_sweep(sao, cfg: SweepConfig, out: Path, shard_index: int = 0, shard_count: int = 1):
    import soundfile as sf

    prompts = select_prompts(pd.read_csv(cfg.prompts), cfg.prompts_per_domain)
    jobs = shard(build_jobs(prompts, cfg), shard_index, shard_count)
    nulls = {name: resolve_null(spec, sao) for name, spec in cfg.nulls.items()}
    zero = torch.zeros(sao.text_shape)
    nulls = {k: zero if v is None else v.float() for k, v in nulls.items()}
    nulls[SHARED] = zero

    log_path = out / f"clips_shard{shard_index}.jsonl"
    done = set()
    if log_path.exists():
        done = {json.loads(line)["clip"] for line in log_path.read_text("utf-8").splitlines()}
    todo = [j for j in jobs if j.clip not in done]
    print(f"shard {shard_index}/{shard_count}: {len(jobs)} jobs, {len(todo)} to do", flush=True)

    t0 = time.perf_counter()
    with open(log_path, "a", encoding="utf-8") as log:
        for start in range(0, len(todo), cfg.batch_size):
            batch = todo[start : start + cfg.batch_size]
            tb = time.perf_counter()
            result = sao.sample(
                [j.text for j in batch],
                [j.seed for j in batch],
                [j.guidance for j in batch],
                torch.stack([nulls[j.null] for j in batch]),
                num_steps=cfg.steps,
                seconds=cfg.seconds,
            )
            per_clip = (time.perf_counter() - tb) / len(batch)
            audio = result.audio.float().cpu()
            for i, job in enumerate(batch):
                wave = audio[i]
                path = out / "audio" / job.clip
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, wave.clamp(-1, 1).numpy().T, result.sample_rate)
                rms = None if result.guidance_rms is None else result.guidance_rms[i].tolist()
                row = asdict(job) | {
                    "clip": job.clip,
                    "peak": float(wave.abs().max()),
                    "clipped_frac": float((wave.abs() > 1).float().mean()),
                    "guidance_rms": rms,
                    "seconds_per_clip": round(per_clip, 2),
                }
                log.write(json.dumps(row) + "\n")
            log.flush()
            n = start + len(batch)
            eta = (time.perf_counter() - t0) / n * (len(todo) - n) / 60
            print(f"shard {shard_index}: {n}/{len(todo)}  eta {eta:.0f} min", flush=True)


def collect(out: Path, nulls: tuple[str, ...]) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for path in sorted(out.glob("clips_shard*.jsonl"))
        for line in path.read_text("utf-8").splitlines()
    ]
    clips = pd.DataFrame(rows)
    clips["guidance_rms_mean"] = [
        None if r is None else sum(r) / len(r) for r in clips["guidance_rms"]
    ]
    clips = expand_shared(clips, nulls)
    return clips.sort_values(["null", "domain", "prompt_id", "sample", "guidance"]).reset_index(
        drop=True
    )
