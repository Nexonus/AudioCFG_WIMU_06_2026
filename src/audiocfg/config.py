"""Load the fixed experiment settings from ``configs/experiment.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment.yaml"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hf_id: str
    sample_rate: int
    channels: int
    sampler: str
    steps: int
    default_w: float
    experiments: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...]
    guidance_scales: tuple[float, ...]
    clip_seconds: float
    loudness_lufs: float
    prompts_csv: Path
    outputs_dir: Path
    results_dir: Path
    figures_dir: Path
    models: dict[str, ModelConfig]
    raw: dict[str, Any] = field(repr=False)

    def section(self, name: str) -> dict[str, Any]:
        """Experiment-specific block, e.g. ``cfg.section("e2_copying")``."""
        return self.raw[name]


def load_config(path: str | Path = DEFAULT_CONFIG) -> ExperimentConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else REPO_ROOT / q

    models = {
        name: ModelConfig(
            name=name,
            hf_id=m["hf_id"],
            sample_rate=int(m["sample_rate"]),
            channels=int(m["channels"]),
            sampler=m["sampler"],
            steps=int(m["steps"]),
            default_w=float(m["default_w"]),
            experiments=tuple(m["experiments"]),
        )
        for name, m in raw["models"].items()
    }
    return ExperimentConfig(
        seeds=tuple(int(s) for s in raw["seeds"]),
        guidance_scales=tuple(float(w) for w in raw["guidance_scales"]),
        clip_seconds=float(raw["clip_seconds"]),
        loudness_lufs=float(raw["loudness_lufs"]),
        prompts_csv=resolve(raw["prompts_csv"]),
        outputs_dir=resolve(raw["outputs_dir"]),
        results_dir=resolve(raw["results_dir"]),
        figures_dir=resolve(raw["figures_dir"]),
        models=models,
        raw=raw,
    )
