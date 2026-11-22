from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound="StageConfig")


class StageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    master_seed: int = 2026
    output_dir: Path = Path("outputs")

    @classmethod
    def from_yaml(cls: type[T], path: str | Path) -> T:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def run_dir(self) -> Path:
        return Path(self.output_dir) / f"{self.name}-{self.config_hash()[:12]}"


class DataConfig(StageConfig):
    sao_fma_tracks: Path
    song_describer_csv: Path
    clotho_captions_csv: Path
    prompts_per_domain: int = 48
    null_train_tracks: int = 2000
    null_val_tracks: int = 300
    sanity_tracks: int = 64


class NullTrainConfig(StageConfig):
    null_tracks: Path = Path("data/null_tracks.csv")
    sanity_tracks: Path = Path("data/sanity_tracks.csv")
    max_seconds: float = 30.0
    inits: tuple[str, ...] = ("zero",)
    steps: int = 1500
    batch_size: int = 8
    lr: float = 5e-3
    warmup: int = 100
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_items: int = 64
    val_draws: int = 4
    sanity_draws: int = 2
    eval_batch: int = 8
