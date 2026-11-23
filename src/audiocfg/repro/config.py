from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import ClassVar, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound="StageConfig")


class StageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    master_seed: int = 2026
    output_dir: Path = Path("outputs")
    unhashed_defaults: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def from_yaml(cls: type[T], path: str | Path) -> T:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def canonical_json(self) -> str:
        data = self.model_dump(mode="json")
        for key, value in self:
            if isinstance(value, Path):
                data[key] = value.as_posix()
        for key in self.unhashed_defaults:
            if data[key] == type(self).model_fields[key].default:
                del data[key]
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

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


class SweepConfig(StageConfig):
    prompts: Path = Path("data/eval_prompts.csv")
    prompts_per_domain: int = 24
    first_prompt: int = 0
    samples_per_prompt: int = 3
    seed_set: str = "A"
    guidance: tuple[float, ...] = (1, 2, 3, 4.5, 6, 7, 9, 12)
    nulls: dict[str, str] = {"zero": "zero"}
    steps: int = 50
    seconds: float = 10.0
    batch_size: int = 4
    # added after pilot A: left out of the hash at its default so pilot A keeps its hash
    unhashed_defaults: ClassVar[tuple[str, ...]] = ("first_prompt",)


class ScoreConfig(StageConfig):
    clap_models: dict[str, str] = {
        "music": "laion/larger_clap_music_and_speech@195c3a3e68faebb3e2088b9a79e79b43ddbda76b",
        "general": "laion/larger_clap_general@ada0c23a36c4e8582805bb38fec3905903f18b41",
    }
    primary_clap: dict[str, str] = {"music": "music", "sfx": "general"}
    aesthetics: str = "facebook/audiobox-aesthetics@9b1dd8e5df9af7216e836a98974fe3b82c56ded6"
    reference_tracks: int = 300
    reference_offset: float = 30.0
    seconds: float = 10.0
    batch_size: int = 16


class CriterionConfig(StageConfig):
    guidance: tuple[float, ...] = (6.0, 7.0, 9.0)
    ccds_margin: dict[str, float]
    n_boot: int = 2000
    level: float = 0.95
