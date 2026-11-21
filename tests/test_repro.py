import json
import random
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from audiocfg.repro.config import StageConfig
from audiocfg.repro.manifest import build_manifest, sha256_file, write_manifest
from audiocfg.repro.seeds import derive_seed, derive_seeds, set_global_seed

# --- seeds ---------------------------------------------------------------------------------


def test_derive_seed_is_deterministic():
    assert derive_seed(2026, "eval_A", "sd_017", 2) == derive_seed(2026, "eval_A", "sd_017", 2)


def test_derive_seed_depends_on_every_input():
    base = derive_seed(2026, "eval_A", "sd_017", 2)
    assert base != derive_seed(2027, "eval_A", "sd_017", 2)
    assert base != derive_seed(2026, "eval_B", "sd_017", 2)
    assert base != derive_seed(2026, "eval_A", "sd_018", 2)
    assert base != derive_seed(2026, "eval_A", "sd_017", 3)


def test_derive_seeds_independent_of_order_and_count():
    assert derive_seeds(2026, "train", 5) == derive_seeds(2026, "train", 10)[:5]
    seeds = derive_seeds(2026, "train", 1000)
    assert len(set(seeds)) == len(seeds)
    assert all(0 <= s < 2**32 for s in seeds)


def test_set_global_seed_repeats_streams():
    set_global_seed(7)
    a = (random.random(), np.random.rand())  # noqa: NPY002
    set_global_seed(7)
    assert (random.random(), np.random.rand()) == a  # noqa: NPY002


# --- config --------------------------------------------------------------------------------


class _Cfg(StageConfig):
    lr: float = 5e-3
    scales: tuple[float, ...] = (1.0, 7.0)


def _yaml(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_config_hash_ignores_key_order_and_comments(tmp_path):
    a = _Cfg.from_yaml(_yaml(tmp_path, "name: x\nlr: 0.005\nscales: [1, 7]\n"))
    b = _Cfg.from_yaml(_yaml(tmp_path, "# comment\nscales: [1, 7]\nlr: 5.0e-3\nname: x\n"))
    assert a.config_hash() == b.config_hash()
    assert a.run_dir() == b.run_dir()


def test_config_hash_changes_with_values(tmp_path):
    a = _Cfg.from_yaml(_yaml(tmp_path, "name: x\nlr: 0.005\n"))
    b = _Cfg.from_yaml(_yaml(tmp_path, "name: x\nlr: 0.004\n"))
    assert a.config_hash() != b.config_hash()


def test_config_rejects_unknown_keys(tmp_path):
    with pytest.raises(ValidationError):
        _Cfg.from_yaml(_yaml(tmp_path, "name: x\nlearning_rate: 0.1\n"))


# --- manifest ------------------------------------------------------------------------------


def test_manifest_records_inputs_and_config(tmp_path):
    data = tmp_path / "prompts.csv"
    data.write_text("id,text\n1,rain\n", encoding="utf-8")
    cfg = _Cfg(name="probe")
    manifest = build_manifest(cfg, inputs={"prompts": data})
    path = write_manifest(tmp_path / "run" / "manifest.json", manifest)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["config_hash"] == cfg.config_hash()
    assert loaded["inputs"]["prompts"]["sha256"] == sha256_file(data)
    assert {"commit", "dirty", "branch"} <= set(loaded["git"])
    assert loaded["packages"]["numpy"] == np.__version__
