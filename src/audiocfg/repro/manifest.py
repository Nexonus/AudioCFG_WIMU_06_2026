from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from audiocfg.repro.config import StageConfig

TRACKED_PACKAGES = (
    "audiocfg",
    "numpy",
    "scipy",
    "pandas",
    "torch",
    "torchaudio",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "audiobox_aesthetics",
    "librosa",
    "soundfile",
)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str, cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state(cwd: Path | None = None) -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD", cwd=cwd)
    status = _git("status", "--porcelain", "--untracked-files=no", cwd=cwd)
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd),
        "dirty": None if status is None else bool(status),
    }


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def device_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch": None}
    info: dict[str, Any] = {"cuda": torch.cuda.is_available(), "cuda_version": torch.version.cuda}
    if torch.cuda.is_available():
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def build_manifest(
    config: StageConfig,
    inputs: dict[str, str | Path] | None = None,
    outputs: dict[str, str | Path] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": sys.argv,
        "git": git_state(),
        "config": config.model_dump(mode="json"),
        "config_class": type(config).__name__,
        "config_hash": config.config_hash(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "device": device_info(),
        "inputs": {
            k: {"path": str(p), "sha256": sha256_file(p)} for k, p in (inputs or {}).items()
        },
        "outputs": {
            k: {"path": str(p), "sha256": sha256_file(p)} for k, p in (outputs or {}).items()
        },
        "extra": extra or {},
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path
