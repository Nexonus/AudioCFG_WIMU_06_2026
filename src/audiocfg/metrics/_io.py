"""Clip loading (loudness-normalised, §4.3) and result writing shared by ``metrics`` and ``copies``."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from audiocfg.audio_io import load_audio, normalize_loudness, resample
from audiocfg.config import REPO_ROOT
from audiocfg.schema import RESULT_FILES


def resolve(path: str | Path, base: Path = REPO_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


def load_clips(paths: list[str | Path], target_lufs: float, sr: int | None = None,
               base: Path = REPO_ROOT) -> tuple[list[np.ndarray], int]:
    """Load mono clips, normalise to ``target_lufs``, resample to a common ``sr``.

    ``sr`` defaults to the first file's rate. Returns ``(audios, sr)`` with 1-D arrays.
    """
    out: list[np.ndarray] = []
    for p in paths:
        a, r = load_audio(resolve(p, base), mono=True)
        a = normalize_loudness(a, r, target_lufs)
        sr = sr or r
        out.append(resample(a, r, sr)[0])
    return out, int(sr or 0)


def existing(paths: list[str], base: Path = REPO_ROOT) -> np.ndarray:
    """Boolean mask: path is a non-empty string pointing at an existing file."""
    return np.array([isinstance(p, str) and bool(p) and resolve(p, base).is_file() for p in paths],
                    dtype=bool)


def write_result(df: pd.DataFrame, results_dir: Path, name: str,
                 replace_on: dict[str, object] | None = None) -> Path:
    """Write ``df`` with exactly ``RESULT_FILES[name]`` columns.

    If the file exists, rows matching ``replace_on`` (e.g. ``{"model": "sao"}``) are dropped
    and the new rows appended, so per-model runs do not overwrite each other.
    """
    cols = list(RESULT_FILES[name])
    df = df[cols]
    path = Path(results_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    reader = pd.read_csv if name.endswith(".csv") else pd.read_parquet
    if path.exists() and replace_on:
        old = reader(path)
        keep = np.ones(len(old), dtype=bool)
        for k, v in replace_on.items():
            keep &= old[k].isin(v if isinstance(v, (list, tuple, set)) else [v]).to_numpy()
        df = pd.concat([old[~keep][cols], df], ignore_index=True)
    if name.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    return path


def warn(msg: str) -> None:
    warnings.warn(msg, stacklevel=2)
    print(f"[metrics] WARNING: {msg}")
