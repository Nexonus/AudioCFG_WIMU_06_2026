"""E3 sanity checks before the full learned-null sweep (PROJECT_DESIGN.md §5.3).

1. Validation loss with the learned null must be lower than with the zero (default) null.
2. Unconditional generations (null branch only) from both nulls: FD_openl3 vs the E3 val set;
   the learned null should give more realistic unconditional samples.
Check 1 is pure python on ``e3_null_training.csv``; check 2 calls generate/ and metrics/
lazily by name (owned by other agents).
"""

from __future__ import annotations

import csv
import importlib
from dataclasses import dataclass
from pathlib import Path

from .train import LogRow, select_best


@dataclass(frozen=True)
class ValLossCheck:
    init: str
    best_step: int
    val_loss: float
    val_loss_zero_null: float

    @property
    def passed(self) -> bool:
        return self.val_loss < self.val_loss_zero_null

    @property
    def relative_gain(self) -> float:
        return 1.0 - self.val_loss / self.val_loss_zero_null


def read_log(path: str | Path) -> list[LogRow]:
    with Path(path).open(encoding="utf-8") as f:
        return [LogRow(r["init"], int(r["step"]), float(r["train_loss"]), float(r["val_loss"]),
                       float(r["val_loss_zero_null"])) for r in csv.DictReader(f)]


def check_val_loss(rows: list[LogRow]) -> dict[str, ValLossCheck]:
    """Per init: selected checkpoint's val loss vs the zero-null val loss (same val noise)."""
    out = {}
    for init in sorted({r.init for r in rows}):
        best = select_best([r for r in rows if r.init == init])
        out[init] = ValLossCheck(init, best.step, best.val_loss, best.val_loss_zero_null)
    return out


def unconditional_fd_check(null_paths: dict[str, Path], val_audio_dir: Path, n_samples: int = 200,
                           seed: int = 0) -> dict[str, float]:
    """FD_openl3 of unconditional samples per null (``{"default": ..., "learned_zeros": ...}``).

    TODO(M1): agree the function names with generate/ and metrics/ owners. Assumed:
    ``audiocfg.generate.sao.generate_unconditional(null=<tensor|None>, n, seed, out_dir)``
    returning wav paths, and ``audiocfg.metrics.fd.fd_from_dirs(gen_paths, ref_dir,
    embedder="openl3")``. A null of ``None`` means SAO's default zero null.
    """
    gen = importlib.import_module("audiocfg.generate.sao")
    fd = importlib.import_module("audiocfg.metrics.fd")
    from .train import load_null

    results = {}
    for name, path in {"default": None, **null_paths}.items():
        null = None if path is None else load_null(path)
        out_dir = Path("outputs/null/unconditional") / name
        clips = gen.generate_unconditional(null=null, n=n_samples, seed=seed, out_dir=out_dir)
        results[name] = float(fd.fd_from_dirs(clips, val_audio_dir, embedder="openl3"))
    return results
