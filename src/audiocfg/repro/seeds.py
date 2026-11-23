"""Deterministic seed derivation.

Every random number in the project comes from one master seed. Seeds are derived with
``numpy.random.SeedSequence`` from (master seed, stream name, keys), so a clip's seed
depends only on *what* it is (e.g. prompt id and sample index), never on the order in
which clips are generated. This keeps the same initial noise x_T for every guidance scale
and null variant, and makes interrupted runs resumable without changing any seed.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np

_UINT32 = 2**32


def _words(text: str) -> list[int]:
    """Stable 32-bit words from a string (Python's hash() is salted per process)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4)]


def derive_seed(master_seed: int, stream: str, *keys: int | str) -> int:
    """One 32-bit seed for (stream, keys), e.g. ``derive_seed(2026, "eval_A", "sd_017", 2)``."""
    entropy = [int(master_seed) % _UINT32, *_words(stream)]
    for key in keys:
        entropy.extend(_words(str(key)))
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def derive_seeds(master_seed: int, stream: str, n: int) -> list[int]:
    """``n`` seeds of one stream: ``derive_seed(master, stream, i)`` for i in 0..n-1."""
    return [derive_seed(master_seed, stream, i) for i in range(n)]


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) torch, e.g. for training runs."""
    random.seed(seed)
    np.random.seed(seed % _UINT32)  # noqa: NPY002 (global state used by libraries)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_generator(seed: int, device: str = "cpu"):
    """A ``torch.Generator`` seeded with ``seed`` (initial noise is drawn on CPU by default
    so that it is identical on any GPU)."""
    import torch

    return torch.Generator(device=device).manual_seed(int(seed))
