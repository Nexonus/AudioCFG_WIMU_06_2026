"""E3 training data: clips from the SAO attribution lists, disjoint from E2's set T (§5.3).

Pure-python selection and splitting (tested); latent pre-encoding with the SAO VAE is lazy.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

ID_COLUMN_CANDIDATES = ("id", "clip_id", "freesound_id", "track_id", "fma_id")


def read_id_list(path: str | Path) -> list[str]:
    """Read ids from a CSV (column ``id``-like, else first column) or a plain one-id-per-line file.

    TODO(M1): confirm the exact column names of the published SAO attribution CSVs
    (Freesound ids and FMA track ids) and whether ids need a source prefix (``fs_``/``fma_``).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8").splitlines()
    lines = [ln for ln in text if ln.strip()]
    if not lines:
        return []
    if path.suffix.lower() != ".csv":
        return [ln.strip() for ln in lines]
    reader = csv.DictReader(lines)
    fields = reader.fieldnames or []
    col = next((c for c in ID_COLUMN_CANDIDATES if c in fields), fields[0])
    return [str(row[col]).strip() for row in reader if str(row[col]).strip()]


def select_clips(candidate_ids: Iterable[str], exclude_ids: Iterable[str], n: int,
                 seed: int) -> list[str]:
    """Pick ``n`` ids from the attribution list, never one of E2's training-neighbour set T.

    Deterministic for a given (set of candidates, exclusion set, n, seed), independent of the
    input order. Returns fewer than ``n`` ids if not enough candidates remain.
    """
    excluded = {str(i) for i in exclude_ids}
    pool = sorted({str(i) for i in candidate_ids} - excluded)
    if n >= len(pool):
        return pool
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=n, replace=False)
    return sorted(pool[i] for i in idx)


def _hash_unit(clip_id: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{clip_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def train_val_split(ids: Sequence[str], val_fraction: float,
                    seed: int) -> tuple[list[str], list[str]]:
    """Deterministic train/val split (exactly ``round(val_fraction * N)`` val ids).

    Ids are ranked by a seeded hash, so the split does not depend on list order and adding
    unrelated ids to the pool does not reshuffle which side an id falls on more than necessary.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    uniq = sorted(set(map(str, ids)))
    n_val = int(round(val_fraction * len(uniq)))
    ranked = sorted(uniq, key=lambda i: (_hash_unit(i, seed), i))
    val = sorted(ranked[:n_val])
    train = sorted(ranked[n_val:])
    return train, val


def write_split(path: str | Path, train: Sequence[str], val: Sequence[str]) -> None:
    """Save the split as CSV ``clip_id,split`` so E3 runs are reproducible and auditable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "split"])
        w.writerows([(i, "train") for i in train] + [(i, "val") for i in val])


def read_split(path: str | Path) -> tuple[list[str], list[str]]:
    with Path(path).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return ([r["clip_id"] for r in rows if r["split"] == "train"],
            [r["clip_id"] for r in rows if r["split"] == "val"])


def encode_latents(clip_paths: dict[str, Path], out_dir: str | Path, *, seconds: float = 10.0,
                   sample_rate: int = 44100, device: str = "cuda") -> list[Path]:
    """Pre-encode 10 s clips to SAO VAE latents (one ``<clip_id>.pt`` per clip). Lazy torch.

    TODO(M1): load SAO via ``stable_audio_tools.get_pretrained_model`` and call
    ``model.pretransform.encode(audio)``; check latent shape (expected ~[64, 215] for 10 s) and
    whether the pretransform scales latents. Crop/pad to ``seconds``; stereo at 44.1 kHz.
    """
    import torch  # noqa: F401  (lazy; heavy)

    from audiocfg.audio_io import load_audio, resample

    from .sao import load_sao  # noqa: WPS433

    model, _ = load_sao(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    n = int(seconds * sample_rate)
    for clip_id, p in clip_paths.items():
        target = out_dir / f"{clip_id}.pt"
        if target.exists():
            written.append(target)
            continue
        audio, sr = load_audio(p)
        audio = resample(audio, sr, sample_rate)
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        audio = audio[:, :n]
        audio = np.pad(audio, ((0, 0), (0, n - audio.shape[1])))
        x = torch.from_numpy(audio)[None].to(device)
        with torch.no_grad():
            z = model.pretransform.encode(x)[0].cpu()  # TODO(M1): verify API
        torch.save(z, target)
        written.append(target)
    return written
