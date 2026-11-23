from __future__ import annotations

import numpy as np
import pandas as pd

from audiocfg.repro.seeds import derive_seed


def fma_large_path(track_id: int) -> str:
    name = f"{int(track_id):06d}"
    return f"{name[:3]}/{name}.mp3"


def split_by_artist(
    tracks: pd.DataFrame, n_train: int, n_val: int, master_seed: int
) -> pd.DataFrame:
    tracks = tracks.drop_duplicates("id").sort_values("id").reset_index(drop=True)
    artists = np.array(sorted(tracks["author"].astype(str).unique()))
    rng = np.random.default_rng(derive_seed(master_seed, "null_data", "artists"))
    artists = artists[rng.permutation(len(artists))]

    counts = tracks["author"].astype(str).value_counts()
    val_artists, n_val_tracks = set(), 0
    for artist in artists:
        if n_val_tracks >= n_val:
            break
        val_artists.add(artist)
        n_val_tracks += int(counts[artist])

    is_val = tracks["author"].astype(str).isin(val_artists)
    parts = []
    for split, pool, n in (("val", tracks[is_val], n_val), ("train", tracks[~is_val], n_train)):
        if len(pool) < n:
            raise ValueError(f"{split}: only {len(pool)} tracks, need {n}")
        pick = np.random.default_rng(derive_seed(master_seed, "null_data", split)).permutation(
            len(pool)
        )[:n]
        part = pool.iloc[np.sort(pick)].copy()
        part["split"] = split
        parts.append(part)

    out = pd.concat(parts, ignore_index=True)
    out["fma_large_path"] = out["id"].map(fma_large_path)
    return out[["id", "author", "license", "split", "fma_large_path"]]
