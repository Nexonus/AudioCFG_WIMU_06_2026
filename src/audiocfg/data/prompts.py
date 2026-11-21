from __future__ import annotations

import re

import numpy as np
import pandas as pd

from audiocfg.repro.seeds import derive_seed

VOCAL_PATTERN = re.compile(
    r"\b(?:vocal|voice|sing|singer|singing|sung|song lyric|lyric|rap|rapp|choir|chant|"
    r"a ?cappella|humm|whisper|spoken|speech|word)",
    re.IGNORECASE,
)

COLUMNS = ["prompt_id", "domain", "rank", "text", "source", "source_id", "ref_audio"]


def _rng(master_seed: int, stream: str) -> np.random.Generator:
    return np.random.default_rng(derive_seed(master_seed, "eval_prompts", stream))


def song_describer_prompts(table: pd.DataFrame, n: int, master_seed: int) -> pd.DataFrame:
    valid = table[(table["is_valid_subset"] == True) & table["caption"].notna()]
    valid = valid[~valid["caption"].str.contains(VOCAL_PATTERN)]
    one_per_track = valid.sort_values("caption_id").groupby("track_id", as_index=False).first()
    one_per_track = one_per_track.sort_values("track_id").reset_index(drop=True)
    if len(one_per_track) < n:
        raise ValueError(f"only {len(one_per_track)} instrumental tracks, need {n}")
    pick = _rng(master_seed, "music").permutation(len(one_per_track))[:n]
    rows = one_per_track.iloc[pick].reset_index(drop=True)
    return pd.DataFrame(
        {
            "prompt_id": [f"sd_{t}" for t in rows["track_id"]],
            "domain": "music",
            "rank": range(n),
            "text": rows["caption"].str.strip(),
            "source": "song_describer",
            "source_id": rows["track_id"].astype(str),
            "ref_audio": [p.replace(".mp3", ".2min.mp3") for p in rows["path"]],
        }
    )


def clotho_prompts(table: pd.DataFrame, n: int, master_seed: int) -> pd.DataFrame:
    files = table.sort_values("file_name").reset_index(drop=True)
    if len(files) < n:
        raise ValueError(f"only {len(files)} Clotho files, need {n}")
    pick = _rng(master_seed, "sfx").permutation(len(files))[:n]
    rows = files.iloc[pick].reset_index(drop=True)
    return pd.DataFrame(
        {
            "prompt_id": [f"clotho_{i:04d}" for i in pick],
            "domain": "sfx",
            "rank": range(n),
            "text": rows["caption_1"].str.strip(),
            "source": "clotho_v2.1_evaluation",
            "source_id": rows["file_name"],
            "ref_audio": rows["file_name"],
        }
    )


def eval_prompts(
    song_describer: pd.DataFrame, clotho: pd.DataFrame, n_per_domain: int, master_seed: int
) -> pd.DataFrame:
    out = pd.concat(
        [
            song_describer_prompts(song_describer, n_per_domain, master_seed),
            clotho_prompts(clotho, n_per_domain, master_seed),
        ],
        ignore_index=True,
    )
    if out["prompt_id"].duplicated().any():
        raise ValueError("duplicate prompt ids")
    return out[COLUMNS]
