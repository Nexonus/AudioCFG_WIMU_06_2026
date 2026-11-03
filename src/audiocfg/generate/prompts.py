"""E0: sample the fixed prompt list ``data/prompts.csv`` (§4.2).

100 SFX captions from the AudioCaps *test* split and 100 music captions from the
non-vocal subset of the Song Describer Dataset, drawn with the fixed
``prompt_sampling.random_seed``. One caption per source recording, so every prompt
has its own real reference clip (used for FD / CLAP calibration, §6).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audiocfg.schema import PROMPT_COLUMNS

# Default column names of the public CSVs.
AUDIOCAPS_COLS = {"id": "youtube_id", "text": "caption"}
SONG_DESCRIBER_COLS = {"id": "track_id", "text": "caption", "path": "path"}


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--audiocaps", required=True, help="AudioCaps test.csv (youtube_id, caption, ...)")
    p.add_argument("--song-describer", required=True, help="Song Describer CSV (track_id, caption, path, ...)")
    p.add_argument("--sd-nonvocal-column", default=None,
                   help="boolean column in the Song Describer CSV that marks non-vocal tracks")
    p.add_argument("--sd-nonvocal-ids", default=None,
                   help="file with one non-vocal track_id per line (alternative to the column)")
    p.add_argument("--audiocaps-audio-dir", default="", help="prefix for reference_audio of SFX prompts")
    p.add_argument("--song-describer-audio-dir", default="", help="prefix for reference_audio of music prompts")
    p.add_argument("--n-sfx", type=int, default=None, help="override prompt_sampling.sfx.n")
    p.add_argument("--n-music", type=int, default=None, help="override prompt_sampling.music.n")
    p.add_argument("--out", default=None, help="output CSV (default: config prompts_csv)")


def _one_caption_per_source(df: pd.DataFrame, id_col: str, text_col: str,
                            rng: np.random.Generator) -> pd.DataFrame:
    """Pick one caption per recording (AudioCaps has 5 per clip, Song Describer ~2 per track)."""
    # Canonical order makes the result independent of the CSV's row order.
    df = (df.dropna(subset=[id_col, text_col]).sort_values([id_col, text_col], kind="mergesort")
          .reset_index(drop=True))
    keys = rng.random(len(df))
    return df.assign(_k=keys).sort_values("_k").drop_duplicates(id_col).drop(columns="_k")


def _sample(df: pd.DataFrame, n: int, id_col: str, text_col: str,
            rng: np.random.Generator) -> pd.DataFrame:
    df = _one_caption_per_source(df, id_col, text_col, rng)
    if len(df) < n:
        raise ValueError(f"only {len(df)} unique sources available, need {n}")
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[np.sort(idx)]


def _join(prefix: str, name: str) -> str:
    return str(Path(prefix) / name) if prefix else ""


def nonvocal_filter(df: pd.DataFrame, column: str | None, ids_file: str | None) -> pd.DataFrame:
    """Keep the non-vocal Song Describer tracks.

    TODO(M1): confirm which official artefact defines the non-vocal subset (SAO's
    evaluation used the "no-singing" subset, distributed as a track-id list). Until
    then pass either ``--sd-nonvocal-ids`` or ``--sd-nonvocal-column``.
    """
    if ids_file:
        ids = {line.strip() for line in Path(ids_file).read_text(encoding="utf-8").splitlines() if line.strip()}
        return df[df[SONG_DESCRIBER_COLS["id"]].astype(str).isin(ids)]
    if column:
        return df[df[column].astype(bool)]
    raise ValueError("non-vocal subset undefined: pass --sd-nonvocal-ids or --sd-nonvocal-column")


def sample_prompts(audiocaps: pd.DataFrame, song_describer: pd.DataFrame, n_sfx: int, n_music: int,
                   seed: int, audiocaps_audio_dir: str = "", sd_audio_dir: str = "") -> pd.DataFrame:
    """Return the prompt table with ``PROMPT_COLUMNS``; deterministic given ``seed``."""
    rng = np.random.default_rng(seed)
    a_id, a_txt = AUDIOCAPS_COLS["id"], AUDIOCAPS_COLS["text"]
    s_id, s_txt = SONG_DESCRIBER_COLS["id"], SONG_DESCRIBER_COLS["text"]

    sfx = _sample(audiocaps, n_sfx, a_id, a_txt, rng)
    music = _sample(song_describer, n_music, s_id, s_txt, rng)
    rows = [
        {"prompt_id": f"sfx_{i:03d}", "domain": "sfx", "text": str(r[a_txt]).strip(),
         "source": "audiocaps_test", "source_id": str(r[a_id]),
         "reference_audio": _join(audiocaps_audio_dir, f"{r[a_id]}.wav")}
        for i, (_, r) in enumerate(sfx.iterrows())
    ] + [
        {"prompt_id": f"music_{i:03d}", "domain": "music", "text": str(r[s_txt]).strip(),
         "source": "song_describer_nonvocal", "source_id": str(r[s_id]),
         "reference_audio": _join(sd_audio_dir, str(r.get(SONG_DESCRIBER_COLS["path"], f"{r[s_id]}.mp3")))}
        for i, (_, r) in enumerate(music.iterrows())
    ]
    return pd.DataFrame(rows, columns=list(PROMPT_COLUMNS))


def run(args: argparse.Namespace) -> None:
    from audiocfg.config import DEFAULT_CONFIG, load_config

    cfg = load_config(args.config or DEFAULT_CONFIG)
    ps = cfg.section("prompt_sampling")
    sd = nonvocal_filter(pd.read_csv(args.song_describer), args.sd_nonvocal_column, args.sd_nonvocal_ids)
    df = sample_prompts(
        pd.read_csv(args.audiocaps), sd,
        n_sfx=args.n_sfx if args.n_sfx is not None else int(ps["sfx"]["n"]),
        n_music=args.n_music if args.n_music is not None else int(ps["music"]["n"]),
        seed=int(ps["random_seed"]),
        audiocaps_audio_dir=args.audiocaps_audio_dir, sd_audio_dir=args.song_describer_audio_dir,
    )
    out = Path(args.out) if args.out else cfg.prompts_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} prompts ({(df.domain == 'sfx').sum()} sfx, {(df.domain == 'music').sum()} music) -> {out}")
