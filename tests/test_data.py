import pandas as pd
import pytest

from audiocfg.data.fma_subset import fma_large_path, split_by_artist
from audiocfg.data.prompts import COLUMNS, eval_prompts, song_describer_prompts


def _song_describer(n_tracks=30):
    rows = []
    for t in range(n_tracks):
        rows.append(
            {
                "caption_id": 2 * t,
                "track_id": 1000 + t,
                "caption": f"calm piano piece number {t}",
                "is_valid_subset": True,
                "path": f"{t % 100:02d}/{1000 + t}.mp3",
            }
        )
        rows.append(
            {
                "caption_id": 2 * t + 1,
                "track_id": 1000 + t,
                "caption": "second caption",
                "is_valid_subset": True,
                "path": f"{t % 100:02d}/{1000 + t}.mp3",
            }
        )
    rows.append(
        {
            "caption_id": 999,
            "track_id": 5000,
            "caption": "female vocals over a guitar",
            "is_valid_subset": True,
            "path": "00/5000.mp3",
        }
    )
    rows.append(
        {
            "caption_id": 998,
            "track_id": 5001,
            "caption": "not validated",
            "is_valid_subset": None,
            "path": "01/5001.mp3",
        }
    )
    return pd.DataFrame(rows)


def _clotho(n=40):
    return pd.DataFrame(
        {"file_name": [f"f{i}.wav" for i in range(n)], "caption_1": [f"c{i}" for i in range(n)]}
    )


def _tracks():
    authors = [f"artist{i % 25}" for i in range(400)]
    return pd.DataFrame({"id": range(1, 401), "author": authors, "license": "cc0"})


def test_song_describer_filters_vocals_and_unvalidated():
    out = song_describer_prompts(_song_describer(), 30, 2026)
    assert not out["source_id"].isin(["5000", "5001"]).any()
    assert out["source_id"].is_unique
    assert (out["text"].str.startswith("calm piano")).all()
    assert out["ref_audio"].str.endswith(".2min.mp3").all()


def test_eval_prompts_deterministic_and_seed_dependent():
    a = eval_prompts(_song_describer(), _clotho(), 10, 2026)
    b = eval_prompts(_song_describer(), _clotho(), 10, 2026)
    c = eval_prompts(_song_describer(), _clotho(), 10, 7)
    assert list(a.columns) == COLUMNS
    pd.testing.assert_frame_equal(a, b)
    assert not a["prompt_id"].equals(c["prompt_id"])
    assert a.groupby("domain").size().to_dict() == {"music": 10, "sfx": 10}


def test_eval_prompts_too_few():
    with pytest.raises(ValueError):
        eval_prompts(_song_describer(5), _clotho(), 10, 2026)


def test_split_by_artist_is_artist_disjoint_and_sized():
    out = split_by_artist(_tracks(), n_train=200, n_val=50, master_seed=2026)
    train, val = out[out.split == "train"], out[out.split == "val"]
    assert len(train) == 200 and len(val) == 50
    assert not set(train.author) & set(val.author)
    assert out["id"].is_unique
    pd.testing.assert_frame_equal(out, split_by_artist(_tracks(), 200, 50, 2026))


def test_fma_large_path():
    assert fma_large_path(2) == "000/000002.mp3"
    assert fma_large_path(123456) == "123/123456.mp3"
