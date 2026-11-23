import numpy as np
import pandas as pd

from audiocfg.metrics.distribution import ccds, frechet_distance
from audiocfg.metrics.score import diversity, fit_length, reference_files, split_ref


def test_ccds_bounds():
    same = np.tile([[1.0, 2.0, 3.0]], (3, 1))
    assert abs(ccds(same)) < 1e-9
    assert abs(ccds(np.eye(3)) - 1) < 1e-9
    assert abs(ccds(np.array([[1.0, 0], [-1.0, 0]])) - 2) < 1e-9
    assert np.isnan(ccds(np.ones((1, 3))))


def test_frechet_distance():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((2000, 4)), rng.standard_normal((2000, 4))
    assert frechet_distance(a, b) < 0.05
    shifted = frechet_distance(a, b + 2)
    assert abs(shifted - 16) < 0.5
    assert frechet_distance(a, 3 * b) > frechet_distance(a, 1.5 * b) > 0.1


def test_split_ref():
    assert split_ref("org/model@abc") == ("org/model", "abc")
    assert split_ref("org/model") == ("org/model", None)


def test_reference_files_deterministic(tmp_path):
    for i in range(10):
        (tmp_path / f"{i:02d}").mkdir()
        (tmp_path / f"{i:02d}" / f"{i}.mp3").touch()
    a = reference_files(tmp_path, 4, 2026)
    assert a == reference_files(tmp_path, 4, 2026) and len(a) == 4
    assert len(reference_files(tmp_path, 50, 2026)) == 10


def test_diversity_groups_by_prompt_guidance_null():
    clips = pd.DataFrame(
        {
            "clip": ["a", "b", "c", "d"],
            "null": ["zero", "zero", "learned", "learned"],
            "domain": "music",
            "prompt_id": "p",
            "guidance": 7.0,
        }
    )
    emb = {"m": np.array([[1.0, 0], [1.0, 0], [1.0, 0], [0, 1.0]])}
    div = diversity(clips, emb, {c: i for i, c in enumerate(clips["clip"])}).set_index("null")
    assert div.loc["zero", "ccds_m"] < 1e-9
    assert abs(div.loc["learned", "ccds_m"] - 1) < 1e-9
    assert (div["n"] == 2).all()


def test_fit_length_pads_and_crops():
    assert fit_length(np.ones(7), 10).tolist() == [1.0] * 7 + [0.0] * 3
    assert len(fit_length(np.ones(12), 10)) == 10
