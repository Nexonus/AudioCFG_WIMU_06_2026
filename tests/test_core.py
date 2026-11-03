import numpy as np

from audiocfg.audio_io import load_audio, normalize_loudness, save_audio
from audiocfg.config import load_config
from audiocfg.schema import CLIP_COLUMNS, ClipSpec


def test_config_matches_design():
    cfg = load_config()
    assert cfg.guidance_scales == (1, 3, 5, 7, 9)
    assert len(cfg.seeds) == 10
    assert set(cfg.models) == {"sao", "audioldm2"}
    assert cfg.models["sao"].default_w == 7


def test_clip_id_encodes_all_varying_settings():
    a = ClipSpec("sao", "sfx_001", "sfx", "rain", 7.0, "default", 3, "dpmpp-3m-sde", 100, 10.0)
    b = ClipSpec("sao", "sfx_001", "sfx", "rain", 7.0, "learned", 3, "dpmpp-3m-sde", 100, 10.0)
    assert a.clip_id != b.clip_id
    row = a.to_row(a.relpath(), 44100, 1.0)
    assert tuple(row) == CLIP_COLUMNS


def test_audio_roundtrip_and_loudness(tmp_path):
    sr = 16000
    x = (0.1 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)[None]
    y = normalize_loudness(x, sr, -23.0)
    save_audio(tmp_path / "a.wav", y, sr)
    z, sr2 = load_audio(tmp_path / "a.wav")
    assert sr2 == sr and z.shape == y.shape
