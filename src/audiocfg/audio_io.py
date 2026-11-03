"""Audio loading, saving and loudness normalisation (-23 LUFS, §4.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: str | Path, mono: bool = False) -> tuple[np.ndarray, int]:
    """Return ``(audio, sr)`` with audio shaped ``(channels, samples)``, float32."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = data.T
    if mono:
        audio = audio.mean(axis=0, keepdims=True)
    return audio, sr


def save_audio(path: str | Path, audio: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.atleast_2d(audio)
    sf.write(str(path), audio.T, sr, subtype="PCM_16")


def resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    from scipy.signal import resample_poly

    g = np.gcd(sr, target_sr)
    return resample_poly(audio, target_sr // g, sr // g, axis=-1).astype(np.float32)


def normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
    """ITU-R BS.1770 loudness normalisation; falls back to RMS if pyloudnorm is missing."""
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio.T)
        if not np.isfinite(loudness):  # silent clip
            return audio
        out = pyln.normalize.loudness(audio.T, loudness, target_lufs).T
    except ImportError:
        rms = np.sqrt(np.mean(audio**2)) + 1e-12
        out = audio * (10 ** (target_lufs / 20) / rms)
    peak = np.max(np.abs(out))
    return (out / peak * 0.99 if peak > 0.99 else out).astype(np.float32)
