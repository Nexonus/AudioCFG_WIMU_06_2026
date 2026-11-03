"""Numpy signal features: STFT, mel filterbank, log-mel and chroma (fallback when librosa is absent).

Used by the stage-2 copy verification (§5.2) and by the deterministic test embedder.
"""

from __future__ import annotations

import numpy as np


def to_mono(audio: np.ndarray) -> np.ndarray:
    """``(samples,)`` or ``(channels, samples)`` -> ``(samples,)`` float64."""
    a = np.asarray(audio, dtype=np.float64)
    return a.mean(axis=0) if a.ndim == 2 else a


def stft_power(x: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Power spectrogram ``(frames, n_fft//2+1)`` with a Hann window (no centre padding)."""
    x = to_mono(x)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(n_fft)[None, :]
    return np.abs(np.fft.rfft(frames, axis=1)) ** 2


def _hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 64, fmin: float = 0.0,
                   fmax: float | None = None) -> np.ndarray:
    """Triangular HTK-style mel filterbank ``(n_mels, n_fft//2+1)``."""
    fmax = fmax or sr / 2
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / max(c - lo, 1e-9)
        down = (hi - freqs) / max(hi - c, 1e-9)
        fb[i] = np.clip(np.minimum(up, down), 0, None)
    return fb


def log_mel(x: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512, n_mels: int = 64) -> np.ndarray:
    """Log-mel spectrogram ``(frames, n_mels)``. TODO(M1): swap for librosa when installed."""
    p = stft_power(x, n_fft, hop)
    return np.log(p @ mel_filterbank(sr, n_fft, n_mels).T + 1e-10)


def chroma(x: np.ndarray, sr: int, n_fft: int = 4096, hop: int = 512,
           fmin: float = 55.0, fmax: float = 5000.0) -> np.ndarray:
    """Log chroma ``(frames, 12)``: STFT power summed per pitch class (for music, §5.2)."""
    p = stft_power(x, n_fft, hop)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    keep = (freqs >= fmin) & (freqs <= fmax)
    pc = np.round(12 * np.log2(freqs[keep] / 440.0) + 69).astype(int) % 12
    m = np.zeros((len(pc), 12))
    m[np.arange(len(pc)), pc] = 1.0
    return np.log(p[:, keep] @ m + 1e-10)
