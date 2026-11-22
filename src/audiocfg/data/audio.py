from __future__ import annotations

import numpy as np
import torch

SAMPLE_RATE = 44100


def load_window(
    path: str, window_samples: int, max_seconds: float = 30.0, offset: float = 0.0
) -> tuple[torch.Tensor, float]:
    import librosa

    audio, _ = librosa.load(
        path, sr=SAMPLE_RATE, mono=False, offset=offset, duration=max_seconds, dtype=np.float32
    )
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    audio = audio[:2, :window_samples]
    out = np.zeros((2, window_samples), dtype=np.float32)
    out[:, : audio.shape[1]] = audio
    return torch.from_numpy(out), audio.shape[1] / SAMPLE_RATE
