from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

CLAP_SR = 48000


class Clap:
    def __init__(self, repo: str, revision: str | None = None, device: str = "cuda"):
        from transformers import ClapModel, ClapProcessor

        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = ClapModel.from_pretrained(repo, revision=revision).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(repo, revision=revision)

    @torch.no_grad()
    def embed_audio(self, waves: Sequence[np.ndarray]) -> np.ndarray:
        """Mono 48 kHz waveforms -> L2-normalised (N, D) embeddings."""
        inputs = self.processor(audio=list(waves), sampling_rate=CLAP_SR, return_tensors="pt").to(
            self.device
        )
        return self.model.get_audio_features(**inputs).pooler_output.float().cpu().numpy()

    @torch.no_grad()
    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True).to(self.device)
        return self.model.get_text_features(**inputs).pooler_output.float().cpu().numpy()


def load_mono(path: str, sr: int = CLAP_SR, offset: float = 0.0, seconds: float | None = None):
    import librosa

    wave, _ = librosa.load(path, sr=sr, mono=True, offset=offset, duration=seconds)
    return wave.astype(np.float32)
