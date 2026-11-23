from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import torch

AXES = ("CE", "CU", "PC", "PQ")


class Aesthetics:
    def __init__(self, repo: str, revision: str | None = None):
        from audiobox_aesthetics.infer import initialize_predictor
        from audiobox_aesthetics.model.aes import AesMultiOutput

        self.predictor = initialize_predictor()
        # The package loads the repo head; reload the pinned revision.
        model = AesMultiOutput.from_pretrained(repo, revision=revision)
        self.predictor.model = model.to(self.predictor.device).eval()

    def score(self, waves: Sequence[torch.Tensor], sample_rate: int) -> pd.DataFrame:
        """(channels, samples) waveforms -> one row of CE, CU, PC, PQ per clip."""
        batch = [{"path": w.float(), "sample_rate": sample_rate} for w in waves]
        return pd.DataFrame(self.predictor.forward(batch))[list(AXES)]
