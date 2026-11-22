from __future__ import annotations

import math

import torch

from audiocfg.repro.seeds import derive_seed


def alpha_sigma(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def noisy_and_target(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
    a, s = (v.view(-1, *([1] * (x0.dim() - 1))) for v in alpha_sigma(t))
    return a * x0 + s * noise, a * noise - s * x0


def v_loss(sao, x0, t, noise, encoder_states, global_states) -> torch.Tensor:
    """Per-sample mean squared error of the v-prediction, shape (B,)."""
    x_t, target = noisy_and_target(x0.float(), t.float(), noise.float())
    out = sao.v_prediction(x_t, t, encoder_states, global_states)
    return (out.float() - target).pow(2).flatten(1).mean(1)


def draws(master_seed: int, stream: str, item_ids, k: int, shape, eps: float = 1e-3):
    """k fixed (t, noise) pairs per item, seeded by item id: identical for every null variant."""
    ts, noises = [], []
    for item_id in item_ids:
        for j in range(k):
            gen = torch.Generator("cpu").manual_seed(derive_seed(master_seed, stream, item_id, j))
            ts.append(eps + (1 - 2 * eps) * torch.rand((), generator=gen))
            noises.append(torch.randn(shape, generator=gen))
    return torch.stack(ts).view(len(item_ids), k), torch.stack(noises).view(
        len(item_ids), k, *shape
    )
