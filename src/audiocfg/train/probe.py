from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from audiocfg.models.sao import build_uncond
from audiocfg.train.objective import v_loss

StatesFn = Callable[[Sequence[int]], torch.Tensor]


def null_states(null_text: torch.Tensor) -> StatesFn:
    return lambda idx: build_uncond(null_text, len(idx))


def text_states(sao, captions: Sequence[str]) -> StatesFn:
    """Real conditioning: caption tokens + real timing tokens (needs seconds per item)."""

    def fn(idx, seconds):
        text = sao.encode_text([captions[i] for i in idx])
        timing, _ = sao.encode_timing(len(idx), 0.0, [float(seconds[i]) for i in idx])
        return torch.cat([text, timing.to(text)], dim=1)

    return fn


@torch.no_grad()
def eval_losses(
    sao,
    latents: torch.Tensor,
    seconds: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    states: Callable,
    batch_size: int = 8,
    needs_seconds: bool = False,
) -> torch.Tensor:
    n, k = t.shape
    flat = [(i, j) for i in range(n) for j in range(k)]
    out = torch.empty(n, k)
    for start in range(0, len(flat), batch_size):
        chunk = flat[start : start + batch_size]
        idx = [i for i, _ in chunk]
        x0 = latents[idx].to(sao.device)
        tt = torch.stack([t[i, j] for i, j in chunk]).to(sao.device)
        eps = torch.stack([noise[i, j] for i, j in chunk]).to(sao.device)
        enc = states(idx, seconds) if needs_seconds else states(idx)
        _, glob = sao.encode_timing(len(idx), 0.0, [float(seconds[i]) for i in idx])
        losses = v_loss(sao, x0, tt, eps, enc.to(sao.device), glob).cpu()
        for (i, j), loss in zip(chunk, losses, strict=True):
            out[i, j] = loss
    return out
