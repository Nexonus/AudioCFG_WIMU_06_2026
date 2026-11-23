from __future__ import annotations

import math
import time
from collections.abc import Callable

import torch

from audiocfg.models.sao import build_uncond
from audiocfg.repro.seeds import derive_seed
from audiocfg.train.objective import v_loss


def lr_at(step: int, total: int, lr: float, warmup: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr * 0.5 * (1 + math.cos(math.pi * progress))


def train_null(
    sao,
    latents: torch.Tensor,
    seconds: torch.Tensor,
    init: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    master_seed: int,
    warmup: int = 100,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    eval_every: int = 0,
    eval_fn: Callable[[torch.Tensor], float] | None = None,
    log: Callable[[dict], None] = print,
) -> tuple[torch.Tensor, list[dict]]:
    null = torch.nn.Parameter(init.detach().float().clone().to(sao.device))
    opt = torch.optim.AdamW([null], lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=sao.dtype == torch.float16)
    sao.transformer.enable_gradient_checkpointing()
    history = []
    n = len(latents)
    t0 = time.perf_counter()

    for step in range(steps):
        gen = torch.Generator("cpu").manual_seed(derive_seed(master_seed, "train_null", step))
        idx = torch.randint(0, n, (batch_size,), generator=gen)
        t = 1e-3 + (1 - 2e-3) * torch.rand(batch_size, generator=gen)
        noise = torch.randn((batch_size, *latents.shape[1:]), generator=gen)
        x0 = latents[idx].to(sao.device)
        _, glob = sao.encode_timing(batch_size, 0.0, seconds[idx].tolist())
        enc = build_uncond(null.to(sao.dtype), batch_size)

        for group in opt.param_groups:
            group["lr"] = lr_at(step, steps, lr, warmup)
        loss = v_loss(sao, x0, t.to(sao.device), noise.to(sao.device), enc, glob).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_([null], grad_clip)
        scaler.step(opt)
        scaler.update()

        row = {
            "step": step,
            "loss": float(loss),
            "grad_norm": float(grad_norm),
            "lr": opt.param_groups[0]["lr"],
            "null_rms": float(null.detach().pow(2).mean().sqrt()),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }
        if eval_fn is not None and eval_every and ((step + 1) % eval_every == 0):
            row["val_loss"] = eval_fn(null.detach())
        history.append(row)
        if step % 25 == 0 or "val_loss" in row:
            log(row)

    sao.transformer.disable_gradient_checkpointing()
    return null.detach().float().cpu(), history
