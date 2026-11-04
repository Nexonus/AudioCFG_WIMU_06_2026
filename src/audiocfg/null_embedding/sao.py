"""Thin adapter around Stable Audio Open for E3 (§5.3). Everything here is lazy and TODO(M1).

What E3 needs from SAO:
- the shape ``(L, d)`` of the raw cross-attention conditioning (T5 tokens + seconds tokens),
  i.e. the tensor that SAO's CFG replaces with ``zeros_like`` in the unconditional branch;
- the global conditioning for a 10 s clip (seconds_start / seconds_total), kept as in the
  unconditional branch of SAO's CFG (only the cross-attention condition is nulled there);
- one call of the frozen DiT returning the v-prediction for ``(x_t, t, cond)``.
"""

from __future__ import annotations

from typing import Any


def load_sao(device: str = "cuda") -> tuple[Any, dict]:
    """Return ``(model, model_config)``; all parameters frozen. TODO(M1): pin the revision."""
    from stable_audio_tools import get_pretrained_model

    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, model_config


def null_shape(model: Any, model_config: dict, seconds: float = 10.0) -> tuple[int, int]:
    """Shape (L, d) of SAO's cross-attention conditioning, read from the model, not hard-coded.

    TODO(M1): verify. Expected: T5-base 128 tokens + seconds_start + seconds_total = 130 tokens,
    d = 768 (``model_config["model"]["conditioning"]["cond_dim"]``). We obtain it by running the
    conditioner once on a dummy prompt, which is robust to config changes.
    """
    cond = model.conditioner([{"prompt": "", "seconds_start": 0, "seconds_total": seconds}],
                             device=next(model.parameters()).device)
    cross, _mask, _glob = _split(model, cond)
    return int(cross.shape[1]), int(cross.shape[2])


def _split(model: Any, cond: dict):
    """Return (cross_attn_cond, cross_attn_mask, global_cond). TODO(M1): exact keys."""
    inputs = model.get_conditioning_inputs(cond)
    return (inputs["cross_attn_cond"], inputs.get("cross_attn_mask"),
            inputs.get("global_cond"))


def make_v_loss_fn(model: Any, seconds: float = 10.0):
    """Build ``loss_fn(latents, cond) -> scalar`` = SAO's own v-objective with condition = cond.

    Uses SAO's training parameterisation (stable-audio-tools ``diffusion_objective="v"``):
    ``alpha = cos(t*pi/2)``, ``sigma = sin(t*pi/2)``, ``x_t = alpha*x + sigma*eps``,
    ``v = alpha*eps - sigma*x``. TODO(M1): confirm t sampling (uniform vs. logit-normal) and
    that ``model.model(...)`` is the DiT wrapper call used in training; pass
    ``cfg_dropout_prob=0`` and ``cfg_scale=1`` so no other null is mixed in. The cross-attention
    mask for the learned null is all ones (the default null uses the prompt's mask).
    """
    import math

    import torch

    device = next(model.parameters()).device
    base = model.conditioner([{"prompt": "", "seconds_start": 0, "seconds_total": seconds}],
                             device=device)
    _cross, _mask, global_cond = _split(model, base)

    def loss_fn(latents: "torch.Tensor", cond: "torch.Tensor") -> "torch.Tensor":
        b = latents.shape[0]
        t = torch.rand(b, device=latents.device)
        alpha = torch.cos(t * math.pi / 2)[:, None, None]
        sigma = torch.sin(t * math.pi / 2)[:, None, None]
        eps = torch.randn_like(latents)
        x_t = alpha * latents + sigma * eps
        v_target = alpha * eps - sigma * latents
        cross = cond[None].expand(b, -1, -1)
        mask = torch.ones(cross.shape[:2], dtype=torch.bool, device=latents.device)
        g = None if global_cond is None else global_cond.expand(b, -1)
        v_pred = model.model(x_t, t, cross_attn_cond=cross, cross_attn_mask=mask,
                             global_embed=g, cfg_scale=1.0, cfg_dropout_prob=0.0)
        return torch.mean((v_pred - v_target) ** 2)

    return loss_fn
