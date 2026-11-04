"""Model backends with explicit classifier-free guidance (§1.1, §1.2, §4.1, §4.3).

Every backend exposes the same four operations so the sweep controls exactly what varies:

* ``encode_prompt(text)``       -> conditioning of the *conditional* branch;
* ``null_condition(variant)``   -> conditioning of the *unconditional* branch (the null, §1.2);
* ``initial_noise(seed, shape)``-> starting noise, a function of the seed only (paired design, §4.3);
* ``generate(spec, null_cond)`` -> ``(audio (channels, samples) float32, sample_rate)``.

The guided prediction is always ``cfg_combine(eps_uncond, eps_cond, w)``. A null variant
replaces the conditioning of the unconditional branch; it is **never** a negative prompt
(no text other than the user prompt is ever encoded for guidance, except the controls
``t5_empty`` / ``mean_caption`` which define the *null*, §5.3).

Heavy imports (torch, stable_audio_tools, diffusers) happen inside methods only.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from audiocfg.config import ModelConfig
from audiocfg.schema import ClipSpec

NULL_VARIANTS = ("default", "learned", "t5_empty", "mean_caption")


def cfg_combine(eps_uncond: Any, eps_cond: Any, w: float) -> Any:
    """Standard CFG (§1.1): ``eps_uncond + w * (eps_cond - eps_uncond)``; ``w = 1`` is conditional only.

    Works for numpy arrays and torch tensors alike.
    """
    return eps_uncond + w * (eps_cond - eps_uncond)


def _check_variant(variant: str, supported: Sequence[str]) -> None:
    if variant not in NULL_VARIANTS:
        raise ValueError(f"unknown null variant {variant!r}; expected one of {NULL_VARIANTS}")
    if variant not in supported:
        raise ValueError(f"null variant {variant!r} not supported by this backend ({supported})")


class GeneratorBackend(ABC):
    """One text-to-audio model with the guidance loop under our control."""

    supported_nulls: tuple[str, ...] = ("default",)

    def __init__(self, model_cfg: ModelConfig, device: str | None = None) -> None:
        self.model_cfg = model_cfg
        self.device = device
        self._cond_cache: dict[str, Any] = {}

    @property
    def sample_rate(self) -> int:
        return self.model_cfg.sample_rate

    def cond(self, text: str) -> Any:
        """Cached ``encode_prompt`` (one prompt is reused for all seeds and w)."""
        if text not in self._cond_cache:
            self._cond_cache[text] = self.encode_prompt(text)
        return self._cond_cache[text]

    @abstractmethod
    def encode_prompt(self, text: str) -> Any: ...

    @abstractmethod
    def null_condition(self, variant: str = "default", path: str | Path | None = None,
                       captions: Sequence[str] | None = None) -> Any:
        """``learned`` needs ``path`` (tensor from E3); ``mean_caption`` needs ``captions``."""

    @abstractmethod
    def initial_noise(self, seed: int, shape: tuple[int, ...] | None = None) -> Any:
        """Deterministic in ``seed`` only, never in ``w`` or the null (§4.3)."""

    @abstractmethod
    def generate(self, spec: ClipSpec, null_cond: Any) -> tuple[np.ndarray, int]: ...


# --------------------------------------------------------------------------- fake


class FakeBackend(GeneratorBackend):
    """Numpy stand-in for tests and the GPU-free Gradio preview.

    "Conditioning" is a sinusoid whose frequency is hashed from the prompt; the null is a
    signal of the same shape. The output is ``noise + cfg_combine(null, cond, w)``, so it is
    deterministic, seed-paired and visibly w-dependent.
    """

    supported_nulls = NULL_VARIANTS

    def __init__(self, model_cfg: ModelConfig, device: str | None = None, seconds: float | None = None) -> None:
        super().__init__(model_cfg, device)
        self.seconds = seconds

    def _n(self, seconds: float) -> int:
        return int(round(seconds * self.sample_rate))

    @staticmethod
    def _freq(text: str) -> float:
        h = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
        return 110.0 + (h % 880)

    def encode_prompt(self, text: str) -> np.ndarray:
        return np.array([self._freq(text)], dtype=np.float64)

    def null_condition(self, variant: str = "default", path: str | Path | None = None,
                       captions: Sequence[str] | None = None) -> np.ndarray:
        _check_variant(variant, self.supported_nulls)
        if variant == "default":
            return np.zeros(1)
        if variant == "learned":
            if path is None:
                raise ValueError("null 'learned' needs --null-path")
            return np.asarray(np.load(path), dtype=np.float64).reshape(-1)[:1]
        if variant == "t5_empty":
            return self.encode_prompt("")
        if not captions:
            raise ValueError("null 'mean_caption' needs captions")
        return np.mean([self.encode_prompt(c) for c in captions], axis=0)

    def initial_noise(self, seed: int, shape: tuple[int, ...] | None = None) -> np.ndarray:
        shape = shape or (self.model_cfg.channels, self._n(self.seconds or 1.0))
        return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)

    def _tone(self, freq: np.ndarray, n: int) -> np.ndarray:
        f = float(freq[0])
        if f == 0.0:
            return np.zeros(n)
        return np.sin(2 * np.pi * f * np.arange(n) / self.sample_rate)

    def generate(self, spec: ClipSpec, null_cond: np.ndarray) -> tuple[np.ndarray, int]:
        n = self._n(self.seconds or spec.seconds)
        noise = self.initial_noise(spec.seed, (self.model_cfg.channels, n))
        guided = cfg_combine(self._tone(null_cond, n), self._tone(self.cond(spec.text), n), spec.w)
        audio = np.tanh(0.05 * noise + 0.1 * guided[None, :])
        return audio.astype(np.float32), self.sample_rate


# --------------------------------------------------------------------------- SAO


class SAOBackend(GeneratorBackend):
    """Stable Audio Open 1.0 via ``stable-audio-tools`` (DiT, T5 + seconds conditioning, §4.1).

    Default null: zeros shaped like the cross-attention conditioning (T5 tokens + seconds
    tokens), as the DiT builds it internally for its unconditional branch.

    TODO(M1): confirm in stable-audio-tools (``models/dit.py`` ``DiffusionTransformer.forward``
    and ``inference/generation.py``) how the unconditional branch is built: whether only
    ``cross_attn_cond`` is zeroed or also ``global_cond`` (seconds), whether a mask is applied,
    and whether ``scale_phi`` (CFG rescale) is on by default — it must be 0 for our formula.
    Injection plan: we do NOT rely on the model's internal CFG. We call the conditioned
    model twice with ``cfg_scale=1`` (once with the prompt, once with ``null_cond`` as the
    cross-attention conditioning) and combine with :func:`cfg_combine` inside our own
    denoiser passed to the k-diffusion sampler. If instead the sampler's
    ``negative_conditioning`` slot is used, it must hold exactly the *unconditional
    conditioning tensor* (our null), never an encoded negative text prompt.
    """

    supported_nulls = NULL_VARIANTS

    def __init__(self, model_cfg: ModelConfig, device: str | None = None) -> None:
        super().__init__(model_cfg, device or "cuda")
        self._model = None
        self._model_config: dict | None = None

    def _load(self):
        if self._model is None:
            from stable_audio_tools import get_pretrained_model

            model, model_config = get_pretrained_model(self.model_cfg.hf_id)
            self._model = model.to(self.device).eval().requires_grad_(False)
            self._model_config = model_config
        return self._model

    def _conditioning(self, text: str, seconds: float) -> dict:
        """Cross-attention + global conditioning tensors for one prompt."""
        model = self._load()
        metadata = [{"prompt": text, "seconds_start": 0, "seconds_total": seconds}]
        tensors = model.conditioner(metadata, self.device)
        # TODO(M1): check key names (cross_attn_cond, cross_attn_mask, global_cond) for SAO 1.0.
        return model.get_conditioning_inputs(tensors)

    def encode_prompt(self, text: str) -> dict:
        return self._conditioning(text, seconds=10.0)  # TODO(M1): take seconds from spec

    def null_condition(self, variant: str = "default", path: str | Path | None = None,
                       captions: Sequence[str] | None = None) -> dict:
        import torch

        _check_variant(variant, self.supported_nulls)
        ref = self._conditioning("", seconds=10.0)  # shapes + seconds tokens
        null = dict(ref)
        if variant == "default":
            null["cross_attn_cond"] = torch.zeros_like(ref["cross_attn_cond"])
        elif variant == "learned":
            if path is None:
                raise ValueError("null 'learned' needs --null-path")
            n = torch.load(path, map_location=self.device)
            null["cross_attn_cond"] = n.reshape(ref["cross_attn_cond"].shape).to(ref["cross_attn_cond"])
        elif variant == "t5_empty":
            pass  # control: T5 embedding of "" (§5.3) — this IS ref
        else:
            if not captions:
                raise ValueError("null 'mean_caption' needs captions")
            stack = torch.stack([self._conditioning(c, 10.0)["cross_attn_cond"] for c in captions])
            null["cross_attn_cond"] = stack.mean(dim=0)
        # TODO(M1): decide whether the null keeps the seconds tokens (likely yes, so only
        # the text part is "absent") and whether cross_attn_mask must be all-ones for the null.
        return null

    def initial_noise(self, seed: int, shape: tuple[int, ...] | None = None):
        import torch

        if shape is None:
            # TODO(M1): latent shape = (1, io_channels=64, sample_size // downsampling_ratio)
            # read from self._model_config; SAO renders 47.55 s and we crop to spec.seconds.
            self._load()
            mc = self._model_config
            shape = (1, self._model.io_channels, mc["sample_size"] // self._model.pretransform.downsampling_ratio)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        return torch.randn(shape, generator=g).to(self.device)

    def generate(self, spec: ClipSpec, null_cond: dict) -> tuple[np.ndarray, int]:
        """TODO(M1): validate against ``generate_diffusion_cond`` at the same w (must match)."""
        import torch
        from stable_audio_tools.inference.sampling import sample_k

        model = self._load()
        cond = self.cond(spec.text)
        noise = self.initial_noise(spec.seed)

        def denoiser(x, sigma, **_):
            # Explicit CFG (§1.1): two passes with cfg_scale=1, combined by us.
            v_c = model.model(x, sigma, **cond, cfg_scale=1.0, cfg_dropout_prob=0.0)
            v_u = model.model(x, sigma, **null_cond, cfg_scale=1.0, cfg_dropout_prob=0.0)
            return cfg_combine(v_u, v_c, spec.w)

        # dpmpp-3m-sde is stochastic: seed its noise too, so the seed fixes the whole trajectory.
        # TODO(M1): confirm sample_k signature and that the Brownian tree seed is honoured.
        torch.manual_seed(int(spec.seed))
        with torch.no_grad():
            latents = sample_k(denoiser, noise, sampler_type=spec.sampler, steps=spec.steps,
                               sigma_min=0.3, sigma_max=500, rho=1.0, device=self.device)
            audio = model.pretransform.decode(latents)
        n = int(spec.seconds * self.sample_rate)
        out = audio[0, :, :n].float().clamp(-1, 1).cpu().numpy()
        return out.astype(np.float32), self.sample_rate


# --------------------------------------------------------------------------- AudioLDM2


class AudioLDM2Backend(GeneratorBackend):
    """AudioLDM2 via ``diffusers.AudioLDM2Pipeline`` (UNet, CLAP + GPT-2/T5, 16 kHz, §4.1).

    Default null: the pipeline's own encoding of the empty string ``""`` (the same tensor
    diffusers builds when no negative prompt is given). We compute it once and pass it through
    the ``negative_*`` embedding arguments purely as the *unconditional conditioning*; no
    negative text prompt is ever used. diffusers applies exactly :func:`cfg_combine`
    (``uncond + w*(text - uncond)``) and skips the uncond pass at ``w <= 1``, i.e. w=1 is
    conditional only, consistent with §1.1.

    TODO(M1): check ``encode_prompt`` return values for the installed diffusers version and
    whether to replace the pipeline loop with our own (then use cfg_combine directly).
    """

    supported_nulls = ("default",)

    def __init__(self, model_cfg: ModelConfig, device: str | None = None) -> None:
        super().__init__(model_cfg, device or "cuda")
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from diffusers import AudioLDM2Pipeline, DDIMScheduler

            pipe = AudioLDM2Pipeline.from_pretrained(self.model_cfg.hf_id, torch_dtype=torch.float16)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)  # eta=0 below
            self._pipe = pipe.to(self.device)
        return self._pipe

    def _encode(self, text: str) -> dict:
        pipe = self._load()
        # do_classifier_free_guidance=False: encode only this text, no implicit uncond.
        embeds, mask, generated = pipe.encode_prompt(
            prompt=[text], device=self.device, num_waveforms_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return {"prompt_embeds": embeds, "attention_mask": mask, "generated_prompt_embeds": generated}

    def encode_prompt(self, text: str) -> dict:
        return self._encode(text)

    def null_condition(self, variant: str = "default", path: str | Path | None = None,
                       captions: Sequence[str] | None = None) -> dict:
        _check_variant(variant, self.supported_nulls)
        e = self._encode("")
        return {"negative_prompt_embeds": e["prompt_embeds"],
                "negative_attention_mask": e["attention_mask"],
                "negative_generated_prompt_embeds": e["generated_prompt_embeds"]}

    def initial_noise(self, seed: int, shape: tuple[int, ...] | None = None):
        import torch

        pipe = self._load()
        if shape is None:
            # TODO(M1): verify against pipe.prepare_latents for 10 s (expected (1, 8, 250, 16)).
            vae_scale = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
            hop = np.prod(pipe.vocoder.config.upsample_rates) / pipe.vocoder.config.sampling_rate
            height = int(10.0 / hop) // vae_scale
            shape = (1, pipe.unet.config.in_channels, height, pipe.vocoder.config.model_in_dim // vae_scale)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        return torch.randn(shape, generator=g).to(self.device, dtype=pipe.unet.dtype)

    def generate(self, spec: ClipSpec, null_cond: dict) -> tuple[np.ndarray, int]:
        pipe = self._load()
        out = pipe(
            **self.cond(spec.text), **null_cond,
            latents=self.initial_noise(spec.seed),
            guidance_scale=spec.w, num_inference_steps=spec.steps, eta=0.0,
            audio_length_in_s=spec.seconds,
            num_waveforms_per_prompt=1,  # >1 triggers CLAP re-ranking: must stay 1
            output_type="np",
        )
        audio = np.asarray(out.audios[0], dtype=np.float32)
        return np.atleast_2d(audio), self.sample_rate


# --------------------------------------------------------------------------- factory

BACKENDS: dict[str, type[GeneratorBackend]] = {
    "fake": FakeBackend,
    "sao": SAOBackend,
    "audioldm2": AudioLDM2Backend,
}


def get_backend(name: str, model_cfg: ModelConfig, **kwargs: Any) -> GeneratorBackend:
    """``name`` is ``fake``, ``sao`` or ``audioldm2``; ``model_cfg`` gives sr/channels/sampler."""
    try:
        cls = BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; expected one of {sorted(BACKENDS)}") from None
    return cls(model_cfg, **kwargs)
