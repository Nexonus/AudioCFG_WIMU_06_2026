from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

MODEL_ID = "stabilityai/stable-audio-open-1.0"
TIMING_SLOTS = 2


def build_uncond(null_text: torch.Tensor, batch_size: int) -> torch.Tensor:
    if null_text.dim() == 2:
        null_text = null_text.unsqueeze(0).expand(batch_size, -1, -1)
    if null_text.shape[0] != batch_size:
        raise ValueError(f"null batch {null_text.shape[0]} != batch {batch_size}")
    timing = null_text.new_zeros(batch_size, TIMING_SLOTS, null_text.shape[-1])
    return torch.cat([null_text, timing], dim=1)


def combine_cfg(uncond: torch.Tensor, cond: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    g = scale.to(cond.dtype).view(-1, *([1] * (cond.dim() - 1)))
    guided = uncond + g * (cond - uncond)
    return torch.where(g == 1, cond, guided)


def guidance_rms(uncond: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    return (cond - uncond).float().pow(2).flatten(1).mean(1).sqrt()


@dataclass
class SampleOutput:
    audio: torch.Tensor
    latents: torch.Tensor
    guidance_rms: torch.Tensor | None
    sample_rate: int


class StableAudio:
    def __init__(self, pipe):
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler

    @classmethod
    def load(
        cls,
        model_id: str = MODEL_ID,
        revision: str | None = None,
        dtype: str = "auto",
        device: str = "cuda",
        offload: str = "auto",
    ) -> StableAudio:
        from diffusers import StableAudioPipeline

        torch_dtype = resolve_dtype(dtype)
        pipe = StableAudioPipeline.from_pretrained(
            model_id, revision=revision, torch_dtype=torch_dtype
        )
        pipe.set_progress_bar_config(disable=True)
        if device == "cuda" and _needs_offload(offload):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        for module in (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.projection_model):
            module.requires_grad_(False).eval()
        return cls(pipe)

    @property
    def device(self) -> torch.device:
        return self.pipe._execution_device

    @property
    def dtype(self) -> torch.dtype:
        return self.transformer.dtype

    @property
    def sample_rate(self) -> int:
        return int(self.vae.config.sampling_rate)

    @property
    def max_seconds(self) -> float:
        return self.transformer.config.sample_size * self.vae.hop_length / self.sample_rate

    @property
    def text_shape(self) -> tuple[int, int]:
        return (
            int(self.pipe.tokenizer.model_max_length),
            int(self.transformer.config.cross_attention_dim),
        )

    @torch.no_grad()
    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        return self.pipe.encode_prompt(list(prompts), self.device, False)

    @property
    def window_samples(self) -> int:
        return int(self.transformer.config.sample_size) * int(self.vae.hop_length)

    @torch.no_grad()
    def encode_audio(self, audio: torch.Tensor, generator: torch.Generator | None = None):
        """(B, 2, window_samples) waveform -> (B, 64, latent_len) sampled VAE latents."""
        audio = audio.to(self.device, self.vae.dtype)
        return self.vae.encode(audio).latent_dist.sample(generator)

    @torch.no_grad()
    def encode_timing(
        self,
        batch_size: int,
        seconds_start: float | Sequence[float],
        seconds_end: float | Sequence[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seconds_start = (
            list(seconds_start) if isinstance(seconds_start, Sequence) else seconds_start
        )
        seconds_end = list(seconds_end) if isinstance(seconds_end, Sequence) else seconds_end
        start, end = self.pipe.encode_duration(
            seconds_start, seconds_end, self.device, False, batch_size
        )
        return torch.cat([start, end], dim=1), torch.cat([start, end], dim=2)

    def conditioning(
        self, prompts: Sequence[str], seconds_start: float, seconds_end: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.encode_text(prompts)
        timing_tokens, global_states = self.encode_timing(len(prompts), seconds_start, seconds_end)
        return torch.cat([text, timing_tokens], dim=1), global_states

    def initial_noise(self, generators: list[torch.Generator]) -> torch.Tensor:
        from diffusers.utils.torch_utils import randn_tensor

        shape = (
            len(generators),
            self.transformer.config.in_channels,
            int(self.transformer.config.sample_size),
        )
        return randn_tensor(shape, generator=generators, device=self.device, dtype=self.dtype)

    def rotary(self, latent_len: int, global_len: int):
        from diffusers.models.embeddings import get_1d_rotary_pos_embed

        return get_1d_rotary_pos_embed(
            self.transformer.config.attention_head_dim // 2,
            latent_len + global_len,
            use_real=True,
            repeat_interleave_real=False,
        )

    def predict(self, latents, t, encoder_states, global_states, rotary) -> torch.Tensor:
        return self.transformer(
            latents,
            t.unsqueeze(0),
            encoder_hidden_states=encoder_states,
            global_hidden_states=global_states,
            rotary_embedding=rotary,
            return_dict=False,
        )[0]

    def v_prediction(self, x_t, t, encoder_states, global_states) -> torch.Tensor:
        rotary = self.rotary(x_t.shape[2], global_states.shape[1])
        return self.transformer(
            x_t.to(self.dtype),
            t.to(self.device, self.dtype),
            encoder_hidden_states=encoder_states.to(self.dtype),
            global_hidden_states=global_states.to(self.dtype),
            rotary_embedding=rotary,
            return_dict=False,
        )[0]

    @torch.no_grad()
    def sample(
        self,
        prompts: Sequence[str],
        seeds: Sequence[int],
        guidance_scale: float | Sequence[float],
        null_text: torch.Tensor | None = None,
        num_steps: int = 50,
        seconds: float = 10.0,
        seconds_start: float = 0.0,
        decode: bool = True,
    ) -> SampleOutput:
        batch = len(prompts)
        if len(seeds) != batch:
            raise ValueError("one seed per prompt")
        scale = torch.as_tensor(guidance_scale, dtype=torch.float32).flatten()
        scale = scale.expand(batch) if scale.numel() == 1 else scale
        use_cfg = bool((scale > 1).any())

        cond_states, global_states = self.conditioning(prompts, seconds_start, seconds)
        if use_cfg:
            if null_text is None:
                null_text = torch.zeros(self.text_shape, dtype=cond_states.dtype)
            uncond_states = build_uncond(null_text.to(cond_states), batch)
            encoder_states = torch.cat([uncond_states, cond_states], dim=0)
            global_states = torch.cat([global_states, global_states], dim=0)
        else:
            encoder_states = cond_states

        self.scheduler.set_timesteps(num_steps, device=self.device)
        generators = [torch.Generator("cpu").manual_seed(int(s)) for s in seeds]
        latents = self.initial_noise(generators) * self.scheduler.init_noise_sigma
        rotary = self.rotary(latents.shape[2], global_states.shape[1])
        scale = scale.to(self.device)
        norms = []

        for t in self.scheduler.timesteps:
            model_in = torch.cat([latents] * 2) if use_cfg else latents
            model_in = self.scheduler.scale_model_input(model_in, t)
            out = self.predict(model_in, t, encoder_states, global_states, rotary)
            if use_cfg:
                uncond, cond = out.chunk(2)
                norms.append(guidance_rms(uncond, cond))
                out = combine_cfg(uncond, cond, scale)
            latents = self.scheduler.step(out, t, latents, generator=generators).prev_sample

        audio = latents
        if decode:
            start, end = int(seconds_start * self.sample_rate), int(seconds * self.sample_rate)
            # One clip at a time: the VAE decodes the full 47 s window and runs out of memory
            # on a 16 GB T4 for batches > 4.
            audio = torch.cat(
                [self.vae.decode(z.unsqueeze(0)).sample[:, :, start:end] for z in latents]
            )
        return SampleOutput(
            audio=audio,
            latents=latents,
            guidance_rms=torch.stack(norms, dim=1).cpu() if norms else None,
            sample_rate=self.sample_rate,
        )


def resolve_dtype(dtype: str) -> torch.dtype:
    if dtype != "auto":
        return getattr(torch, dtype)
    if not torch.cuda.is_available():
        return torch.float32
    name = torch.cuda.get_device_name(0)

    if "GTX 16" in name:
        return torch.float32
    return torch.float16


def _needs_offload(offload: str) -> bool:
    if offload in ("on", "off"):
        return offload == "on"
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    return total < 12
