import pytest
import torch

from audiocfg.models.sao import build_uncond, combine_cfg, guidance_rms
from audiocfg.nulls.variants import load_null, resolve_null, save_null


def test_build_uncond_zero_timing_slots():
    null = torch.randn(128, 768)
    out = build_uncond(null, batch_size=3)
    assert out.shape == (3, 130, 768)
    assert torch.equal(out[:, :128], null.expand(3, -1, -1))
    assert torch.count_nonzero(out[:, 128:]) == 0


def test_build_uncond_per_sample_null_and_batch_check():
    nulls = torch.randn(2, 128, 768)
    assert torch.equal(build_uncond(nulls, 2)[1, :128], nulls[1])
    with pytest.raises(ValueError):
        build_uncond(nulls, 3)


def test_combine_cfg_matches_pipeline_formula_per_sample():
    uncond, cond = torch.randn(3, 64, 10), torch.randn(3, 64, 10)
    scale = torch.tensor([1.0, 3.0, 7.0])
    out = combine_cfg(uncond, cond, scale)
    assert torch.equal(out[0], cond[0])
    for i, g in enumerate(scale.tolist()[1:], start=1):
        assert torch.equal(out[i], uncond[i] + g * (cond[i] - uncond[i]))


def test_guidance_rms():
    uncond = torch.zeros(2, 4, 5)
    cond = torch.ones(2, 4, 5) * torch.tensor([1.0, 2.0]).view(2, 1, 1)
    assert torch.allclose(guidance_rms(uncond, cond), torch.tensor([1.0, 2.0]))


def test_null_roundtrip_and_resolve(tmp_path):
    null = torch.randn(128, 768)
    path = save_null(tmp_path / "n.safetensors", null, {"init": "zero"})
    assert torch.equal(load_null(path), null)
    assert torch.equal(resolve_null(str(path)), null)
    assert resolve_null("zero") is None
    with pytest.raises(ValueError):
        resolve_null("nope")


@pytest.mark.gpu
def test_zero_null_equals_stock_pipeline():
    from audiocfg.models.sao import StableAudio

    sao = StableAudio.load()
    prompt, seed, steps, g, seconds = "rain on a tin roof", 1234, 8, 7.0, 5.0

    gen = torch.Generator("cpu").manual_seed(seed)
    stock = sao.pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=g,
        audio_end_in_s=seconds,
        generator=[gen],
        output_type="latent",
    ).audios
    ours = sao.sample([prompt], [seed], g, null_text=None, num_steps=steps, seconds=seconds)
    zeros = sao.sample(
        [prompt], [seed], g, null_text=torch.zeros(sao.text_shape), num_steps=steps, seconds=seconds
    )
    again = sao.sample([prompt], [seed], g, null_text=None, num_steps=steps, seconds=seconds)
    assert torch.equal(ours.latents, stock)
    assert torch.equal(zeros.latents, stock)
    assert torch.equal(again.latents, ours.latents)
    assert ours.guidance_rms.shape == (1, steps)
