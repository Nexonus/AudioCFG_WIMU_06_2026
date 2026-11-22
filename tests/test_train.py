import math

import numpy as np
import pytest
import torch

from audiocfg.analysis.bootstrap import paired_ci
from audiocfg.train.loop import lr_at, train_null
from audiocfg.train.objective import draws, noisy_and_target
from audiocfg.train.probe import eval_losses, null_states

SAO_SCHEDULER = dict(
    sigma_min=0.3,
    sigma_max=500,
    sigma_data=1.0,
    prediction_type="v_prediction",
    sigma_schedule="exponential",
    solver_order=2,
    solver_type="midpoint",
    final_sigmas_type="zero",
)


def test_objective_matches_diffusers_edm_preconditioning():
    from diffusers import CosineDPMSolverMultistepScheduler

    sched = CosineDPMSolverMultistepScheduler(**SAO_SCHEDULER)
    x0, eps = (
        torch.randn(3, 64, 16, dtype=torch.float64),
        torch.randn(3, 64, 16, dtype=torch.float64),
    )
    t = torch.tensor([0.2, 0.5, 0.95], dtype=torch.float64)
    x_t, v = noisy_and_target(x0, t, eps)
    for i in range(3):
        sigma = torch.tan(t[i] * math.pi / 2)
        x_edm = x0[i] + sigma * eps[i]
        assert torch.allclose(sched.precondition_inputs(x_edm, sigma), x_t[i])
        assert torch.allclose(sched.precondition_noise(sigma), t[i])
        assert torch.allclose(sched.precondition_outputs(x_edm, v[i], sigma), x0[i])


def test_draws_depend_only_on_item_id():
    t1, n1 = draws(2026, "val", ["a", "b"], 2, (4, 8))
    t2, n2 = draws(2026, "val", ["b"], 2, (4, 8))
    assert torch.equal(t1[1], t2[0]) and torch.equal(n1[1], n2[0])
    assert not torch.equal(n1[0, 0], n1[0, 1])
    assert ((t1 > 0) & (t1 < 1)).all()


def test_lr_schedule():
    assert lr_at(0, 100, 1.0, 10) == pytest.approx(0.1)
    assert lr_at(10, 100, 1.0, 10) == pytest.approx(1.0)
    assert lr_at(100, 100, 1.0, 10) == pytest.approx(0.0)


def test_paired_ci():
    ci = paired_ci(np.full(50, 0.3) + np.random.default_rng(0).normal(0, 0.01, 50))
    assert ci["lo"] < 0.3 < ci["hi"] and ci["lo"] > 0.29


class _FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Linear(4, 4)
        self.frozen.requires_grad_(False)

    def enable_gradient_checkpointing(self):
        pass

    def disable_gradient_checkpointing(self):
        pass


class _FakeSAO:

    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self):
        self.transformer = _FakeTransformer()

    def encode_timing(self, batch, start, end):
        return torch.zeros(batch, 2, 32), torch.zeros(batch, 1, 64)

    def v_prediction(self, x_t, t, enc, glob):
        return enc[:, 0, :].reshape(x_t.shape) + 0 * self.transformer.frozen.weight.sum()


def test_train_null_only_moves_null_and_lowers_loss():
    sao = _FakeSAO()
    frozen_before = sao.transformer.frozen.weight.clone()
    latents = torch.ones(16, 4, 8)
    seconds = torch.full((16,), 30.0)
    init = torch.zeros(5, 32)
    t, noise = draws(1, "val", list(range(16)), 2, (4, 8))
    before = eval_losses(sao, latents, seconds, t, noise, null_states(init)).mean()
    null, hist = train_null(
        sao,
        latents,
        seconds,
        init,
        steps=200,
        batch_size=8,
        lr=0.05,
        master_seed=1,
        warmup=10,
        log=lambda row: None,
    )
    after = eval_losses(sao, latents, seconds, t, noise, null_states(null)).mean()
    assert after < before
    assert torch.equal(sao.transformer.frozen.weight, frozen_before)
    assert null.shape == init.shape and len(hist) == 200
    assert torch.count_nonzero(null[1:]) == 0 
