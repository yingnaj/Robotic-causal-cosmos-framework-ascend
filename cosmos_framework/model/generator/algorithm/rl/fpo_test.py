# SPDX-License-Identifier: OpenMDW-1.1
"""Numerical objective, trust-region gradients and chunk credit assignment."""

import math

import pytest
import torch

from cosmos_framework.model.generator.algorithm.rl.fpo import (
    FPOConfig,
    discounted_chunk_returns,
    fpo_loss,
    group_relative_advantages,
)


def test_aspo_matches_scalar_reference_and_detaches_behavior():
    current = torch.tensor([[0.7, 1.4], [1.5, 0.8]], requires_grad=True)
    old = torch.ones_like(current, requires_grad=True)
    advantage = torch.tensor([2.0, -3.0], requires_grad=True)
    loss, metrics = fpo_loss(current, old, advantage)
    expected = []
    for row, a in zip(current.detach().tolist(), advantage.detach().tolist(), strict=True):
        for c in row:
            ratio = math.exp(1 - c)
            if a >= 0:
                value = min(ratio * a, max(0.8, min(1.2, ratio)) * a)
            else:
                value = ratio * a - abs(a) * (ratio - 1) ** 2 / 0.4
            expected.append(value)
    torch.testing.assert_close(loss, torch.tensor(-sum(expected) / len(expected)))
    loss.backward()
    assert torch.isfinite(current.grad).all()
    assert old.grad is None and advantage.grad is None
    assert not any(value.requires_grad for value in metrics.values())
    assert not any("kl" in key or "entropy" in key for key in metrics)


@pytest.mark.parametrize("advantage", [-1.0, 1.0])
def test_initial_ratio_and_gradient_equal_vanilla_fpo(advantage):
    gradients = []
    for per_probe in (False, True):
        current = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        loss, metrics = fpo_loss(
            current,
            current.detach().clone(),
            torch.tensor([advantage]),
            config=FPOConfig(per_probe_ratio=per_probe),
        )
        assert metrics["fpo/surrogate_ratio"] == 1
        loss.backward()
        gradients.append(current.grad)
    torch.testing.assert_close(*gradients)
    torch.testing.assert_close(gradients[0], torch.full((1, 3), advantage / 3))


def test_per_probe_clipping_does_not_cancel_opposite_probe_changes():
    current = torch.tensor([[0.5, 1.5]])
    old = torch.ones_like(current)
    per_probe, _ = fpo_loss(current, old, torch.ones(1), config=FPOConfig(asymmetric=False))
    vanilla, _ = fpo_loss(current, old, torch.ones(1), config=FPOConfig(per_probe_ratio=False, asymmetric=False))
    assert vanilla == -1
    torch.testing.assert_close(per_probe, torch.tensor(-(1.2 + math.exp(-0.5)) / 2))


def test_negative_spo_restores_ratio_where_ppo_has_zero_gradient():
    gradients = []
    for asymmetric in (False, True):
        current = torch.tensor([[2.0]], requires_grad=True)
        loss, _ = fpo_loss(current, torch.ones(1, 1), -torch.ones(1), config=FPOConfig(asymmetric=asymmetric))
        loss.backward()
        gradients.append(current.grad.item())
    assert gradients[0] == 0
    assert gradients[1] > 0  # Descent lowers CFM, pulling exp(old-current) back up.


def test_mask_excludes_nonfinite_padding_and_zero_advantage_has_no_gradient():
    current = torch.tensor([[1.0, 2.0], [float("nan"), float("inf")]], requires_grad=True)
    old = current.detach().clone()
    loss, _ = fpo_loss(current, old, torch.tensor([0.0, float("nan")]), valid=torch.tensor([True, False]))
    assert loss == 0
    loss.backward()
    assert current.grad.eq(0).all()


@pytest.mark.parametrize("cfm", [100.0, 1e20])
def test_extreme_ratio_reports_saturation_and_keeps_restoring_gradient(cfm):
    current = torch.tensor([[cfm]], requires_grad=True)
    loss, metrics = fpo_loss(current, torch.zeros(1, 1), -torch.ones(1))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(current.grad).all()
    assert metrics["fpo/log_ratio_saturation"] == 1
    torch.testing.assert_close(metrics["fpo/surrogate_ratio"], torch.tensor(math.exp(-10)))
    assert current.grad.item() > 0


@pytest.mark.parametrize("kwargs", [{"clip_epsilon": 0}, {"ratio_scale": -1}, {"max_abs_log_ratio": 100}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        FPOConfig(**kwargs)


def test_returns_respect_terminals_padding_and_truncation_bootstrap():
    rewards = torch.tensor([[1.0, 2.0, float("nan")], [1.0, 2.0, 3.0]])
    discounts = torch.tensor([[0.5, 0.0, float("nan")], [0.5, 0.5, 0.5]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    result = discounted_chunk_returns(rewards, discounts, valid=valid, bootstrap=torch.tensor([99.0, 4.0]))
    torch.testing.assert_close(result, torch.tensor([[2.0, 2.0, 0.0], [3.25, 4.5, 5.0]]))
    with pytest.raises(ValueError, match="suffix"):
        discounted_chunk_returns(rewards, discounts, valid=torch.tensor([[True, False, True], [True, True, True]]))


def test_group_advantages_use_same_position_and_exclude_missing_chunks():
    returns = torch.tensor([[[1.0, 5.0], [3.0, float("nan")]], [[7.0, 4.0], [7.0, 4.0]]])
    valid = torch.isfinite(returns)
    advantages = group_relative_advantages(returns, valid=valid)
    torch.testing.assert_close(advantages, torch.tensor([[[-1.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]))
