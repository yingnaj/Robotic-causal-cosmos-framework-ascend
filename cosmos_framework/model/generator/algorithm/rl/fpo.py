# SPDX-License-Identifier: OpenMDW-1.1
"""FPO / FPO++ surrogates (arXiv:2602.02481, equations 8--12).

CFM differences are surrogate log ratios, never action log probabilities or KL.
The two switches allow vanilla FPO and per-probe PPO ablations of FPO++.
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FPOConfig:
    clip_epsilon: float = 0.2
    ratio_scale: float = 1.0
    per_probe_ratio: bool = True
    asymmetric: bool = True
    max_abs_log_ratio: float = 10.0

    def __post_init__(self):
        if not math.isfinite(self.clip_epsilon) or not 0 < self.clip_epsilon < 1:
            raise ValueError("clip_epsilon must be finite and in (0, 1)")
        if not math.isfinite(self.ratio_scale) or self.ratio_scale <= 0:
            raise ValueError("ratio_scale must be finite and positive")
        if not math.isfinite(self.max_abs_log_ratio) or not 0 < self.max_abs_log_ratio <= 20:
            raise ValueError("max_abs_log_ratio must be finite and in (0, 20]")


def fpo_loss(
    current_cfm: torch.Tensor,
    old_cfm: torch.Tensor,
    advantages: torch.Tensor,
    *,
    config: FPOConfig = FPOConfig(),
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Minimization loss for CFM [chunks, probes] and advantages [chunks].

    Each chunk has equal weight and probes are averaged within a chunk. Old CFM
    and advantages are detached. Invalid chunks are excluded before arithmetic.
    An STE bound on log ratios prevents exponential overflow while retaining the
    SPO restoring gradient; saturation is reported and should normally be zero.
    """
    if current_cfm.ndim != 2 or old_cfm.shape != current_cfm.shape or current_cfm.shape[1] < 1:
        raise ValueError("current_cfm and old_cfm must have matching [chunks, probes] shapes")
    if advantages.shape != current_cfm.shape[:1]:
        raise ValueError("advantages must have shape [chunks]")
    if valid is None:
        valid = torch.ones_like(advantages, dtype=torch.bool)
    if valid.dtype != torch.bool or valid.shape != advantages.shape or not valid.any():
        raise ValueError("valid must be a boolean [chunks] mask with at least one valid chunk")
    if any(x.device != current_cfm.device for x in (old_cfm, advantages, valid)):
        raise ValueError("CFM, advantages and valid must be on the same device")
    current = current_cfm[valid].float()
    old = old_cfm.detach()[valid].float()
    advantage = advantages.detach()[valid].float()[:, None]
    if not all(torch.isfinite(x).all() for x in (current, old, advantage)):
        raise ValueError("Valid CFM and advantages must be finite")
    if (current < 0).any() or (old < 0).any():
        raise ValueError("CFM losses must be nonnegative")
    log_ratio = config.ratio_scale * (old - current)
    if not torch.isfinite(log_ratio).all():
        raise ValueError("Scaled CFM difference overflowed; reduce ratio_scale or check CFM")
    if not config.per_probe_ratio:
        log_ratio = log_ratio.mean(dim=1, keepdim=True)
    bounded = log_ratio.clamp(-config.max_abs_log_ratio, config.max_abs_log_ratio)
    ratio = (bounded.detach() + (log_ratio - log_ratio.detach())).exp()
    clipped = ratio.clamp(1 - config.clip_epsilon, 1 + config.clip_epsilon)
    ppo = torch.minimum(ratio * advantage, clipped * advantage)
    if config.asymmetric:
        spo = ratio * advantage - advantage.abs() * (ratio - 1).square() / (2 * config.clip_epsilon)
        objective = torch.where(advantage >= 0, ppo, spo)
    else:
        objective = ppo
    loss = -objective.mean()
    metrics = {
        "fpo/loss": loss.detach(),
        "fpo/current_cfm": current.detach().mean(),
        "fpo/old_cfm": old.mean(),
        "fpo/surrogate_ratio": ratio.detach().mean(),
        "fpo/surrogate_clip_fraction": ((ratio.detach() - 1).abs() > config.clip_epsilon).float().mean(),
        "fpo/log_ratio_saturation": (log_ratio.detach().abs() > config.max_abs_log_ratio).float().mean(),
        "fpo/negative_advantage_fraction": (advantage < 0).float().mean(),
    }
    return loss, metrics


def discounted_chunk_returns(rewards, discounts, *, valid=None, bootstrap=None):
    """Return-to-go along the final dimension, using explicit chunk discounts.

    For a chunk executing h controls use gamma**h, or use a fixed gamma for a
    chunk-time objective. Set discount=0 at true terminals; truncations may use
    a bootstrap value. Padding must be a contiguous suffix of each trajectory.
    """
    if rewards.ndim < 1 or rewards.shape != discounts.shape or rewards.shape[-1] < 1:
        raise ValueError("rewards and discounts must have matching nonempty [..., chunks] shapes")
    if valid is None:
        valid = torch.ones_like(rewards, dtype=torch.bool)
    if valid.shape != rewards.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean mask matching rewards")
    if ((~valid[..., :-1]) & valid[..., 1:]).any():
        raise ValueError("Padding must be a contiguous suffix")
    r = rewards.detach().float().masked_fill(~valid, 0)
    d = discounts.detach().float().masked_fill(~valid, 0)
    if not torch.isfinite(r).all() or not torch.isfinite(d).all() or ((d < 0) | (d > 1)).any():
        raise ValueError("Rewards must be finite and discounts must be in [0, 1]")
    carry = torch.zeros_like(r[..., 0]) if bootstrap is None else bootstrap.detach().float()
    if carry.shape != r.shape[:-1] or not torch.isfinite(carry).all():
        raise ValueError("bootstrap must be finite with shape rewards.shape[:-1]")
    result = torch.zeros_like(r)
    for k in range(r.shape[-1] - 1, -1, -1):
        carry = torch.where(valid[..., k], r[..., k] + d[..., k] * carry, carry)
        result[..., k] = torch.where(valid[..., k], carry, 0)
    return result


def group_relative_advantages(returns, *, valid=None, epsilon=1e-6):
    """Normalize [reset_groups, trajectories, chunks] over trajectories.

    Groups must share the same reset and task. Equal-return and singleton groups
    give zero advantages; missing/padded chunks do not affect group statistics.
    """
    if returns.ndim != 3 or not all(returns.shape):
        raise ValueError("returns must have nonempty [reset_groups, trajectories, chunks] shape")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if valid is None:
        valid = torch.ones_like(returns, dtype=torch.bool)
    if valid.shape != returns.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean mask matching returns")
    values = returns.detach().float().masked_fill(~valid, 0)
    if not torch.isfinite(values).all():
        raise ValueError("Valid returns must be finite")
    count = valid.sum(dim=1, keepdim=True).clamp_min(1)
    mean = values.sum(dim=1, keepdim=True) / count
    centered = (values - mean).masked_fill(~valid, 0)
    std = (centered.square().sum(dim=1, keepdim=True) / count).sqrt()
    return centered / (std + epsilon)
