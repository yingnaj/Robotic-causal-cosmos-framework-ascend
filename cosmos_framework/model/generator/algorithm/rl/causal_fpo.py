# SPDX-License-Identifier: OpenMDW-1.1
"""Causal Cosmos3 rollout capture and differentiable action-CFM replay.

The native joint video/action sampler is unchanged. One record is one causal
block, with observed history and measured state copied from the sampling pack.
Replay keeps joint video inputs and CFG, but reduces error over valid actions.
"""

import copy
import math
from dataclasses import dataclass, fields, is_dataclass, replace

import torch

from cosmos_framework.data.generator.action.block_state import prediction_block_span
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.model.generator.algorithm.rl.fpo import FPOConfig, fpo_loss


def _snapshot(value, device=None):
    """Copy detached tensor trees, preserving packed metadata and tensor dtypes."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device or value.device).clone()
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{f.name: _snapshot(getattr(value, f.name), device) for f in fields(value) if f.init})
    if isinstance(value, dict):
        return {k: _snapshot(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_snapshot(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(v, device) for v in value)
    return copy.deepcopy(value)


@dataclass(frozen=True)
class CausalFPORollout:
    branches: dict[str, PackedSequence]
    endpoint_vision: torch.Tensor
    endpoint_action: torch.Tensor
    vision_mask: torch.Tensor
    action_mask: torch.Tensor
    current_block: int
    guidance: float
    guidance_interval: tuple[float, float] | None
    normalize_cfg: bool
    timestep_scale: float
    num_steps: int
    shift: float
    policy_version: str

    @classmethod
    def from_trace(cls, trace: dict, *, policy_version: str, storage_device="cpu"):
        """Own a snapshot independent of subsequent sampling/session mutations."""
        if not isinstance(policy_version, str) or not policy_version:
            raise ValueError("policy_version must be a nonempty string")
        with torch.inference_mode(False):
            record = cls(**_snapshot(trace, storage_device), policy_version=policy_version)
        record.validate()
        return record

    def validate(self):
        v, a = self.endpoint_vision, self.endpoint_action
        if v.ndim != 5 or v.shape[0] != 1 or a.ndim != 2:
            raise ValueError("Endpoints must be vision [1,C,T,H,W] and action [N,D]")
        for endpoint, mask in ((v, self.vision_mask), (a, self.action_mask)):
            if mask.shape != endpoint.shape or mask.dtype != torch.bool or endpoint.requires_grad:
                raise ValueError("Endpoints must be detached with matching boolean generation masks")
            if not torch.isfinite(endpoint).all():
                raise ValueError("Endpoints must be finite")
        if not self.action_mask.any() or not self.vision_mask.any():
            raise ValueError("Causal FPO requires joint video/action generation")
        if not math.isfinite(self.guidance) or self.guidance < 0:
            raise ValueError("guidance must be finite and nonnegative")
        if not math.isfinite(self.timestep_scale) or self.timestep_scale <= 0:
            raise ValueError("timestep_scale must be finite and positive")
        if self.guidance_interval is not None:
            lo, hi = self.guidance_interval
            if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
                raise ValueError("guidance_interval must have finite increasing bounds in model timestep units")
        expected_branches = {"cond"} if self.guidance == 1 else {"cond", "uncond"}
        if set(self.branches) != expected_branches:
            raise ValueError("Recorded CFG branches do not match guidance")
        for packed in self.branches.values():
            if len(packed.sample_lens) != 1 or packed.teacher_forcing is None or packed.sound is not None:
                raise ValueError("Each trace must contain a single causal action sample without sound")
            if packed.vision is None or packed.action is None or not packed.causal_action_metadata:
                raise ValueError("Trace requires vision, action and measured-state metadata")
            if len(packed.vision.tokens) != 1 or len(packed.action.tokens) != 1:
                raise ValueError("Trace must contain one vision and one action payload")
            if packed.vision.tokens[0].shape != v.shape or packed.action.tokens[0].shape != a.shape:
                raise ValueError("Packed payloads must match endpoint shapes")
            m = packed.causal_action_metadata[0]
            if not 0 <= self.current_block < len(m.states) or not m.state_mask[self.current_block].any():
                raise ValueError("Current block needs measured state")
            start, end = prediction_block_span(self.current_block, m.block_size)
            if end > v.shape[2]:
                raise ValueError("Causal FPO requires a complete block")
            ids = m.action_frame_ids.to(a.device)
            current_actions = (ids >= start) & (ids < end)
            if packed.vision.condition_mask[0][start:end].bool().any() or (
                packed.action.condition_mask[0].to(a.device)[current_actions].bool().any()
            ):
                raise ValueError("Joint policy rollouts cannot condition on current-block future video or actions")
            expected_v = torch.zeros_like(self.vision_mask)
            expected_v[:, :, start:end] = True
            expected_v &= ~packed.vision.condition_mask[0].to(v.device).bool()
            expected_a = current_actions[:, None] & m.action_mask.to(a.device)
            expected_a &= ~packed.action.condition_mask[0].to(a.device).bool()
            if not torch.equal(expected_v, self.vision_mask) or not torch.equal(expected_a, self.action_mask):
                raise ValueError("Generation masks must select the current block's valid coordinates")
            if not packed.vision.condition_mask[0][:start].bool().all():
                raise ValueError("Visual history must consist of observed conditions")


@dataclass(frozen=True)
class CausalFPOProbes:
    sigmas: torch.Tensor
    vision_noise: torch.Tensor
    action_noise: torch.Tensor

    @classmethod
    def sample(cls, rollout: CausalFPORollout, *, num_probes: int, seed: int):
        """Independent CPU RNG; persist actual probes rather than replaying a seed.

        Uniform sigma in [0.005, 0.995] is independent of the rollout's shifted
        UniPC step schedule. Both modalities use the same sigma for each probe.
        """
        if not isinstance(num_probes, int) or num_probes < 1:
            raise ValueError("num_probes must be a positive integer")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return cls(
            sigmas=0.005 + 0.99 * torch.rand(num_probes, generator=generator),
            vision_noise=torch.randn((num_probes, *rollout.endpoint_vision.shape), generator=generator),
            action_noise=torch.randn((num_probes, *rollout.endpoint_action.shape), generator=generator),
        )

    def validate(self, rollout):
        if self.sigmas.ndim != 1 or self.sigmas.numel() < 1:
            raise ValueError("sigmas must have nonempty [probes] shape")
        q = len(self.sigmas)
        if self.vision_noise.shape != (q, *rollout.endpoint_vision.shape):
            raise ValueError("vision_noise shape must be [probes, *endpoint_vision.shape]")
        if self.action_noise.shape != (q, *rollout.endpoint_action.shape):
            raise ValueError("action_noise shape must be [probes, *endpoint_action.shape]")
        if not all(torch.isfinite(x).all() for x in (self.sigmas, self.vision_noise, self.action_noise)):
            raise ValueError("Probes must be finite")
        if ((self.sigmas <= 0) | (self.sigmas >= 1)).any():
            raise ValueError("Probe sigmas must be strictly between zero and one")


@dataclass(frozen=True)
class CausalFPOExample:
    rollout: CausalFPORollout
    probes: CausalFPOProbes
    old_cfm: torch.Tensor


def _require_eval(net):
    if any(module.training for module in net.modules()):
        raise ValueError("FPO sampling and replay require net.eval(); eval mode still permits gradients")


def sample_causal_fpo_rollout(model, data_batch, *, policy_version, seed, storage_device="cpu", **sampling_options):
    """Sample one native UniPC block and return (ordinary_outputs, rollout).

    Caller owns environment actions, reward, history commits and episode reset.
    The ordinary result retains its original normalized action representation.
    """
    net = sampling_options.get("net")
    net = model.net if net is None else net
    _require_eval(net)
    if data_batch.get("causal_action_preview", False):
        raise ValueError("Ground-truth future-state previews cannot be used for policy rollouts")
    parallel = getattr(model, "parallel_dims", None)
    if parallel is not None and (parallel.cp_enabled or parallel.cfgp_enabled):
        raise ValueError("Causal FPO v1 does not support context/CFG parallelism")
    result = model.generate_samples_from_batch(
        dict(data_batch, causal_action_return_trace=True), seed=seed, **sampling_options
    )
    result = dict(result)
    trace = result.pop("sampling_trace")
    rollout = CausalFPORollout.from_trace(trace, policy_version=policy_version, storage_device=storage_device)
    return result, rollout


def compute_action_cfm(model, rollout: CausalFPORollout, probes: CausalFPOProbes, *, net=None):
    """Differentiable FP32 per-probe [Q] masked-mean CFM, with no sampler graph.

    Replays the effective guided velocity, including the joint video/action norm
    when normalize_cfg=True. No inference attention cache is installed. Inputs
    are detached; freezing parameter weights does not disable input gradients.
    """
    net = model.net if net is None else net
    _require_eval(net)
    rollout.validate()
    probes.validate(rollout)
    device = torch.device(model.tensor_kwargs["device"])
    dtype = model.tensor_kwargs["dtype"]
    record = _snapshot(rollout, device)
    v, a = record.endpoint_vision.float(), record.endpoint_action.float()
    vm, am = record.vision_mask, record.action_mask
    losses = []
    for q in range(len(probes.sigmas)):
        sigma = probes.sigmas[q].detach().to(device=device, dtype=torch.float32)
        eps_v = probes.vision_noise[q].detach().to(device=device, dtype=torch.float32)
        eps_a = probes.action_noise[q].detach().to(device=device, dtype=torch.float32)
        noisy_v = torch.where(vm, (1 - sigma) * v + sigma * eps_v, v)
        noisy_a = torch.where(am, (1 - sigma) * a + sigma * eps_a, a)
        timestep = sigma * record.timestep_scale

        def branch(name):
            # Each graph owns its pack: reusing a mutable pack across probes can
            # change activation-checkpoint recomputation during backward.
            packed = _snapshot(record.branches[name])
            packed.vision = replace(
                packed.vision, tokens=[noisy_v.to(dtype)], timesteps=torch.ones_like(packed.vision.timesteps) * timestep
            )
            packed.action = replace(
                packed.action, tokens=[noisy_a.to(dtype)], timesteps=torch.ones_like(packed.action.timesteps) * timestep
            )
            out = model.denoise(net=net, data_batch_packed=packed)
            # Match the sampler's concatenate/mask dtype before float32 CFM.
            return torch.cat([out["preds_vision"][0].reshape(-1), out["preds_action"][0].reshape(-1)]).masked_fill(
                ~torch.cat([vm.flatten(), am.flatten()]), 0
            )

        conditional = branch("cond")
        use_cfg = record.guidance != 1.0
        if record.guidance_interval is not None:
            use_cfg = use_cfg and record.guidance_interval[0] < float(timestep) < record.guidance_interval[1]
        velocity = conditional
        if use_cfg:
            unconditional = branch("uncond")
            velocity = unconditional + record.guidance * (conditional - unconditional)
            if record.normalize_cfg:
                velocity = velocity * (conditional.norm() / (velocity.norm() + 1e-8)).clamp(0, 1)
        action_velocity = velocity[v.numel() :].reshape_as(a).float()
        # Boolean selection excludes all padding/unused embodiment dimensions.
        losses.append((action_velocity[am] - (eps_a - a)[am]).square().mean())
    return torch.stack(losses)


@torch.no_grad()
def prepare_fpo_example(model, rollout, *, num_probes, probe_seed, behavior_version, net=None):
    """Compute old CFM BEFORE changing behavior weights; cache on CPU.

    The version check prevents accidental batch mixing; the caller must actually
    keep those weights unchanged until every old loss has been recorded.
    """
    if rollout.policy_version != behavior_version:
        raise ValueError("Rollout policy_version differs from behavior_version")
    probes = CausalFPOProbes.sample(rollout, num_probes=num_probes, seed=probe_seed)
    old_cfm = compute_action_cfm(model, rollout, probes, net=net).detach().cpu().clone()
    return CausalFPOExample(rollout, probes, old_cfm)


def causal_fpo_loss(model, examples, advantages, *, config=FPOConfig(), net=None):
    """Build one update loss; optimizer/backward/distributed ownership is external."""
    if not examples:
        raise ValueError("At least one FPO example is required")
    if len({x.rollout.policy_version for x in examples}) != 1:
        raise ValueError("An update batch must use one behavior policy version")
    if len({len(x.probes.sigmas) for x in examples}) != 1:
        raise ValueError("An update batch must use the same probe count")
    current = torch.stack([compute_action_cfm(model, x.rollout, x.probes, net=net) for x in examples])
    old = torch.stack([x.old_cfm for x in examples]).to(current.device)
    return fpo_loss(current, old, advantages.to(current.device), config=config)


def configure_fpo_trainable_parameters(net, *, train_state=False):
    """Select native action interfaces before constructing an optimizer/FSDP.

    No new weights or checkpoint keys are introduced. Shared generation weights
    stay frozen; state interfaces are an explicit opt-in. Return selected names.
    """
    roots = {"action2llm", "llm2action", "action_modality_embed"}
    if train_state:
        roots |= {"state2llm", "state_modality_embed"}
    named = list(net.named_parameters())
    found = {name.split(".")[0] for name, _ in named}
    if not roots <= found:
        raise ValueError(f"Expected unwrapped causal network interfaces; missing {sorted(roots - found)}")
    selected = []
    for name, parameter in named:
        enabled = name.split(".")[0] in roots
        parameter.requires_grad_(enabled)
        parameter.grad = None
        if enabled:
            selected.append(name)
    net.eval()
    return tuple(selected)
