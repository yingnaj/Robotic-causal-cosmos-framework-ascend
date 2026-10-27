# SPDX-License-Identifier: OpenMDW-1.1
"""CPU integration with native UniPC, causal packing and action/state interfaces.

Only the pretrained backbone and accelerator transfer are replaced. These
tests do not claim pretrained checkpoint or environment performance coverage.
"""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from cosmos_framework.data.generator.action.block_state import BlockStateMetadata
from cosmos_framework.data.generator.sequence_packing.causal_action import dense_action_mask
from cosmos_framework.data.generator.sequence_packing.modality import ModalityData
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.inference.causal_action.sampling import sample_prepared
from cosmos_framework.model.generator.algorithm.rl.causal_fpo import (
    CausalFPOProbes,
    causal_fpo_loss,
    compute_action_cfm,
    configure_fpo_trainable_parameters,
    prepare_fpo_example,
    sample_causal_fpo_rollout,
)
from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler, UniPCSamplerConfig
from cosmos_framework.model.generator.mot.causal_action_network import CausalActionNetwork
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


class TinyNetwork(CausalActionNetwork):
    """Production action encode/decode and a small dense causal attention layer."""

    def __init__(self):
        nn.Module.__init__(self)
        self.action_dim, self.hidden_size = 3, 8
        self.timestep_scale = 1.0
        self._create_action_interfaces()
        self._init_action_interfaces()
        self.action_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))
        self.vision_input = nn.Linear(1, self.hidden_size)
        self.vision_output = nn.Linear(self.hidden_size, 1)
        self.text_embedding = nn.Embedding(3, self.hidden_size)
        self.time_embedder = nn.Linear(1, self.hidden_size)

    def _embed_packed_timesteps(self, timesteps, packed_seq):
        return self.time_embedder(timesteps[:, None] / 1000)

    def forward(self, packed_seq, **kwargs):
        p = packed_seq
        x = torch.zeros(p.sequence_length, self.hidden_size)
        x[p.text_indexes] = self.text_embedding(p.text_ids)
        x[p.vision.sequence_indexes] = self.vision_input(p.vision.tokens[0].reshape(-1, 1))
        x[p.vision.mse_loss_indexes] += self._embed_packed_timesteps(p.vision.timesteps, p)
        x[p.teacher_forcing.layout.clean_token_indexes] = self.vision_input(
            p.teacher_forcing.clean_vision_tokens[0].reshape(-1, 1)
        )
        self._encode_action(p, x, torch.float32)
        q = p.teacher_forcing.layout.gen_query_indexes
        hidden = x.clone()
        hidden[q] = F.scaled_dot_product_attention(
            x[q][None, None], x[None, None], x[None, None], attn_mask=dense_action_mask(p.teacher_forcing.layout)
        )[0, 0]
        result = {"preds_vision": [self.vision_output(hidden[p.vision.sequence_indexes]).reshape(1, 5, 1, 1)]}
        self._decode_action(p, hidden, result)
        return result


def make_pack(text_id=1):
    t, n, d = 5, 8, 3
    vc = torch.tensor([1, 1, 1, 0, 0], dtype=torch.float32)[:, None, None]
    ac = torch.zeros(n, 1)
    vi, ai = torch.arange(1, 1 + t), torch.arange(1 + t, 1 + t + n)
    am = torch.ones(n, d, dtype=torch.bool)
    am[:, -1] = False
    meta = BlockStateMetadata(
        block_size=2,
        history_blocks=1,
        action_frame_ids=torch.arange(1, t).repeat_interleave(2),
        action_mask=am,
        states=torch.tensor([[0.1, 0.2, 0.0], [0.3, 0.4, 0.0]]),
        state_mask=torch.tensor([[True, True, False], [True, True, False]]),
        state_action_indexes=torch.tensor([0, 4]),
        anchors=torch.zeros(2, d),
        state_latent_indexes=torch.tensor([0, 2]),
    )
    vision = ModalityData(
        sequence_indexes=vi,
        timesteps=torch.zeros(2),
        mse_loss_indexes=vi[3:],
        token_shapes=[(t, 1, 1)],
        tokens=[torch.arange(t).float().reshape(1, 1, t, 1, 1) * vc],
        condition_mask=[vc],
        noisy_frame_indexes=[torch.tensor([3, 4])],
    )
    action = ModalityData(
        sequence_indexes=ai,
        timesteps=torch.zeros(n),
        mse_loss_indexes=ai,
        token_shapes=[(n,)],
        tokens=[torch.zeros(n, d)],
        condition_mask=[ac],
        noisy_frame_indexes=[torch.arange(n)],
        domain_id=[torch.zeros(1, dtype=torch.long)],
    )
    return PackedSequence(
        sample_lens=[1 + t + n],
        split_lens=[1, t + n],
        attn_modes=["causal", "full"],
        sequence_length=1 + t + n,
        text_ids=torch.tensor([text_id]),
        text_indexes=torch.tensor([0]),
        position_ids=torch.zeros(3, 1 + t + n),
        vision=vision,
        action=action,
        causal_action_metadata=[meta],
    )


class TinyModel:
    _update_inference_pack_template = OmniMoTModel._update_inference_pack_template
    _copy_timestep_to_template = OmniMoTModel._copy_timestep_to_template

    def __init__(self):
        self.net = TinyNetwork().eval()
        self.tensor_kwargs = dict(device="cpu", dtype=torch.float32)
        self.precision = torch.float32
        self.parallel_dims = None
        self.config = SimpleNamespace(
            teacher_forcing_dense_mode="grouped_tnd",
            rectified_flow_inference_config=SimpleNamespace(scheduler_type="unipc"),
        )
        self.sampler = UniPCSampler(UniPCSamplerConfig(), self.tensor_kwargs)
        self.calls = []

    def _pack_input_sequence(self, sequence_plans, text, data, timesteps, **kwargs):
        return make_pack(text[0][0])

    def denoise(self, *, net=None, data_batch_packed):
        p = data_batch_packed
        self.calls.append(copy.deepcopy(p))
        return (self.net if net is None else net)(p)

    def generate_samples_from_batch(self, batch, *, seed, **options):
        packed = make_pack()
        data = GenerationDataClean(
            batch_size=1,
            is_image_batch=False,
            x0_tokens_vision=packed.vision.tokens,
            x0_tokens_action=packed.action.tokens,
        )
        initial = torch.randn(29, generator=torch.Generator().manual_seed(seed[0]))
        reference = torch.cat([data.x0_tokens_vision[0].flatten(), data.x0_tokens_action[0].flatten()])
        condition = torch.cat([packed.vision.condition_mask[0].flatten(), torch.zeros(24)])
        return sample_prepared(
            self,
            net=None,
            sampler=None,
            guidance=options.get("guidance", 1.0),
            guidance_interval=options.get("guidance_interval"),
            velocity_postprocess_builder=None,
            num_steps=3,
            shift=1.0,
            sigma_max=1.0,
            skip_text_tokens_for_cfg=False,
            normalize_cfg=options.get("normalize_cfg", False),
            sequence_plans=[SimpleNamespace(causal_action_metadata=packed.causal_action_metadata[0])],
            gen_data_clean=data,
            cond_tokens=[[1]],
            uncond_tokens=[[2]],
            initial_noise=[initial],
            condition_reference=[reference],
            condition_mask=[condition],
            has_noisy_actions=True,
            causal_num_blocks=2,
            causal_block_size=2,
            causal_history_blocks=1,
            causal_use_kv_cache=False,
            data_batch=dict(batch, causal_action_current_block=1),
        )


@pytest.fixture
def model(monkeypatch):
    torch.manual_seed(42)
    # The production pack transfer is accelerator-specific; all test tensors are CPU.
    monkeypatch.setattr(PackedSequence, "to_cuda", lambda self: None)
    return TinyModel()


def capture(model, **options):
    return sample_causal_fpo_rollout(model, {}, policy_version="behavior-0", seed=[42], **options)


def test_native_trace_does_not_change_samples_and_owns_history(model):
    plain = model.generate_samples_from_batch({}, seed=[42])
    result, record = capture(model)
    torch.testing.assert_close(plain["action"][0], result["action"][0], rtol=0, atol=0)
    torch.testing.assert_close(plain["vision"][0], result["vision"][0], rtol=0, atol=0)
    assert "sampling_trace" not in plain and "sampling_trace" not in result
    assert not record.action_mask[:4].any() and record.action_mask[4:, :2].all()
    assert not record.action_mask[:, -1].any()
    # Clean stream was captured BEFORE sampled future latents entered history.
    assert record.branches["cond"].teacher_forcing.clean_vision_tokens[0][:, :, 3:].eq(0).all()
    before = record.endpoint_action.clone()
    result["action"][0].fill_(999)
    torch.testing.assert_close(before, record.endpoint_action)


@pytest.mark.parametrize(
    "guidance,normalize,interval",
    [
        (1.0, False, None),
        (2.0, False, None),
        (2.0, True, None),
        (2.0, True, (0, 10)),
    ],
)
def test_replay_matches_independent_guided_velocity(model, guidance, normalize, interval):
    _, record = capture(model, guidance=guidance, normalize_cfg=normalize, guidance_interval=interval)
    probes = CausalFPOProbes.sample(record, num_probes=2, seed=13)
    model.calls.clear()
    actual = compute_action_cfm(model, record, probes)
    expected = []
    call = 0
    active = torch.cat([record.vision_mask.flatten(), record.action_mask.flatten()])
    for q, sigma in enumerate(probes.sigmas):
        pack = model.calls[call]
        call += 1
        v, a = record.endpoint_vision, record.endpoint_action
        torch.testing.assert_close(pack.vision.tokens[0][~record.vision_mask], v[~record.vision_mask])
        torch.testing.assert_close(pack.action.tokens[0][~record.action_mask], a[~record.action_mask])
        torch.testing.assert_close(pack.action.timesteps, torch.full_like(pack.action.timesteps, sigma * 1000))
        torch.testing.assert_close(pack.vision.timesteps, torch.full_like(pack.vision.timesteps, sigma * 1000))
        torch.testing.assert_close(
            pack.action.tokens[0][record.action_mask],
            ((1 - sigma) * a + sigma * probes.action_noise[q])[record.action_mask],
        )

        def velocity(p):
            out = model.net(p)
            return torch.cat([out["preds_vision"][0].flatten(), out["preds_action"][0].flatten()]) * active

        conditional = velocity(pack)
        guided = conditional
        if guidance != 1 and (interval is None or interval[0] < sigma * 1000 < interval[1]):
            unconditional = velocity(model.calls[call])
            call += 1
            guided = unconditional + guidance * (conditional - unconditional)
            if normalize:
                guided = guided * (conditional.norm() / (guided.norm() + 1e-8)).clamp(0, 1)
        predicted = guided[v.numel() :].reshape_as(a)
        expected.append(((predicted - (probes.action_noise[q] - a))[record.action_mask]).square().mean())
    torch.testing.assert_close(actual, torch.stack(expected))
    assert call == len(model.calls)


def test_fixed_probes_replay_and_optimizer_only_update_selected_parameters(model):
    names = configure_fpo_trainable_parameters(model.net, train_state=True)
    _, record = capture(model)
    example = prepare_fpo_example(model, record, num_probes=3, probe_seed=7, behavior_version="behavior-0")
    torch.manual_seed(999)
    current = compute_action_cfm(model, record, example.probes)
    torch.testing.assert_close(current, example.old_cfm, rtol=0, atol=0)
    before = {name: p.detach().clone() for name, p in model.net.named_parameters()}
    optimizer = torch.optim.SGD([p for p in model.net.parameters() if p.requires_grad], lr=1e-3)
    loss, metrics = causal_fpo_loss(model, [example], torch.ones(1))
    assert metrics["fpo/surrogate_ratio"] == 1
    loss.backward()
    assert model.net.action2llm.weight.grad.abs().sum() > 0
    assert model.net.state2llm.weight.grad.abs().sum() > 0
    assert not model.net.llm2action.weight.grad[-1].any()
    assert not model.net.action2llm.weight.grad[:, -1].any()
    assert not model.net.state2llm.weight.grad[:, -1].any()
    optimizer.step()
    changed = {name for name, p in model.net.named_parameters() if not torch.equal(p, before[name])}
    assert changed and changed <= set(names)
    assert "vision_input.weight" not in changed
    assert compute_action_cfm(model, record, example.probes).mean() < example.old_cfm.mean()


def test_replay_uses_joint_video_and_current_state_but_no_future_clean_leak(model):
    _, record = capture(model)
    probes = CausalFPOProbes.sample(record, num_probes=1, seed=4)
    original = compute_action_cfm(model, record, probes)
    changed_video = replace(probes, vision_noise=probes.vision_noise + 5 * record.vision_mask)
    assert not torch.allclose(original, compute_action_cfm(model, record, changed_video))
    changed_state = copy.deepcopy(record)
    changed_state.branches["cond"].teacher_forcing.state_tokens[0][1, 0] += 5
    assert not torch.allclose(original, compute_action_cfm(model, changed_state, probes))
    future = copy.deepcopy(record)
    future.branches["cond"].teacher_forcing.clean_vision_tokens[0][:, :, 3:] = 999
    torch.testing.assert_close(original, compute_action_cfm(model, future, probes), rtol=0, atol=0)


def test_probe_invalid_dimensions_do_not_affect_loss(model):
    _, record = capture(model)
    probes = CausalFPOProbes.sample(record, num_probes=2, seed=7)
    altered = replace(probes, action_noise=probes.action_noise.masked_fill(~record.action_mask, 999))
    torch.testing.assert_close(compute_action_cfm(model, record, probes), compute_action_cfm(model, record, altered))


def test_record_serialization_and_version_checks(model, tmp_path):
    _, record = capture(model)
    example = prepare_fpo_example(model, record, num_probes=2, probe_seed=2, behavior_version="behavior-0")
    path = tmp_path / "trusted_local_rollout.pt"
    torch.save(example, path)
    restored = torch.load(path, weights_only=False)
    torch.testing.assert_close(compute_action_cfm(model, restored.rollout, restored.probes), example.old_cfm)
    with pytest.raises(ValueError, match="behavior_version"):
        prepare_fpo_example(model, record, num_probes=1, probe_seed=1, behavior_version="different")
    mixed = replace(example, rollout=replace(record, policy_version="different"))
    with pytest.raises(ValueError, match="one behavior"):
        causal_fpo_loss(model, [example, mixed], torch.ones(2))


def test_rejects_dropout_mode_preview_and_bad_mask(model):
    _, record = capture(model)
    probes = CausalFPOProbes.sample(record, num_probes=1, seed=1)
    with pytest.raises(ValueError, match="Generation masks"):
        compute_action_cfm(model, replace(record, action_mask=torch.ones_like(record.action_mask)), probes)
    leaked = copy.deepcopy(record)
    leaked.branches["cond"].vision.condition_mask[0][3] = 1
    with pytest.raises(ValueError, match="current-block future"):
        compute_action_cfm(model, leaked, probes)
    with pytest.raises(ValueError, match="previews"):
        sample_causal_fpo_rollout(model, {"causal_action_preview": True}, policy_version="v", seed=[1])
    model.net.train()
    with pytest.raises(ValueError, match="net.eval"):
        compute_action_cfm(model, record, probes)


def test_default_trainable_scope_preserves_checkpoint_keys_and_freezes_state(model):
    keys = set(model.net.state_dict())
    names = configure_fpo_trainable_parameters(model.net)
    assert "state2llm.weight" not in names and "action2llm.weight" in names
    assert keys == set(model.net.state_dict())
    assert not model.net.state2llm.weight.requires_grad


def test_checkpointed_multi_probe_backward_matches_eager(model, monkeypatch):
    from torch.utils.checkpoint import checkpoint

    configure_fpo_trainable_parameters(model.net, train_state=True)
    _, record = capture(model, guidance=2.0, normalize_cfg=True)
    example = prepare_fpo_example(model, record, num_probes=3, probe_seed=7, behavior_version="behavior-0")
    loss, _ = causal_fpo_loss(model, [example], torch.ones(1))
    loss.backward()
    expected = {name: p.grad.clone() for name, p in model.net.named_parameters() if p.grad is not None}
    model.net.zero_grad(set_to_none=True)
    original = model.net.forward
    monkeypatch.setattr(model.net, "forward", lambda p: checkpoint(original, p, use_reentrant=False))
    replayed, _ = causal_fpo_loss(model, [example], torch.ones(1))
    replayed.backward()
    torch.testing.assert_close(replayed, loss)
    for name, parameter in model.net.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad, expected[name])
