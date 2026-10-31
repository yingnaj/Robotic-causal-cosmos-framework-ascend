# SPDX-License-Identifier: OpenMDW-1.1
"""Build a policy block from measured state and an observed camera prefix."""

import json
from dataclasses import replace

import torch

from cosmos_framework.data.generator.action.block_state import BlockStateMetadata
from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.action.video_view import VideoViewConfig


class ObservedCausalBatchBuilder:
    """Re-encode bounded real history; future pixels/actions are placeholders only.

    Camera frames include the window's initial image and one image per executed
    control. Historical state/action slots stay invalid; only the fresh anchor
    conditions the requested block. No predicted video is accepted as history.
    """

    def __init__(
        self,
        *,
        template,
        statistics,
        state_mask,
        action_mask,
        actions_per_block=32,
        video_stride=4,
        history_blocks=1,
        resolution="480",
        fps=30.0,
        cameras=None,
        viewpoint="concat_view",
        temporal_compression_factor=4,
    ):
        self.template, self.statistics = template, statistics
        self.state_mask = template.validate_valid_mask(state_mask).cpu()
        self.action_mask = template.validate_valid_mask(action_mask).cpu()
        statistics.validate(template.width, self.state_mask, self.action_mask)
        self.geometry = CausalBlockGeometry(
            actions_per_block=actions_per_block,
            video_stride=video_stride,
            temporal_compression_factor=temporal_compression_factor,
        )
        if type(history_blocks) is not int or history_blocks < 1:
            raise ValueError("history_blocks must be a positive integer")
        self.history_blocks, self.resolution, self.fps = history_blocks, resolution, self.geometry.action_fps(fps)
        self.views = VideoViewConfig(
            cameras=cameras or {"head": "head_camera", "left": "left_camera", "right": "right_camera"}
        )
        self.viewpoint = viewpoint
        self.transform = ActionTransformPipeline(
            max_action_dim=template.width,
            cfg_dropout_rate=0,
            format_prompt_as_json=True,
            video_temporal_downsample=temporal_compression_factor,
        )

    def __call__(self, *, frames, state, prompt):
        g = self.geometry
        if not frames or (len(frames) - 1) % g.actions_per_block:
            raise ValueError("Observed history must contain an initial image and complete executed blocks")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty task instruction is required")
        frames = frames[-(self.history_blocks * g.actions_per_block + 1) :]
        current = (len(frames) - 1) // g.actions_per_block
        videos = {}
        for key in self.views.camera_keys(viewpoint=self.viewpoint):
            images = [torch.as_tensor(frame[key]) for frame in frames]
            if any(x.dtype != torch.uint8 or x.ndim != 3 or x.shape[-1] != 3 for x in images):
                raise ValueError("Camera frames must be uint8 RGB [H,W,3]")
            videos[key] = torch.stack(images).permute(0, 3, 1, 2)
        video = self.views.compose(videos, len(frames), viewpoint=self.viewpoint)[:, :: g.video_stride]
        future = video.new_zeros(3, g.actions_per_block // g.video_stride, *video.shape[-2:])
        count = (current + 1) * g.actions_per_block
        sample = self.transform(
            dict(
                video=torch.cat((video, future), dim=1),
                action=torch.zeros(count, self.template.width),
                ai_caption=prompt,
                mode="policy",
                viewpoint=self.viewpoint,
                additional_view_description=self.views.describe(viewpoint=self.viewpoint),
                conditioning_fps=torch.tensor(g.video_fps(self.fps)),
            ),
            self.resolution,
        )
        state = torch.as_tensor(state).detach().float().cpu()
        if state.shape != (self.template.width,) or not torch.isfinite(state[self.state_mask]).all():
            raise ValueError("Expected one finite measured template anchor")
        state = self.template.sanitize(state, self.state_mask)
        anchors = torch.zeros(current + 1, self.template.width)
        anchors[current] = state
        sm = torch.zeros_like(anchors, dtype=torch.bool)
        sm[current] = self.state_mask
        normalized, clipped = self.statistics.state.normalize(anchors, sm)
        am = self.action_mask.expand(count, -1).clone()
        metadata = BlockStateMetadata(
            block_size=g.latent_frames_per_block,
            history_blocks=self.history_blocks,
            action_frame_ids=torch.arange(1, 1 + (current + 1) * g.latent_frames_per_block).repeat_interleave(
                g.actions_per_latent_frame
            ),
            action_mask=am,
            states=normalized,
            state_mask=sm,
            anchors=anchors,
            state_action_indexes=torch.arange(current + 1) * g.actions_per_block,
            state_latent_indexes=torch.arange(current + 1) * g.latent_frames_per_block,
            statistics=self.statistics,
            template=self.template,
            state_clip_fraction=clipped,
        )
        plan = replace(
            sample["sequence_plan"],
            causal_action_metadata=metadata,
            action_start_frame_offset=1,
            condition_frame_indexes_vision=list(range(1 + current * g.latent_frames_per_block)),
        )
        return dict(
            video=[sample["video"].unsqueeze(0)],
            action=[sample["action"]],
            image_size=[sample["image_size"].unsqueeze(0)],
            ai_caption=[json.dumps(sample["ai_caption"])],
            conditioning_fps=[torch.tensor(g.video_fps(self.fps))],
            conditioning_fps_action=[torch.tensor(self.fps)],
            sequence_plan=[plan],
            raw_action_dim=[self.template.width],
            causal_action_current_block=current,
            causal_action_preview=False,
        )
