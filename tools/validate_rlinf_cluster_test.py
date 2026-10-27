# SPDX-License-Identifier: OpenMDW-1.1
"""Probe payload corruption checks without Ray or accelerator initialization."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transport_probe_detects_value_dtype_shape_and_metadata_changes():
    probe = load_script("validate_rlinf_cluster")
    payload = probe.probe_payload()
    digest = probe.payload_digest(payload)
    assert digest == probe.payload_digest(deepcopy(payload))
    changed = deepcopy(payload)
    changed["actions"][0, 0, 0] += 1
    assert digest != probe.payload_digest(changed)
    changed = deepcopy(payload)
    changed["obs"]["states"] = changed["obs"]["states"].double()
    assert digest != probe.payload_digest(changed)
    changed = deepcopy(payload)
    changed["actions"] = changed["actions"].flatten()
    assert digest != probe.payload_digest(changed)
    changed = deepcopy(payload)
    changed["identity"]["version"] = "probe-1"
    assert digest != probe.payload_digest(changed)
    assert probe.payload_digest([1, 23]) != probe.payload_digest([12, 3])
    changed["actions"].requires_grad_(True)
    with pytest.raises(ValueError, match="detached CPU"):
        probe.payload_digest(changed)


def test_environment_probe_saves_rgb_and_rejects_invalid_state(tmp_path):
    from PIL import Image

    probe = load_script("validate_robotwin_env")
    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    obs = {"state": np.arange(14, dtype=np.float32), "full_image": rgb, "instruction": "test"}
    result = probe.check_observation(obs, tmp_path, "initial")
    assert result["images"]["full_image"]["shape"] == [8, 10, 3]
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "initial_full_image.png")), rgb)
    obs["state"][0] = float("nan")
    with pytest.raises(ValueError, match="finite native state"):
        probe.check_observation(obs, tmp_path, "invalid")
