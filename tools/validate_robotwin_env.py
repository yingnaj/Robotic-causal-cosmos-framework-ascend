# SPDX-License-Identifier: OpenMDW-1.1
"""Check RoboTwin VectorEnv rendering and optional physical joint-action replay."""

import argparse
import json
import multiprocessing as mp
import traceback
from pathlib import Path

import numpy as np


def check_observation(observation, directory, tag):
    from PIL import Image

    state = np.asarray(observation["state"])
    if state.ndim != 1 or not np.isfinite(state).all():
        raise ValueError("Expected a finite native state vector")
    images = {}
    for key in ("full_image", "left_wrist_image", "right_wrist_image"):
        value = observation.get(key)
        if value is None:
            continue
        image = np.asarray(value)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected {key} uint8 RGB [H,W,3], got {image.shape}/{image.dtype}")
        Image.fromarray(image).save(directory / f"{tag}_{key}.png")
        images[key] = {"shape": list(image.shape), "std": float(image.std())}
    if "full_image" not in images:
        raise ValueError("Missing full_image")
    return {"state": state.tolist(), "images": images, "instruction": str(observation.get("instruction", ""))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-config", required=True, type=Path, help="RLinf standalone env YAML, not the whole training recipe"
    )
    parser.add_argument("--assets-path", required=True, type=Path, help="RoboTwin root containing assets/")
    parser.add_argument("--embodiment", help="Optional single embodiment name, e.g. aloha-agilex")
    parser.add_argument("--wrist-cameras", action="store_true", help="Enable both wrist cameras in the task config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--actions-npy", type=Path, help="Trusted physical qpos commands [T,D]; never normalized/55D endpoints"
    )
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "seed": args.seed, "scope": "native_vector_env_only", "state_source": "unverified"}
    destination = args.output / "env_result.json"
    destination.write_text(json.dumps(report, indent=2))
    env = None
    try:
        import os

        from omegaconf import OmegaConf
        from robotwin.envs.vector_env import VectorEnv

        if args.chunk_size < 1 or not (args.assets_path / "assets").is_dir():
            raise ValueError("Expected positive chunk-size and a RoboTwin root containing assets/")
        if not hasattr(VectorEnv, "step_with_full_obs"):
            raise ValueError("This RoboTwin version lacks VectorEnv.step_with_full_obs")
        cfg = OmegaConf.load(args.env_config)
        if args.embodiment:
            cfg.task_config.embodiment = [args.embodiment]
        if args.wrist_cameras:
            cfg.task_config.camera.collect_wrist_camera = True
        report["task_config"] = OmegaConf.to_container(cfg.task_config, resolve=True)
        os.environ["ASSETS_PATH"] = str(args.assets_path.resolve())
        mp.set_start_method("spawn", force=True)
        env = VectorEnv(task_config=report["task_config"], n_envs=1, env_seeds=[args.seed])
        env.reset(env_idx=None, env_seeds=[args.seed])
        initial = env.get_obs()[0]
        report["initial"] = check_observation(initial, args.output, "initial")
        if args.wrist_cameras and not all(
            key in report["initial"]["images"] for key in ("left_wrist_image", "right_wrist_image")
        ):
            raise ValueError("Both wrist camera images are required when --wrist-cameras is set")
        report["chunks"] = []
        if args.actions_npy:
            actions = np.load(args.actions_npy, allow_pickle=False)
            width = len(report["initial"]["state"])
            if (
                actions.ndim != 2
                or actions.shape[1] != width
                or width == 55
                or not len(actions)
                or not np.isfinite(actions).all()
            ):
                raise ValueError(f"Expected finite physical qpos commands [T,{width}], not a unified55 endpoint")
            for start in range(0, len(actions), args.chunk_size):
                chunk = actions[start : start + args.chunk_size].astype(np.float32)
                observations, rewards, terminated, truncated, infos = env.step_with_full_obs(chunk[None])
                count = np.asarray(infos[0]["executed_action_count"]).reshape(-1)
                if (
                    count.size != 1
                    or not np.isfinite(count[0])
                    or count[0] != int(count[0])
                    or not 0 <= count[0] <= len(chunk)
                ):
                    raise ValueError(f"Invalid executed_action_count: {count}")
                count = int(count[0])
                if len(observations) < count:
                    raise ValueError("Missing post-action observations for executed commands")
                term, trunc = bool(np.asarray(terminated).reshape(-1)[0]), bool(np.asarray(truncated).reshape(-1)[0])
                if count < len(chunk) and not (term or trunc):
                    raise ValueError("Partial execution without termination/truncation")
                last = (
                    check_observation(observations[count - 1][0], args.output, f"chunk_{start:06d}") if count else None
                )
                report["chunks"].append(
                    {
                        "requested": len(chunk),
                        "executed": count,
                        "terminated": term,
                        "truncated": trunc,
                        "reward": np.asarray(rewards).tolist(),
                        "last_valid_obs": last,
                    }
                )
                destination.write_text(json.dumps(report, indent=2))
                if term or trunc:
                    break
        report["status"] = "replay_passed" if args.actions_npy else "reset_render_passed_only"
        report["limitations"] = (
            "Does not verify measured-state provenance, wrapper reward bookkeeping, causal policy or RL."
        )
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if env is not None:
            try:
                env.close(True)
            except Exception as error:
                report["cleanup_error"] = repr(error)
        destination.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
