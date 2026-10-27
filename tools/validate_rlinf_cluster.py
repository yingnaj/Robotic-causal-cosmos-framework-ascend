# SPDX-License-Identifier: OpenMDW-1.1
"""Production RLinf transport and NPU collective probes; no Cosmos model load."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
import uuid
from pathlib import Path


def payload_digest(payload):
    import torch

    digest = hashlib.sha256()

    def add(data):
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    def visit(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or value.requires_grad:
                raise ValueError("Probe messages must contain detached CPU tensors")
            add(str((value.dtype, tuple(value.shape))).encode())
            add(value.contiguous().view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            add(b"dict")
            for key in sorted(value):
                add(key.encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            add(str((type(value).__name__, len(value))).encode())
            for item in value:
                visit(item)
        else:
            add(json.dumps(value, sort_keys=True).encode())

    visit(payload)
    return digest.hexdigest()


def probe_payload():
    import torch

    return {
        "obs": {
            "images": torch.arange(2 * 64 * 64 * 3).remainder(256).to(torch.uint8).reshape(2, 64, 64, 3),
            "states": torch.arange(28, dtype=torch.float32).reshape(2, 14),
        },
        "actions": torch.arange(2 * 32 * 14, dtype=torch.float32).reshape(2, 32, 14) / 1000,
        "feedback": {"count": torch.tensor([32, 7]), "done": torch.tensor([False, True])},
        "identity": {"episode": [10, 11], "version": "probe-0", "task": "transport_only"},
    }


def worker_class():
    from rlinf.scheduler import Worker

    class ProbeWorker(Worker):
        def inventory(self, expected_device):
            import rlinf
            import torch

            if self.torch_device_type != expected_device:
                raise ValueError(f"Expected {expected_device}, found {self.torch_device_type}")
            self.torch_platform.set_device(int(os.environ["LOCAL_RANK"]))
            device = torch.device(self.torch_device_type, self.torch_platform.current_device())
            if (torch.ones(8, device=device) + 2).sum().item() != 24:
                raise ValueError("Device arithmetic failed")
            root = Path(rlinf.__file__).resolve().parent
            files = [
                "models/embodiment/cosmos3/causal_policy.py",
                "models/embodiment/cosmos3/robotwin_adapter.py",
                "models/embodiment/cosmos3/fpo.py",
                "workers/actor/fsdp_fpo_actor_worker.py",
            ]
            return {
                "hostname": platform.node(),
                "python": sys.executable,
                "python_version": platform.python_version(),
                "node_rank": self._cluster_node_rank,
                "rank": self._rank,
                "device": str(device),
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "ray", "cloudpickle")},
                "rlinf": str(root),
                "code_hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
                "visible_devices": {
                    name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES")
                },
            }

        def send_and_check(self, outgoing, incoming):
            payload = probe_payload()
            expected = payload_digest(payload)
            outgoing.put(payload)
            received = incoming.get()
            actual = payload_digest(received)
            if actual != expected:
                raise ValueError("Round-trip CPU payload differs from the source")
            return {"sha256": actual, "round_trip": "passed"}

        def echo(self, incoming, outgoing):
            payload = incoming.get()
            digest = payload_digest(payload)
            outgoing.put(payload)
            return {"sha256": digest, "received_cpu_payload": True}

        def npu_fsdp(self):
            import torch
            import torch.distributed as dist
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import fully_shard

            self.inventory("npu")
            device = torch.device("npu", self.torch_platform.current_device())
            dist.init_process_group(backend="hccl", timeout=datetime.timedelta(seconds=120))
            try:
                value = torch.tensor([self._rank + 1.0], device=device)
                dist.all_reduce(value)
                expected = self._world_size * (self._world_size + 1) / 2
                if value.item() != expected:
                    raise ValueError(f"HCCL sum must be {expected}, got {value.item()}")
                torch.manual_seed(42)
                model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)).to(device)
                fully_shard(model, mesh=init_device_mesh("npu", (self._world_size,)))
                before = [p.detach().to_local().clone() for p in model.parameters()]
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
                for _ in range(2):
                    optimizer.zero_grad()
                    loss = model(torch.ones(4, 8, device=device)).square().mean()
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite FSDP probe loss")
                    loss.backward()
                    optimizer.step()
                if not any(not torch.equal(old, p.detach().to_local()) for old, p in zip(before, model.parameters())):
                    raise ValueError("FSDP probe did not update any parameters")
                return {
                    "rank": self._rank,
                    "world_size": self._world_size,
                    "sum": value.item(),
                    "loss": loss.item(),
                    "updated": True,
                }
            finally:
                dist.destroy_process_group()

    return ProbeWorker


def wait_result(handle, timeout):
    deadline = time.monotonic() + timeout
    while not handle.done():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Probe worker did not finish within {timeout} seconds")
        time.sleep(0.1)
    return handle.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("transport", "npu-fsdp"), default="transport")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    report = {"status": "running", "stage": args.stage, "scope": "infrastructure_only"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    groups = []
    channels = []
    try:
        import ray
        from omegaconf import OmegaConf
        from rlinf.scheduler import Channel, Cluster, ClusterConfig, PackedPlacementStrategy

        cfg = OmegaConf.load(args.config)
        schema = ClusterConfig.from_dict_cfg(cfg.cluster)
        if schema.num_nodes != 2 or args.timeout < 1:
            raise ValueError("Expected a two-node cluster and a positive timeout")
        labels = {group.label: group.node_ranks for group in schema.node_groups}
        if labels.get("sim") != [0] or labels.get("ascend") != [1]:
            raise ValueError("Expected sim at node 0 and ascend at node 1")
        sim_device, start, end = cfg.probe.sim_device, cfg.probe.npu_start, cfg.probe.npu_end
        if any(type(value) is not int or value < 0 for value in (sim_device, start, end)) or end <= start:
            raise ValueError("Expected nonnegative device indexes and at least two contiguous NPUs")
        report["config"] = OmegaConf.to_container(cfg, resolve=True)
        if args.check_config:
            report["status"] = "config_validated_only"
            return
        if os.environ.get("RAY_ADDRESS", "auto") in ("local", ""):
            raise ValueError("Use the existing production Ray cluster; RAY_ADDRESS cannot be local/empty")
        os.environ.setdefault("RAY_ADDRESS", "auto")
        Cluster.NAMESPACE = "CausalProductionProbe_" + uuid.uuid4().hex[:12]
        report["namespace"] = Cluster.NAMESPACE
        cluster = Cluster(cluster_cfg=cfg.cluster)
        probe = worker_class()

        def launch(name, first, last, label):
            group = probe.create_group()
            groups.append(group)
            return group.launch(
                cluster, name=name, placement_strategy=PackedPlacementStrategy(first, last, node_group=label)
            )

        if args.stage == "transport":
            sim = launch("probe_sim", sim_device, sim_device, "sim")
            npu = launch("probe_npu", start, start, "ascend")
            reports = wait_result(sim.inventory("cuda"), args.timeout) + wait_result(npu.inventory("npu"), args.timeout)
            report["workers"] = reports
            if [r["node_rank"] for r in reports] != [0, 1]:
                raise ValueError("Worker node placement differs from sim=0, ascend=1")
            if reports[0]["code_hashes"] != reports[1]["code_hashes"]:
                raise ValueError("RLinf causal code differs between nodes")
            if reports[0]["python_version"] != reports[1]["python_version"] or any(
                reports[0]["versions"][key] != reports[1]["versions"][key] for key in ("ray", "cloudpickle")
            ):
                raise ValueError("Match Python, Ray and cloudpickle versions between nodes")
            to_npu = Channel.create("probe_to_npu", node_rank=1)
            channels.append(to_npu)
            to_sim = Channel.create("probe_to_sim", node_rank=0)
            channels.append(to_sim)
            echo = npu.echo(to_npu, to_sim)
            sender = sim.send_and_check(to_npu, to_sim)
            report["transport"] = wait_result(sender, args.timeout) + wait_result(echo, args.timeout)
        else:
            npu = launch("probe_fsdp", start, end, "ascend")
            report["workers"] = wait_result(npu.inventory("npu"), args.timeout)
            report["fsdp"] = wait_result(npu.npu_fsdp(), args.timeout)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        # Kill only this probe's workers; never stop the shared Ray cluster.
        for group in groups + [channel._channel_worker_group for channel in channels]:
            for worker in group.worker_info_list:
                try:
                    ray.kill(worker.worker, no_restart=True)
                except Exception as error:
                    report.setdefault("cleanup_errors", []).append(repr(error))
        if "ray" in locals() and ray.is_initialized():
            ray.shutdown()
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
