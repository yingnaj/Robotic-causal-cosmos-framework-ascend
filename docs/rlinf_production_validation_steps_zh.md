# 代码迁移后的首轮生产验证：执行步骤

日期：2026-10-27。GPU 节点已有 RoboTwin 依赖，本轮不重装。先完成下列基础验证，不需要正式 causal checkpoint。节点 0 为 GPU/head，节点 1 为 NPU；先使用 GPU 0、NPU 0–1。

新增脚本已经通过本地语法、配置 schema 和辅助函数测试；真实 RLinf Worker/Channel 的单机 CPU 载荷往返也已通过，记录位于 `outputs/production_probe_development_20261027/`。尚未在生产双节点或真实 RoboTwin 上运行。它们用于环境和训练基础设施验收，不是在线 FPO 训练入口。

## 1. 核对迁移内容与路径

两节点均部署相同 RLinf 代码；为了运行这次的工具，Cosmos 源码也部署到两节点，但 GPU 节点无需安装完整 Cosmos 模型依赖。以下统一假设：

```bash
export RLINF_ROOT=/opt/rlprod/RLinf
export COSMOS_ROOT=/opt/rlprod/Cosmos
```

GPU 节点另设 `ROBOTWIN_PATH` 为现有 RoboTwin 根目录。在各自已激活的环境内核对 Python、Ray、cloudpickle；本轮 probe 要求两端版本完全相同。CUDA/NPU torch 的构建后缀可以不同。

```bash
python -c 'import sys, platform, ray, cloudpickle, torch; print(sys.executable); print(platform.python_version(), ray.__version__, cloudpickle.__version__, torch.__version__)'
```

确认两端 RLinf 都包含新增的 `causal_policy.py`、`robotwin_adapter.py`、`fpo.py`、`fsdp_fpo_actor_worker.py`，Cosmos 包含 `utils/robotwin.py`、`model/generator/algorithm/rl/` 和本次的 `tools/validate_*.py`。未跟踪文件不会随普通 clone/pull 到达；需要从当前工作区生成完整发布包。后续 transport probe 会比较上述四个 RLinf 核心文件的 SHA256，但它不替代完整发布清单。

在 NPU 节点复用已验证环境，运行 CPU 回归（只用 CPU，不启动 Ray）：

```bash
export PYTHONPATH="$RLINF_ROOT:$COSMOS_ROOT"
cd "$COSMOS_ROOT"
python -m pytest --noconftest \
  cosmos_framework/utils/robotwin_test.py \
  cosmos_framework/model/generator/algorithm/rl/fpo_test.py \
  cosmos_framework/model/generator/algorithm/rl/causal_fpo_test.py \
  cosmos_framework/inference/causal_action/session_test.py \
  "$RLINF_ROOT/tests/unit_tests/test_cosmos3.py" -q
```

通过标准：当前版本为 56 项通过，无 skip。若出现 skip，不能用通过数替代依赖检查。这里不重新创建 Conda 环境；若使用现有激活脚本，其 `conda.sh` 初始化路径必须与生产安装一致。

## 2. GPU 节点独立验证 RoboTwin

在已有 RoboTwin 环境中执行。`adjust_bottle + aloha-agilex` 是参考配置的候选示例，换任务/机器人时使用匹配的 env YAML 和 embodiment；seed 应选择该环境已知可初始化的值。

```bash
export ROBOTWIN_PATH=/opt/rlprod/RoboTwin
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="$RLINF_ROOT:$COSMOS_ROOT:$ROBOTWIN_PATH"
cd "$COSMOS_ROOT"

CUDA_VISIBLE_DEVICES=0 python tools/validate_robotwin_env.py \
  --env-config "$RLINF_ROOT/examples/embodiment/config/env/robotwin_adjust_bottle.yaml" \
  --assets-path "$ROBOTWIN_PATH" \
  --embodiment aloha-agilex --wrist-cameras --seed 42 \
  --output outputs/production_probe/env_reset
```

检查 `env_reset/env_result.json` 的 `status` 为 `reset_render_passed_only`，并打开 `initial_full_image.png`、两个 wrist PNG，确认图像内容和相机布局正常。脚本验证图像 dtype/shape、状态有限值；黑屏与相机位置仍需看图。用 `nvidia-smi` 确认实际渲染资源；CUDA 可见设备不自动证明 Vulkan 渲染卡号正确。

这一步只验证原生 `VectorEnv.reset/get_obs` 和渲染，结果明确标记 `state_source: unverified`，不宣称测到了真实关节状态。

若已有**同任务、同机器人、同 reset seed 的专家/已知有效 qpos 动作序列**，再执行：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/validate_robotwin_env.py \
  --env-config "$RLINF_ROOT/examples/embodiment/config/env/robotwin_adjust_bottle.yaml" \
  --assets-path "$ROBOTWIN_PATH" \
  --embodiment aloha-agilex --wrist-cameras --seed 42 \
  --actions-npy /absolute/path/to/expert_joint_actions.npy \
  --chunk-size 32 --output outputs/production_probe/env_replay
```

动作文件是物理 qpos `[T,D]`，双 6-DOF 臂时通常为 `[T,14]`，顺序为左关节/左夹爪/右关节/右夹爪；不能传 normalized endpoint 或原始 unified55。该 probe 暂不执行 EEF 命令。没有匹配动作文件时先完成 reset/render，不用随机动作伪造专家回放。

通过标准：`status=replay_passed`；每块都有合法 `executed_action_count`、reward、termination/truncation，以及最后一张真实有效观测。遇到提前终止时 count 小于请求长度，脚本停止且不使用 padding 的最后一帧。它没有检验 RLinf wrapper 的奖励落点/步数修复，也没有证明 causal 模型能控制任务。

## 3. 填写双节点 probe 配置

复制 [probe YAML](../examples/rlinf/robotwin_production_probe.yaml) 为本次实验配置：

```bash
cd "$COSMOS_ROOT"
mkdir -p outputs/production_probe
cp examples/rlinf/robotwin_production_probe.yaml outputs/production_probe/cluster.yaml
```

编辑 `cluster.yaml`：

- `sim` 的 `python_interpreter_path` 填 GPU 节点现有 RoboTwin Python 绝对路径。
- `ascend` 的解释器填生产 NPU 环境路径。
- 两个节点的 `PYTHONPATH` 均改为实际仓库路径；GPU 端增加 RoboTwin 根目录。
- 设备由 `probe.sim_device`、`probe.npu_start/npu_end` 选择。默认 GPU 0、NPU 0–1，先确认空闲。本 probe 用这些字段创建 Worker；`component_placement` 是 schema 所需的角色描述，不单独决定 probe 卡号。

先在 GPU/head 校验配置，不会启动 Ray Worker：

```bash
python tools/validate_rlinf_cluster.py \
  --config outputs/production_probe/cluster.yaml --check-config \
  --output outputs/production_probe/config_check.json
```

通过标准：`status=config_validated_only`。这只检查配置结构，不检查远端解释器和资产是否存在。

## 4. 启动双节点 Ray

使用 [runbook 中的逐节点命令](rlinf_robotwin_production_runbook_zh.md) 启动，或按下面的最小流程。各节点先激活自己的环境，并在 `ray start` 前设置本节点的 `PYTHONPATH`、`RLINF_NODE_RANK` 和网卡。将占位变量设置成实际值；不要将 GPU 端 CUDA 动态库路径复制到 NPU 端。

GPU/head：

```bash
: "${SIM_IP:?Set GPU head IP}"
: "${SIM_IFACE:?Set GPU node interface}"
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES="$SIM_IFACE"
export GLOO_SOCKET_IFNAME="$SIM_IFACE"
ray start --head --node-ip-address="$SIM_IP" --port=6379
```

NPU：

```bash
: "${SIM_IP:?Set GPU head IP}"
: "${NPU_IP:?Set NPU node IP}"
: "${NPU_IFACE:?Set NPU node interface}"
export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES="$NPU_IFACE"
export GLOO_SOCKET_IFNAME="$NPU_IFACE"
export PYTHONPATH="$RLINF_ROOT:$COSMOS_ROOT"
export COSMOS_FRAMEWORK_PATH="$COSMOS_ROOT"
ray start --address="$SIM_IP:6379" --node-ip-address="$NPU_IP"
```

回到 GPU/head 执行 `ray status`，确认两节点在线。已有符合本次环境/节点编号的 Ray 集群可直接使用；不要为了本 probe 对共享机器执行 `ray stop`。上述 env probe 的 `CUDA_VISIBLE_DEVICES=0` 是单命令前缀，启动 Ray 时不应意外继承一张卡的全局限制。

## 5. 验证实际 RLinf 跨节点通信

只在 GPU/head 上执行一次：

```bash
cd "$COSMOS_ROOT"
RAY_ADDRESS=auto python tools/validate_rlinf_cluster.py \
  --config outputs/production_probe/cluster.yaml --stage transport \
  --output outputs/production_probe/transport.json
```

脚本在 GPU 0 和 NPU 0 各创建一个真实 RLinf Worker，各执行小张量设备运算；建立两个 Channel，向 NPU 发送 CPU 图像/state/action/执行长度/episode/version 嵌套数据，并原样返回 GPU。数据是合成测试载荷，不是 RoboTwin 实际采集，也不是 CausalFPOBatch。

通过标准：`status=passed`；`workers` 中节点 rank 分别为 0/1、设备为 cuda/npu、解释器正确、四个核心文件 hash 一致；`transport` 两侧 SHA256 一致，round_trip 为 passed。失败时查 JSON 的 error/traceback 和 RLinf 日志。

脚本拒绝 `RAY_ADDRESS=local`，使用唯一 namespace，只清理自身 Worker/Channel，不停止生产 Ray 集群。方法调用有默认 180 秒超时；集群初始化/Worker 启动时间不受该方法超时覆盖，若卡在启动须检查节点日志与网络。

## 6. 验证 NPU 双卡 HCCL/FSDP2

继续在 GPU/head 上执行，计算只发生在 NPU 节点：

```bash
RAY_ADDRESS=auto python tools/validate_rlinf_cluster.py \
  --config outputs/production_probe/cluster.yaml --stage npu-fsdp \
  --output outputs/production_probe/npu_fsdp.json
```

通过标准：`status=passed`；默认两个 rank 的 all-reduce `sum=3.0`、loss 有限、`updated=true`。该步骤使用微型 MLP 做两次 AdamW 更新，验证 RLinf 放置、HCCL 和 FSDP2，**没有加载 Cosmos3**。可以先用 2 卡，再改变 `probe.npu_end` 测试 4/8 卡；所有选中卡必须可用。

## 7. 当前验收边界与后续步骤

把 config、JSON、PNG、代码/环境版本、完整日志汇总到本次实验目录。1–6 通过后，可以确认迁移的基本环境、转换回归、Ray/RLinf 通信和 NPU 分布式训练基础正常；仍不表示双节点在线 RL 通过。

下一阶段才是：正式/工程 checkpoint 的真实 Cosmos 固定 replay 更新 → 冻结策略驱动 RoboTwin → 两轮在线 FPO 更新及恢复。当前真实模型验证仍依赖本机 `outputs/.../validate_actor.py` 的特定路径和 compatibility reader，尚未整理为本文件新 probe 的阶段；不要将其直接当作上述生产 Ray 集群的入口。

2026-10-28 已新增完整块 causal batch builder、实测反馈、专用 rollout/collector、Runner 与 `train_causal_fpo.py`；实现、配置、冻结策略、两轮更新和恢复命令见 [在线 FPO 实现与验收](rlinf_online_fpo_implementation_zh.md)。这些新路径尚未进行生产 GPU/NPU 闭环验收；本文件的基础设施通过结果不能替代该验收。
