# 8×RTX 3090 + 8×Ascend 910B：RoboTwin 在线 causal FPO 训练

更新：2026-10-31。本文给出正式训练配置、必须填写的生产资产、运行顺序和恢复方法。原验证 1–5 已由用户完成；原第 6 项和后续真实模型闭环需按实际进度完成。新增的八环境路径需要单独完成八卡验收。

配置文件位于 **RLinf**：`examples/embodiment/config/robotwin_causal_production_8gpu_8npu.yaml`。使用 `examples/embodiment/train_causal_fpo.py`，不是普通 PPO 入口。配置是自包含 YAML，只保留此执行路径使用的参数；不会继承普通 PPO 或通用 FSDP 的无效调参项。

## 资源与实际并行方式

| 节点                  | 放置                       | 实际行为                                                         |
| --------------------- | -------------------------- | ---------------------------------------------------------------- |
| GPU/head，8×3090 24GB | Env rank 0–7 → GPU 0–7     | 八个进程、八个独立原生 RoboTwin task，同时推进八条轨迹           |
| NPU，8×910B 64GB      | Rollout rank 0–7 → NPU 0–7 | 每卡一个完整模型副本，每个副本固定请求同 rank 的 GPU 环境        |
| 同一 NPU 节点         | Actor rank 0–7 → NPU 0–7   | 八 rank FSDP2；与 Rollout 分阶段共享同一组设备                   |
| 同一 NPU 节点         | Collector 放置在 NPU 0     | CPU 汇总完整组 replay 并产生八份不同 Actor shard，不占额外一张卡 |

每个 rank 的组大小 G=4，组内四个成员顺序执行，并使用相同 task/reset/instruction/version、不同策略噪声。八个 rank 同时采集，因此一轮是 **八个并行环境、八个组、32 个 episode**，不是 32 个同时运行的环境。GPU/Env 与 NPU/Rollout 之间仅传 CPU 图像、实测状态和物理动作；大的 FPO replay 留在 NPU 节点。

本次补充的 Runner 为每个 Env 建立独立请求 Channel，Rollout rank `r` 固定路由到 Env rank `r % env_world_size`；配置中的 world size 均为 8，因此一一对应。每个 Env 独立进程中的 Python/NumPy/SAPIEN 状态不会被其他 Env 的 reset 修改。原有单 Env、双 Rollout 配置继续可用。

RLinf 的隔离放置在进程启动前设置单卡 `CUDA_VISIBLE_DEVICES`；Env 在导入 RoboTwin/SAPIEN 前设置本进程设备。SAPIEN 维护者说明了这种 [子进程可见设备选择方法](https://github.com/haosulab/SAPIEN/discussions/99)。仍需用每个 Worker 的放置日志和 GPU 渲染进程核实生产机实际选择，不能只看八个 Python PID。

Actor 训练时 Rollout 已 offload，采样时 Actor 已 offload。八张 GPU 和八张 NPU 都参与这个同步训练作业，但不会在每个阶段同时达到满利用率。此配置不做异步流水线，也不承诺吞吐按卡数线性增长。24GB/64GB 是设备容量；NPU 主机还必须容纳八个 Rollout 的 CPU 模型、Actor 分片、初始化瞬时副本、replay 和同步缓冲，需在八卡验收中实测主机内存峰值。

## 生产机必须填写的项目

复制配置为 `examples/embodiment/config/robotwin_causal_production.yaml` 后修改。文件里的 `/opt/rlprod/...` 是示例部署布局，`???` 是强制必填项；不要将开发机 checkpoint 路径当生产默认值。

| 字段                                     | 填写内容                                                                                                  |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| sim 的 `python_interpreter_path`         | GPU 节点已有且验证通过的 RoboTwin Python                                                                  |
| ascend 的 `python_interpreter_path`      | NPU 节点已有且验证通过的 Cosmos/RLinf Python                                                              |
| 两节点 `PYTHONPATH`                      | 各节点实际 RLinf/Cosmos 路径；GPU 另加 RoboTwin。NPU 的 `COSMOS_FRAMEWORK_PATH` 同时修改                  |
| `actor.model.cosmos_sft_toml`            | NPU 绝对路径，配套 causal SFT TOML；模型必须是 `OmniMoTCausalActionModel`                                 |
| `actor.model.model_path`                 | NPU 完整 causal DCP 根目录，根下有 `.metadata`；包含动作与状态接口，使用已在 RoboTwin 域验证的初始化      |
| `actor.model.wan_vae_path`               | 若 TOML/环境未提供正确本地 VAE，则填写 NPU 本地文件路径；否则保持 null                                    |
| `actor.model.cosmos_overrides`           | 仅用于 TOML 中仍需覆盖的本地 tokenizer/processor 等资产；使用配套 TOML 接受的真实配置键，默认空列表       |
| `actor.model.robotwin.statistics_path`   | NPU 对应机器人的 block statistics 文件                                                                    |
| `actor.model.robotwin.robot_config_path` | NPU 对应机器人的 joint/state/action masks 与 DOF 映射 JSON                                                |
| `causal.instruction`                     | 对所有所选 reset 都成立的任务目标指令                                                                     |
| `causal.reset_seeds`                     | 首轮建议填写八个已逐个验证的 uint32 seed；模板强制填写，不提供未经验证的 seed                             |
| `env.assets_path`                        | GPU 的 RoboTwin 根目录，含 `assets/`                                                                      |
| `env.task_config`                        | 本配置以 `adjust_bottle / aloha-agilex / D435` 为具体任务；换任务必须一起改指令、机器人、统计、映射和时长 |
| `runner.output_dir`                      | GPU/head 的 resolved config 与 metrics 输出；示例为 `/opt/rlprod/outputs/robotwin_rl_run001`              |
| `env.evidence_dir`                       | GPU 端视频、物理命令、实测反馈和 Worker 放置记录；示例后缀 `_env`                                         |
| `env.task_config.save_path`              | GPU 端 native task 的输出根目录；native demonstration 保存关闭，主要证据由专用 Env 写入                   |
| `runner.checkpoint_dir`                  | NPU 端 checkpoint 根目录，八个 Actor rank 必须能访问同一文件系统；无需与 GPU 同名目录共享                 |
| `runner.resume`                          | 首次训练 null；恢复时填写 NPU 已提交的 `version_XXXXXX` 目录                                              |

完整 DCP 能加载不代表动作已适配 RoboTwin。之前 AgiBot 固定 replay 的硬件验证不能替代 RoboTwin 域的初始化与统计验收。新模型单副本也必须通过单卡 64GB 的采样验收；八张卡的总 HBM 不会合并为单个 Rollout 的显存。

reset seed 选择按 `((version + round_index) * rollout_world_size + rank) % len(reset_seeds)`，不再受最大补采轮数的取模周期影响。使用八个固定训练 seeds 时，每个版本覆盖同一组八个 reset；使用 32 个已验证 seeds 时，四个版本的首轮采样依次覆盖全部 32 个。补采轮使用下一个窗口，同组四个成员仍共享 reset。修改 seed 列表会改变恢复配置指纹，需另开实验。

## 参数对应的训练含义

| 参数                                        | 当前值                | 含义与选取理由                                                                                           |
| ------------------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------- |
| `runner.max_versions`                       | 100                   | 首批正式训练到第 100 个已提交行为边界；不是 100 次 optimizer step，也不是任务效果已收敛的承诺            |
| `runner.timeout_seconds`                    | 14400                 | 每次等待一个阶段最多四小时；根据八卡验收实测耗时调整，不是总训练时长                                     |
| `causal.group_size`                         | 4                     | 同 reset 的四个独立采样用于组内 suffix 优势；保持此前 G=4 基线                                           |
| `causal.max_collect_rounds`                 | 4                     | 有效 chunks 不足 global batch 才追加完整组；正常首轮已有至少 32 个 chunk，不等于每版本固定采四轮         |
| `causal.gamma`                              | 1                     | 有限 episode、成功奖励 0/1、无额外时间偏好；成功和超时均结束 episode、零 bootstrap                       |
| `causal.num_probes`                         | 2                     | 每个完整采样块保存两个 CFM probes，Actor 复用；沿用已验证基线                                            |
| `causal.cfm_rtol/atol`                      | 0.02 / 1e-5           | old CFM 在 Rollout/Actor 间的一致性验收阈值；不是 loss 权重，失败需查数值/模型一致性                     |
| `causal.compatible_dcp`                     | true                  | 模型加载期间的旧 pathlib 元数据只读兼容；不修改公共 checkpoint                                           |
| `batch.actions_per_block`                   | 32                    | 完整动作 endpoint 的长度；FPO 样本单位是块，不是每个控制目标                                             |
| `batch.video_stride/compression`            | 4 / 4                 | 观测采样和 VAE 时间压缩；必须与 checkpoint/TOML 一致                                                     |
| `batch.history_blocks`                      | 1                     | 下一块使用最多一个已执行块的真实图像历史和最新实测状态，控制单次上下文与 replay 大小                     |
| `batch.resolution/viewpoint/cameras`        | 480 / 三视角拼接      | 头部和双腕按训练布局拼接；原图不做额外中心裁剪                                                           |
| `batch.fps`、`env.fps`                      | 30                    | 模型视频条件与名义控制时间轴；不是 SAPIEN physics timestep 或实测控制吞吐                                |
| `sampling.num_steps/guidance/shift`         | 4 / 1 / 5             | 原生 UniPC、无额外 CFG 分支、沿用已验证采样参数；`normalize_cfg=false` 保留为采样接口的明确设置          |
| `env.max_episode_steps`                     | 400                   | 使用本机 `adjust_bottle` 原生 eval 上限；生产核对 `_eval_step_limit.yml`。实际为原生上限与本值的最小值   |
| `env.action_dim`                            | 14                    | 左六关节+夹爪、右六关节+夹爪；夹爪是按配置归一化的物理控制量，六关节 qpos 使用机器人定义的单位           |
| Env 四个 false 契约开关                     | false                 | 显式 reset、不忽略终止、不跳过中间渲染、不做额外 crop；专用入口会拒绝违反契约的值                        |
| `env.record_video`                          | true                  | 此实现中同时开启视频、命令和反馈证据；关闭会一起关闭这些 episode 证据                                    |
| `task_config.save_freq`                     | null                  | 关闭 native demonstration 自动保存；模型每个真实执行控制后的相机观测仍由专用 Env 收集                    |
| `task_config.data_type`                     | RGB 开，其余关闭      | 仅收需要的 RGB；实测状态直接读 articulation，不读取 native 缓存 command qpos；避免深度/点云/分割输出     |
| 相机、embodiment、planner                   | D435/aloha/mplib      | 原生 task 的机器人、相机和初始化配置，须与通过验收的场景一致                                             |
| `domain_randomization`                      | 低随机化              | 首批聚焦固定任务；背景、杂物、光照、桌高与相机额外扰动关闭。`clean_background_rate=1` 与关闭随机背景配套 |
| `actor.micro/global_batch_size`             | 1 / 32                | 8 rank × 每 rank 1 chunk × 累积 4 次；每个优化批次 32 个有效 chunk                                       |
| `actor.seed`                                | 42                    | 独立于环境 reset；策略与 probe seed 还包含行为版本、组、成员和块身份                                     |
| `model.train_state`                         | false                 | 只更新原生动作接口，state 与共享网络冻结；不是全参数 RL 或 LoRA                                          |
| `model.precision`、FSDP param/reduce        | bf16 / bf16 / fp32    | 构建精度、计算精度与梯度归约精度；`cast_forward_inputs=false` 保留 replay 输入 dtype                     |
| `model.is_lora/load_to_device`              | false / false         | 普通原生动作接口；模型先在 CPU 创建，后续由 Actor 或 Rollout 控制设备放置                                |
| `disable_cosmos_parallelism/ema/compile`    | true / false / false  | RLinf 独立管理 FSDP；行为模型不使用 EMA；保持已验收的 eager 执行                                         |
| `actor.enable_offload`                      | true                  | Actor/Rollout 阶段切换时 offload；不是同时把两个模型留在每张 NPU 上                                      |
| `fsdp.strategy/reshard_after_forward`       | fsdp2 / true          | 八卡分片；子模块前向后重新分片，root 保持代码中既定策略                                                  |
| `fsdp.cpu_offload`                          | false                 | 不额外开启逐层 CPU swapping；阶段间 offload 仍开启                                                       |
| `fsdp.gradient_checkpointing`               | false                 | 不通过 RLinf 再包装；保留配套 Cosmos TOML 自带的 activation-checkpoint wrappers                          |
| `fsdp.enable_gradient_accumulation`         | true                  | 前三个 microstep 不做最终归约，第四个 microstep 同步                                                     |
| `amp_autocast/grad_scaler.enabled`          | false / false         | 使用 FSDP BF16 精度策略，不叠加全局 AMP 或 loss scaling                                                  |
| `save_full_model_weights`                   | false                 | 保存分片模型、optimizer、scheduler、RNG 与 FPO 状态；每个行为边界保存一次                                |
| `optim.lr/betas/eps/weight_decay/clip`      | 1e-5/.9,.999/1e-8/0/1 | AdamW；动作接口学习率沿用验收基线，关闭额外 weight decay，裁剪梯度范数到 1；不是调优后的最优超参数       |
| `optim.lr_scheduler/lr_warmup_steps`        | constant / 0          | 固定学习率；每版本 optimizer 步数随有效块数变化，因此不添加虚假的 `total_training_steps=100`             |
| `algorithm.update_epoch`                    | 1                     | 当前行为版本每个保留 chunk 使用一次；无跨版本 replay                                                     |
| `algorithm.entropy_bonus`                   | 0                     | CFM/FPO++ 不使用伪动作 logprob、entropy、value 或 KL                                                     |
| `fpo.per_probe_ratio/asymmetric/clip/scale` | true/true/0.2/1       | 逐 probe 的 FPO++ 非对称目标、clip 宽度与 CFM 差值缩放，保持原算法默认                                   |
| `weight_syncer.bucket_*`                    | 128MiB/null/CPU       | 完整参数与 buffers 通过 CPU buckets 同步，保留原始 dtype；全部 Rollout 的 hash/version ACK 通过才采样    |

`group_name` 是 Worker/通信组标识，`pipeline_stage_num=1` 是同步执行约束，`hydra.run.dir=.` 与 `output_subdir=null` 保持 driver 工作目录、避免 Hydra 写额外输出目录。`runner.check_config` 是只校验模式。`runner.resume` 控制边界恢复。节点 label、rank 和 Channel 放置决定真实通信方向，不需要另加未被 Runner 消费的 `num_envs`、`rollout_batch_size`、`save_interval` 或 `eval_interval`。

400 步上限意味着未提前结束的 episode 最多 13 个块，最后一块实际执行 16 步但保留完整 endpoint。每轮最多 32×13=416 个有效 chunk；在 global batch=32、update_epoch=1 时最多 13 个 optimizer steps。提前成功会改变块数，整除不了 32 的尾部在完整组优势计算后丢弃并记录。版本数、episode 数、chunk 数与 optimizer 步数必须分别统计。全零优势可以推进版本和保存边界，但不发生学习更新。

## 逐步运行

### 1. 部署代码、选择生产配置

部署本次修改的两个仓库：仅有旧提交 `da3c36e` / `5960e95a` 不包含本次八环境路由和新配置。RoboTwin 代码不需要为本配置做修改。两节点复用已验收解释器与依赖，生产配置放在 GPU/head 的 RLinf config 目录。

GPU/head：

```bash
export RLINF_ROOT=/opt/rlprod/RLinf
export COSMOS_ROOT=/opt/rlprod/Cosmos
export ROBOTWIN_PATH=/opt/rlprod/RoboTwin
export PYTHONPATH="$RLINF_ROOT:$COSMOS_ROOT:$ROBOTWIN_PATH"
export ROBOT_PLATFORM=ALOHA
cd "$RLINF_ROOT"
cp examples/embodiment/config/robotwin_causal_production_8gpu_8npu.yaml \
  examples/embodiment/config/robotwin_causal_production.yaml
```

完成上表所有生产值，尤其强制填写模型、统计、机器人映射、指令与八个已验证 seed。确认 `actor/rollout/env` 三个 placement 都为各自 node group 内 `0-7`。保持 `runner.resume=null`。

### 2. 复用已通过验证的 Ray 集群

GPU/head 执行 `ray status`，核实两个节点在线且各八张设备均可见。GPU 节点不能从此前单卡 probe 意外继承全局 `CUDA_VISIBLE_DEVICES=0`；各 Env 的单卡隔离由 RLinf 设置。NPU 节点同样不能继承只开放两张卡的全局限制。

若尚未启动集群，在各节点激活已验收环境并设置已有网络配置后执行。变量须填写实际 IP/网卡；NPU 先按已有流程加载 CANN 环境，不复制 GPU 的 CUDA 库路径。

GPU/head：

```bash
: "${SIM_IP:?Set GPU head IP}"
: "${SIM_IFACE:?Set GPU network interface}"
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES="$SIM_IFACE"
export GLOO_SOCKET_IFNAME="$SIM_IFACE"
ray start --head --node-ip-address="$SIM_IP" --port=6379
```

NPU：

```bash
export RLINF_ROOT=/opt/rlprod/RLinf
export COSMOS_ROOT=/opt/rlprod/Cosmos
export PYTHONPATH="$RLINF_ROOT:$COSMOS_ROOT"
export COSMOS_FRAMEWORK_PATH="$COSMOS_ROOT"
export COSMOS_DEVICE=npu
: "${SIM_IP:?Set GPU head IP}"
: "${NPU_IP:?Set NPU node IP}"
: "${NPU_IFACE:?Set NPU network interface}"
export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES="$NPU_IFACE"
export GLOO_SOCKET_IFNAME="$NPU_IFACE"
ray start --address="$SIM_IP:6379" --node-ip-address="$NPU_IP"
```

已有配置正确的生产集群直接复用，不执行机器级 `ray stop`。原来的双卡 HCCL probe 通过后，还应先将 probe 的 `npu_end` 改为 7、确认其余卡空闲，运行八卡 `npu-fsdp` 阶段。

### 3. 只检查配置

GPU/head 的 RLinf 根目录：

```bash
python examples/embodiment/train_causal_fpo.py \
  --config-name robotwin_causal_production --cfg job --resolve
python examples/embodiment/train_causal_fpo.py \
  --config-name robotwin_causal_production runner.check_config=true
```

此阶段检查整数类型、有限正数、视频/VAE 时间几何、相机映射和绝对路径，不读取远端模型、不启动 Worker；不能替代资产检查和八卡验收。

### 4. 八环境冻结策略验收

八卡部署首次运行，先使用与正式训练相同的 400 步上限，只关闭优化器更新：

```bash
RAY_ADDRESS=auto python examples/embodiment/train_causal_fpo.py \
  --config-name robotwin_causal_production \
  runner.mode=frozen runner.max_versions=1 \
  runner.output_dir=/opt/rlprod/outputs/robotwin_8env_frozen \
  env.evidence_dir=/opt/rlprod/outputs/robotwin_8env_frozen_env
```

在 GPU evidence 的 `<run_id>/env_worker_0.json` 至 `env_worker_7.json` 核对不同 PID、单卡 `cuda_visible_devices`；结合 `nvidia-smi` 核实八张 3090 都有对应渲染进程。检查八个组、共 32 个 episode 的 seed/member/version，打开头部和双腕拼接视频，核对实测状态与动作语义、成功/超时和部分末块。NPU 八个 Rollout 的完整权重 ACK 必须通过。记录 GPU HBM、NPU HBM、两节点主机内存、replay/同步耗时；不能用上一版单 Env 的验收替代。

### 5. 正式配置先运行两个版本

首次使用独立、空的 run001 checkpoint 目录；保持正式 seed、400 步和所有算法参数不变。Actor 在启动 Rollout 前预检目标版本目录，拒绝覆盖已有边界，并检查目标版本确实晚于恢复边界：

```bash
set -o pipefail
mkdir -p /opt/rlprod/outputs/robotwin_rl_run001
RAY_ADDRESS=auto python examples/embodiment/train_causal_fpo.py \
  --config-name robotwin_causal_production runner.max_versions=2 \
  2>&1 | tee /opt/rlprod/outputs/robotwin_rl_run001/driver_first2.log
```

查看 `metrics.jsonl`：行为 CFM 对齐通过、loss/grad 有限、`trained_chunks` 为 32 的倍数、每个 Actor 收到不同的有效 shard。要验收实际学习，至少应出现非零优势且 `fpo/optimizer_steps_this_version > 0`；若组内全成功或全失败产生零优势，优先调查任务与初始化，不修改优势伪造更新。生产成功率效果使用独立固定评测集检查，不直接把训练组成功率当泛化评测。

NPU 端检查 `version_000001`、`version_000002` 的 `COMMITTED.json` 与 `fpo_state.json`。恢复源是 RLinf 完整边界目录；不是 `actor.model.model_path` 的初始 DCP。量一下单个 checkpoint 实际大小与八卡主机内存峰值，再评估后续 100 个边界的磁盘/内存余量。当前每版本保存完整分片状态和全部训练视频，无自动 checkpoint 保留或视频降采样开关。

### 6. 从边界 2 正式训练到边界 100

保持同一份配置，只变更 `runner.resume` 与目标版本数。GPU/head 启动一次，不另起 NPU torchrun：

```bash
nohup env RAY_ADDRESS=auto python examples/embodiment/train_causal_fpo.py \
  --config-name robotwin_causal_production \
  runner.max_versions=100 \
  runner.resume=/opt/rlprod/checkpoints/robotwin_rl_run001/version_000002 \
  > /opt/rlprod/outputs/robotwin_rl_run001/driver_to100.log 2>&1 &
```

这里继续 98 个行为版本。若同一八环境配置已经完成冻结、两轮与恢复验收，可以从空目录直接用 `runner.max_versions=100 runner.resume=null` 启动首批正式训练。

### 7. 监控、边界停止与异常恢复

GPU/head 查看 `runner.output_dir/metrics.jsonl`、driver 日志及 GPU evidence；NPU 查看 checkpoint、`npu-smi info`、主机内存与磁盘。GPU 用 `nvidia-smi` 查看八个渲染进程和显存。参数 `timeout_seconds` 可按实测阶段耗时调整；它不是 optimizer 步数、训练总时长或吞吐目标。

当前 Runner 通过 `max_versions` 在边界正常退出，没有热修改配置或“收到信号后保存当前物理现场”的功能。建议分目标版本运行，例如 2→20→100；异常中断后从最后已提交边界重新采样。

例如边界 37 已提交时：

```bash
RAY_ADDRESS=auto python examples/embodiment/train_causal_fpo.py \
  --config-name robotwin_causal_production \
  runner.max_versions=100 \
  runner.resume=/opt/rlprod/checkpoints/robotwin_rl_run001/version_000037
```

边界目录先写 `.partial-*`，完成全部 Actor 保存和 manifest 后才原子改名。只恢复带 `COMMITTED.json` 的正式目录；不覆盖已有正式边界。恢复检查配置指纹、模型资产和 Actor world size；除 `runner` 字段外，种子、步数、超参数、设备布局和 `env.evidence_dir` 等须保持一致，调参另开实验。保留原始 DCP/统计/TOML 不可变，避免恢复资产指纹变化。

### 8. 复查训练后的策略

训练 driver 退出后，可以加载已提交策略进行无更新复查。例如完成边界 100 后：

```bash
RAY_ADDRESS=auto python examples/embodiment/train_causal_fpo.py \
  --config-name robotwin_causal_production \
  runner.mode=frozen runner.max_versions=101 \
  runner.resume=/opt/rlprod/checkpoints/robotwin_rl_run001/version_000100 \
  runner.output_dir=/opt/rlprod/outputs/robotwin_rl_run001_review100
```

它从边界 100 加载策略，采集一轮行为版本 100，不执行 optimizer、不保存新 checkpoint。该 driver 的 metrics 边界标记会推进到 101，不能据此宣称训练出了 `version_000101`。保持原 `env.evidence_dir`，新 run ID 自动隔离复查视频。

这是同训练 seeds 的固定任务复查。当前专用 Runner 没有自动评测调度或独立 held-out seed override；恢复指纹会拒绝直接修改 `causal.reset_seeds`。独立泛化评测需要后续配套评测入口，不能用本命令的训练种子成功率代替。阶段性复查可在 20、50、100 等正常退出边界手动运行，再恢复训练。

## 本地验证范围

八环境开发验证产物存于 `outputs/rlinf_8gpu_8npu_development_20261029/`。CPU 回归与真实 RLinf CPU Worker/Channel 测试覆盖严格配置、八 Env 请求路由、两轮 reset、实测反馈、终止与重复命令、长 seed 列表轮转、checkpoint 目标预检，以及既有 FPO replay 分片。2026-10-31 CPU 回归通过 86 项，复查产物位于 `outputs/rlinf_production_review_20261031/`。通信 smoke 显式使用 CPU Worker 放置并校验设备身份，即使主机存在 NPU 也不将 Env 测试放到 NPU 上。SAPIEN 是测试边界的替身；本轮本地结果不代表八张 3090 的真实渲染、八卡 HCCL 或完整模型在线训练已经通过。按上文完成生产八卡验收后，再评估吞吐和学习效果。
