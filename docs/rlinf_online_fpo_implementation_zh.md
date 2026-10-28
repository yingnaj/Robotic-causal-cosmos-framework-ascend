# RLinf + RoboTwin causal FPO：实现与生产验收

更新：2026-10-28。本文接续 `RLINF_CAUSAL_COSMOS3_RL_INTEGRATION_PLAN.md`、`causal_cosmos3_robotwin_rl_plan.md`、`robotwin_adapter_and_branch_audit_zh.md` 和 `rlinf_robotwin_production_runbook_zh.md`。基础设施 probe 已由用户在生产环境验证；本文新增实现尚未进行生产 GPU/NPU 闭环验收。

## 已实现的执行链

```text
GPU/head：Hydra driver，只收配置、状态和指标
GPU 0：CausalRoboTwinEnvWorker → native RoboTwin task
             CPU reset / physical commands / measured state / RGB feedback
NPU：CausalFPORolloutWorker → 原生 UniPC → 原始 CausalFPOExample
NPU：CausalFPOCollectorWorker → 完整组 suffix 优势 → 独立 Actor shards
NPU：OnlineCausalFPOActor → FSDP2 / CFM / FPO++ 更新
NPU：完整 CPU bucket 权重同步 → hash/version ACK → 下一行为版本
```

保留原生 attention、UniPC、CFM/FPO++ 和动作参数训练范围。没有 value head、伪 logprob、KL 或 entropy。普通 PPO 入口和 RoboTwin wrapper 保持原行为；新入口明确选择专用数据路径。`predict_action_batch` 的普通 HF 接口仍不适用。

首版一个 GPU 环境服务，每个 rollout rank 有一个稳定会话，组内成员顺序采集，rank 间可交替请求。默认两个独立 NPU rollout 副本，每个采集一组 G=4；不是四个并行环境，也不是分片 rollout。环境执行序列化以避免原生仿真和进程全局随机状态并发。Actor/Rollout 通过 offload 分阶段共享设备。

环境路径复用当前 RoboTwin 的 task 类、机器人配置与原生 `take_action(qpos)`；不通过普通 `VectorEnv` 的 command-state、自动换 seed 和 block padding。每个物理目标后读取 articulation 的实际关节 qpos 和真实相机，夹爪按机器人配置反解为归一化观测值，实际执行一步记一步。RoboTwin 的 `real_jointState` getter 仍拼接缓存夹爪目标，本实现不使用该目标作为状态。reset 失败直接报错，不换 seed。这个新执行路径仍须做真实任务验收，原来的 VectorEnv probe 不能替代它。

## 文件地图

以下 RLinf 路径相对其仓库根目录。

| 文件                                                                               | 职责                                                          |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `rlinf/envs/sim/robotwin/causal_env.py`                                            | 实测状态、严格 seed、逐步执行、终止、去重、观测视频和反馈证据 |
| `rlinf/workers/env/causal_robotwin_worker.py`                                      | GPU 环境服务与 CPU 请求/响应 Channel                          |
| `rlinf/models/embodiment/cosmos3/online_fpo.py`                                    | episode/chunk/version 协议、完整组验证与有效 chunk 分片       |
| `rlinf/workers/rollout/causal_fpo_worker.py`                                       | 独立策略副本、真实历史、采样、NPU collector、权重 ACK         |
| `rlinf/workers/actor/fsdp_fpo_actor_worker.py`                                     | 原 Actor 加上生产 bootstrap、行为分数校验、完整同步和提交恢复 |
| `rlinf/models/embodiment/cosmos3/runtime.py`                                       | Worker 内 NPU 初始化、作用域内只读 DCP reader、资产指纹       |
| `rlinf/runners/causal_fpo_runner.py`                                               | 同步阶段编排、超时、配置校验、边界保存恢复、只清理自身 Worker |
| `examples/embodiment/train_causal_fpo.py`                                          | Hydra 启动入口                                                |
| `examples/embodiment/config/robotwin_adjust_bottle_causal_cosmos3_fpo_2nodes.yaml` | 完整双节点 recipe；资产与指令必须填写                         |
| Cosmos `cosmos_framework/data/generator/action/online_batch.py`                    | 同训练模板/统计的相机拼接、真实历史与 fresh anchor packing    |

## 1. 发布与配置

两节点部署本次 Cosmos 与 RLinf 新增文件及修改。普通 diff 不含未跟踪文件；先核对发布清单，不要只复制 YAML。GPU 环境必须能导入 RLinf 和 native RoboTwin；NPU 使用已经验证的环境。

在 GPU/head 的 RLinf 仓库复制配置：

```bash
export RLINF_ROOT=/opt/rlprod/RLinf
export COSMOS_ROOT=/opt/rlprod/Cosmos
cd "$RLINF_ROOT"
cp examples/embodiment/config/robotwin_adjust_bottle_causal_cosmos3_fpo_2nodes.yaml \
  examples/embodiment/config/robotwin_causal_production.yaml
```

修改 `robotwin_causal_production.yaml`：

| 字段                                                               | 内容/所属节点                                                                     |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| `cluster.node_groups[*].env_configs`                               | 两端实际解释器与 PYTHONPATH；沿用通过 probe 的配置，保留 sim=0/ascend=1           |
| `cluster.component_placement`                                      | 默认 env=sim/0，actor/rollout=ascend/0-1；仅选择空闲卡                            |
| `causal.collector_device`                                          | NPU 节点上 collector 的放置卡号；它只做 CPU replay 汇总                           |
| `actor.model.cosmos_sft_toml`                                      | NPU 上与 checkpoint 配套的 TOML 绝对路径                                          |
| `actor.model.model_path`                                           | NPU 完整 causal DCP 目录，不是动作差分文件                                        |
| `actor.model.wan_vae_path`                                         | 可选 VAE 覆盖；也可在配套 TOML 填写                                               |
| `actor.model.cosmos_overrides`                                     | 如需离线 tokenizer，在此覆盖本地 processor/tokenizer 位置                         |
| `actor.model.robotwin.statistics_path`                             | NPU 上目标机器人 block statistics                                                 |
| `actor.model.robotwin.robot_config_path`                           | NPU 上正确的关节 DOF、state/action masks 映射 JSON                                |
| `causal.instruction`                                               | 当前任务的明确自然语言指令，同组必须完全相同；不自动生成任务指令                  |
| `causal.reset_seeds`                                               | 已知可初始化且与指令相符的 seeds；默认 42 只是示例                                |
| `env.assets_path`                                                  | GPU RoboTwin 根目录（含 assets/）                                                 |
| `env.task_config`                                                  | task、embodiment、相机和随机化；默认 adjust_bottle / aloha-agilex、低随机化       |
| `env.action_dim`                                                   | native 关节命令维度，双 6-DOF 臂加夹爪为 14                                       |
| `env.actions_per_block`、`causal.batch.actions_per_block`          | 必须相同且与 TOML 配套，默认 32                                                   |
| `causal.batch.video_stride/resolution/temporal_compression_factor` | 与 TOML、VAE、预训练相机契约一致；默认 4 / 480 / 4                                |
| `causal.batch.cameras/viewpoint`                                   | 默认头部加双腕 concat_view；必须匹配训练布局                                      |
| `env.fps`、`causal.batch.fps`                                      | 相同的名义控制采样频率；不是实测 SAPIEN wall-clock 帧率                           |
| `env.max_episode_steps`                                            | 首轮 64；成功或超时立即结束，不能用于最终成功率对比                               |
| `runner.output_dir`                                                | GPU/head 的配置、指标输出目录                                                     |
| `env.evidence_dir`                                                 | GPU 环境视频与命令/实测状态/逐步 reward 日志目录                                  |
| `runner.checkpoint_dir`                                            | NPU checkpoint 输出目录，两 Actor rank 都能访问                                   |
| `actor.optim.lr/clip_grad`                                         | 首轮示例 1e-5 / 1；须独立验证，不能当作已调优超参数                               |
| `causal.cfm_rtol/cfm_atol`                                         | rollout old CFM 与 Actor 重算的验收阈值，默认 0.02 / 1e-5；失败应调查，不自动放宽 |

若启用 `joint_vector`，mask 不能启用 EEF state；首版不执行 EEF 命令。`causal.compatible_dcp: true` 仅在模型加载期间安装旧 pathlib 元数据 reader，退出后还原；不写公共 checkpoint。

完整 CPU bucket 同步保持 `bucket_dtype: null`，避免改变 FP32 buffer；同步后比较 Actor/所有 Rollout 的完整 state hash、资产指纹和版本，再开始采样。资产指纹包含 TOML、统计、机器人映射、DCP metadata/config 的 SHA256，以及 DCP 文件清单/大小/mtime，不是所有 DCP 权重字节的 SHA256；基座须保持不可变。

## 2. 配置检查

```bash
cd "$RLINF_ROOT"
export PYTHONPATH="$RLINF_ROOT:$COSMOS_ROOT:/opt/rlprod/RoboTwin"
python examples/embodiment/train_causal_fpo.py \
  --config-name robotwin_causal_production --cfg job --resolve
python examples/embodiment/train_causal_fpo.py \
  --config-name robotwin_causal_production runner.check_config=true
```

检查不会启动 Ray Worker，不证明远端文件存在。实际启动时由 NPU Worker 读取模型/统计，并核对 block、stride、resolution 和 VAE。继续使用此前已通过 probe 的双节点 Ray；driver 只在 GPU/head 启动一次，不执行 `ray stop`。

## 3. 冻结策略闭环

```bash
RAY_ADDRESS=auto python examples/embodiment/train_causal_fpo.py \
  --config-name robotwin_causal_production \
  runner.mode=frozen runner.max_versions=1 \
  runner.output_dir=/opt/rlprod/outputs/causal_frozen \
  env.evidence_dir=/opt/rlprod/outputs/causal_frozen_env
```

这仍构建 Actor 并向 Rollout 同步完整权重，也生成 CFM records，但不调用 optimizer 更新。用于验收与训练相同的采样/反馈/记录路径；不是最省资源的独立推理服务。

检查 GPU/head 的 `resolved.yaml`、`metrics.jsonl`，以及 GPU 环境目录各 `<run_id>/v*_g*_m*/` 的 `observed.mp4`、`episode.json`、`feedback.jsonl`、`chunk_*_commands.npy`。视频包含初始帧和每个真实执行帧；反馈的 `control_timestamps` 是按配置 FPS 计算的控制时间轴，`execution_seconds` 是块的 wall-clock 耗时。

必须看到至少两个连续完整 block 的任务样本；第二块使用第一块真实执行后的图像及 fresh measured state。提前成功只记录已执行步，保持完整模型 endpoint 作为 FPO 样本，终止后不继续该 session。当前不做命令限幅；保存原始物理命令供核对，不能将“没有限幅操作”解释为动作都在机器人有效范围内。

## 4. 两个在线版本

冻结策略与动作语义验收通过后，使用独立、空的 checkpoint 和证据目录：

```bash
RAY_ADDRESS=auto python examples/embodiment/train_causal_fpo.py \
  --config-name robotwin_causal_production \
  runner.mode=train runner.max_versions=2 \
  runner.output_dir=/opt/rlprod/outputs/causal_train \
  env.evidence_dir=/opt/rlprod/outputs/causal_train_env \
  runner.checkpoint_dir=/opt/rlprod/checkpoints/causal_train
```

每版本：完整 reset 组采样 → 按实际执行区间计算 reward/discount → suffix 优势 → Actor 对齐 old CFM → 更新 → 推进行为版本 → 保存提交边界。下一版本等待完整权重同步与 ACK 后重新采样。

组内保持 task/reset/instruction/version 一致，成员噪声独立。真实奖励使用 task success，成功落在实际执行的最后一步；终止与超时均按有限 episode、零 bootstrap 处理。按 gamma 的逐控制时间折扣汇总，不引入 value head。

优势计算先覆盖完整组。有效 chunks 不足一个 global batch 时，在同一行为版本继续采集；达到上限仍不足就报错。达到最小批量后，无法整除 global batch 的尾部 chunks 会被丢弃并记录 `discarded_tail_chunks`，不伪造 padding 或复制样本。Actor ranks 得到不同且等量的有效 records。

`fpo/optimizer_steps_this_version` 区分实际优化步数与行为版本推进。组内奖励一致时零优势是合法结果，消耗该版本并推进，但不会发生学习更新；不能修改优势来伪造通过。若所有任务都失败，应先改善目标域 SFT 初始化或核实奖励。

## 5. 冷恢复

两个版本通过后，结束该 driver，再从 NPU 已提交目录恢复到第三个版本；保持同一配置、Actor world size、基础模型和统计资产：

```bash
RAY_ADDRESS=auto python examples/embodiment/train_causal_fpo.py \
  --config-name robotwin_causal_production \
  runner.mode=train runner.max_versions=3 \
  runner.output_dir=/opt/rlprod/outputs/causal_train \
  env.evidence_dir=/opt/rlprod/outputs/causal_train_env \
  runner.checkpoint_dir=/opt/rlprod/checkpoints/causal_train \
  runner.resume=/opt/rlprod/checkpoints/causal_train/version_000002
```

`max_versions` 是目标边界编号，不是额外运行次数。每个边界先写独立 `.partial-*` 暂存目录，全部 Actor 保存完成并写好 `COMMITTED.json` 后才原子改名为正式目录。恢复检查配置/资产/world-size，加载模型、optimizer、scheduler、RNG、行为版本和已消费版本；创建新环境并重新采样，不恢复任意物理现场。已有正式 checkpoint 目录拒绝覆盖；失败留下的 `.partial-*` 目录不能用作恢复源。恢复使用新 run ID 保存观测证据，不会与失败尝试的同版本视频目录冲突。

## 本地验证与限制

本轮运行 CPU 单元/回归测试，覆盖原生 UniPC/CFM 与两轮本地 Actor 更新、真实历史、分组/分片、部分终止、重复命令、旧版本拒绝、配置与恢复拒绝条件。使用仿真 task 测试替身检查实测 getter 与计数，不等于真实 SAPIEN 验收。

额外运行真实 RLinf CPU Worker/Channel：原生小模型生成 FPO records → 专用 collector → 两个不同 shard，检查优势和 endpoint SHA256。命令：

```bash
python tests/e2e_tests/embodied/causal_fpo_transport_smoke.py \
  --output /absolute/path/to/transport.json
```

这个测试自行创建本地 CPU Ray，不加入生产 Ray，也不停止其他集群。证据位于 Cosmos `outputs/rlinf_online_fpo_development_20261028/`。本轮未进行真实 RoboTwin、完整 Cosmos3 在线双节点训练或新同步路径的 NPU 验收；已有固定 replay 的硬件报告仍是之前版本的证据，不能冒充本轮闭环结果。先完成上述生产冻结/两轮/恢复验收，再扩充环境并发、全卡、长历史与多任务。
