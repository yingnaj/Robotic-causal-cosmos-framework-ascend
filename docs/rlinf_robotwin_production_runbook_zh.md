# Causal Cosmos3 FPO++：RoboTwin 双节点生产验证步骤

日期：2026-10-27。检查对象是当前工作区代码，包括尚未提交的修改。本文是迁移检查和执行方案；本轮没有连接生产节点，没有修改训练实现，也没有启动生产实验。

后续进展：已参考 `feat/robotwin-client-server@b4e90f0` 完成原生状态/unified55/native 命令转换及 RLinf 配置接入，见 [适配实现与分支复核](robotwin_adapter_and_branch_audit_zh.md:1)。下文关于相机、实测状态采集、环境反馈、collector 和生产启动的缺口仍成立；动作转换不再需要从零实现。

**结论：可以迁移做环境准备和单模块验证；尚不能直接启动双节点 RoboTwin 在线 FPO++。** 原生状态/动作转换已实现，仍需接入环境反馈、causal rollout/collector 和 FPO 训练入口。已有 NPU 单/双卡验证可作为模型侧回归基线，不能替代环境闭环验收。

具体执行命令见 [迁移后的首轮验证步骤](rlinf_production_validation_steps_zh.md:1)。GPU 节点已具备 RoboTwin 依赖，直接复用；新提供环境 probe、RLinf 跨节点 transport 和 NPU FSDP2 probe，尚待生产机实际运行。

## 当前迁移就绪检查（动作适配完成后）

本次重新检查了两仓库当前代码、adapter 调用位置、配置与启动文件是否存在，以及本机验证脚本的路径/设备/Ray 设置。本次是静态就绪检查，没有访问生产节点或重复启动模型训练；上一轮 56 项 CPU 测试结果见 [适配验证](robotwin_adapter_and_branch_audit_zh.md:90)。

| 测试目标                               | 当前是否可开始                 | 迁移要求或阻塞点                                                                                            |
| -------------------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| 安装、设备识别、Ray 双节点加入         | 可以                           | 按实际 IP/网卡/解释器修改第 4–6 节命令；成功加入不等于业务通信通过                                          |
| RoboTwin 状态/动作转换单测             | 可以                           | 两仓库新文件完整部署，运行适配文档的 CPU 回归命令；不需要正式 checkpoint                                    |
| RoboTwin 独立 reset/step/render        | 可先做环境自检                 | 需要 GPU 上兼容的 RoboTwin 和资产；已有 reset/render 和 qpos 回放 probe；完整终止语义、实测状态来源仍待验收 |
| NPU 单/双卡固定 replay 更新            | 有已验证实现，脚本不能原样部署 | 迁移资产、依赖和 reader，改掉原路径/卡号；它是独立本地验证，不是双节点 RL                                   |
| B：RLinf 异构 Worker/Channel 数据往返  | 已有 probe，尚未实机验收       | 已有 YAML 驱动的 transport probe；需在生产节点验证设备隔离、解释器、CPU 张量与嵌套测试消息                  |
| C：causal Cosmos 冻结策略控制 RoboTwin | 不可直接启动                   | 缺相机/history batch builder、实测状态采集接入、per-env session 和反馈路由                                  |
| D/E：在线 FPO 更新及环境闭环恢复       | 不可直接启动                   | 缺在线 collector、完整 FPO Runner/recipe、权重同步与阶段生命周期验收                                        |
| F/G：全卡扩展、约五任务效果实验        | 暂不具备                       | 先通过 C–E，并确定目标域 checkpoint、统计和任务契约                                                         |

确认仍存在的直接阻塞证据：

1. [builder](/data5T/bangyan/workspace/RLinf/rlinf/models/embodiment/cosmos3/__init__.py:268) 只将 adapter 挂到 policy。`pack_states/decode_actions` 当前在测试中被调用，环境/rollout 业务路径尚未消费它们；[predict_action_batch](/data5T/bangyan/workspace/RLinf/rlinf/models/embodiment/cosmos3/causal_policy.py:135) 仍抛出 `NotImplementedError`。
2. [普通训练入口](/data5T/bangyan/workspace/RLinf/examples/embodiment/train_embodied_agent.py:96) 仍选择普通 Actor 和 HF rollout；第 7 节的 `train_causal_fpo.py` 及完整双节点 FPO YAML 仍不存在。现有 [模型片段](/data5T/bangyan/workspace/RLinf/examples/embodiment/config/model/cosmos3_causal_fpo.yaml:5) 还有 `???`，且 `robotwin: null`，不能直接运行。
3. [环境 wrapper](/data5T/bangyan/workspace/RLinf/rlinf/envs/sim/robotwin/robotwin_env.py:198) 仍直接使用 `obs["state"]`，尚未核实/改为实测采集；[chunk_step](/data5T/bangyan/workspace/RLinf/rlinf/envs/sim/robotwin/robotwin_env.py:388) 仍按请求动作数计步。转换函数不能自动修正这些环境语义。
4. [本机 Actor 验证脚本](../outputs/rlinf_causal_fpo_validation_20261027/validate_actor.py:10) 强制 `RAY_ADDRESS=local`，在第 152 行固定 NPU 6、7，并依赖旧目录内的 metadata reader、replay 和参考 checkpoint。把它放到已启动的生产 Ray 集群上执行，不能得到所需双节点实验。
5. 关键实现仍有未跟踪文件，普通 clone/pull 不会携带；依赖快照与验证产物位于 `outputs/`，也须显式归档。激活脚本除环境目录外，还固定 [Conda 初始化路径](../tools/activate_rlinf_cosmos3_ascend.sh:10) 为 `/root/miniconda3/etc/profile.d/conda.sh`；生产 Conda 根目录不同须调整初始化，单设 `COSMOS_RLINF_CONDA_ENV` 不足以覆盖该路径。

下一批最小交付顺序：先运行新增环境/异构通信/微型 FSDP probe，并整理真实模型固定 replay 入口；再接 measured state 与块内反馈、完整块 batch builder 和 session；最后接 collector、FPO Runner、独立 Rollout 权重同步与完整 YAML。前一项可让生产机先开展基础设施验证，后两项完成后才能验证用户目标中的双节点在线 RL。

## 1. 拓扑与首轮范围

- 节点 0：8 张 RTX 3090，运行 Ray head、轻量 driver、RoboTwin EnvWorker。首轮只使用 GPU 0。
- 节点 1：8 张 Ascend 910B，运行 Actor、Rollout、FPO replay/优势汇总。首轮只使用 NPU 0、1；确认这些卡在生产机空闲。
- 首轮同步 on-policy，Actor 与 Rollout 分阶段使用相同 NPU。两套 Worker 不等于两套模型能同时常驻；必须验证参数/优化器 offload 与采样阶段切换。
- GPU ↔ NPU 只传 CPU 图像、状态、环境命令、奖励和身份元数据；完整 FPO replay 留在 NPU 节点。Actor FSDP/HCCL 组仅包含 NPU。
- 暂以参考配置中的 `adjust_bottle`、`aloha-agilex` 为首个候选。任务、控制接口、正式 causal checkpoint 尚未最终确定，这些选择须写入实验 manifest 后才能进行策略效果实验。

参考配置的 `env.placement: 0` 只使用一张 GPU；它的 `actor/rollout: 0-7` 才表示各 8 个 NPU rank。[参考配置](/data5T/bangyan/workspace/RLinf/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05_2nodes.yaml:17)。CPU 跨设备通信已有 Gloo 实现，仍需在生产机验证：[通信选择](/data5T/bangyan/workspace/RLinf/rlinf/scheduler/collective/multi_channel_pg.py:107)。

## 2. 必须补齐的实现

| 优先级 | 修改位置                                   | 当前问题与具体改造                                                                                                                            | 验收条件                                                                       |
| ------ | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| P0     | RLinf：状态/动作 adapter 与输入构造        | 状态/动作转换和配置接入已实现；仍需将相机、instruction、实测 state/history 组装为原生 causal batch，并接入环境控制链                          | 专家动作往返映射和环境执行一致；无二次归一化；缺少映射/统计时明确报错          |
| P0     | RLinf：`robotwin_env.py`                   | 保留逐步执行图像、实际执行长度、terminal observation、success、终止原因；按实际执行长度累计步数和奖励位置；训练与评估都保留有效长度           | 完整块、提前成功、超时、reset 均对齐；无 padding 帧进入真实历史                |
| P0     | RLinf：`env_worker.py` 与消息协议          | 当前普通 policy 输入没有完整块内反馈；增加稳定 env/episode/chunk/group/member/reset/version 身份与时间戳；块内反馈独立于 LeRobot 录制开关     | 并发环境交换顺序后不串历史；重复消息不重复执行；CPU 传输不带设备张量           |
| P0     | RLinf：causal policy 与专用 rollout worker | 普通 `predict_action_batch` 尚未实现；新增 per-env session、prepared batch builder、`sample_fpo_batch` 调用、完整 endpoint 保存和物理动作输出 | 一个 episode 连续至少两个 block；下一块使用真实执行视频与新实测 state          |
| P0     | RLinf：FPO collector                       | 按同 task/reset/goal 分组，独立策略噪声产生成员；汇总 reward/discount/valid 元数据，计算 suffix 优势，再形成等长有效 `CausalFPOBatch` shards  | 无跨任务分组、重复成员、缺失成员、跨版本 replay；不复制同一 shard 冒充数据并行 |
| P0     | RLinf：`config.py`、训练入口、Runner       | 增加 FPO 配置校验、Actor/rollout/collector 选择、同步版本 ACK、阶段调度、边界保存恢复；不走 PPO 的 logprob/value/GAE 流程                     | 两次真实采样→奖励→更新→同步；新进程恢复后继续；旧 replay 被拒绝                |
| P0     | RLinf：NPU Worker 初始化与模型加载         | 验证脚本中的 NPU bootstrap、公共 DCP pathlib 兼容目前未变成统一生产入口                                                                       | Actor/Rollout 都可从新进程加载同一资产；GPU head 不导入 NPU 扩展               |
| P1     | Cosmos：目标域数据/资产                    | 缺少 RoboTwin 域 checkpoint 时补数据 reader、统计和必要 SFT；保持当前 attention、UniPC、FPO++ 与 checkpoint key                               | 冻结策略有可解释的 RoboTwin 行为；模型/相机/动作模板一致                       |
| P1     | RLinf：独立 rollout 权重同步               | 复用 bucket sync，先完整同步并比较参数名/形状/hash；确认基座相同后再考虑只同步动作参数                                                        | rollout 在版本 ACK 后开始新采样；Actor/Rollout 的同输入 CFM 在验收容差内一致   |

### 2.1 代码证据与容易遗漏的边界

1. **55D 不能直接截成 14D。** 模板左/右关节槽分别为 `0:7`、`7:14`，夹爪位于 `28:30`。对于实际为双臂各 6 关节加夹爪的接口，要显式按机器人关节名映射并屏蔽未用槽；不能猜测目标机器人维度。见 [模板](../cosmos_framework/data/generator/action/action_state_template.py:78)。环境当前对 RoboTwin 动作直接透传，不会自动转换：[prepare_actions](/data5T/bangyan/workspace/RLinf/rlinf/envs/action_utils.py:372)。模型 55D 与环境控制维度必须分别表达。
2. **块内图像已有基础，但还没接到 causal rollout。** `skip_intermediate_renders: false` 调用 `step_with_full_obs`；wrapper 已读取 `executed_action_count`，但将它移出 infos，训练时只输出 `valid_action_mask`，评估时不输出该 mask。见 [chunk_step](/data5T/bangyan/workspace/RLinf/rlinf/envs/sim/robotwin/robotwin_env.py:336)。应保留显式 count 和有效帧序列，不依赖评估/录制开关。
3. **提前终止的记账需要修正。** `_cal_chunk_rewards` 中 `n_steps_to_run` 固定为零，chunk 的 elapsed_steps 仍增加请求长度，termination/truncation 放在最后一格；不能直接用于执行时刻对齐。见 [奖励](/data5T/bangyan/workspace/RLinf/rlinf/envs/sim/robotwin/robotwin_env.py:228)、[步数](/data5T/bangyan/workspace/RLinf/rlinf/envs/sim/robotwin/robotwin_env.py:388)。保留已有 `final_observation/final_info`，避免把 reset 后图像当作上一 episode 的末帧。
4. **full obs 返回值不等于已经传到策略。** EnvWorker 将块内观测保存在独立 payload，但普通 `PolicyInput` 只传当前 obs 等字段；部分执行 mask 还受 `enable_online_lerobot` 控制。见 [环境处理](/data5T/bangyan/workspace/RLinf/rlinf/workers/env/env_worker.py:400)、[PolicyInput](/data5T/bangyan/workspace/RLinf/rlinf/workers/env/env_worker.py:850)。需要明确新增 causal feedback 消息契约。
5. **session 只接受完整块反馈。** [CausalActionSession](../cosmos_framework/inference/causal_action/session.py:90) 拒绝部分块 commit。首版提前终止后销毁该 session，保留完整 FPO endpoint，只按实际执行区间计奖励，不继续该 episode 的部分块重规划。普通 `for_model` 走常规生成；RL rollout 要把 `sample_fpo_batch` 的 trace/record 一起保存，不能只调用普通 session 后丢掉训练样本。
6. **默认 HF rollout 不兼容。** [causal policy](/data5T/bangyan/workspace/RLinf/rlinf/models/embodiment/cosmos3/causal_policy.py:126) 明确抛出未实现；默认 [HF worker](/data5T/bangyan/workspace/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py:614) 仍依赖 `prev_logprobs`。CFM 不能填入该字段充当 logprob。保留专用 FPO 数据路径。
7. **现有 Actor 有严格分片要求。** [FPO Actor](/data5T/bangyan/workspace/RLinf/rlinf/workers/actor/fsdp_fpo_actor_worker.py:115) 要求各 rank 有相同有效 chunk 数、probe 数和版本。补配置检查 `global_batch_size % actor_world_size == 0`，并保证每 rank batch 可被 microbatch 整除。collector 先按完整组计算优势，再从有效 chunks 中形成等长训练批；不足一个全局批时在同一行为版本继续收集，不能伪造 padding 样本。
8. **原 PPO 配置里的 200/25 不可照搬。** 原 `num_action_chunks=25` 与 `max_steps_per_rollout_epoch=200` 相容；若目标 causal 配方为 32 actions/block，则 200 不再整除 32，当前 [配置校验](/data5T/bangyan/workspace/RLinf/rlinf/config.py:1278) 会拒绝。首个短闭环可用 64 步上限，后续候选上限 192；模型 block 长度以 checkpoint/TOML 为准。人为缩短上限会改变任务难度，不用于最终成功率对比。
9. **零优势是有效结果。** sparse success 下同组全部失败会得到零优势，Actor 跳过更新；Runner 仍须推进已完成的采集版本，防止重复消费。不得用手填优势把生产闭环标成学习成功。若长期没有组内奖励差异，先改善 SFT 初始化或验证奖励设计。

Cosmos 核心暂不需要再改 attention 或重写 FPO loss。主要工程量在 RLinf 的机器人/执行协议与训练编排；NPU 已通过的模型检查见 [验证报告](rlinf_causal_fpo_npu_validation.md:5)。

## 3. 发布包和资产准备

1. 两节点部署同一份 RLinf 修订代码。当前工作区包含未跟踪的 causal policy、FPO Actor、模型 YAML 和测试，单独 `git pull` 或只传 `git diff` 都会遗漏；发布包必须包含这些文件及 SHA256 清单。保留两仓库 commit、dirty diff 和实际文件 hash。
2. 统一约定代码路径，例如两节点均使用 `/opt/rlprod/RLinf`；NPU 节点另有 `/opt/rlprod/Cosmos`。下文路径是生产部署约定，须按实际安装位置替换。配置中的解释器、TOML、checkpoint、processor、VAE、stats 必须使用拥有该资产节点上的绝对路径。
3. NPU 节点准备完整 causal DCP、配套 TOML、VAE、processor/tokenizer、RoboTwin state/action stats 和模板 mask。完整模型只加载一次正确基座；不能用仅含动作差分的文件冒充完整 checkpoint。
4. GPU 节点准备 RoboTwin `RLinf_support` 兼容版本及资产、训练/评估 seeds。`assets_path` 指 RoboTwin 根目录，即 `assets/` 的上一级。
5. 保存 task/embodiment、控制单位/频率、夹爪范围、相机名称/拼接/裁剪/分辨率、动作模板版本、训练/评估 split、奖励定义。先冻结相机与随机化。约 5 个 task 后续逐个独立验收，首轮不混任务分组。
6. 迁移既有模型验证证据时保留 `replay.pt`、对应 checkpoint、配置和脚本。当前 `outputs/.../validate_actor.py` 固定本地 Ray、路径和卡号，不能直接加入生产 Ray 集群；先将其需要的验证逻辑整理进读取 YAML 的正式 probe，或者只在独立的迁移前检查中运行。

最终 checkpoint 未确定不妨碍安装、渲染和通信验证；可以用 Subtasks 做加载/反向工程检查，但不能据此宣称 RoboTwin 策略有效。

## 4. 分别准备两个节点的运行环境

### 4.1 GPU 节点

使用独立 CUDA 环境；不要复制 aarch64 昇腾 Conda 环境到 GPU 机器。优先复用已有的 RoboTwin 环境。当前安装器的 env-only 分支没有 RoboTwin，单独 `embodied --env robotwin` 会被拒绝，见 [install_env_only](/data5T/bangyan/workspace/RLinf/requirements/install.sh:3117)。需要新建环境时，可以先使用已有 OpenPI/RoboTwin 安装分支；它会额外安装 OpenPI 依赖，但环境节点不加载 OpenPI 模型。下列是候选命令，须结合 GPU 驱动与可用 CUDA wheel 验证，本轮未在 3090 上执行：

```bash
cd /opt/rlprod/RLinf
bash requirements/install.sh embodied --model openpi --env robotwin \
  --python 3.12.14 --torch 2.10.0 \
  --venv .venv-sim --no-flash-attn --install-rlinf
source .venv-sim/bin/activate
uv pip install --python .venv-sim/bin/python 'ray==2.59.0' 'cloudpickle==3.1.2'
```

选择与已验证 NPU 环境相同的 Python/Ray/cloudpickle 版本；CUDA 和 NPU 扩展各自安装。安装器在 NVIDIA 路径还会构建 cuRobo 等组件，即使 task 选择 mplib；它不是轻量的纯环境 wheel 安装。相关逻辑：[install_robotwin_env](/data5T/bangyan/workspace/RLinf/requirements/install.sh:3682)。后续若需要精简部署，再为 `install_env_only` 增加 RoboTwin 分支并验证依赖；不把它当作目前已有能力。

按 [RLinf RoboTwin 安装说明](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html) 获取 `RLinf_support` 分支和资产，然后固定已验证 commit，不在实验间自动跟随分支更新。选择 `planner_backend: mplib` 作为首轮配置；验收真实 SAPIEN 离屏渲染，不把 MuJoCo 的 EGL 环境变量当作渲染已通过。

```bash
export ROBOTWIN_PATH=/opt/rlprod/RoboTwin
export PYTHONPATH=/opt/rlprod/RLinf:/opt/rlprod/RoboTwin
python -c 'import sys, ray, torch, sapien, mplib; print(sys.version, ray.__version__, torch.__version__); print(torch.cuda.device_count())'
python -c 'from robotwin.envs.vector_env import VectorEnv; assert hasattr(VectorEnv, "step_with_full_obs")'
nvidia-smi
```

通过条件：CUDA 识别 8 卡；实际 reset/专家动作/step/render 的图像有效；`executed_action_count` 在提前结束时小于请求长度且准确。最后一项需要实际环境测试，import 检查不能代替。

### 4.2 NPU 节点

复用当前已验证的 `cosmos3-rlinf-ascend` 环境版本和激活钩子。依赖快照见 `outputs/rlinf_cosmos3_env_setup/{environment.yml,conda-explicit.txt,pip-freeze.txt}`；处理 editable 安装的路径变化。CANN、驱动、架构和本地编译的 torchcodec 必须匹配，不直接运行默认 RLinf 安装器覆盖已经验证的 torch/torch-npu 栈。

```bash
cd /opt/rlprod/Cosmos
source tools/activate_rlinf_cosmos3_ascend.sh
export PYTHONPATH=/opt/rlprod/RLinf:/opt/rlprod/Cosmos
python -c 'import sys, ray, torch, torch_npu; print(sys.version, ray.__version__, torch.__version__, torch_npu.__version__); print(torch.npu.device_count())'
npu-smi info
```

若只有 Conda 环境目录变化，在 source 前设置 `COSMOS_RLINF_CONDA_ENV`；若 Conda 根目录也变化，还须调整激活脚本第 10 行的 `conda.sh` 路径。不要清空 NPU 的 `LD_LIBRARY_PATH`。当前已验证组合为 Python 3.12.14、Ray 2.59.0、torch 2.10.0、torch-npu 2.10.0.post6、CANN 9.1.0，见 [环境快照](../outputs/rlinf_cosmos3_env_setup/README.txt:18)。

生产 Actor/Rollout 都须在各自 NPU 进程中完成设备兼容初始化；不得在 GPU head 全局执行 `transfer_to_npu`。公共 DCP 的跨 Python pathlib 兼容要变成明确可测试的 reader，或预先生成兼容 checkpoint 副本，不修改公共源文件。首轮检查：完整加载、单卡 sample/CFM/backward、双卡 FSDP/HCCL、保存恢复。

## 5. 双节点 cluster 配置

以下 **cluster 片段使用现有 RLinf schema**，只描述放置；它不是完整 FPO 训练 recipe。每个节点解释器路径按实际环境修改。

```yaml
cluster:
  num_nodes: 2
  component_placement:
    env:
      node_group: sim
      placement: "0"
    actor:
      node_group: ascend
      placement: "0-1"
    rollout:
      node_group: ascend
      placement: "0-1"
  node_groups:
    - label: sim
      node_ranks: 0
      env_configs:
        - node_ranks: 0
          python_interpreter_path: /opt/rlprod/RLinf/.venv-sim/bin/python
          env_vars:
            - PYTHONPATH: /opt/rlprod/RLinf:/opt/rlprod/RoboTwin
            - ROBOTWIN_PATH: /opt/rlprod/RoboTwin
            - ROBOT_PLATFORM: ALOHA
    - label: ascend
      node_ranks: 1
      env_configs:
        - node_ranks: 1
          python_interpreter_path: /root/miniconda3/envs/cosmos3-rlinf-ascend/bin/python
          env_vars:
            - PYTHONPATH: /opt/rlprod/RLinf:/opt/rlprod/Cosmos
            - COSMOS_FRAMEWORK_PATH: /opt/rlprod/Cosmos
```

不把两个节点的动态库路径写成同一个值；分别在启动 Ray 前激活对应环境。RLinf 会合并 head 在 Ray 启动后修改的环境变量，所以 driver 启动前不要再切换另一套 CUDA/CANN 环境；逐节点 `env_vars` 最终覆盖的语义见 [node.py](/data5T/bangyan/workspace/RLinf/rlinf/scheduler/cluster/node.py:399)。

完整 FPO recipe 需补以下配置和检查，字段最终由新增入口校验；不复制参考 PPO 的 `gae`、value head、entropy、staleness 参数：

- Actor：`cosmos_variant: causal_action`、`training_mode: fpo`、FSDP2、BF16 参数/FP32 reduce、`disable_cosmos_parallelism: true`、无 EMA、`train_state: false`。完整加载所有接口再冻结共享网络。
- 算法：初始 `update_epoch: 1`；FPOConfig 使用 `per_probe_ratio: true`、`asymmetric: true`、`clip_epsilon: 0.2`、`ratio_scale: 1.0`，记录 saturation。它们是当前默认值，见 [FPOConfig](../cosmos_framework/model/generator/algorithm/rl/fpo.py:15)。
- 首轮 2 个 Actor rank：microbatch=1，global batch=4，至少 4 个真实有效 chunks；G=4，4 个环境，1 个 EnvWorker，pipeline=1。每 rank 2 chunks。生产学习率从独立小规模试验确定，不沿用数值验证脚本的 `lr=1e-3/clip_grad=1e6`。
- 采样工程检查可先用既有 4 UniPC steps、2 probes、history=1、无 KV cache；这是接口验证设置。评估行为及学习效果应使用目标 checkpoint 的合适采样参数，并让 Actor 重算与 Rollout 严格一致。
- 环境：`skip_intermediate_renders: false`、`ignore_terminations: false`、相机/分辨率与模型契约一致；先由 collector 明确处理 episode/group reset，验证后再开启自动 reset。短闭环用 64 步/2 blocks；任务正式上限单独验收。
- 奖励先用真实 task success，区分成功、失败和超时；success 与 termination 关系必须核对底层 task。后续再增加视觉奖励，不默认引入 Ctrl-World 或把当前策略预测视频当作冻结奖励目标。
- 权重同步优先现有 `type: bucket`、`bucket.bucket_device: cpu`；先完整同步确认正确，再考虑 patch/delta。同步配置不决定 Env↔Rollout 消息设备，后者必须独立保证 CPU。
- `enable_offload` 与同步阶段生命周期配套测试；关闭异步预取和环境 bootstrap overlap，避免未确认版本的采样或阶段内存峰值重叠。

## 6. 启动 Ray 集群：当前可以执行的部署步骤

先确认两节点 IP 双向可达、路由/网卡正确；开放选定 Ray 端口及节点间 Worker/Gloo 通信所需端口，而不只是 head 的 6379。两节点挂载/本地路径按角色核对。下面命令需先设置实际 `SIM_IP`、`NPU_IP`、`SIM_IFACE`、`NPU_IFACE`。

**节点 0（GPU/head），在已经激活 `.venv-sim` 的终端：**

```bash
: "${SIM_IP:?Set the GPU node IP}"
: "${SIM_IFACE:?Set the GPU node network interface}"
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES="$SIM_IFACE"
export GLOO_SOCKET_IFNAME="$SIM_IFACE"
export PYTHONPATH=/opt/rlprod/RLinf:/opt/rlprod/RoboTwin
export EMBODIED_PATH=/opt/rlprod/RLinf/examples/embodiment
export REPO_PATH=/opt/rlprod/RLinf
export ROBOTWIN_PATH=/opt/rlprod/RoboTwin
export ROBOT_PLATFORM=ALOHA
ray start --head --node-ip-address="$SIM_IP" --port=6379
```

**节点 1（NPU），在已经激活 `cosmos3-rlinf-ascend` 的终端：**

```bash
: "${SIM_IP:?Set the GPU head IP}"
: "${NPU_IP:?Set the NPU node IP}"
: "${NPU_IFACE:?Set the NPU node network interface}"
export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES="$NPU_IFACE"
export GLOO_SOCKET_IFNAME="$NPU_IFACE"
export PYTHONPATH=/opt/rlprod/RLinf:/opt/rlprod/Cosmos
export COSMOS_FRAMEWORK_PATH=/opt/rlprod/Cosmos
ray start --address="$SIM_IP:6379" --node-ip-address="$NPU_IP"
```

回到节点 0 执行以下检查：

```bash
ray status
python - <<'PY'
import ray
ray.init(address="auto")
nodes = [node for node in ray.nodes() if node["Alive"]]
assert len(nodes) == 2, nodes
for node in nodes:
    print(node["NodeManagerAddress"], node["Resources"])
ray.shutdown()
PY
```

这只证明 Ray 有两个活节点，不证明 RLinf 的 3090/910B 资源映射正确。下一步须运行 RLinf Worker probe，逐节点记录 `sys.executable`、代码路径/hash、hostname、node rank、设备类型/卡号、动态库环境和 CPU 往返数据 hash。probe 要采用上述 cluster placement，不通过普通 Ray 任意调度替代 RLinf 验收。现已提供 `tools/validate_rlinf_cluster.py --stage transport`，具体命令见 [首轮验证步骤](rlinf_production_validation_steps_zh.md:1)；真实业务反馈与 FPO replay 仍需后续验收。

节点 rank 必须在 `ray start` **之前**设置，训练 driver 只启动一次。依据 [RLinf 多节点说明](/data5T/bangyan/workspace/RLinf/docs/source-zh/rst_source/guides/multi_node.rst:28) 和 [Ray 自建集群文档](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html)。

## 7. 补齐代码后的训练启动与验收顺序

建议交付的正式文件（**当前尚不存在，以下不是现成命令**）：

- `examples/embodiment/train_causal_fpo.py`：读取 Hydra YAML，复用 RLinf Cluster/Worker/FSDP/Channel，选择 causal rollout、FPO collector 和现有 FPO Actor。
- `examples/embodiment/config/robotwin_adjust_bottle_causal_cosmos3_fpo_2nodes.yaml`：完整 recipe。
- 独立环境/跨节点 probe 现已提供，见 [首轮验证步骤](rlinf_production_validation_steps_zh.md:1)。上述训练入口的 `--cfg job --resolve` 将仅打印配置，业务契约仍需专门校验。

上述文件实现、测试并部署后，才在 GPU/head 上执行预期入口：

```bash
cd /opt/rlprod/RLinf
python examples/embodiment/train_causal_fpo.py \
  --config-name robotwin_adjust_bottle_causal_cosmos3_fpo_2nodes \
  --cfg job --resolve

python examples/embodiment/train_causal_fpo.py \
  --config-name robotwin_adjust_bottle_causal_cosmos3_fpo_2nodes
```

当前不要用普通 `run_embodiment.sh` 替代：它固定调用 [train_embodied_agent.py](/data5T/bangyan/workspace/RLinf/examples/embodiment/run_embodiment.sh:5)，后者仍使用普通 HF rollout 和默认 Actor，尚未选择 FPO 分支。

| 阶段        | 运行内容                                                               | 通过条件                                                                            |
| ----------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| A：环境独立 | GPU 0、单 task、专家/回放动作；测试正常结束、提前成功和超时            | 动作/实测状态一致，图像可读，实际 count/时间戳/terminal obs 正确                    |
| B：异构传输 | GPU 0 ↔ NPU 0，CPU 图像/state/action/reward 往返；同时测试独立进程环境 | 字段/dtype/hash 一致，无跨设备 Tensor 反序列化错误；无 NPU/CUDA 动态库串用          |
| C：冻结策略 | GPU 0、NPU 0，固定 reset seeds，至少两个连续 block 和完整 episode      | 真实反馈进入下一 block；无串 session/提前 reset；记录视频、成功率、限幅率和延迟     |
| D：最小 RL  | GPU 0 上 4 env/G=4，Actor/Rollout 各 2 rank 分阶段；至少两个采集版本   | old/current CFM 初始对齐，梯度有限，仅动作参数改变，下一版本重新采样，权重 ACK 完整 |
| E：恢复     | D 的 update 边界保存，销毁 Worker 后重建加载并重置环境                 | 模型/optimizer/scheduler/version/RNG 恢复；旧 replay 拒绝；新 rollout 后可继续更新  |
| F：容量扩展 | NPU 2→4→8 rank；GPU EnvWorker 1→2→4→8；分次增加环境数/历史/probes      | 无死锁/OOM；逐 rank 内存、CPU RSS、通信量和耗时可解释；至少连续 20 个采集周期       |
| G：任务效果 | 约 5 个任务分别固定训练/评估 split；SFT baseline 与 RL 后相同评估协议  | 每任务多随机种子、成功率/置信区间、样本量和真实更新次数；不只看平均训练奖励         |

阶段 D 如果奖励完全相同，正确结果是零优势跳过；这通过数据流检查，但不通过“发生有效学习更新”的验收。应单独报告两者，不能更改优势来掩盖缺少有效奖励。

### 7.1 扩展到两节点全部 16 张卡

在阶段 D/E 通过后，将三个角色的 placement 分别改为：`env: sim/0-7`、`actor: ascend/0-7`、`rollout: ascend/0-7`。保持 8 个 EnvWorker 各自使用一张 3090；验证 SAPIEN 子进程实际渲染卡号，不能仅凭 `CUDA_VISIBLE_DEVICES` 推断 Vulkan 使用的设备。

首个全卡容量候选：32 个环境、G=4、pipeline=1，即每个 EnvWorker 4 个环境；Actor world size=8、microbatch=1、global batch=8 或 16。各 Actor rank 仍须拿到等量有效 chunks。Rollout 的 8 个 rank 通常是 8 个模型副本，不是一个跨 8 卡切分的模型；每个副本必须能独立容纳采样峰值。若目标 checkpoint 放不下单卡，应先实现分片 rollout。

不要从“单/双卡约 7–10 GiB”推算最终容量：该证据对应 Edge 验证资产、history=1、2 probes、4 steps。真实相机、历史、模型、同时驻留和环境数量会改变 HBM/CPU 内存。记录每阶段 allocated/reserved HBM、GPU 显存、主机 RSS、replay 大小、网络 MB/s、env FPS、采样/重算/反向/同步耗时和等待时间。

### 7.2 保存与故障恢复

首版只承诺 update 边界恢复。保存模型、optimizer、scheduler、RNG、行为版本、已消费版本、optimizer_steps，同时保存 task/reset/asset/config manifest 和完整提交标记。失败在 update 中途时回到最近完整 checkpoint，丢弃未提交 replay，重置环境并重新采样；不声称可恢复任意 RoboTwin 物理状态。

恢复先在原 world size 验证，再另测重分片恢复。状态加载检查保持严格一致；NPU 后续反向可能不逐位确定，沿用原生重复性基线和独立数值阈值，不放宽 checkpoint 完整性检查。

## 8. 交付完成的判定

迁移可按顺序交付三项：**生产可用环境/通信 probe → 单任务两次在线 FPO 更新与冷恢复 → 全卡扩展和约五任务效果实验**。目前第一项中的安装、Ray 启动可以按本文开始；后两项受第 2 节代码缺口和目标域资产约束，不能以已有固定 replay 验证代替。
