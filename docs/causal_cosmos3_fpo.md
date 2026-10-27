# Causal Cosmos3 FPO++（模型侧第一版）

本实现提供 `原生 UniPC 采样 → 固定 probes → old/current 动作 CFM → FPO++ loss → backward` 的模型侧链路。环境执行、奖励、轨迹调度和优化器由调用方管理；尚未接入 RLinf。

## 设计与范围

- 保留联合视频/动作生成、causal attention、32 actions/block 等原 checkpoint 配置；没有新增模型权重或改变 checkpoint key。
- 默认只训练 `action2llm`、`llm2action`、`action_modality_embed`；可显式开放 `state2llm`、`state_modality_embed`。
- 每份 rollout 对应一个完整 causal block，保存联合 latent endpoint、实际采样的条件/无条件 packed inputs、观测历史、当前实测 state、有效维度、CFG 和策略版本。
- 首版支持原生 UniPC、batch size 1 的 joint policy rollout。多个记录可以组成更新 minibatch，但目前逐记录、逐 probe 前向。preview、纯 FD/ID、context/CFG parallelism 不在支持范围。
- 本版没有新增独立 action expert、LoRA 或隔离 mask；动作参数更新仍可能通过联合 attention 改变视频输出。若后续引入视觉奖励，应另设冻结参考策略，不能把当前视频分支当作固定奖励目标。

入口与实现：

| 文件                                                                                                   | 职责                                               |
| :----------------------------------------------------------------------------------------------------- | :------------------------------------------------- |
| [omni_mot_causal_action_model.py](../cosmos_framework/model/generator/omni_mot_causal_action_model.py) | `sample_fpo_rollout()`、`fpo_training_step()`      |
| [causal_fpo.py](../cosmos_framework/model/generator/algorithm/rl/causal_fpo.py)                        | rollout/probe/example 数据契约、CFM 重算、参数冻结 |
| [fpo.py](../cosmos_framework/model/generator/algorithm/rl/fpo.py)                                      | FPO++、消融配置、chunk return 和组内优势           |
| [sampling.py](../cosmos_framework/inference/causal_action/sampling.py)                                 | 可选采样记录；普通推理默认关闭，无 RL 依赖         |

## 目标函数

约定 `sigma=0` 为数据，`sigma=1` 为噪声，保持 Cosmos 的速度方向 `epsilon - endpoint`。

对每个 probe，视频和动作使用同一个 sigma，只对当前 block 的生成区域加噪；历史、state 和其他条件保持不变。输入必须保留联合视频 endpoint，不能只保存动作。

```text
z_sigma = where(generation_mask, (1 - sigma) * z + sigma * epsilon, z)
ell[q]  = mean_valid_action((v_theta(z_sigma, sigma) - (epsilon - z)) ** 2)
rho[q]  = exp(ratio_scale * (ell_old[q] - ell_current[q]))

A >= 0: objective[q] = min(rho[q] * A, clip(rho[q], 1-eps, 1+eps) * A)
A <  0: objective[q] = rho[q] * A - abs(A) * (rho[q]-1)**2 / (2*eps)
loss = -mean_chunks(mean_probes(objective))
```

这是动作 CFM 的 **surrogate ratio**，不是真实 action likelihood、KL 或 entropy。默认按有效动作元素取均值；`ratio_scale` 显式控制尺度，不能把其他实现的求和损失超参数直接搬过来。

CFM 使用 FP32 误差和 ratio。probe 为 CPU 独立随机源生成的固定高斯噪声与均匀 `sigma ∈ [0.005, 0.995]`，与 UniPC 的采样初始噪声、shifted timestep grid 分开。old/current 必须复用同一组实际 probe 张量。

重算使用与采样一致的 CFG 组合和 guidance interval；启用 `normalize_cfg` 时，仍以联合视频/动作的有效速度计算 norm。区间单位为模型 timestep，例如默认 UniPC 为 `sigma * 1000`。

默认 `FPOConfig()` 开启逐 probe ratio 与 ASPO；可用以下配置做公平消融：

| 方法         | `per_probe_ratio` | `asymmetric` |
| :----------- | :---------------- | :----------- |
| FPO          | `False`           | `False`      |
| 逐 probe PPO | `True`            | `False`      |
| FPO++        | `True`            | `True`       |

`clip_epsilon=0.2`、`ratio_scale=1.0` 是接口初值，未经预训练 Cosmos3 RL 调参验证。log ratio 默认以 STE 限幅到 `[-10, 10]`，避免指数溢出并保留回拉梯度；`fpo/log_ratio_saturation` 应接近零，持续饱和时应检查更新幅度与尺度。

## 调用方式

前提：已加载正确的 `OmniMoTCausalActionModel` 和预训练动作/state 权重，并由现有数据适配器构建 `batch`。继续训练时不要沿用基座初始化 recipe 中跳过动作/state 权重的加载设置。

```python
import torch

from cosmos_framework.model.generator.algorithm.rl.causal_fpo import (
    configure_fpo_trainable_parameters,
    prepare_fpo_example,
)
from cosmos_framework.model.generator.algorithm.rl.fpo import FPOConfig

# 在创建 optimizer / FSDP 前选择参数。eval() 保证 old/current 无 dropout，仍可反传。
trainable_names = configure_fpo_trainable_parameters(model.net, train_state=False)
optimizer = torch.optim.AdamW(
    [p for p in model.net.parameters() if p.requires_grad],
    lr=1e-5,
    weight_decay=0.0,
)

metadata = batch["sequence_plan"][0].causal_action_metadata
outputs, rollout = model.sample_fpo_rollout(
    batch,
    policy_version="iteration-0",
    seed=[42],  # 各候选轨迹使用不同 seed。
    causal_block_size=metadata.block_size,
    causal_history_blocks=metadata.history_blocks,
    guidance=1.5,  # 按已验证的原策略采样设置固定。
    num_steps=35,
    shift=5.0,
)
example = prepare_fpo_example(
    model, rollout,
    num_probes=2,
    probe_seed=10042,  # 与 rollout 初始噪声独立。
    behavior_version="iteration-0",
)
```

`outputs["action"]` 是模型归一化空间动作。执行前使用 [real_actions()](../cosmos_framework/inference/causal_action/outputs.py) 还原物理目标，再通过环境适配器转成控制命令；FPO endpoint 始终保留采样时的原始模型动作。

调用方收集环境反馈，将若干 `example` 与对应的 `advantages` 对齐后更新：

```python
# examples: 同一 behavior_version、同一 probe 数量的 CausalFPOExample 列表。
# advantages: 环境回报算出的 [len(examples)] 张量，不能用常量代替真实反馈。
# 所有 example.old_cfm 必须在旧权重尚未变化时完成计算。
optimizer.zero_grad(set_to_none=True)
metrics, loss = model.fpo_training_step(examples, advantages, config=FPOConfig())
loss.backward()
torch.nn.utils.clip_grad_norm_([p for p in model.net.parameters() if p.requires_grad], 1.0)
optimizer.step()
```

版本字符串检查防止混合不同采样批次，但不能证明实际权重版本。调用方负责在计算完全部 old CFM 前保持 behavior weights 不变，并在下一轮采样时同步新权重、更新版本。old CFM 和 probes 在本轮多次更新期间保持固定。

回报工具接受显式 `discounts`：每个 chunk 执行 h 个控制步时，可采用 `gamma**h`；若按 chunk 定义时间，则采用固定 gamma。chunk 内 reward 也应按相同时间约定聚合。真正终止置 discount 为零，时间截断可提供 bootstrap。

```python
from cosmos_framework.model.generator.algorithm.rl.fpo import (
    discounted_chunk_returns,
    group_relative_advantages,
)

# [reset_group, trajectory, chunk]；同组必须共享 reset 状态与任务。
returns = discounted_chunk_returns(rewards, discounts, valid=valid, bootstrap=bootstrap)
group_advantages = group_relative_advantages(returns, valid=valid)
# 按与 examples 完全一致的顺序选出有效 chunk；不要把 padding 放入 examples。
```

组内回报相同或只剩一个有效成员时，优势为零。调用方应跳过没有学习信号的更新。中途终止需记录实际执行步数及终止类型，保留完整采样 endpoint；不能把动作截短后冒充原策略样本。现有 session 的完整 block 反馈限制仍然适用。

## 验证与限制

测试使用原生 UniPC、causal packing、动作/state 投影和小型 dense attention backbone，覆盖：普通采样不变、固定 probes 重算、CFG 与归一化、联合视频依赖、历史因果性、无效维度、参数冻结、SPO 回拉梯度、一次优化更新、多 probe 梯度检查点与可信本地记录的保存/恢复。测试不加载完整预训练 checkpoint，不代表机器人任务成功率已有提升。

在已有训练依赖环境中执行：

```bash
export LD_LIBRARY_PATH=''
# 本机安装了 torch-npu；先加载其依赖库。纯 CPU/CUDA 环境省略这一行。
source /usr/local/Ascend/cann-9.1.0/set_env.sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest --noconftest \
  cosmos_framework/model/generator/algorithm/rl/fpo_test.py \
  cosmos_framework/model/generator/algorithm/rl/causal_fpo_test.py -q
```

`--noconftest` 用于这些独立 CPU 测试，避免仓库全局 fixture 初始化设备。计算 CFM 时关闭 inference KV cache，并重新建立可微 pack；冻结 backbone 权重不等于可以对 backbone 使用 `no_grad()`。当前实现保留 minibatch 全部 probe 的反向图，规模化时需要外部 microbatch/梯度累积；不要直接使用很大的轨迹组和 probe 数。

后续接 RLinf 时，新增模型 rollout/CFM 调用、episode 历史管理与奖励/优势对齐；沿用 RLinf 的 optimizer、分布式和权重同步。需要先做原生预训练 checkpoint 的固定输入数值验证，再做 RLinf 对齐和 LIBERO 闭环。不要把负 CFM 填进现有 PPO `logprobs` 字段，也不要复用其 KL/entropy 指标。

方法来源：[FPO++ 论文](https://arxiv.org/html/2602.02481)、[Cosmos3 + FPO 的联合 endpoint 方法](https://arxiv.org/html/2609.33832)。本实现采用 Cosmos 的时间方向、有效动作均值归一化及原生 CFG 重算。
