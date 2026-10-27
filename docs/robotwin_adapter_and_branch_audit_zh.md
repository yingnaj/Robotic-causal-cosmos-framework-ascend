# RoboTwin 动作适配与 client-server 分支复核

日期：2026-10-27。参考本地远端引用 `origin/feat/robotwin-client-server`，固定提交 `b4e90f0278d9797343ed79fce25bdbb0780970c0`。本轮选择性迁入状态/动作转换，未合并整个分支，未改变现有 FPO 的完整块采样行为。

## 1. 已实现的转换层

- [共享 RobotwinAdapter](../cosmos_framework/utils/robotwin.py:13)：来自分支的 `inference/causal_action/robotwin.py`，保留关节 DOF、source masks、EEF frame、statistics coverage 的检查，以及 world/base 变换。放在 `utils` 供训练和推理共同使用，不让 RLinf 训练路径依赖推理服务模块。
- [推理兼容入口](../cosmos_framework/inference/causal_action/robotwin.py:1)：保留参考分支的 import 路径，导出同一个类，没有复制两份转换实现。
- [RLinf RobotwinCausalAdapter](/data5T/bangyan/workspace/RLinf/rlinf/models/embodiment/cosmos3/robotwin_adapter.py:25)：加载配置/统计、批量 state 打包、归一化 endpoint 解码、输出 CPU 环境命令。可选配置由 causal policy builder 加载，不影响未配置 RoboTwin 的固定 replay 路径。

以双臂各 6 个关节为例：

| 数据      | 原生 RoboTwin 顺序        | unified55 位置                                  |
| --------- | ------------------------- | ----------------------------------------------- |
| 左臂关节  | `left_arm[0:6]`           | `0:6`；槽 6 无效                                |
| 右臂关节  | `right_arm[0:6]`          | `7:13`；槽 13 无效                              |
| 左/右夹爪 | 各自紧跟该臂关节或 EEF    | 28 / 29                                         |
| 左/右 EEF | `xyz + wxyz`，world frame | `14:21` / `21:28`，`xyz + xyzw`；frame 显式配置 |

关节命令输出为 `[left joints, left gripper, right joints, right gripper]`，默认 14D；EEF 命令为 `[left xyz+wxyz, left gripper, right xyz+wxyz, right gripper]`，16D。DOF 可以显式配置为 1–7，不能把 55D 连续前 14 维当作环境命令。

动作解码顺序固定为：**归一化 endpoint → action statistics 反归一化 → 用采样时的实测 anchor 恢复绝对目标 → world/base 与四元数转换 → native 命令**。夹爪是绝对量，不加 anchor。输入 endpoint 不被修改；该接口不裁剪执行前缀，不替换 FPO replay，不做额外动作限幅。

状态打包返回未归一化的物理 anchor；状态归一化仍由 causal batch builder 完成。builder 必须与 adapter 使用相同 template、source masks、statistics。不能把已解码的物理动作再次送入 `decode_actions`。

## 2. 如何在 RLinf 配置和调用

模型配置片段 [cosmos3_causal_fpo.yaml](/data5T/bangyan/workspace/RLinf/examples/embodiment/config/model/cosmos3_causal_fpo.yaml:19) 新增可选的 `robotwin`，默认为 null。需要使用时填写：

```yaml
actor:
  model:
    cosmos_variant: causal_action
    training_mode: fpo
    robotwin:
      statistics_path: /absolute/path/to/robotwin_block_statistics.json
      robot_config_path: /absolute/path/to/robotwin_robot.json
      control_mode: joint
      state_format: joint_vector
```

这是模型配置片段，不是完整训练 recipe。`robot_config_path` 可省略，此时要求 statistics 的所有 source masks 完全一致；不会把统计中的 valid 并集当作机器人 mask。机器人 JSON 使用参考分支相同 schema：`state_mask`、`action_mask`、可选 `left_arm_dof/right_arm_dof`、`eef_pose_frame` 和 `T_world_base`。启用 EEF 时必须明确 frame，或从指定 `dataset_root/meta/info.json` / statistics provenance 读取，不猜测坐标系。

调用方式（`policy` 为已经构建好的 causal policy）：

```python
adapter = policy.robotwin_adapter
anchors = adapter.pack_states(env_obs["states"], state_source="measured")
# 用这些物理 anchors 与相机/真实历史构造 causal batches，并调用 sample_fpo_batch。
# normalized_actions: [B,T,55]，每个 batch item 对应一个完整 block。
commands = adapter.decode_actions(normalized_actions, anchors)
# commands: CPU [B,T,14]，可交给已核对 qpos 语义的 RoboTwin 环境。
```

`state_format: native` 接受每个环境一份 `joint_action/endpose` 字典，支持 EEF；`joint_vector` 接受 RLinf 格式的批量关节/夹爪向量，不能提供 EEF state，若 mask 启用 EEF state 会立即拒绝。

`state_source="measured"` 是调用方必须核实的契约，不是把命令目标变成实测值的开关。adapter 拒绝 `command` 来源，但无法从浮点数组推断来源真伪。当前 RLinf wrapper 的 `states` 来自底层 `obs["state"]`，生产环境仍须核实它是否为实测关节；参考分支的 measured helper 可用于补齐该环节。

EEF 的**转换已实现**，环境执行仍需核对 `VectorEnv` 是否支持 `action_type="ee"`。当前 RLinf RoboTwin wrapper 的 `venv.step(actions)` 没有传控制模式；不能只把 `control_mode` 改为 eef 就宣称 EEF 闭环可用。关节首轮也须核对选定机器人和 VectorEnv 的动作顺序。

`predict_action_batch` 仍明确报错，原因已经更新为缺少相机/历史 batch 构造和 FPO collector。本轮完成转换接口，不把尚未实现的在线采集伪装成可运行。

## 3. 参考分支还完成了哪些工作

以下位置均指固定提交 `b4e90f0` 的文件，不代表这些文件已迁入当前工作区。可用 `git show origin/feat/robotwin-client-server:<path>` 查看。

| 原待实现项                     | 分支已有实现                                                                                        | 与当前 RLinf/FPO 的关系                                                                      | 本轮处理                                                               |
| ------------------------------ | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 原生状态与 unified55 转换      | `inference/causal_action/robotwin.py:13`                                                            | 可直接复用，包含 EEF world/base 修正                                                         | 已迁入共享层，并增加 RLinf 批量接口                                    |
| 相机整理与实际状态读取         | `robotwin_client.py:28` 的 `camera_images`，`:41` 的 `native_state`                                 | 三视角命名可复用；measured 模式调用真实关节 getter，但默认来源是 command                     | 已核对；后续迁入 EnvWorker 的观测适配，不复制 command 默认值           |
| 仿真 reset/动作执行/反馈       | `robotwin_client.py:85` 的 `rollout`                                                                | 每次 take_action 后取图像、支持 qpos/ee、成功/超时停止；是单个原生环境，不是 RLinf VectorEnv | 执行与测试语义可复用；仍需 typed feedback、terminal/count 和多环境路由 |
| Session 校验与清理             | `policy.py:15` 的 `CausalPolicy`                                                                    | 检查 prompt/control mode、observation_step、执行长度；失败后强制 reset                       | 复用约束；没有 RLinf env/episode/group/version 管理                    |
| 相机历史与 causal 输入 packing | `online.py:21/52/109/143`                                                                           | 已有流式 VAE、三相机视图处理和 compact pack 构造                                             | 有现成参考，不能直接当当前 prepared FPO batch builder 使用             |
| 真实观测 KV cache              | `streaming_cache.py:45`，`online.py:214`                                                            | 只保留真实观测 KV，预测视频不写历史；使用 cache 的单独 UniPC 路径                            | 暂不迁入 FPO；须先解决 cache/条件记录与 CFM 重算一致性                 |
| 双机服务协议                   | `client.py`、`server.py:19`、`wire.py`                                                              | 已有 NumPy/msgpack WebSocket 通信，server 明确只允许一个客户端                               | 可作独立推理对照；不能替代 RLinf Worker/Channel/多环境 collector       |
| RoboTwin 数据 reader           | `data/generator/action/datasets/robotwin_segment_lerobot_dataset.py:29`、factory 注册、sources 配方 | 已支持 unified state/action/mask 与相机映射；注释明确 state 可能是命令目标代理               | 不必从零重写 reader；迁入时同时审查数据的 state 来源和统计             |
| 专家 EEF 数据回放              | `examples/action_pretrain/robotwin_eef_replay.py`                                                   | 可独立于策略检查 EEF 数据、坐标系和仿真执行，支持 prepare-only                               | 可作为生产环境适配验收工具，后续按需迁入                               |
| FPO replay、优势与训练         | 分支没有 RLinf FPO collector/Actor/runner                                                           | 当前工作区自己的 FPO 实现承担这些职责                                                        | 仍需接在线数据，不能从该分支直接获得                                   |

上述简写的 `robotwin_client.py`、`policy.py`、`online.py`、`streaming_cache.py`、`client.py`、`server.py`、`wire.py` 均位于 `cosmos_framework/inference/causal_action/`。

### 为什么不直接合并 online engine

分支 `CausalPolicyEngine` 默认预测 32 步、执行 16 步；反馈一次必须覆盖一个 VAE latent，即 `video_stride × temporal_compression`。它持有流式 VAE 和各层 KV，`predict()` 单独调用 sampler，返回 normalized action 与 predicted video，但没有当前 `CausalFPORollout` 的完整条件快照/probes/版本契约。

因此，“分支已有闭环”应理解成**已有独立推理闭环实现**，不能推导出 RLinf 在线 FPO 已完成。若下一步采用 16 步重规划，必须同时设计：完整 32 步 endpoint 的奖励归属、16 步执行区间、缓存/位置编码的行为条件快照，以及 Actor 重算等价性；当前完整块 FPO 首版不隐式切换此语义。

分支也包含与此次转换无关的 prompt JSON 修改、dataset symlink discovery 和抽样回调变动。本轮没有整分支合并，以免改变当前训练输入和验证基线。

## 4. 验证与下一步

本轮新增/迁入的测试覆盖：原生关节和 EEF 顺序、mask 与 source statistics 校验、world/base 旋转和平移、批量归一化动作解码、夹爪绝对语义、未改变完整 endpoint、拒绝 command/非有限值/错误形状，以及配置接入 causal policy。

```bash
source tools/activate_rlinf_cosmos3_ascend.sh
python -m pytest --noconftest \
  cosmos_framework/utils/robotwin_test.py \
  cosmos_framework/model/generator/algorithm/rl/fpo_test.py \
  cosmos_framework/model/generator/algorithm/rl/causal_fpo_test.py \
  cosmos_framework/inference/causal_action/session_test.py \
  /data5T/bangyan/workspace/RLinf/tests/unit_tests/test_cosmos3.py -q
```

最终上述 56 项 CPU 测试通过，包括转换/适配 29 项及已有 FPO/session 回归 27 项；没有运行真实 RoboTwin、GPU/NPU 模型或生产双节点。Ruff 与 Markdown 格式检查通过。分支内客户端/流式 cache 的测试本轮只做源码审查，不计入当前迁入代码的通过数量。

下一步优先复用分支的 measured state 和相机处理来扩展 RLinf 环境反馈；随后实现完整块 causal batch builder、session 路由和 collector。任务域统计、checkpoint、VectorEnv 实测状态与控制器仍需生产验收，详见 [生产步骤](rlinf_robotwin_production_runbook_zh.md:1)。
