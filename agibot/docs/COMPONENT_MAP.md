# 目录与组件

## 核心流程

- `tasks/<task-id>/`、`templates/right_place/`：新右臂放置任务的数据配置、名单和复用母本；
- `scripts/prepare_g2_task.py`、`prepare_g2_right_place_training.py`：按 JSON 任务配置运行原始复查、分集转换和训练前验证；
- `scripts/verify_g2_training_math.py`：官方 loader/处理器、完整 H16 和训练统计量验证；
- `scripts/build_g2_place_review.py`：生成所有选定 episode 的释放/回撤关键帧图和抽查索引，不自动判定物理成功；
- `bin/groot-g2`、`scripts/team_cli.py`：数据、训练和推理的统一入口；
- `templates/task_profile/`：新项目配置模板；
- `training/preflight_1xa10080.sh`、`launch_1xa10080.sh`：单卡 A100 训练；
- `robot/g2_groot_right_observation_bridge.py`：G2 图像、右 EEF 和夹爪观测；
- `robot/g2_groot_continuous_action_bridge.py`：`10.20.15.194` 当前默认的单 owner 动作服务，支持预热后显式激活；
- `robot/g2_groot_continuous_controller.py`、`g2_groot_continuous_sender.py`：50 Hz 官方 GDK 联合 EEF/夹爪请求、反馈补偿和同步插值；
- `robot/g2_groot_persistent_h1_action_bridge.py`、`g2_groot_persistent_right_arm_controller.py`：保留的 legacy 服务/控制器，也提供候选桥接复用的公共定义；
- `robot/g2_groot_place_action_mux.py`：旧版机械臂/夹爪端点交接路径，仅供明确回退与历史复盘；
- `robot/g2_groot_gripper_state_machine.py`、`g2_groot_gripper_runtime.py`：夹爪控制；
- `tools/g2_gr00t_shadow_adapter.py`、`g2_groot_live_observation_client.py`：模型适配。

这些组件对应相同的数采协议、G2 和 GR00T N1.7。当前真机记录覆盖右臂任务；部署时按
task profile 选择机械臂，并使用目标机器人的 GDK、workspace 和夹爪配置，不能据此推断
左臂或其他 GDK 组合已完成验证。

`10.20.15.194` 的 continuous 单 owner 组合在 2026-09-09 完成两轮完整真机推理，现场
确认第一轮完整、第二轮“很流畅”；后续该机器使用这条路径，见
[IP 运行入口](robots/10.20.15.194.md)和[当前运行流程](G2_RUNTIME_FLOW.md)。每轮 48 条动作、
真实开爪与回撤均有记录，但两次 PASS 不等于长期可靠性或精确落座验收。

9 月 8 日的 raw-native 联合后端、关节持位实验和恢复 legacy 的记录仍用于追溯；
[旧联合调用说明](G2_NATIVE_TRAJECTORY.md)不是本次已验证 continuous 服务的启动说明。
旧 mux 不会自动回退，也不得与当前单 owner 服务并行运行。
通用 `groot-g2 robot action` 仍硬编码 legacy 入口；本次只明确该机器的运行方式，
没有修改此 CLI。`10.20.15.194` 的动作服务使用上述 IP 文档中的 continuous 命令。

## 抓取任务实现

文件名包含 `xichong` 的转换、筛选、审核、训练和评估脚本组成已完成抓取案例。新任务可
复用其中的数据转换和训练逻辑，并按任务目标调整成功条件、数据选择和评估。

`run_g2_groot_full_protected_inference.py` 和 `recover_g2_groot_training_start.py` 是抓取案例的
推理与复位 runner。task profile 用 `inference_runner`、`recovery_runner` 指定项目实际入口。

## 放置任务实现

`scripts/run_g2_groot_full_place_inference.py` 执行官方 GR00T 完整 H16 动作块，通过当前
continuous 服务同步执行右臂与夹爪，并依据真实开爪反馈和回撤距离记录任务完成。
具体初始位姿、模型与启动命令见[放置案例](../examples/xichong_right_place_r0002/README.md)，
两轮结果及历史失败边界见[任务状态](../examples/xichong_right_place_r0002/STATUS.md)。

## GDK 检查工具

名称包含 `probe`、`roundtrip`、`zero_hold` 的文件用于 GDK、轨迹跟踪和夹爪调试；名称包含
`h1`、`h2`、`receding` 的 runner 保留用于历史问题复盘。通用流程从 `groot-g2` 入口运行；
`10.20.15.194` 的动作服务按 IP 文档启动。适配 GDK 时再按
[`G2_ADAPTATION_GUIDE.md`](G2_ADAPTATION_GUIDE.md) 选择相应工具。
