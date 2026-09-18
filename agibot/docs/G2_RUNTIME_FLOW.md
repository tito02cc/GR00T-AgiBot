# G2 推理运行流程

## 当前默认：10.20.15.194 的 continuous 单 owner 桥接

2026-09-09，该机器完成两轮完整放置推理，均返回
`PASS_PLACED_RELEASED_AND_RETRACTED`。现场确认第一轮完整、第二轮“很流畅”；
用户指定以后 `10.20.15.194` 使用本次桥接。两轮各完成 3 个 H16（48 条动作），每条
执行 5 个联合 tick，真实开爪后分别回撤 234.892 / 269.941 mm，无记录到的桥接 fault。
这是该机器、当前 GDK 与右臂放置任务的验证结果，不代表其他机器/任务/版本也已验证。

- 模型：官方 `gr00t/eval/run_gr00t_server.py`。
- 完整推理：`agibot/scripts/run_g2_groot_full_place_inference.py`。
- 唯一动作入口：`g2_groot_continuous_action_bridge.py`。
- 控制器与发送器：`g2_groot_continuous_controller.py`、
  `g2_groot_continuous_sender.py`；一个常驻 Robot/TF、一个 50 Hz 工作线程。
- 官方 GDK：`Robot.trajectory_tracking_control` 的同一请求同时携带右臂
  `ABS_POSE` 和右夹爪 `ABS_JOINT`。不切换旧 EEF/工具发布进程。

该机器的部署目录、夹爪名和具体入口见
[10.20.15.194 运行记录](robots/10.20.15.194.md)，完整命令见
[放置任务案例](../examples/xichong_right_place_r0002/README.md)。
旧 mux 保留供明确回退；不会自动切回，也不得与 continuous 服务同时运行。
通用 `groot-g2 robot action` 仍使用 legacy 入口，不能代替上述 IP 记录中的动作服务命令；
这次默认运行方式的记录没有改动 CLI 的后端选择。

## 当前执行顺序

```text
模型与观测就绪 → continuous service standby（机械臂未接管）
→ 官方 get_action 预热，丢弃预热动作，policy.reset（不是机器人复位）
→ runner 显式激活 → live link3 参考初始化 + 2 s 反馈标定
→ 新鲜观测 → 模型完整 H16
→ 每条 10 Hz waypoint：5 个 50 Hz EEF/夹爪联合 tick
→ 同一 owner 持位，继续下一次观测和完整 H16
→ 真实开爪与回撤完成，或报错退出；停止发布，不额外开爪/复位
```

EEF 使用既有插值/反馈补偿方法，但针对联合接口从零校准，不复用旧接口的固定偏置。
2 s 标定允许毫米级启动收敛，不将 raw API 的第一帧静止视为必然：第一轮最大标定瞬态
1.9991 mm，最终位置误差 0.0573 mm。标定成功仍需满足任务的初始位姿条件。

放置任务以闭爪持件开始；启动闭合指令为 `0.0`，模型阶段夹爪按每条输出执行，
包含中间开度，没有“等端点后停止/重启机械臂”的交接。推理/网络耗时期间沿用同一
控制 owner 的持位 tick，并不意味着模型每次计算都恰好只需 100 ms。

保留完整 H16 回执、实际夹爪反馈、模型准备后激活和错误清理。GDK 报错、碰撞或通信
结果不确定时，不自动重试动作、不自行清错或切换后端。停止发布不等同于硬件急停，
退出推理也不等于应松开当前工件。

## 验证范围与后续使用

两轮程序 PASS 和现场反馈确认了本次完整机械臂/夹爪协同流程。它们不证明工件精确落座、
所有瞬态完全消失或长期可靠性，也没有给出历史 GDK 碰撞/IK 行为的根因结论。
本轮之间的闭爪复位由用户单独要求；第二轮为录像作了明确提示及 5 s 等待，
都不是策略中自动加入的动作阶段。

证据、当前状态及后续重复验证要求见
[任务状态](../examples/xichong_right_place_r0002/STATUS.md)和
[联合控制记录](G2_CONTINUOUS_CONTROL.md)。

## 历史：2026-09-08 恢复 legacy 与端点交接

以下保留当时的实现与限制，供复盘或人工选择回退使用；不是 10.20.15.194 当前默认入口。
9 月 8 日 raw-native/持位实验与 9 月 9 日经过反馈标定的 continuous 服务不是同一验收版本。

2026-09-08：按用户要求恢复原 legacy 机械臂路径，只修正夹爪交接。
模型、EEF/夹爪解码、任务初始位姿、10 Hz waypoint 和 50 Hz 插值不变。

### 当时入口

- 模型：官方 `gr00t/eval/run_gr00t_server.py`。
- 完整推理：`agibot/scripts/run_g2_groot_full_place_inference.py`。
- 唯一动作入口：`g2_groot_place_action_mux.py`。
- 机械臂：配套的 `g2_groot_persistent_h1_action_bridge.py` 和
  `g2_groot_persistent_right_arm_controller.py`，使用原 GDK
  `end_effector_pose_control`，原补偿及插值方式。
- 夹爪：`g2_groot_right_gripper_command_daemon.py`，调用官方 `move_ee_pos`。

当时的实验性 native 后端、新增关节持位判停及零补偿默认改动已退出正式代码路径。
之前的实验记录保留用于追溯，不能作为 2026-09-09 continuous 服务的启动说明。

### 当时执行顺序

```text
模型与观测就绪 → mux standby（机械臂未接管）
→ 官方 get_action 预热，丢弃预热动作，policy.reset（不是机器人复位）
→ runner 激活机械臂 → 实时观测 → 模型完整 H16
→ 逐行执行；需要改变夹爪端点时：
  当前行 EEF 执行一次 → 等该行完成回执 → 停机械臂发布
  → 夹爪命令与实际反馈确认 → 恢复机械臂 → 继续下一行
→ 任务完成或报错退出，不额外开爪
```

机械臂行不能在开爪之后才执行，也不能重启后重复执行。
夹爪交接期间属于同一自动运行会话，不需要人工续跑；但这条 legacy 分进程交接路径有
实测秒级等待，不能承诺完全无停顿或夹爪/机械臂逐步同步。中间开度仍延后到端点处理。

保留回执缓存、完整状态读取、模型准备后激活和错误清理。GDK 报错、碰撞或通信结果
不确定时不自动重试、不自行清错。停止发布不等同于硬件急停。

### 当时验证边界

恢复依据是已验证能够执行机械臂运动、放下后续跑回撤的 legacy 基线，不是一个已经
完成单次连续全任务验收的版本。恢复代码也不代表历史 GDK 碰撞报警已解决。

以上只保留 legacy 历史，不是新的动作启动指令。当前 continuous 启动命令见
[放置任务案例](../examples/xichong_right_place_r0002/README.md)，
各版本部署与测试结果见[任务状态](../examples/xichong_right_place_r0002/STATUS.md)。
