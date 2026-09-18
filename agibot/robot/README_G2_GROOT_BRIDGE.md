# G2 GR00T bridge 部署说明

**10.20.15.194 当前入口（2026-09-09）：** 使用已完成两轮完整真机推理、现场确认
流畅的 `g2_groot_continuous_action_bridge.py`，配套 continuous controller/sender。
机械臂与夹爪走同一个官方联合请求。配置与命令见
[194 记录](../docs/robots/10.20.15.194.md)。下文 legacy action 命令不适用于该机器的
当前放置任务，不要同时启动，也不会自动切回。

本目录包含 G2 机器人侧的只读 observation bridge、右臂/右夹爪 action bridge、持久
50 Hz controller 以及校准/诊断工具。

当前提交完成并验证了右臂任务实现。task profile 可按任务选择机械臂；左臂任务配置对应的
observation/action mapping、GDK bridge 和测试。

团队总入口是 [`../README.md`](../README.md)，每台机器人的适配步骤见
[`../docs/G2_ADAPTATION_GUIDE.md`](../docs/G2_ADAPTATION_GUIDE.md)，现场准入见
[`../docs/SAFETY_AND_COMMISSIONING.md`](../docs/SAFETY_AND_COMMISSIONING.md)。

## 通用与 legacy 组件

| 文件 | 作用 |
|---|---|
| `g2_groot_right_observation_bridge.py` | 只读双相机、右 EEF、右夹爪和 health |
| `g2_groot_persistent_h1_action_bridge.py` | loopback-only 右臂/夹爪 action service |
| `g2_groot_persistent_right_arm_controller.py` | 持久 50 Hz EEF 跟踪 |
| `g2_groot_gripper_state_machine.py` | omnipicker 状态语义 |
| `g2_groot_gripper_runtime.py` | 非阻塞夹爪目标执行 |
| `g2_groot_h1_bridge_client.py` | action bridge 协议客户端 |

其余 `probe`、`roundtrip` 和 `zero_hold` 文件是 commissioning/诊断工具，不是完整策略
runner。

## 部署

将正式组件和所需诊断脚本同步到机器人专用目录，例如：

```text
/home/agi/vla_ct/groot_right_arm_shadow
```

机器人端使用 GDK 自带 Python 环境，不使用工作站 `.venv`。启动前：

```bash
source /home/agi/app/env.sh
cd /home/agi/vla_ct/groot_right_arm_shadow
```

只读 observation bridge：

```bash
python3 -u g2_groot_right_observation_bridge.py \
  --bind-host 127.0.0.1 \
  --port 9100 \
  --camera-timeout-ms 500 \
  --max-camera-skew-ms 100 \
  --max-state-camera-skew-ms 50 \
  --model-input-jpeg-quality 92
```

旧版完整策略 action bridge（不是 194 当前入口）：

```bash
python3 -u g2_groot_persistent_h1_action_bridge.py \
  --enable-control \
  --enable-gripper \
  --enable-approach-session \
  --enable-protected-close \
  --confirm ENABLE_G2_GROOT_RIGHT_ARM_SINGLE_PROTECTED_CLOSE_SESSION \
  --session-limit-s 1800
```

两个服务都必须绑定 loopback，通过 SSH tunnel 访问。不要暴露到局域网。

## 不变量

- observation bridge 不包含任何控制 API；
- 当前右臂 action bridge 只接收右臂和右 omnipicker；
- 左臂、头、腰和底盘字段不在这个右臂 bridge 的 action 协议中；
- 一个持久 GDK owner 独占控制；
- policy 10 Hz，每个 waypoint 由 controller 执行 5 个 50 Hz tick；
- 夹爪训练范围为 `[-0.785, 0]`，方向必须在目标机器人重新确认；
- 抓取案例使用 `agibot/scripts/run_g2_groot_full_protected_inference.py` 的完整 H16；
- 其他任务必须在 task profile 中声明自己的 inference/recovery runner 和完成条件；
- 硬件 health、finite、workspace、age、session 和 loopback 检查不能删除。

## 每台机器人重新校准

代码中的 reference 数值来自已经验证的 G2，不是所有 G2 的出厂保证。部署到新机器前必须
重新确认：

- GDK 版本、mode/error 和控制 owner；
- `base_link`、`arm_r_end_link`、XYZW；
- omnipicker joint name、raw endpoints、方向和 status；
- 相机身份/尺寸/同步；
- 起始位姿、workspace、WBC compensation；
- 0.5 mm roundtrip 和 shutdown 后控制释放。

校准结果写入 `agibot/configs/local/<robot-id>.toml`，并通过：

```bash
agibot/bin/groot-g2 doctor \
  --robot-config agibot/configs/local/<robot-id>.toml
```

历史阶段性结论和问题复盘统一保留在
[`../examples/xichong_right_single_grasp/RUNBOOK.md`](../examples/xichong_right_single_grasp/RUNBOOK.md)，
不再在本文件混写。
