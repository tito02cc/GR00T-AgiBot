# G2 机器人 10.20.15.194：已成功运行的两个放置任务

保存两个放置任务的实测桥接源码与配置，版本固定于 2026-09-14。
公共实验脚本的更新不会影响此目录；后续控制逻辑改动另建版本验证。

## 运行依赖

此目录包含推理桥接和任务参数。模型权重、工作站环境与厂商 GDK 需单独配置。
训练与数据处理见 [操作手册](../../README.md)，实机演示见
[推理结果视频](../../inference_videos/README.md)。

推理需要对应模型包的 `model/` 和 `backbone/`，不依赖原始训练数据。
换机器后更新 `model/config.json` 与 `model/processor_config.json` 中活动 `model_name`
路径，指向本机 Cosmos 骨干，保留各任务训练时的归一化参数。

推理工作站需要完整同版本 Isaac-GR00T 仓库；固定包仍依赖官方 `gr00t/` 代码。
`commands.py --repo` 指定仓库位置，`--robot-root` 指定机器人上的包位置，均只影响路径。
它们不自动适配另一 IP/GDK；10.20.15.194 地址仍出现在生成的 SSH 命令中，另一机器须独立检查修改。
适配其他机器人时另建部署版本。

## 版本与实测记录

| 任务 | 固定桥接 | 依据 | 关键差异 |
|---|---|---|---|
| [xichong_right_place_r0002](xichong_right_place_r0002/profile.json) | continuous_20260909 | 9 月 9 日两轮 48 步，现场确认完整、流畅 | 逐行提交 H16，运行中自适应补偿；不用后来的 transport/RTC 实验版 |
| [zhewan_right_place_r0003](zhewan_right_place_r0003/profile.json) | h16_collision_latched_20260911 | 9 月 11 日 64 步，现场确认成功、轨迹平滑 | 原子 H16、精简通信、校准后冻结补偿、原生碰撞事件锁存 |

两者都是官方 GR00T N1.7 30k 权重 + GDK 联合 arm/tool，10 Hz 模型动作 / 50 Hz 执行。
右夹爪保留 native radians：0 闭合、−0.785 张开。均不启用 RTC、新增滤波或固定放置轨迹。
**不要将 zhewan 的 workspace、模型、归一化或补偿设置套到 xichong。**

## 内容

- 每个任务的 `profile.json`：模型路径、训练 prompt、初始位姿、工作范围、完整推理开关。
- 每个任务的 `robot/`：从 10.20.15.194 机器人已部署版本取回的 bridge/controller/sender，以及递归导入的
  G2 依赖和只读观测服务。虽有 probe/recover 命名的依赖，日常只运行 `robot_bridge.sh`，
  不直接运行依赖脚本。`sources.json` 记录每个文件的来源。
- 每个任务的 `robot_bridge.sh`：默认只读 standby；`control` 允许完整 runner 随后激活；
  `observation` 只启动相机/状态服务。启动脚本不会清错、复位、修改碰撞配置或自动开始推理。
- `workstation_runtime/`：完整 place runner、动作解码/观测客户端及其本地依赖。
  两任务共用这份快照，命令显式传入各自参数，不依赖 runner 的默认任务值。
  xichong 的模型 modality 配置同样固定；官方 GR00T 包和模型 processor 仍使用原仓库/权重。
- `commands.py`：只打印命令，不连接机器人、不执行命令。

共享 G2 依赖取自 9 月 14 日部署环境，工作站 runner 为 9 月 11 日实测后的兼容版本，
并非 9 月 9 日完整系统快照。迁移到固定目录后完成了静态和导入检查，未重新进行实机测试。

## 使用

在推理工作站仓库根目录，选择一个任务打印配套命令，报告路径每次换新：

```bash
.venv/bin/python agibot/deployments/g2_194/commands.py xichong_right_place_r0002 --report agibot/local_reports/xichong_next/inference.json
.venv/bin/python agibot/deployments/g2_194/commands.py zhewan_right_place_r0003 --report agibot/local_reports/zhewan_next/inference.json
```

按输出说明在不同终端启动所选模型服务、观测服务、动作服务和 SSH 转发。两个任务使用
相同端口，不可并行；已有服务须先确认归属，不盲目杀进程。`standby` 与 `control` 二选一，
不要同时运行。完整 runner 需要 `control`，但它仍先预热模型，再显式激活控制器。

机器人固定包目录：`/home/agi/vla_ct/releases/g2_194_20260914/`；已有 `candidates/` 目录
保留，未覆盖。若重新安装，将整个 `g2_194` 目录上传到新的目录，使用 `--robot-root`
指定新位置；脚本通过自身位置找依赖，不会从旧共享 bridge 目录导入。

模型文件在仓库 `agibot/models/` 下，实际位置不同可修改实例 profile 或用 `--repo` 指向
正确仓库。xichong 保留 percentile 归一化，zhewan 保留 min/max，以各自 checkpoint 的
processor 配置为准；不要混用模型服务。原始数据无需保留在推理机，初始参考已写入 profile。

## 现场操作边界

启动前确认当前任务、初始状态、工件持稳、工作范围及急停值守，并通知录屏。不得因为上次
成功就自动启动、闭爪或复位。完成/异常后停止本轮，不自动重试或额外开闭爪。

zhewan 成功时原生碰撞配置为开启 / 灵敏度 3 / 恢复等待 500 ms。脚本要求保护开启且
恢复等待至少 500 ms；实际配置可能在重启后变化，不能把历史值当当前读回值。
xichong 固定版没有新增的应用侧原生碰撞锁存，**这不是关闭机器人原生保护的指令**；
不为追求统一而修改其已验证控制行为。新版保护若要移植，需另建版本并验证。

成功次数不等于长期成功率，未触发碰撞也不证明真实碰撞的停机效果。工作范围是任务边界，
不是碰撞安全认证。

完整历史：[10.20.15.194 机器人记录](../../docs/robots/10.20.15.194.md)、
[xichong 记录](../../examples/xichong_right_place_r0002/README.md)、
[zhewan 记录](../../tasks/zhewan_right_place_r0003/BRIDGE_DEPLOYMENT_20260911.md)。

## 固定包检查结果

2026-09-14：本地包检查 6 项通过；明确导入固定 runtime 后，place runner 回归
39 项 + 62 子测试通过。首次该回归少配了测试夹具的裸模块搜索路径，导致两个夹具导入
失败；补齐测试 PYTHONPATH 后通过，没有修改运行算法。10.20.15.194 机器人上两任务源码 Python 3.10
编译、shell 语法和固定目录导入检查通过。厂商绑定导入会初始化 DDS，但没有构造 Robot、
启动控制服务或发送机械臂/夹爪命令。原已部署目录未覆盖，固定包已同步到上述 releases 目录。
