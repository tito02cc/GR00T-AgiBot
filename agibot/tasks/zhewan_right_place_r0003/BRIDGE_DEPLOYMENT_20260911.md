# 194：H16 整块传输桥接部署

日期：2026-09-11。用户授权部署新版桥接，未授权在本次部署中启动机械臂推理。

## 部署结果

新目录：

```text
/home/agi/vla_ct/bridges/10.20.15.194/candidates/h16_batch_transport_20260911/
  g2_groot_continuous_action_bridge.py
  g2_groot_continuous_controller.py
  g2_groot_continuous_sender.py
  g2_groot_persistent_h1_action_bridge.py
```

前三个文件为联合控制桥接，第四个包含新增的逐行目标验证参数 `previous_target`。
其余共享依赖继续从 `/home/agi/vla_ct/bridges/10.20.15.194/` 加载；这是增量部署，
不能仅复制这四个文件到一台空白机器就运行。

加载环境必须保留 GDK 路径，并使候选目录优先：

```bash
source /home/agi/app/env.sh
export PYTHONPATH="/home/agi/vla_ct/bridges/10.20.15.194/candidates/h16_batch_transport_20260911:/home/agi/vla_ct/bridges/10.20.15.194:${PYTHONPATH:-}"
cd /home/agi/vla_ct/bridges/10.20.15.194/candidates/h16_batch_transport_20260911
```

本机 `inference.json` 的 `robot_bridge_directory` 已指向该目录，
`native_chunk_submission=true`；`inference.py print-live-command` 会附加
`--native-chunk-submission`。该入口仅打印命令，不会部署、启动或使能远端服务。
权重、训练配置、动作映射、控制器和联合 sender 算法未改，不启用 RTC。

## 已做检查

- 194 系统 Python 3.10 编译四个文件成功，真实 GDK 导入/初始化成功。
- 临时服务监听 `127.0.0.1:9200`，**没有传 `--enable-control`**。
- 接口报告 `execute_h16_gripper`、`atomic_h16_native_v1`、10 Hz / 50 Hz / 5 tick。
- `activation_state=standby`、`enable_control=false`、`arm_owner_active=false`、
  `queue_depth=0`、`fatal_error=null`。
- 初始化读取的右夹爪反馈 −0.004611875 rad，motor/whole end error 均为 0；
  末端 XYZ 约 `[0.474754, -0.245390, 0.922593]` m。
- 客户端仅发送 `info`、`status`、`shutdown`，未发送 activate 或动作请求。
  临时检查服务已正常退出，未设置自启动，未关闭现有只读观测服务。

检查时沿用原 XYZ 包围盒只为构造只读服务，它**仍不是经过确认的接触安全边界**。
以上不是实机运动、放置精度或碰撞安全验证。首轮危险下压的待处理事项仍见
[现场记录](LIVE_TEST_20260911.md)，`robot_scene_and_gdk_verified_for_this_task` 保持 false。

## 回退

旧目录 `candidates/baseline_transport_20260909/` 和共享依赖没有覆盖。
在动作服务停止后，将任务 `robot_bridge_directory` 改回旧目录，
`native_chunk_submission` 设为 false，并从旧目录启动服务即可恢复原传输方式。
必须同时回退路径和开关；新 runner 会在 activate 前拒绝不支持 H16 的旧服务。
回退也不表示危险下压已解决，不应据此直接重跑首轮动作。

## 接触检查候选暂停（同日后续）

用户要求补充下压停止逻辑后，在本地增加可选 `g2_groot_contact_guard.py`、sender
发送前检查、runner 配置一致性检查及 `contact_guard.json` 待确认模板。
本地测试：机器人相关 89 项 + 76 子测试、place runner 36 项 + 60 子测试通过。
这些是合成限值的测试，不能把测试数字用于真机。

随后用户表示不知道现场阈值，要求先不要继续复杂的碰撞检测工作。因此：

- 接触检查候选 **未部署、未激活**，本任务默认不启用；现场阈值仍为 null。
- 本任务重新指向上文已部署的 `h16_batch_transport_20260911/`，只启用整块传输。
- 本地 action bridge/sender 含尚未部署的接触检查改动，不能将本地最新文件与远端
  H16 目录称为完全一致；将来重新部署前需明确版本及配置。
- 只读核实到官方 `get_collision_detection_config()` 返回 `is_enabled=false`、
  `sensitivity=3`、`checkout_timeout_ms=5`。未调用任何碰撞配置 setter、控制模式 setter、
  力矩标定或运动接口。
- 已能读取 `arm_r_end_link` 对应的 wrench，单位按安装版 GDK 文档为 N / Nm；
  Python MotionControlStatus 不暴露源时间戳，普通成功读取不构成新鲜度保证。
- 用户这次要求暂停复杂化，不等于确认危险下压已消除或授权立即重试。

本地候选停止语义只是拒绝后续 SDK 发布并作废队列，不会自动松爪或复位，不能取消
已经下发的控制点或代替硬件急停。官方碰撞检测也不等于此候选，二者没有被开启。

## 再次测试前的准备（用户确认有现场保护后，仅准备）

- 2026-09-11 约 12:00，本机 5564 已加载本任务 30k 模型，使用实时观测完成一次 H16
  预热并丢弃输出；warmup 1.647 s，未执行任何动作，不作为稳态速度指标。
- SSH 本地端口 19100/19200 转发至 194 的 9100/9200；新版 H16 服务在 9200 待机，
  **未传 `--enable-control`**，没有控制线程。待机服务从启动计最多存在 1800 秒，
  后续需检查是否过期；不得将其误认为已经激活。
- 待机读取的初始误差约 0.582 mm / 0.140°，右夹爪 −0.004611875 rad。
- 再读 GDK 碰撞配置仍为关闭；用户表示现场保护有效，但具体独立保护机制尚未说明。
  因而记录为“准备完成、未放行运动”，不修改 `robot_scene_and_gdk_verified_for_this_task`。
- 同时存在多个 GDK 只读进程时，待机进程打印一次 GetCollisionDetectionConfigResponse
  uuid mismatch；发起读取的客户端返回了配置。该日志保留为待核对项，不当作运动成功证据。
- 准备阶段没有发送机械臂、夹爪、复位、activate 或安全配置 setter 请求。

本机记录：`agibot/local_reports/zhewan_live_20260911/prepared_h16_retry/`，包含
`preparation.json`、`model_server.log` 和 `observation/` 两路现场图像及元数据。

## 执行链路收尾（同日后续）

用户补充确认“能及时阻止危险下压”，要求只完善其他部分。记录为现场人员的确认，
不是软件已验证某种自动保护有效，也不更改之前读到的 GDK 碰撞开关状态。
本轮没有再扩展或部署自定义接触检查，没有执行机械臂/夹爪动作。

- 194 仍保留已部署的 H16 整块传输版，模型及映射不变，控制仍为原生联合 arm/tool。
- 本机 place runner 在 H16 完成后复用**刚取得且已核验 16 条完成回执**的状态，
  去掉一次重复的网络 status 请求。异常、缺失、不健康回执不能继续；
  下一轮推理仍重新读取相机和机器人状态，不使用旧观测推理。
- 该优化只涉及本机 runner，无需覆盖 194 的已部署文件；原逐行模式保持原行为。
- 最新软件测试：机器人相关 89 项 + 76 子测试，共享 runner 28 项 + 2 子测试，
  place runner 38 项 + 62 子测试，共 155 项 + 140 子测试通过；ruff 检查通过。
- 含假机器人的真实 TCP/生产 controller/sender 连续 H16 测试，不是新一轮实机验收。
  同步推理等待与模型轨迹变化仍然存在，不能承诺消除全部抖动。
- 本轮末检查：模型 5564 仍监听；机器人 9200 待机已到时退出。下次现场操作前需重新
  启动指定部署版本并读取当前状态，不能沿用“服务一直在运行”的假设。

## 官方碰撞检测已开启（同日最新配置）

用户明确要求“启用碰撞保护”后，在无动作服务运行、9200 未监听时，调用一次
`Robot.set_collision_detection_config`，仅把 `is_enabled` 从 false 改为 true。
保留原有 `sensitivity=3`、`checkout_timeout_ms=5`，未改灵敏度或恢复时间。

- setter 返回 0；随后 getter 读回 **true / 3 / 5**，与请求一致。
- 读回 motion mode=1、control_mode=3、motion_error=0、right_arm_error=0。
- 右夹爪反馈仍为 −0.004611875 rad，错误码 0。
- 未调用运动、夹爪、控制模式切换、清错或力矩标定接口。

安装版 GDK 文档说明：触发后切换到位置控制，超出恢复窗口且无新触发会自动恢复。
因此原有 **5 ms 自动恢复窗口不等于故障锁停**；50 Hz 应用轮询也不能保证捕捉如此短的
模式变化。此次完成的是配置开启与读回，**不是碰撞触发/停机效果或接触安全验证**。
仍需处理并验证触发后的不续跑行为，不能仅据开关已开启宣称完整实测已放行。
官方文档同时说明该碰撞配置不跨机器人重启持久化，重启后须重新读取确认。

## H16 + 原生碰撞锁存 + 补偿冻结（2026-09-11 后续版本）

当前任务指向 `candidates/h16_collision_latched_20260911/`。旧部署目录未覆盖。
本轮只有软件更新、保护配置修改和只读 GDK 检查，没有机械臂/夹爪运动。

- 官方碰撞检测保持开启、灵敏度 3；恢复等待从 5 ms 改为 500 ms，setter 返回 0，
  getter 确认 true / 3 / 500。该窗口便于应用观察模式变化，不是允许持续碰撞 500 ms。
- `NativeCollisionLatch` 在激活前读取配置，在每次 SDK 发布前读取运动状态。
  碰撞对、错误、离开 servo/预期阻抗模式或读失败都会锁存故障，作废剩余动作，
  不自动清错、恢复、松爪或复位。固件恢复原模式也不会使本次服务自动继续。
- 这是停止后续发布，不是硬件急停，不保证取消已发出的控制点。SDK 没有源时间戳，
  所以读取成功不等于已经证明状态新鲜或真实碰撞能及时停止。
- 未启用额外自定义力阈值或最低 Z 接触 guard。用户要求避免轻触误判；安装版 GDK
  文档说明灵敏度 1–3，通常从 2 按负载调试。当前保留既有 3，不凭“轻碰正常”擅自
  下调，也不把该等级换算成未经标定的 N/Nm。后续应根据实际误触发证据再调试。
- 抖动优化：沿用原子 H16、10 Hz 动作/50 Hz 插值、同请求 arm/tool，以及已核验完成
  回执复用。新增 `freeze_compensation_after_calibration=true`，只在启动校准期间
  学习静态位姿偏差，运行中冻结，避免把动态跟踪滞后持续积分成额外位移。
  不改变模型 H16 目标、夹爪方向或开度，不启用 RTC。不能据此断言抖动已消除。

远端部署包括 bridge/controller/sender/persistent H1 依赖、native latch、可选但未启用
的 contact guard 和 `robot_bridge.sh`。Python 3.10 编译通过。启动脚本在 source 厂商
env.sh 前清空位置参数，避免厂商脚本把 standby/control 误当作应用根目录。

真实 GDK 只读检查：配置合格，连续 20 次运动状态检查通过，单次最慢约 0.151 ms；
没有发送运动指令。新服务报告 `atomic_h16_native_v1`、`calibration_only`、
`g2_native_collision_latch_v1`；`enable_control=false`、standby。待机有 1800 秒有效期，
不能假定后续一直在线。任务入口会在 activate 前拒绝错误版本/补偿模式，激活后确认锁存已 armed。

本任务现场状态仍未验收，`robot_scene_and_gdk_verified_for_this_task=false` 保持不变。

### 本版本软件回归结果

- 机器人测试：97 项 + 81 子测试通过，包含生产 controller/sender、假机器人、真实
  TCP 的连续 4×H16、原子批提交、原逐行兼容，以及碰撞故障后清队列/拒绝恢复。
- place runner：39 项 + 62 子测试通过；共享 protected runner：28 项 + 2 子测试通过。
  两个 runner 测试须分进程执行：它们为不同任务注册同一个 NEW_EMBODIMENT，放在同一
  pytest 进程会发生重复注册的收集错误，不是机器人运行错误。
- 共 164 项 + 145 子测试通过；本轮修改的 Python ruff/编译及 bash 语法检查通过。
- 初次整套机器人回归中，旧逐行 TCP 测试出现 20 ms 发布截止超时；诊断记录到测试
  进程 GC 暂停约 33–190 ms。将热路径 MagicMock 改为不记录调用历史的函数桩后，
  同一整套测试通过。未放宽任何生产截止时间，也未关闭生产 GC。
  该结果不能保证真机进程没有调度停顿，现场仍需观察实际 tick/chunk 间隔。
- 最后只读读取 194 接口仍为 standby / enable_control=false / contact_guard=null，
  碰撞锁存等待显式激活；未启动新一轮实机推理。

## 完整实机复测：用户确认成功（2026-09-11 12:50）

用户确认现场可开始后，刷新双相机和机器人观测，初始夹爪闭合、关节无错误；通知现场
录屏。关闭仅本任务旧只读待机服务，以 `robot_bridge.sh control` 启动同一部署版本，
运行任务对应的完整 place runner。未修改模型、阈值或动作目标。

- 版本：`h16_collision_latched_20260911`；原子 H16、50 Hz 联合 arm/tool，
  `calibration_only`，无 RTC，无自定义 contact guard。
- 激活时原生碰撞配置 true / 3 / 500，锁存 armed；本轮未报告碰撞、模式错误或锁停。
- 完成 4×H16，共 64 步；4 轮累计 10.325 s（不含模型预热和启动校准）。
  各轮模型计算约 0.508 / 0.441 / 0.332 / 0.295 s。
  各 H16 提交至完成约 1.662 / 1.769 / 1.702 / 1.765 s。
- 收到实际开爪事件，结束反馈 −0.767939 rad，开爪后回撤 231.453 mm。
  程序结果 `PASS_PLACED_RELEASED_AND_RETRACTED`；用户随后明确评价“这次可以”，
  因此记为本任务一轮现场确认成功，而不只是程序完成条件通过。
- runner 和远端动作服务均退出码 0；未自动复位、闭爪或再启动下一轮。
  末态只读反馈夹爪约 −0.766518 rad，右臂及夹爪错误码为 0。
- 保留此版本与配置作为已成功运行参照；一轮成功不构成成功率估计、完全无抖动，
  或真实碰撞触发保护的验收。末次观测与动作状态有约 13.8 mm 位置差异诊断，仍保留
  原始日志，不为获得通过而改掉诊断值。现场保护资格标记未自动改成已验证。

报告与前后画面（仓库相对路径）：
`agibot/local_reports/zhewan_live_20260911/collision_latched_retry/`，包括
`inference.json`、`runner.log`、`observation_before/`、`observation_after/`。
