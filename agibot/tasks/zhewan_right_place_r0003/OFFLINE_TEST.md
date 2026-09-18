# 本机离线推理检查 · 2026-09-10

结论：**本机模型服务、处理器、H16 解码及联合请求映射通过；模型效果有待现场验证，
不是无人看护或无需现场确认的真机放行。** 未连接机器人、未发送运动或开闭爪命令。

## 实际测试范围

- RTX 4090 24 GB，现有 `.venv`；GPU 总占用采样约 8.9 GB（包含桌面，不是模型峰值）。
- 完整加载本机 checkpoint-30000 三片权重及 Cosmos 骨干。
- 使用原版 `gr00t/eval/run_gr00t_server.py`，`Gr00tPolicy` → 官方 `PolicyClient`，
  仅监听 `127.0.0.1:5574`，BF16 推理、4 次去噪；未启用 RTC、平滑或修改权重。
- 8 条等间隔训练样本、全部 58 条留出样本，UUID 无交集；共 4,907 个真实帧，
  338 次 H16 请求，另有 1 次预热。按每 16 行重新输入记录观测，覆盖整个 episode。
- 每次全部 16 行均检查；末块超出原轨迹的尾部不计误差，但仍检查输出/解码结构。
- 保存的 min/max 配置保持不变，未套用旧 xichong 的 percentile 或统计量。
- 用真实输出经过同一个 `decode_action_chunk` 与 `NativeTrajectorySender`，
  在内存假 GDK 上验证 5,408 条完整 waypoint、27,040 次联合请求。
  每次同时携带 `right_arm/ABS_POSE` 和 `right_effector/ABS_JOINT`，每行 5 tick。
  **该模拟不模拟物理反馈、IK、碰撞、DDS 或实时调度。**

范围内夹爪值映射不变，0 闭合、−0.785 张开，中间开度均保留；仅端点 float32 误差
约 2.6e−8 rad 被现有范围修正处理。XYZ 未被二次缩放，Rot6D → XYZW 与独立旋转矩阵
计算一致。实际采集观测与 live adapter 的图像、state、prompt 输入结构一致。

## 性能与开放环误差

| 指标 | 8 条训练样本 | 58 条留出样本 |
|---|---:|---:|
| 每块第 1 行 XYZ 参考误差均值 | 3.74 mm | 7.77 mm |
| H16 全部有效行 XYZ 参考误差均值 | 33.29 mm | 54.69 mm |
| H16 全部有效行 XYZ 参考误差 P95 | 166.09 mm | 196.41 mm |
| 旋转参考误差均值 | 1.64° | 2.40° |
| 开爪时机绝对差均值 | 0.075 s | 0.312 s |
| 预测首次开爪位置与示范首次开爪位置距离均值 | 9.15 mm | 19.57 mm |
| 没有预测开爪的 episode | 0 | 0 |

首次预热约 2.16 s；后续官方服务往返平均 **0.381 s/块**、P95 0.533 s，最大 0.954 s。
这是本机 loopback 的模型请求耗时，不包含机器人相机采集、SSH、GDK 或 H16 执行时间。
原同步策略每块执行约 1.6 s 后再取观测，仍会有模型/通信等待，不宣称零间隔。

上述是**记录观测下的开放环预测误差**。动作不会生成下一帧图像，下一块仍输入示范
轨迹状态；模型动作节奏与示范不同会产生很大的同时间索引误差，不能当作真实机械臂
跟踪误差、真实放置精度或成功率。误差随块内预测长度增加，不能只引用较小的第 1 行。
留出首次开爪位置参考距离 P95 43.18 mm、最大 70.51 mm，需关注实际落座精度。

后续 [异常复查](ANOMALY_REVIEW.md) 确认：长等待段会放大同时间索引误差；最大
70.51 mm 样本的充分开爪跨阈值恰好发生在记录观测重置点，不能直接当作真机落点误差。
原结果保留，补充诊断不改变模型或实际开爪阈值。

### 时序异常与重复测试

留出 `episode_000004` 示范首次达到 −0.72 rad 在 step 126；第一轮预测 step 109。
两个额外随机推理复测分别是 109、108，提前约 1.7～1.8 s。
拼接预测中开爪后再闭合出现在 **step 112 的新 H16 边界**：此时重新输入的是示范中
仍闭爪的观测，不是上一块模型开爪后的反馈。三轮均有此现象。

全量首轮的“再次闭合”只有这一例，且不是同一 H16 内反转。它不是夹爪桥接翻转、
丢失开爪或强制闭爪的证据，也不能据此断言真机一定再闭合；提前释放预测仍应关注。
本轮没有通过锁死夹爪、增加固定动作或改权重来隐藏这些时序差异。

## 软件回归

- continuous 联合 bridge/controller/sender：65 项通过，另含 60 个 subtest。
- 放置 runner：34 项通过，另含 56 个 subtest。
- 共享 H16/夹爪 runner：25 项通过，另含 2 个 subtest。
- 新离线评估器：5 项通过。合计 **129 项**，另有 118 个 subtest。

第一次将所有历史测试放进同一个 pytest 进程时，出现机器人模块导入路径缺失，以及
抓取/放置配置重复注册 `NEW_EMBODIMENT` 的测试收集冲突；按原部署 `PYTHONPATH`
运行机器人测试、将两个任务测试分别运行后均通过。没有为使测试通过修改生产桥接。

## 明天的入口与剩余现场工作

新增 `inference.json` 保存本任务模型、提示词、训练初始参考、native 夹爪约定和端口，
避免旧 runner 默认落入 xichong 参数。旧任务与机器人桥接文件未修改。

本机启动正式模型服务（不动机器人）：

```bash
cd /path/to/Isaac-GR00T
.venv/bin/python agibot/tasks/zhewan_right_place_r0003/inference.py server
```

生成本任务完整 runner 命令（**只打印，不执行**）：

```bash
.venv/bin/python agibot/tasks/zhewan_right_place_r0003/inference.py \
  print-live-command --report agibot/local_reports/zhewan_live/inference.json
```

生成的命令使用原同步完整 H16、`--max-cycles 0`、`--optimize-transport`，不使用 RTC。
初始参考 XYZ 为 `[0.474645, -0.245275, 0.922031]` m，完整 XYZW 与夹爪来自训练参考文件。
该参考不是自动复位命令。开爪后回撤判据仍为 0.13 m；400 条训练示范对应距离最低
0.140739 m，符合这个起始判据，但不能用它证明工件已正确落座。

**现场尚待完成：**

1. 对齐新折弯工位的初始图像/姿态，确认工件稳定夹持，无卡挂，现场可测试。
2. 确认 194 的 GDK 与已验证联合桥接部署一致，只有一个控制 owner。
3. 更新机器人服务的任务 workspace。旧 xichong 的最低 Z=0.975 m 会排除本任务
   初始 Z≈0.922 m；不能直接复制旧命令。`inference.json` 的训练 XYZ envelope
   仅是数据范围，不是已经确认无碰撞的现场工作空间。
4. 告知现场开始录像后再启动实际执行。停止后不自动开爪或复位。

因此本机软件准备已完成，但新工位不能跳过以上现场适配而直接宣称一定成功。

## 复现与证据

保持上述 server 运行，在另一终端：

```bash
.venv/bin/python agibot/tasks/zhewan_right_place_r0003/inference.py offline \
  --report agibot/local_reports/zhewan_offline_repeat
```

本次报告目录：`agibot/local_reports/zhewan_offline_20260910/`。
`evaluation/summary.json`、`analysis.json`、逐 episode JSON 和 `heldout_examples.png`
保存原始预测、误差与映射证据；`repeat_1/`、`repeat_2/` 保存异常轨迹的复测。
大报告不进入 Git；本文和可复用脚本进入任务资料。
