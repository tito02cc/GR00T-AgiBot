# 折弯：右臂放置、释放、回撤（r0003）

30k 模型已训练完成，2026-09-11 完整复测成功，现场确认动作完整、轨迹平滑。
推理桥接于 9 月 14 日固定。
日常推理使用 [10.20.15.194 机器人固定包](../../deployments/g2_194/README.md)，不使用本目录旧入口代替固定版本。

首轮失败及危险下压的记录保留在 [现场复盘](LIVE_TEST_20260911.md)，后续修正和成功复测见
[部署记录](BRIDGE_DEPLOYMENT_20260911.md)。各阶段结果按记录日期区分，尚未统计长期成功率。

环境安装、数据目录和模型配置见 [操作手册](../../README.md)。

## 任务和输入

使用原有 G2 数采格式及 GR00T N1.7。当前任务验证右臂，动作从夹持工件开始，放置后张开
夹爪并回撤。训练名单 `train.txt` 400 条，独立采集会话留出名单 `heldout.txt` 58 条。
这两个文件是唯一选择入口，不纳入原始目录中早期下载留下的其他部分 episode。

源批次 `zhewan_right_place_r0003_job01` 的目录/提示词仍有旧 Xichong 名称。用户在
2026-09-09 查看 episode_000409 后确认其为本折弯任务。原始资料保持不变，转换时根据
`agibot/configs/zhewan_right_place_r0003_prescreen.json` 明确映射到 bending workstation。

## 数据准备

```bash
.venv/bin/python agibot/scripts/prepare_g2_task.py \
  --task-config agibot/tasks/zhewan_right_place_r0003/prepare.json
```

`prepare.json` 是本机这批数据的实例配置；其他机器应修改数据/输出位置，或另存一份本地
配置。可复用母本见 [`../../templates/right_place/README.md`](../../templates/right_place/README.md)。

配置定义：

- 两路图像：head_color、hand_right；640×480，10 Hz，H.264 CRF18。
- EEF：源 XYZ+XYZW → 绝对 XYZ+Rot6D；Rot6D 使用旋转矩阵前两行。
- action：源世界系 delta 重建下一步绝对 EEF，再由官方 processor 转为相对动作。
- 夹爪：保留绝对 native radians，约 0 闭合、-0.785 张开，保留全部中间开度。
- H16；最后一行保留采集器的 next observation，不人为置零或删除。
- normalization 只拟合 400 条 train，58 条 heldout 使用同一份统计量。
- 本任务 `normalization_bounds=minmax`，对应官方 `--no-use-percentiles`。旧 q01/q99
  设置会将少数回撤 state 截断最多约 65 mm，因此不继续沿用。详见本轮结果记录。

未来训练的 `dataset_path` 必须指向上述 `train` 子目录，不要将包含两个分集的父目录
或原始数据目录作为训练输入。留出集只供离线评估。

输出（仓库相对路径）：

| 内容 | 位置 |
|---|---|
| 原始选定数据 | `agibot/data/zhewan_right_place_r0003_job01` |
| 训练数据 | `agibot/gr00t_data/zhewan_right_place_r0003_400/train` |
| 留出数据 | `agibot/gr00t_data/zhewan_right_place_r0003_400/heldout` |
| 本轮报告和日志 | `agibot/local_reports/zhewan_prepare_20260910` |

## 相比之前流程的调整

1. 沿用经过训练/真机验证的数值转换，不更换动作表示或归一化逻辑。
2. 用右臂放置筛查器代替旧抓取/抬升条件，不依赖 `xichong_terminal` 字段名称。
3. 显式名单和计数，发现选中数据有问题就报告，不悄悄跳过、删掉或补入其他样本。
4. 保留源任务信息和实际采集时间；标准任务描述同时写入两个分集。
5. 视频对应校验从首/中/尾抽查改为全部帧；画质门槛 30 dB 是本流程的经验检查值。
6. 官方 loader 全量读取，独立重算统计量，所有完整 H16 窗口检查 EEF 往返。
7. 报告区分自动数据检查、视觉检查、模型训练和真机验证，不将前一阶段 PASS 当成后一阶段成功。
8. 审计真实处理器的 state 截断影响后改用训练 min/max；训练启动器读取
   `meta/training_preprocessing.json` 自动选取归一化参数，显式环境变量冲突时拒绝启动。

## 当前阶段

2026-09-14：已按用户要求保存 [10.20.15.194 机器人 / zhewan 推理固定包](../../deployments/g2_194/README.md)。
后续日常实机使用固定包的命令入口和源码快照；本目录旧 `inference.py` / `robot_bridge.sh`
保留为历史入口，不作为固定版本入口。不启用 RTC 或额外滤波。

**最新实测：2026-09-11 12:50，当前 H16 + 碰撞锁存 + 补偿冻结版完成完整推理，
用户现场确认“这次可以”。** 共 4×H16 / 64 步，确认开爪后回撤 231.45 mm；
动作服务已正常退出，没有自动复位或额外开闭爪。本版本作为本任务已成功运行的参照，
保留当前参数。详细证据见 [部署记录](BRIDGE_DEPLOYMENT_20260911.md) 最后一节。
这是一轮现场成功，不代表不同场景成功率或碰撞保护触发效果已验证。

2026-09-11 更新：首轮真机出现抖动和危险下压，不能把下文离线检查理解为真机成功。
当前候选使用整块 H16、本机 50 Hz 联合 arm/tool、校准后冻结位姿补偿，以及原生
碰撞事件锁存；任务配置已配套更新。部署后先完成只读检查，随后完成上文现场复测。
当日历史入口为本目录 `robot_bridge.sh`（默认 `standby`）；现已切换固定包入口。详细版本、保护
语义及验收边界见 [部署记录](BRIDGE_DEPLOYMENT_20260911.md)。不要混用旧动作服务。

云端路径、单卡训练配置、旧产物清理和后续启动方法见 [云端训练记录](CLOUD_TRAINING.md)。
上传和无卡检查不等于 GPU 测试通过；启动训练前先通知用户。

2026-09-10：数据转换、全帧对应、训练统计量、官方 loader/处理器全量检查完成，采用
min/max 的最终检查通过。规模、归一化改进、视觉抽查范围及剩余边界见
[数据处理结果](DATA_PREPARATION.md)；机器报告为本轮 `preparation_status.json`。
已完成修复后的 100 step GPU 短测及 checkpoint 处理器重载检查，详见云端训练记录。
30k 正式训练已于 2026-09-10 完成，退出码 0；最终 5k 平均训练 loss 为 0.034975。
30k 推理包已于当天 18:44 下载完成，位于
`agibot/models/zhewan_rplace_r0003_n1d7_checkpoint-30000/`。
本机官方 GPU 推理、全部 58 条留出轨迹与联合请求映射检查已完成，
存在预测时序和开放环误差需要现场观察；当时尚未开展实机，后续结果见本文顶部。
历史离线结果见 [离线测试](OFFLINE_TEST.md)，当时任务参数在 `inference.json`；
现在的配套推理命令以固定包为准。
原始画面、示范等待段和开爪时序的进一步检查见 [异常复查](ANOMALY_REVIEW.md)。
不修改此前 10.20.15.194 机器人的已验证控制桥接。云端查看进度方法见 [训练记录](CLOUD_TRAINING.md)。

后续真机阶段可复用 10.20.15.194 机器人的 GDK 联合控制实现，但应按本任务重新核对初始图像、工作范围
和夹爪接口。`train/meta/initial_pose_reference.json` 是从真实训练起点选出的参考，不是
自动复位命令，也不应直接沿用上个架子放置任务的坐标和 workspace 参数。
