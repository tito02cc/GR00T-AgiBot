# 数据处理结果 — 2026-09-10

## 已完成

自动数据检查通过。原始数据未删除、改写、平滑或裁帧，未进行额外数据/模型 SHA 扫描。
本轮未运行新模型训练、GPU forward 或机器人控制。

| 内容 | 训练集 | 留出集 |
|---|---:|---:|
| episode | 400 | 58 |
| 帧 / Parquet 行 | 27,980 | 4,326 |
| Parquet 文件 | 400 | 58 |
| 两路 MP4 | 800 | 116 |
| 完整 H16 窗口 | 21,980 | 3,456 |
| 源 JPEG 与视频逐帧比较 | 55,960 | 8,652 |
| 最低源图像 PSNR | 36.262 dB | 36.128 dB |

合计 32,306 帧，916 个视频，转换数据约 2.04 GB。帧数与选定原始 episode 完全一致，
不是只传了图片或只转换了轨迹片段。源数据的左腕图像保留在原始目录，但本右臂策略只
使用 head_color / hand_right 两路。MP4 仍是 CRF18 有损存储；PSNR 不是模型效果保证。

400/58 名单无交叉；留出集来自独立完整采集会话，不参与统计量拟合。按名单在本机重新
检查全部 NPZ/JSONL/元数据、同步和轨迹语义，并完整解码 64,612 张训练用源 JPEG，
458 条全部仍为 candidate。未新增剔除或替换。

## 相比旧流程实际修正的问题：state 归一化

数据转换本身正确，但直接沿用旧的 q01/q99 归一化，会截掉这批数据中的合法尾部 state，
主要出现在部分回撤轨迹，不能简单归因于初始位置。

| 诊断 | 旧 percentile | 最终训练 min/max |
|---|---:|---:|
| 训练 state XYZ 最大变化 | 65.036 mm | 约 0，仅 1.37e-8 m 数值舍入 |
| 训练位置截断超过 1 cm 的帧 | 456 | 0 |
| 留出 state XYZ 最大变化 | 68.759 mm | 11.861 mm |
| 留出位置截断超过 1 cm 的帧 | 471 | 2 |

因此本任务采用：

```text
normalization_bounds = minmax
use_percentiles = false
clip_outliers = true
use_relative_action = true
official finetune flag: --no-use-percentiles
```

这是根据训练集发现的问题选择的参数，不用留出数据扩张训练边界。源轨迹不变，stats
仍由官方脚本仅从训练集计算。相对 EEF 本来就使用 relative min/max；夹爪的 q01/q99
恰好等于 min/max（约 -0.785 至 0），因此这两类 action 的归一化边界没有改变。
训练和留出中的夹爪动作均无超界截断，数值往返误差约 2.30e-8 rad。

两个分集的 `meta/training_preprocessing.json` 保存最终参数。公共 A100 启动器自动读取
它并传入官方开关；若 `GROOT_USE_PERCENTILES` 与数据记录冲突则拒绝启动。没有这份
元数据的旧任务仍保留旧默认值，不修改之前已训练模型的 processor 配置。

留出集还剩 8 帧的位置超出训练 min/max 超过 1 mm，其中 2 帧超过 1 cm，最大约
11.9 mm。这是需要在后续 held-out 评估里保留的分布边界，不删掉这些样本、不用它们
重算训练统计量，也不把数据 PASS 当成泛化或真机成功率证明。

## 数值、加载和处理器检查

- 转换状态和动作相对原始数据的最大元素误差约 2.98e-8，主要来自 float32 存储。
- Rot6D 前两行正交性、时间戳、帧索引、source episode 映射和任务描述检查通过。
- 400/58 的所有 episode 均经官方 `LeRobotEpisodeLoader` 加载，两路视频全量解码。
- 全部 25,436 个完整 H16 窗口经过官方 EEF 相对/绝对往返和实际 `StateActionProcessor`。
- 位姿绝对往返最大误差约 3.55e-8；独立公式与官方反归一化最大误差约 1.11e-15。
- 独立重算 state/action 统计量误差为 0；相对 EEF 统计量最大误差约 5.96e-8。
- 54 项相关回归测试通过；包括中间帧错配、非零最后一步目标、任务提示词映射、分集
  隔离、官方处理器、小型端到端数据以及训练归一化开关冲突；Ruff、shell 语法检查通过。

这些是离线数值误差，不是机器人实际定位精度。未声称已运行新模型 forward、A100 显存
检查或真机闭环；这些属于接下来的训练/推理阶段。

## 视觉检查范围

生成了全部 458 条的头部/右腕释放和回撤关键帧页。实际查看 18 条（12 训练、6 留出），
覆盖运动/稳定性极值及采集时间跨度，每条 6 个时刻、两路画面，共 216 个图像面板。
抽查未发现明确需要新增剔除的问题；有正常光照差异，夹具遮挡和反光限制精确落座判断。
没有将未看的 440 条自动标为视觉 PASS，也没有进行 458 条逐帧物理成功验收。

## 复现与证据

从仓库根目录运行：

```bash
.venv/bin/python agibot/scripts/prepare_g2_task.py \
  --task-config agibot/tasks/zhewan_right_place_r0003/prepare.json
```

已有输出请用 `--stage verify` 复查；默认转换拒绝覆盖旧数据。对本次已完成的产物不需要
再跑一次。若改变名单、图像、坐标、任务描述或处理器设置，重新验证对应环节。

本机报告目录：`agibot/local_reports/zhewan_prepare_20260910/`：

- `local_raw_audit.json`：458 条本机原始数据复查。
- `train_conversion_audit.json`、`heldout_conversion_audit.json`：全帧视频与低维转换审计。
- `official_loading_and_math.percentile_baseline.json`：保留的旧 percentile 对照，不是最终配置。
- `official_loading_and_math.json`：最终 min/max 全量结果。
- `preparation_status.json`：最终状态及边界说明。
- `pipeline.log`、`minmax_verification.log`、`tests.log`：实际执行记录。
- `visual_review/index.html`、`visual_review/INSPECTION.md`：浏览页和实际抽查记录。

数据位置见 [README](README.md)。初始位姿参考写入训练数据的
`meta/initial_pose_reference.json`，引用真实 source episode_000809；仅供下一阶段核对，
不是直接可执行的复位命令。
