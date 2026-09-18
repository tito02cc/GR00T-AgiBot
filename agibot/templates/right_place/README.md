# 右臂放置任务配置母本

从仓库根目录运行；公共实现位于 `agibot/scripts/prepare_g2_task.py` 和
`prepare_g2_right_place_training.py`。不需要复制公共脚本。

1. 建立 `agibot/tasks/<task-id>/`，复制本目录的 `prepare.template.json` 为 `prepare.json`。
2. 填写路径；相对路径一律相对于仓库根目录，与启动命令所在目录无关。
3. 放入明确的 `train.txt`、`heldout.txt`（每行一个 `episode_NNNNNN`），按采集会话隔离。
   不要遍历原始目录自动纳入全部 episode，也不要随机拆分同一 episode 的帧。
4. 将 `prescreen.template.json` 复制为 `prescreen.json`，填写数采批次、源任务/提示词白名单、标准任务描述、相机及筛查阈值。
   可参考折弯案例的 prescreen，但阈值是任务经验值，不是 GR00T 官方通用标准。
   若有旧名称，必须记录确认来源和映射原因；绝不修改原始元数据。
5. 复制本目录的 `modality.py`，或引用已验证且语义完全相同的公共配置：10 Hz、H16、右 EEF
   XYZ+Rot6D、夹爪绝对 native radians。开/闭方向必须以当前数采定义为准。
6. 运行：

```bash
.venv/bin/python agibot/scripts/prepare_g2_task.py \
  --task-config agibot/tasks/<task-id>/prepare.json
```

流水线重新审计选定原始数据及全部两路 JPEG，调用原有转换数值逻辑，生成 train/heldout，
只用 train 计算官方绝对/相对统计量，然后逐帧核对源图像、检查所有 H16 完整窗口的官方
EEF 相对/绝对往返，并通过官方 loader 解码所有 episode。

`--stage convert` 仅转换；`--stage verify` 对现有输出重跑验证。默认 `all`。转换拒绝覆盖
既有数据目录；调整名单/参数后应使用新输出目录，不将旧视频或统计缓存混入新版本。
重新拟合统计量只针对 train；heldout 使用同一份训练统计量。官方 stats 内部保留其 schema
缓存指纹机制，不做额外原始数据/模型 SHA256 扫描。

输出中的 `meta/source_episode_map.json` 保留原始 episode、UUID、任务映射和采集时间。
Parquet/MP4 使用固定 10 Hz 网格，但没有插帧、删帧、平滑、姿态限幅或夹爪二值化。
CRF18 H.264/yuv420p 沿用成功案例：有损，不承诺对训练效果零影响；逐帧 PSNR 是数据
对应和画质指标，不能替代工件是否正确放置的视觉审核。

官方处理器验证同时检查真实 `StateActionProcessor` 的归一化、反归一化和 EEF 恢复。
`normalization_bounds` 可选 `minmax`（本母本默认）或 `percentile`，对应官方
`use_percentiles=False/True`，保留 `clip_outliers=True` 和 `use_relative_action=True`。
当前官方实现中，相对 EEF 使用各 horizon 的 relative min/max，state 和绝对夹爪根据
所选模式使用训练 min/max 或 q01/q99。超界截断本身不具有可逆性，所以报告会区分纯位姿往返误差、与官方数学
结果的误差，以及训练/留出数据被该处理实际改变的幅度。不会为了得到 PASS 修改官方算法。

不能无条件沿用旧任务的 percentile 设置：折弯案例在完整轨迹已通过结构/同步检查的情况下，
仍有 456 个训练 state 的位置被 q01/q99 截断超过 1 cm，最大约 65 mm；改为 minmax 后
训练集此类截断为零。判断依据来自训练集，不使用留出样本拟合边界。其他任务应查看自己的
报告，不能把 minmax 当作能修复未清洗异常值的手段。

验证通过后，两个数据分集各写入 `meta/training_preprocessing.json`。公共
`training/launch_1xa10080.sh` 会读取其中的 `use_percentiles`，自动传入官方
`--no-use-percentiles` 或 `--use-percentiles`；显式 `GROOT_USE_PERCENTILES` 与元数据
冲突时拒绝启动。没有该元数据的旧案例保留原来的 percentile 默认值。直接运行官方训练
脚本时也必须传入对应开关，不能只复制数据而遗漏处理器设置。

本机数据、模型、完整报告不提交 Git。可提交代码、配置母本、脱敏任务记录及纯 episode
名单；不要将机器人密码或云 token 放进任务目录。训练/推理配置应在相应阶段填入并记录
验证结果，数据检查通过不意味着新模型已训练或真机已验证。

## 从新原始数据生成名单

下面以 400 条训练数据为例，数量不是强制标准。先按上面的步骤创建任务目录与
`prescreen.json`。命令从仓库根目录执行：

```bash
TASK_ID=my_right_place
RAW_ROOT=/path/to/raw_episodes
REPORT_ROOT=agibot/local_reports/$TASK_ID

.venv/bin/python agibot/scripts/audit_g2_right_place_raw.py \
  --raw-root "$RAW_ROOT" --profile "agibot/tasks/$TASK_ID/prescreen.json" \
  --output-dir "$REPORT_ROOT/audit"

.venv/bin/python agibot/scripts/decode_g2_prescreen_images.py \
  --raw-root "$RAW_ROOT" --episode-list "$REPORT_ROOT/audit/candidates.txt" \
  --report "$REPORT_ROOT/image_decode.json" --workers 4

.venv/bin/python agibot/scripts/select_g2_prescreen_candidates.py \
  --audit-report "$REPORT_ROOT/audit/report.json" \
  --image-report "$REPORT_ROOT/image_decode.json" \
  --output-dir "$REPORT_ROOT/selection" --counts 400 --heldout-target 60

cp "$REPORT_ROOT/selection/train_400.txt" "agibot/tasks/$TASK_ID/train.txt"
cp "$REPORT_ROOT/selection/heldout.txt" "agibot/tasks/$TASK_ID/heldout.txt"
```

筛选器保留完整采集会话作为留出集，实际数量不一定恰好 60（折弯案例为 58）。将实际
数量填入 `prepare.json` 的 `expected_train_count` / `expected_heldout_count`。
检查报告和关键画面后再接受名单；candidate 不是自动证明物理放置成功。

数据在数采机时，可以先在那里运行这三步，再按名单传输完整 episode（NPZ、JSONL、
元数据、参数文件和图像），而不是只传 JPEG。选定数据到训练准备机后，运行
`prepare_g2_task.py` 做本机复查及转换。本流水线不访问 GDK，也不控制机器人。

需要浏览释放/回撤画面时：

```bash
.venv/bin/python agibot/scripts/build_g2_place_review.py \
  --raw-root "$RAW_ROOT" --audit "$REPORT_ROOT/audit/report.json" \
  --train-manifest "agibot/tasks/$TASK_ID/train.txt" \
  --heldout-manifest "agibot/tasks/$TASK_ID/heldout.txt" \
  --output "$REPORT_ROOT/visual_review"
```

`visual_review/index.html` 包含所有选定 episode 的关键帧；`sample_*.jpg` 是按轨迹极值和
时间跨度选出的抽查页。脚本只生成材料，不替审核者填入成功结论。
