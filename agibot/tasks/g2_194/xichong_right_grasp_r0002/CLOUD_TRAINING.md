# 10.20.15.194 右臂抓取：单卡 A100 80GB 训练

任务：`xichong_right_grasp_r0002`。本次从 GR00T-N1.7-3B 基座开始后训练，
不接着旧放置任务的 checkpoint 训练。数据清洗与转换见 [DATA_PREPARATION.md](DATA_PREPARATION.md)。

## 2026-09-20 准备结果

- 11:21 上传完成，退出码 0。训练集 1,211 文件 / 1,950,808,831 字节；
  留出集 283 文件 / 475,070,846 字节。完整相对路径与逐文件大小和本机一致，没有缺失或多余文件。
- 云端状态为 `CPU_CONFIG_AND_INVENTORY_PASS_GPU_SMOKE_PENDING`，配置解析、训练/留出隔离、
  训练统计复用检查通过；GR00T/Cosmos 的模型索引、头部及文件长度检查通过，未做模型负载扫描。
- 官方 loader 抽测训练第 0/399 条（90/105 帧）、留出第 0/90 条（187/77 帧）通过；
  这是部署抽测，本地全量图像/数学核对已在数据处理阶段完成。
- Python 3.12.14，PyTorch 2.9.0+cu128，CUDA runtime 12.8；CPU 复核时未开卡。
- 上传后数据盘可用约 107 GiB。当时尚未启动正式训练；后续 GPU 试跑和正式启动结果见下节。
- 配置、传输分区及磁盘预算相关测试 26 项通过。

实际解析结果位于云端复核报告目录的 `resolved_training_config.json`。
其中 `max_action_horizon=40`、`max_state_dim=max_action_dim=132` 是模型容量上限，
不是本任务的数据维度；任务实际为 H16、状态/动作各 10 维，由模态和 mask/padding 配合处理。
本机复核副本位于 `agibot/local_reports/g2_194/xichong_right_grasp_r0002_prepare_20260920/cloud/`。
单流出现降速后改为两个不重叠名单并行续传，旧日志 `upload.single_stream.log` 的退出码 20
是主动切换传输方式留下的记录；最终状态看 `upload.log`，退出码为 0。

## 2026-09-20 GPU 试跑结果

开卡后确认 GPU 为单卡 NVIDIA A100-SXM4-80GB（81,920 MiB），驱动 580.126.09。
11:36 启动独立 100-step smoke，正常完成并退出，`SMOKE_EXIT_CODE=0`。

- 实际使用本任务 400 条训练数据，batch16 × 梯度累积2；不使用留出集训练。
- 10 组 loss / grad_norm / learning_rate 日志均为有限数；前 10 步平均 loss 1.2374，
  最后 10 步平均 0.9409，总平均 1.06374276；记录的 grad_norm 范围为 0.350553–0.953364。
- 训练期间抽测显存 40,329 MiB（约 39.4 GiB），无 OOM。此值是采样值，不是精确峰值。
  首批缓存后约 0.75–0.9 秒/step；100 步 Trainer 耗时 177.993 秒，包含首次缓存和 checkpoint 保存，
  不含随后根目录模型导出。不能直接将该平均速度作为正式训练的精确 ETA。
- `checkpoint-100` 保存了模型、optimizer、scheduler、RNG 和 trainer_state，global_step=100。
  checkpoint 和最终根目录导出均为 3 个 safetensors 分片，索引/头部/文件长度检查通过；没有做 SHA 全盘扫描。
- checkpoint 的 processor 重载通过：`use_percentiles=false`、relative action / clip_outliers 与数据配置一致，
  实际 state 归一化边界匹配本任务训练统计，dataset_episodes=400。
  此检查验证 processor，不声称已重新执行保存后模型的 GPU 前向。
- smoke 使用 100-step 自身的 warmup/cosine 调度，只用于链路检查；正式 30k 仍从基座重新开始，
  warmup 为 1,500 步，不从 smoke 续训。试跑 loss 不是留出评估或真机抓取成功率。

日志及小型配置归档保存在上述云端复核目录和本机 `cloud/` 报告目录：
`smoke.exit`、`smoke_normalization.log`、`smoke_metadata.tgz`、`gpu_smoke_result.json`。
完整训练日志名为 `xichong_rgrasp_r0002_n1d7_e400_f10_h16_1xa10080_s30k_ckpt5k_v1_smoke.smoke.log`。
为腾出正式训练空间，验证后仅删除本次 smoke 的 6 个权重分片和 optimizer.pt，约 35.5 GiB；
原处小型配置/统计/状态和日志继续保留。该 smoke 已不能用于权重推理或断点续训；大文件未备份。
基座、数据和其他任务均未改动。试跑完成时未启动正式训练，之后的启动记录如下。

## 正式训练启动：2026-09-20 12:12（北京时间）

收到正式启动指令后，于 12:12:20 在云端 tmux 会话 `groot_xichong_grasp_r0002_30k`
启动 `train.sh baseline`。从 GR00T-N1.7-3B 基座初始化，不续接 smoke。
实际写入输出目录的 `experiment_cfg/config.yaml` 已核对：30,000 step、batch16 × 累积2、
warmup 5%、每5,000步保存、保留2个、每10步日志、min/max 归一化，仅加载本任务 train/。

12:13:54 已进入第32步，前三组10步平均 loss 为 1.3921、1.3813、1.3805，
loss/梯度均有限，无 OOM；抽测显存40,329 MiB。正式调度尚在1,500步 warmup 中，
不应与100步 smoke 的短调度 loss 直接对比。此处是启动快照，不代表训练已完成。

SSH 登录云端后查看：

```bash
tmux attach -t groot_xichong_grasp_r0002_30k
```

按 `Ctrl+B` 后按 `D` 可退出查看而不停止训练；不要按 `Ctrl+C`。
进程退出后，复核报告目录的 `baseline.exit` 会记录实际退出码。
训练日志及 checkpoint 路径见下文。

## 路径与文件

### 2026-09-20 训练完成与本机下载

19:21:25 正式训练正常结束，`BASELINE_EXIT_CODE=0`、global_step=max_steps=30,000。
Trainer 耗时25,692.883秒，最后1,000步平均loss 0.030617，最后5,000步平均0.0308342，
全部3,000组loss/梯度记录均为有限数。保留checkpoint-25000、checkpoint-30000。
30k模型分片索引/头部/长度及processor重载检查通过，实际min/max匹配400条训练数据。

切换无卡后，20:01在本机tmux会话 `groot_xichong_grasp_r0002_download` 启动30k推理包下载。
本机目录：`agibot/models/xichong_rgrasp_r0002_n1d7_checkpoint-30000/`，加载入口为 `model/`。
下载量12,583,679,653字节（含小型实验记录），不下载优化器；配套骨干复用本地副本。
入口为本目录 `download_model.py --ssh-control <已认证的SSH控制socket>`，支持按原始权重文件续传。

**此处记录的是下载启动，不是完成确认。** 状态文件在
`agibot/local_reports/g2_194/xichong_right_grasp_r0002_prepare_20260920/download/status.json`。
只有状态 `DOWNLOAD_COMPLETE_PROCESSOR_RELOAD_PASS` 才表示传输、分片头部/尺寸和本地processor检查全部通过。
云端完整checkpoint不删除；本机没有自动执行模型前向或机器人动作。

服务器数据盘根目录为 `/root/gpufree-data/GR00T`。

| 内容 | 数据盘内相对路径 |
| --- | --- |
| 仓库 / Python 环境 | `Isaac-GR00T/`、`Isaac-GR00T/.venv/` |
| GR00T 基座 | `models/GR00T-N1.7-3B/` |
| Cosmos 骨干 | `models/Cosmos-Reason2-2B/` |
| 训练集 | `datasets/xichong_right_grasp_r0002_400/train/` |
| 留出集 | `datasets/xichong_right_grasp_r0002_400/heldout/` |
| 任务配置 | `Isaac-GR00T/agibot/tasks/g2_194/xichong_right_grasp_r0002/` |
| 文件清单 | `manifests/xichong_right_grasp_r0002_{train,heldout}.inventory.json` |
| 配置复核报告 | `logs/xichong_right_grasp_r0002_preparation_20260920/` |
| 正式训练输出 | `outputs/xichong_rgrasp_r0002_n1d7_e400_f10_h16_1xa10080_s30k_ckpt5k_v1/` |

训练集 400 条 / 35,108 帧，留出集 91 条 / 7,358 帧，共约 2.43 GB 转换数据。
只上传转换后的两组完整数据和配置，不上传约 21.90 GB 原始文件，也不重复下载基座。
训练入口只传 `train/` 路径，不能把 `heldout/` 拼到 dataset-path。

## 训练参数

具体变量保存在 `train_1xa10080.env`，入口为 `train.sh`，复用已在该服务器跑通的
`agibot/training/launch_1xa10080.sh`。新任务有独立名称、模态和输出，不改旧任务配置。

| 参数 | 设置 |
| --- | --- |
| GPU | 单卡 A100 80GB，CUDA_VISIBLE_DEVICES=0 |
| 频率 / 动作窗口 | 10 Hz / H16（1.6 秒） |
| 视频 | head_color + hand_right |
| 状态 / 动作 | 右臂 EEF 9 维 + 夹爪 1 维 |
| 模型动作表达 | 当前窗口参考姿态下的相对 EEF；夹爪绝对值 |
| 每次前向 batch | 16 |
| 梯度累积 | 2，有效 batch=32 / optimizer step |
| 总步数 | 30,000 optimizer step |
| 保存 | 每 5,000 step，保留最新 2 个 checkpoint |
| 续训状态 | 保存 optimizer / scheduler / RNG，不是 save-only-model |
| loss 日志 | 每 10 optimizer step；本地日志，默认不启用 W&B |
| 优化器 / 学习率 | AdamW torch / 1e-4 |
| 调度 / warmup | cosine / 5%，正式训练 1,500 step |
| weight decay / 梯度裁剪 | 1e-5 / max_grad_norm=1.0 |
| 精度 | BF16 + TF32 |
| 训练模块 | projector、diffusion/action 模块；保留既有 tune_vlln=true |
| 冻结模块 | LLM 主体、视觉编码器，沿用上次实际成功配置 |
| state dropout | 0.2 |
| 图像颜色增强 | brightness 0.3 / contrast 0.4 / saturation 0.5 / hue 0.08 |
| 数据加载 | 4 workers，shard_size=1024，episode_sampling_rate=0.1 |
| 归一化 | min/max，显式 `--no-use-percentiles`，训练集独立统计 |

基座/骨干离线加载，不访问 HF 下载新权重。任务提示词来自本批源数据，不使用放置任务提示词。
`episode_sampling_rate=0.1` 是加载器随机采样参数，不是只选固定 40 条 episode。

### 参数选择依据

本任务与此前成功训练的任务均为右臂、两路相机、10 Hz、H16、同一基座，因此保留已在
单卡 A100 80GB 跑通的 batch、优化器、学习率和冻结策略。任务差异由新数据、提示词、
模态入口和独立统计量体现，不凭同名 `xichong` 复用旧任务统计。

本次 35,108 帧，多于此前 400 条放置数据的 27,980 帧。30k × 有效 batch32 约为 96 万
训练样本曝光，相当于约 27.3 倍保存帧数；实际为随机采样，并非严格按顺序的 27.3 个 epoch。
这是一套有依据的起始配置，不是已完成超参数搜索的最优值；30k 也不保证优于更早 checkpoint。
后续需看 loss、留出离线结果和实机效果，不以训练 loss 代替抓取成功率。

## 上传与复核

本机从仓库根目录运行（SSH 端口以当前实例为准，不把密码/token 写入文件）：

```bash
CLOUD_SSH_TARGET=root@YOUR_HOST CLOUD_SSH_PORT=YOUR_PORT \
  bash agibot/tasks/g2_194/xichong_right_grasp_r0002/upload.sh
```

该脚本要求已建立 SSH 密钥或连接复用认证，可选 `CLOUD_SSH_CONTROL_PATH`。
支持 rsync 断点续传，不重新压缩 MP4，不删除远端其他数据。
若单连接受丢包影响，可额外设置 `CLOUD_SSH_CONTROL_PATH_2` 使用第二条已认证连接。
`partition_upload.py` 按字节量生成两个不重叠、并集等于完整清单的文件列表，分别续传；
不会让两个连接同时写同一个文件。两个连接都成功后才核对完整清单。
双流进度分别在 `cloud/upload_part_0.log`、`cloud/upload_part_1.log`，总体结果仍在 `cloud/upload.log`。
完成标志是上传日志中的 `UPLOAD_COMPLETE_INVENTORY_PASS` 和 `UPLOAD_EXIT_CODE=0`。
核对相对文件名和逐文件字节数，不额外扫描整个数据/模型的 SHA。

云端 CPU 复核命令：

```bash
cd /root/gpufree-data/GR00T/Isaac-GR00T
.venv/bin/python agibot/tasks/g2_194/xichong_right_grasp_r0002/check_cloud_ready.py \
  --report /root/gpufree-data/GR00T/logs/xichong_right_grasp_r0002_preparation_20260920/cloud_ready.json
```

它检查上传文件清单、分集、训练统计来源、基座 safetensors 的索引/头部/尺寸，并调用实际
训练入口 `audit`，核对解析后的配置。**不执行训练，不加载 GPU 权重，不声称张量负载已完整验证。**
成功状态为 `CPU_CONFIG_AND_INVENTORY_PASS_GPU_SMOKE_PENDING`。

## 开卡后的运行顺序（本次配置步骤不自动执行）

先确认实际 GPU 是 A100 80GB，再执行：

```bash
cd /root/gpufree-data/GR00T/Isaac-GR00T
bash agibot/tasks/g2_194/xichong_right_grasp_r0002/train.sh smoke
```

smoke 为 100 step，输出目录为正式 RUN_ID 加 `_smoke`，不会混入正式 30k 目录。
须确认 loss / 梯度有限、显存足够、能保存 checkpoint、保存后 processor 仍为本任务 min/max。
CPU dry-run 不能替代 GPU smoke。保存后可用 `agibot/training/check_saved_normalization.py`
检查 smoke checkpoint 的 processor 与本任务 train/ 统计是否一致。

确认通过并获得正式训练指令后：

```bash
tmux new-session -s groot_xichong_grasp_r0002_30k \
  'cd /root/gpufree-data/GR00T/Isaac-GR00T && bash agibot/tasks/g2_194/xichong_right_grasp_r0002/train.sh baseline'
```

查看：`tmux attach -t groot_xichong_grasp_r0002_30k`。
正式训练日志：`/root/gpufree-data/GR00T/logs/xichong_rgrasp_r0002_n1d7_e400_f10_h16_1xa10080_s30k_ckpt5k_v1.baseline.log`。
断点恢复使用 `train.sh resume`；不要对已有正式输出再次使用 `baseline`。

### 空间与保留策略

清理旧 zhewan 后数据盘可用约 108.6 GiB；上传本批数据占用约 2.3 GiB。
上次同架构完整 checkpoint 每份约 24 GiB，两份及最终根目录权重约 60 GiB；
保存新 checkpoint 后再轮换旧文件会有临时峰值，不能只按常驻两份估算。
100-step smoke 除约 24 GiB 的 checkpoint 外，还可能在输出根目录额外导出约 12 GiB 权重，
合计接近 36 GiB。当前空间不能同时保留全部 smoke 大文件并保证正式训练轮换峰值的余量。
smoke 验证通过后，正式训练前须先归档其小型日志/配置，并确认处理不再需要的大权重；
不能删除正在用的恢复点或其他任务。

`train.sh` 会在真正训练前检查可用空间：baseline 至少 80 GiB、smoke 至少 40 GiB、
resume 至少 32 GiB；不满足时明确退出，不会自动删除文件。audit 不要求这些训练空间。
这些是基于上次同架构产物的余量预算，不是对所有未来模型尺寸的通用保证。

本次仅配置上传和训练入口，不修改 GDK/推理桥接、不启动机器人，也不把数据处理通过等同于实机成功。
