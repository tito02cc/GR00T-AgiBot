# 云端准备与训练

2026-09-10 状态：**正式 30k 已完成，退出码 0；30k 推理包已于 18:44 下载完成，官方本机 GPU 推理测试通过。**
上传进程退出码 0；远端与本机文件清单和逐文件字节数一致：训练集 1,211 个文件 /
1,698,718,265 字节，留出集 184 个文件 / 344,715,989 字节，合计 2,043,434,254 字节。
无缺失、多余或大小不匹配。上传后数据盘可用约 108 GiB。

## 本次配置

服务器使用数据盘 `/root/gpufree-data/GR00T`，不是 `/ct`。连接端口可能随实例变化，
以现场提供的 SSH 信息为准；凭据不写入仓库。任务参数在
[train_1xa10080.env](train_1xa10080.env)，入口为 [train.sh](train.sh)。

| 内容 | 数据盘下的位置 |
|---|---|
| 仓库及环境 | `Isaac-GR00T/`、`Isaac-GR00T/.venv/` |
| 原始预训练模型 | `models/GR00T-N1.7-3B/` |
| 视觉语言骨干 | `models/Cosmos-Reason2-2B/` |
| 训练集 | `datasets/zhewan_right_place_r0003_400/train/` |
| 留出集 | `datasets/zhewan_right_place_r0003_400/heldout/` |
| 上传清单 | `manifests/zhewan_right_place_r0003_{train,heldout}.inventory.json` |
| 新任务产物 | `outputs/zhewan_rplace_r0003_n1d7_e400_f10_h16_1xa10080_s30k_ckpt5k_v1/` |
| 日志 | `logs/` 下同名任务日志 |

上传转换后的完整两个分集，不上传约 16 GB 原始 JPEG/其他相机数据。不将 heldout 路径
传给训练。每份 inventory 由本机数据生成，按相对文件名和字节数核对远端，不做额外
SHA 扫描。这是上传完整性检查，不替代此前已经完成的数值/全帧校验。

## 训练参数与依据

- 单卡 A100 80 GB；每次前向 batch 16，梯度累积 2，有效 batch **32**，不是 16。
- 30,000 optimizer step；每 5,000 step 保存，保留最新两个；每 10 step 记录 loss。
- 保留 optimizer/scheduler/RNG，支持恢复训练；不是仅保存推理权重。
- AdamW，学习率 `1e-4`，cosine，warmup 5%（1,500 step），weight decay `1e-5`。
- 冻结 LLM 和视觉编码器；训练 projector、diffusion/action 模块，保留既有 VLLN 设置。
- BF16/TF32、state dropout 0.2、颜色增强沿用上次成功训练；不新增 RTC 或控制改动。
- 400 条训练、27,980 帧、10 Hz、两路图像、H16。
- **本任务使用 `--no-use-percentiles`**，与 `meta/training_preprocessing.json` 一致。
  不修改基座文件，也不修改旧任务的已训练 processor。
- `episode_sampling_rate=0.1` 是原有数据加载采样参数，不表示只取固定 40 条训练。

这是基于上次成功训练的起始配置，不是已通过本任务训练验证的最优超参数。按每步
32 个样本粗算约 96 万次样本曝光（约 34 倍帧数）；实际是随机采样，不是顺序 34 个
完整 epoch。30k 不保证优于所有较早 checkpoint，后续保留两份进行效果对照。

## 分阶段执行

无卡模式只做上传、文件/配置/依赖检查。2026-09-10 实例 cgroup 上限为 0.5 CPU / 2 GB
内存；`free -h` 显示宿主机内存，不能据此判断容器可以加载模型。

用户开卡后完成 GPU 冒烟检查，并已明确确认启动正式训练。
以下为操作方法，实际执行结果见文末记录：

```bash
cd /root/gpufree-data/GR00T/Isaac-GR00T
# 配置 dry-run：不执行模型前向或优化器更新。
bash agibot/tasks/zhewan_right_place_r0003/train.sh audit
# 开卡并确认后，先独立 100 step 冒烟，验证模型加载、数据 batch、loss 和 checkpoint。
bash agibot/tasks/zhewan_right_place_r0003/train.sh smoke
# 冒烟通过并确认后，正式训练从原始基座重新开始，不从 smoke checkpoint 续训。
tmux new-session -s groot_zhewan_r0003_30k \
  'bash agibot/tasks/zhewan_right_place_r0003/train.sh baseline'
```

从另一个终端查看：`tmux attach -t groot_zhewan_r0003_30k`。若当前没有启动任务，
`no sessions` 是正常情况。恢复中断的正式训练使用 `train.sh resume`，不是 baseline。
独立 smoke 产物也占空间；记录其结果后再决定是否删除，不自动删除其他任务。

## 2026-09-10 清理记录

已删除旧 xichong r0002 云端的 checkpoint-25000、checkpoint-30000、输出根目录的三片
最终权重，以及 train_600 / heldout_100 数据目录，释放约 63 GiB。空闲空间由约 47 GiB
增至约 110 GiB（上传新数据前）。保留旧任务脚本、日志、processor 和实验配置；删除前
将小型 metadata 备份到 `archives/xichong_rplace_r0002_metadata_before_cleanup_20260910.tgz`。

本机已有此前验证的 xichong 30k 推理包，本次未改动。云端 optimizer 已删除，**本机推理包
不能恢复旧任务的优化器续训状态**。原始采集数据未动。独立的 `s101_green_triangle` 项目
及 `grr/` 未删除；GR00T 基座、Cosmos 骨干、Python 环境和依赖缓存均保留。

模型检查仅读取已有 safetensors 头部、索引和文件长度：GR00T 两片 / 1,031 个 tensor、
Cosmos 一片 / 626 个 tensor 均一致。此检查没有重新下载模型、扫描完整权重或运行模型；
真正的 GPU 加载仍待开卡测试。

本机使用官方 `launch_finetune.py --dry-run` 解析相同参数已通过（不加载权重、不训练），
确认有效 batch 32、H16、relative EEF / absolute gripper、min/max、30k/5k/保留两份。
日志为本机 `agibot/local_reports/zhewan_prepare_20260910/cloud/local_config_dry_run.log`，
云端副本为 `logs/zhewan_rplace_r0003.local_config_dry_run.log`。关键
`state_action_processor.py`、`pose.py`、`lerobot_episode_loader.py` 经文本比较与云端一致。
本轮新增检查与数据准备相关测试合计 18 项通过，shell 语法与 Ruff 检查通过。

## 开卡检查与处理器参数修复

实际硬件：A100-SXM4-80GB，81,920 MiB 显存，12 CPU 核，容器内存上限 120 GiB。
云端任务入口 `train.sh audit` 通过，实际 GPU 短测能够加载基座、读取 21,980 个 H16 窗口、
完成 100 step 并写出 checkpoint。然而第一轮短测的实际处理器仍为 percentile，**不能作为
最终 min/max 训练放行依据**。第一轮 loss 1.2098 → 0.9814 仅证明那轮运行能完成。

根因：`Gr00tN1d7Processor.from_pretrained` 的覆盖列表遗漏 `use_percentiles`，虽然 CLI、
实验 config 和 dry-run 为 false，加载基座 processor 时仍保留 true。修复包括：

1. 将 `use_percentiles` 加入显式覆盖列表；不传覆盖值时继续尊重已有 checkpoint，旧模型
   推理语义不变，不修改原始基座权重或配置文件。
2. 模型加载时同步保存此配置，并在建立训练数据前检查外层 processor 与实际
   StateActionProcessor 都采用所要求的值。
3. 新增 6 种保存值/覆盖值组合的加载、运行时属性、保存及重载测试；修复前其中 2 种失败，
   修复后整个处理器测试文件 17 项通过。

第一轮临时测试大权重已删除；日志和小型 metadata 留在数据盘 `archives/` 下
`zhewan_smoke_percentile_mismatch_20260910.log` 和
`zhewan_smoke_100_percentile_mismatch_metadata_20260910.tgz`。
修复前云端源文件也备份在 `archives/`。修复后从原始基座重新执行 100 step 短测，
不从第一轮错误归一化的测试权重继续。

### 修复后实测结果

- 完整运行 100 optimizer step，进程退出码 0；首个/最后一个 10-step 平均 loss 为
  1.2098 / 0.9744，全程平均 loss 1.06178，梯度有限，未出现 NaN、CUDA OOM 或 cgroup OOM。
- GPU 显存采样约 40,289 MiB（39.34 GiB）；稳定计算段约 0.75～0.9 秒/step。
  首次缓存约 50 秒，首步含初始化约 70 秒；Trainer 计时含 checkpoint 保存约 198 秒，
  不含全部进程初始化及最后输出根目录权重保存，不能直接据此保证正式训练用时。
- checkpoint-100 的三片权重索引、头部及文件长度一致，共 1,030 个 tensor；
  optimizer、scheduler、RNG、trainer_state、statistics 和 processor 配套文件均写出。
- 实际训练处理器、保存的 processor 和重新加载后的 StateActionProcessor 均为
  `use_percentiles=false`；重载后的实际 state min/max 与本次 400 条训练统计量一致。
- 训练数据的文件清单和字节数检查仍通过；未改动原始/转换数据或已有正式模型。
- 临时 smoke 大权重和 optimizer 已清理，恢复磁盘空间；日志保留，小型配置/状态
  归档为 `archives/zhewan_smoke_100_minmax_pass_metadata_20260910.tgz`。

本机证据在 `agibot/local_reports/zhewan_prepare_20260910/cloud/`：
`smoke_minmax.log`、`smoke_minmax_checkpoint_metadata/`、`smoke_minmax_processor_reload.log`。
此前未修复版本的 `smoke_checkpoint_metadata/` 只作问题记录，不作为正确配置示例。

后续正式 checkpoint 可用同一检查器验证保存/重载，**不传归一化覆盖值**，检查权重包自己
保存的实际配置（该命令仅重载处理器，不加载模型张量、不做推理）：

```bash
.venv/bin/python agibot/training/check_saved_normalization.py \
  --model /path/to/task/checkpoint-30000 \
  --dataset /root/gpufree-data/GR00T/datasets/zhewan_right_place_r0003_400/train
```

短测通过表示当前数据→训练→checkpoint→处理器重载链路可运行，不代表任务成功率或
泛化已经验证。正式 30k 从原始 GR00T 基座开始，不使用临时 smoke 权重。

## 正式训练启动记录

2026-09-10 10:31:27 +08:00，通过 `train.sh baseline` 从原始基座启动，未从 smoke 续训。
tmux 会话：`groot_zhewan_r0003_30k`。训练在服务器上独立运行，断开 SSH 或关闭本机终端
不会停止训练；训练进行中不能关闭云端实例或切回无卡模式。目前训练已完成，已切回无卡模式。

```bash
ssh -p 30263 root@120.209.70.195
tmux attach -t groot_zhewan_r0003_30k
```

退出查看但保留训练：按 `Ctrl+B`，松开，再按 `D`。不要按 `Ctrl+C`。
会话保留结束后的输出，并打印 `TRAIN_EXIT_CODE`。日志为
`/root/gpufree-data/GR00T/logs/zhewan_rplace_r0003_n1d7_e400_f10_h16_1xa10080_s30k_ckpt5k_v1.baseline.log`。
此节保留启动时快照；完成结果见下节。

启动后已观察到至少 24 step：step 10/20 的区间平均 loss 为 1.3414 / 1.3231，
梯度有限、warmup 学习率正常，显存采样 40,289 MiB；实际 processor 保存为 min/max。
此处是启动时快照，不会自动随训练刷新。

## 正式训练完成与模型下载

2026-09-10 已确认 `global_step=max_steps=30000`、`TRAIN_EXIT_CODE=0`，保留
checkpoint-25000 和 checkpoint-30000。Trainer 计时约 6 小时 57 分钟，全程平均 loss
0.0587804，最后一次 10-step 平均 0.0339，最后 1,000 step 平均 0.034417。
记录的 loss 和梯度均为有限值。

| step 区间 | 平均训练 loss |
|---|---:|
| 1–5,000 | 0.1288662 |
| 5,001–10,000 | 0.0583276 |
| 10,001–15,000 | 0.0495486 |
| 15,001–20,000 | 0.0430776 |
| 20,001–25,000 | 0.0378936 |
| 25,001–30,000 | 0.0349750 |

最后 5k 较前一个 5k 下降约 7.7%。这是训练损失，不是留出成功率或真机效果。
30k 保存的模型/processor 均为本任务 min/max（`use_percentiles=false`）。

无卡模式下于当天启动本机后台下载：
`agibot/models/zhewan_rplace_r0003_n1d7_checkpoint-30000/`，推理入口为其中 `model/`。
只拉取推理权重及小型 metadata，复用本机 Cosmos 骨干副本，不下载 optimizer。
下载于 18:44 完成、退出码 0；包内 `inference_bundle_manifest.json` 状态为
`DOWNLOAD_COMPLETE_PROCESSOR_RELOAD_PASS`。后续本机前向及留出结果见 [离线测试](OFFLINE_TEST.md)。

本机查看进度：`tmux attach -t groot_zhewan_download_30k`；日志和恢复脚本位于
`agibot/local_reports/zhewan_prepare_20260910/download/`。该目录中的 `trainer_state.json`
是从正式 checkpoint-30000 取回的 loss 原始记录。下载完成当时保留了云端模型；后续清理见下文。

## 2026-09-20 云端旧数据与权重清理

按用户要求，为新的 `xichong_right_grasp_r0002` 任务腾出空间，已删除云端本任务
`datasets/zhewan_right_place_r0003_400/`、`checkpoint-25000/`、`checkpoint-30000/`
以及输出根目录三个重复 safetensors 文件。释放约 61.2 GiB，数据盘可用约 108.6 GiB。

清理前的小型元信息（含 loss / Trainer 记录、配置、processor、数据 meta）保存在
`/root/gpufree-data/GR00T/archives/zhewan_r0003_before_cleanup_20260920.tgz`。
本机原始/转换数据及 30k 推理模型未删除，GR00T/Cosmos 基座和训练环境也未修改。
旧云端输出目录现在只有小型配置文件，不能再作为模型加载目录。
优化器状态没有另行备份，因此不能从本机推理包恢复已删除的完整续训 checkpoint。
