# 10.20.15.194：xichong_right_grasp r0002

这是新的右臂抓取任务，不是之前的 `xichong_right_single_grasp` 或右臂放置任务。
本目录只记录本批数据，旧任务的筛选名单、运动阈值、夹爪判定和初始位姿均不直接沿用。

## 来源

- 机器人：10.20.15.194，采集记录 robot_id 为 `G2A0104C300179`。
- 原始路径：`/data/agi/g2_datasets/g2_pi05_g2a0104c300179_operator_v1/xichong_right_grasp/right_grasp_lift/revision_0002`。
- 2026-09-20 首次盘点发现 851 条 episode，采集于 2026-09-18、2026-09-19。
- 本地原始数据路径：`agibot/data/g2_194/xichong_right_grasp_r0002_job01/`。
- 原始 episode 包含三路图像、JSONL、NPZ、采集参数和收据，拉取时保持完整，不重新编码。

## 当前处理范围

`inventory_raw.py` 是只读盘点工具：检查文件、数组形状、数值有限性、帧序号、时间递增、
图像索引和文件存在性，并汇总本批轨迹分布及采集端标签。它不套用历史任务阈值，
不筛选固定条数，不删除轨迹，不修改机器人状态。

采集端标记成功、图片文件存在，均不等于独立确认了物理抓取成功。
本轮没有完成逐条视觉任务审核；自动检查的范围、转换状态和固定名单见下文。

## 先筛再传（2026-09-20）

最初的全量传输已停止，后续只传用户选定的名单；源数据没有删除或修改。

- 基础盘点：851 条、76,560 个保存帧、39,079,006,808 字节。
- `screen_numeric.py`：851 条均通过 NPZ/JSONL 一致性、动作重建和本批采集合同的数值复核。
  本批合同的抓取抬升要求为 0.1 m、末段稳定半径上限为 0.01 m；直接读取本批记录，
  不是从历史任务配置复制。通过这些检查不能单独证明抓到了工件。
- `decode_g2_prescreen_images.py`：已在机器人端完整解码 head_color 与 hand_right，851 条全部通过。
- `compare_subsets.py`：图像解码完成后比较不同样本量的轨迹覆盖和传输体积；
  按完整采集会话隔离留出集，特征缩放只从本批训练候选池估计。不自动拍板训练条数。
- `pull_raw.py`：后续必须显式指定 `--episode-list`；仅在明确要求全量时才传 `--all-episodes`。

本地报告：`agibot/local_reports/g2_194/xichong_right_grasp_r0002_20260920/`。
机器人端报告：`/home/agi/vla_ct/data_preparation/g2_194/xichong_right_grasp_r0002_20260920/`。

## 后续数据处理

### 已确认的传输名单

用户已选择 400 条训练 + 91 条留出，共 491 条、21,896,397,988 字节原始文件。
名单固定为本目录的 `train.txt`、`heldout.txt`、`transfer.txt`。
训练数据包含 2026-09-18 的 227 条及 2026-09-19 的 173 条，覆盖 7 个采集会话。
留出集为第一段完整采集会话 `episode_000000`–`episode_000090`，与训练无重叠。
851 条的 153,120 张头部/右腕图像已全部解码通过。
额外抽看了 000091、000401、000607、000716 的起始头部图像，未据此宣称逐条物理抓取成功。

### 无损分包传输

`pack_selected.py` 仅对名单中的完整 episode 打包，约 512 MiB 原始文件一包，共 43 包。
使用 tar + zstd level 1，无损保留 JPEG、NPZ、JSONL 和所有采集参数，不改变图像质量或动作值。
`pull_packed.py` 按包续传和自动解包，保留已匹配源清单的本地完整文件；校验解压路径、文件大小、
每条 episode 文件数量及总字节数。训练原始数据仍落到上文规定路径。

2026-09-20 已完成 491 条完整 episode 的下载、解包以及文件路径、数量和大小核对，
共 21,896,397,988 字节原始文件。两端均为 Wi-Fi，传输采用单流无损分包。
完成记录以本地报告目录的 `packed_pull.status.json` 和 `packed_pull.log` 为准；
旧 `pull.status.json` 是已暂停的全量传输记录，不代表当前进度。
远端分包位于 `/data/agi/groot_transfer/xichong_right_grasp_r0002_400_91_20260920/`，
本地缓存位于报告目录的 `transfer_cache/`。两端缓存暂时保留，不自动删除源数据。

## 本地清洗与 GR00T 转换

入口是本目录的 `prepare.py`，参数见 `prepare.json`，模型模态见 `modality.py`。
从 Isaac-GR00T 仓库根目录执行：

```bash
.venv/bin/python agibot/tasks/g2_194/xichong_right_grasp_r0002/prepare.py
```

输出为 `agibot/gr00t_data/g2_194/xichong_right_grasp_r0002_400/{train,heldout}/`。
训练集 400 条、35,108 帧；留出集 91 条、7,358 帧。每帧包含头部与右腕图像。
只有这两个相机进入本次模型；原始数据中的左腕图像继续保留，不删除。

处理过程与注意事项见 [DATA_PREPARATION.md](DATA_PREPARATION.md)。
报告目录为 `agibot/local_reports/g2_194/xichong_right_grasp_r0002_prepare_20260920/`，
`preparation_status.json` 为实际完成状态。转换完成后仍需通过全量视频/数值和官方加载验证，
只有最终状态 `AUTOMATED_DATA_CHECKS_PASS` 表示本轮自动数据检查完成。

2026-09-20 10:47 已完成，最终状态为 `AUTOMATED_DATA_CHECKS_PASS`。
491 个 Parquet、982 段视频的全量对应核对，以及官方 loader/processor 的 35,101 个完整 H16
窗口数学验证全部通过。转换数据约 2.426 GB（2.26 GiB，按文件逻辑字节统计）。
原始数据没有修改，未启动训练。保留 1 条采样延迟警告和少量留出集归一化越界诊断，详见处理记录。

仅重新验证已有输出（不会重新编码视频）：

```bash
.venv/bin/python agibot/tasks/g2_194/xichong_right_grasp_r0002/prepare.py --stage verify
```

脚本拒绝覆盖已有转换目录，不删除原始 episode，不修改固定名单。
若调整名单或编码配置，应使用新的输出和报告目录。不要同时运行多个准备进程。
自动数据检查不等同于物理抓取成功认证，也不是模型推理成功率评估。
正式训练的 GPU 配置、模型前向/反向试跑和实机部署不在本次数据转换步骤内。

## 云端训练准备

单卡 A100 80GB 配置为 `train_1xa10080.env`，入口为 `train.sh`，详见
[CLOUD_TRAINING.md](CLOUD_TRAINING.md)。上传工具为 `upload.sh`；CPU 部署检查为
`check_cloud_ready.py`。训练只使用本任务的 `train/`，不使用旧放置模型作为初始化。

当前服务器数据盘为 `/root/gpufree-data/GR00T`，新任务数据目标目录为
`datasets/xichong_right_grasp_r0002_400/`。本机上传日志位于处理报告目录的
`cloud/upload.log`。文件传输完成以 `UPLOAD_COMPLETE_INVENTORY_PASS` 为准，
云端 `cloud_ready.json` 通过表示 CPU 文件/配置/加载抽测完成，不代表 GPU smoke 已运行。
没有收到正式训练指令前，不自动启动 30k 训练。

2026-09-20 已完成本任务两组数据上传和云端 CPU 复核：1,494 个文件 / 2,425,879,677 字节，
状态为 `CPU_CONFIG_AND_INVENTORY_PASS_GPU_SMOKE_PENDING`。
实际参数为单卡 A100 80GB、batch16 × 累积2、30k step、每5k保存、保留两个、每10 step记录loss。
同日开卡后，单卡 A100 80GB 的 100-step GPU smoke 已通过，退出码 0：
前向/反向、loss/梯度有限性、checkpoint 保存及 processor 重载检查均通过；
loss 从前 10 步平均 1.2374 到最后 10 步平均 0.9409，抽测显存约 39.4 GiB。
日志/配置已归档，临时 smoke 大权重已清理。
2026-09-20 12:12（北京时间）已获指令并启动正式 30k 训练，云端 tmux 会话为
`groot_xichong_grasp_r0002_30k`。12:13:54 确认已进入第32步，首批 loss/梯度正常；
这只是启动记录，不是完成状态。正式训练从基座开始，不续接 smoke。
试跑结果与局限、空间安排详见 [CLOUD_TRAINING.md](CLOUD_TRAINING.md)。

## 30k 下载与离线测试

2026-09-20 正式30k训练完成，19:21正常退出；21:04本机推理包下载及processor重载检查通过。
模型入口：`agibot/models/xichong_rgrasp_r0002_n1d7_checkpoint-30000/model/`。
对外分发的 [30k 推理包](https://huggingface.co/Minth-Group/GR00T-G2-xichong-right-grasp-r0002-30k)
和 [400/91 条转换数据](https://huggingface.co/datasets/Minth-Group/GR00T-G2-xichong-right-grasp-r0002-data)
分别下载；原始 491 条 episode 未公开，重做清洗仍需内部取得。
本机实际GPU前向、8条训练样本和全部91条留出轨迹的H16开放环评估已完成，
无解码失败或原始夹爪数值越界，仍有闭合时机与长窗口预测误差警告。
详细指标、异常复测及复现命令见 [OFFLINE_TEST.md](OFFLINE_TEST.md)。未连接机器人或改变生产桥接。

## 194机器人推理入口

2026-09-21已建立本任务 `inference.json`、`inference.py`、`run_inference.py`，
机器人包装入口部署到 `/home/agi/vla_ct/tasks/g2_194/xichong_right_grasp_r0002/`。
复用此前成功的固定联合GDK控制器，完整H16闭环、无固定cycle/总时长上限、不启用RTC。
2026-09-24用户通过VR调整右臂后，完成7个H16动作块，现场确认抓取成功。
runner随后加入仅针对开爪低速微调的轻微平滑，并并行读取观测/状态；
闭爪时机、抬升轨迹仍由模型决定。从另一个VR选定的新起点再次完成6个H16动作块，
软件反馈闭爪并抬升，但6块均未触发平滑，物理抓取结果待现场确认；
块间等待也未观察到缩短。因此不能将本轮表现归功于平滑或并行读取。
后续软件记录已细分平滑跳过原因和观测链路耗时；本轮的38mm接近动作没有
被重新归类为微调，闭爪/抬升映射保持不变。已验证的历史回执不再随下一块
的状态读数重复传输；该传输优化及新增诊断尚未再次实机验证耗时收益。
任务版观测桥接还加入同像素JPEG Huffman优化；本机验证通过，但2026-09-24
SSH恢复后两个任务文件已同步至194机器人，只读观测验证通过。
用户确认的新起点比训练初始参考偏约12cm。2026-09-24从该起点执行了6个完整H16动作块：现场确认抓起工件，
但未进入夹爪卡槽。运行报告及桥接时序分析见 [BRIDGE_DEPLOYMENT.md](BRIDGE_DEPLOYMENT.md)；
本任务的反馈抬升完成状态不能代替入槽判定。
详情、报告和启动命令见 [BRIDGE_DEPLOYMENT.md](BRIDGE_DEPLOYMENT.md)。

随后两轮加入头部/右腕原始帧记录的完整实机测试：第一轮执行 6×H16，软件反馈闭爪并抬升，但现场判定距离不足、未抓到；第二轮执行 7×H16，现场确认抓起且进入夹爪卡槽。两轮均由模型输出接近、闭爪和抬升，未在运行中写死动作目标；逐轮准备时的人工复位/摆件不属于模型动作。对应 `inference.json`、逐帧录像和场景尺度分析的路径见 [BRIDGE_DEPLOYMENT.md](BRIDGE_DEPLOYMENT.md)。这是少量个例，不是稳定成功率，也不能把“闭爪＋抬升”的软件状态当作物理入槽认证。
