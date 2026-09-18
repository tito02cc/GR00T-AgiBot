# Agibot G2 右臂 GR00T N1.7 端到端 Runbook

> 本文记录 `xichong_right_single_grasp` 的完整实测流程。新项目从
> [`../../templates/task_profile/`](../../templates/task_profile/) 创建配置，并按照
> [`../../docs/TASK_ADAPTATION.md`](../../docs/TASK_ADAPTATION.md) 填写任务、数据、训练和
> GDK 参数。

> 用途：团队复现“原始数据清洗 → GR00T 数据转换 → 单卡 A100 微调 → 离线验证 → G2 真机完整推理”。
> 当前基线：右臂单工件抓取，300 条训练轨迹，GR00T-N1.7-3B，30k checkpoint。
> 验证状态：数据、训练、离线推理、真机完整抓取和复位均已实际通过。
> 最后核对：2026-08-31；官方基线提交：`51d4c89`。
> 维护原则：本文是唯一运行手册，不再维护单独的 “V1_FREEZE” 文档。

团队首次使用请先阅读 [`README.md`](README.md)，并优先通过
`agibot/bin/groot-g2` 运行已支持流程。本文保留完整证据、参数和问题复盘；机器人差异必须
记录到独立 `configs/local/<robot-id>.toml`，供目标机器人部署使用。

本文只描述当前可复现路径。历史试验、失败方案和修复原因集中在第 10 节，不能把其中的
H1/H2/H4、分段接近或夹爪门控脚本当成正式推理流程。

## 1. 当前结论和适用边界

### 1.1 已完成

| 环节 | 已验证结果 |
|---|---|
| 原始数据 | 636/636 episode 结构可用；63,470 帧；两路训练图像 126,940 张均可解码 |
| GR00T 转换 | 636 episode、63,470 Parquet 行、1,272 个 H.264 MP4，帧/状态/动作一一对应 |
| 训练集 | 从 576 条严格同步候选中分层 + 轨迹多样性选出 300 条，共 29,052 帧 |
| 数据门禁 | 300/300 语义通过；300/300 AI 视觉审核通过；processor 与 G2 shadow 门禁通过 |
| 训练 | 单卡 A100 80 GB 完成 30,000 steps，约 6 h 43 min，无 OOM/NaN |
| 模型 | 30k 推理包完整，3,144,016,000 个 BF16 参数，云端与本地 SHA256 一致 |
| 离线评估 | train/held-out 多 seed 完成；服务、动作 shape、协议和安全故障注入通过 |
| 真机 | 两次完整 chunk 闭环抓取均 PASS；随后张开夹爪并精确回到训练起点 |
| 自动测试 | 当前数据/配置/桥接核心测试 50/50 通过 |

### 1.2 不能过度声明

- 两次真机成功不是统计意义上的成功率；团队正式验收仍应连续做更多 trials。
- 训练起始位姿几乎固定，当前模型只具备同分布泛化能力。底盘、架子或工件大幅换位时，
  应补采有系统位姿扰动的数据并重新训练。
- 当前基线按模型原始动作执行。第二次成功末段仍出现约 49.326 mm/100 ms、
  5.120°/100 ms 的动作尖峰；这是已知模型输出特征，不应写成“已解决”。
- 接触检测目前未启用，报告中的 `contact_latched=false` 不等于没有抓住物体。
  实物是否被夹住必须由现场人员和图像确认。

## 2. 唯一协议：任何一项改变都必须重做门禁

| 项目 | 固定值 |
|---|---|
| Embodiment | `NEW_EMBODIMENT` |
| 机器人自由度 | 本案例使用右臂 EEF + 右夹爪 |
| 图像键 | `head_color`、`hand_right` |
| 图像格式 | RGB `uint8`，640×480 |
| 数据/策略频率 | 10 Hz |
| G2 底层插值 | 每个 10 Hz action 用 5 个 50 Hz 控制 tick 跟踪 |
| state.right_eef | 9D：XYZ + Rot6D |
| state.right_gripper | 1D |
| action.right_eef | `RELATIVE + EEF + XYZ_ROT6D`，16-step chunk |
| action.right_gripper | `ABSOLUTE + NON_EEF + DEFAULT` |
| 数据坐标标签 | `base_link_tf` |
| G2 实时坐标 | `base_link`，四元数顺序 XYZW |
| Rot6D 顺序 | `r00 r01 r02 r10 r11 r12`，即旋转矩阵前两行 |
| 夹爪训练/命令范围 | `[-0.785, 0]`；`-0.785` 张开，`0` 闭合 |
| G2 原始反馈 | 约 `0` 张开，约 `120` 闭合 |
| 语言键 | `annotation.human.task_description` |

固定 prompt：

```text
starting with the right gripper open and the left arm and left gripper stationary, grasp the metal_workpiece with the right gripper, lift it at least 3 cm clear of the pickup position, and hold the right arm stable
```

训练起始 EEF 参考位姿（XYZ + XYZW）：

```text
[0.5081030780841185, -0.25893071977192195, 1.04842646506165,
 0.6620299522145109, -0.020616324660443348,
 0.7490960119476248, 0.01210266138135472]
```

原始采集动作的语义是“基座坐标系下的下一步增量”：

```text
target_xyz = current_xyz + delta_xyz_world
R_target   = Exp(delta_rotvec_world) @ R_current
gripper    = next absolute gripper position
```

转换器必须先重建绝对目标 EEF；GR00T processor 再相对当前 EEF 生成训练 action。
禁止把原始 delta 直接声明为 GR00T `RELATIVE` action，否则会发生二次做差。

Rot6D 不是四元数的替代传感器格式，而是模型训练中的连续旋转表示。数据源和 G2 bridge
仍用 XYZW 四元数，转换时执行 `quaternion ↔ rotation matrix ↔ Rot6D`。已测得 processor
往返最大误差约 `5.14e-8`，姿态重建最大误差约 `5.91e-6°`。

## 3. 目录、版本和产物

本地仓库根目录：

```text
/path/to/Isaac-GR00T
```

团队主要入口：

```text
agibot/
├── configs/xichong_right_single_grasp_config.py
├── data/xichong_right_single_grasp/
├── gr00t_data/
│   ├── xichong_right_single_grasp/
│   └── xichong_right_single_grasp_300/
├── models/xichong_rgrasp_n1d7_checkpoint-30000/model/
├── reports/
├── reviews/
├── robot/
├── scripts/
│   ├── convert_xichong_right_single_grasp.py
│   ├── select_xichong_training_subset.py
│   ├── validate_xichong_gr00t_training_ready.py
│   ├── run_g2_groot_full_protected_inference.py
│   └── recover_g2_groot_training_start.py
├── tools/g2_gr00t_shadow_adapter.py
└── training/
    ├── preflight_xichong_1xa10080.sh
    └── launch_xichong_1xa10080.sh
```

云端所有内容必须放在数据盘：

```text
/data/ct/
├── datasets/
├── models/
├── outputs/
├── logs/
├── cache/
└── scripts/
```

不要把模型、数据、checkpoint 或缓存放进系统盘。当前基线模型版本：

- GR00T-N1.7-3B revision：`2fc962b973bccdd5d8ce4f67cc63b264d6886495`
- Cosmos backbone revision：`9ce19a195e423419c349abfc86fd07178b230561`

账号、SSH 密码、Hugging Face token 不得写入仓库或本文。使用环境变量、SSH key 或交互式
登录；日志分享前也要检查是否含凭据。

## 4. 数据清洗、转换和 300 条训练集

以下命令都从仓库根目录运行。

### 4.1 创建本地环境

```bash
uv python install 3.12
uv sync
.venv/bin/python --version
```

必须是 Python 3.12。官方示例若还是 Git LFS 指针，先安装 `git-lfs` 并执行
`git lfs pull`；本任务自己的数据处理不依赖示例数据。

### 4.2 原始数据审计

原始目录：

```text
agibot/data/xichong_right_single_grasp
```

已验收结果：

- episode `000000–000635`，636/636 结构可用，0 条因结构错误删除；
- 共 63,470 帧；
- 用于训练的 `head_color` 和 `hand_right` 共 126,940 张 JPEG 全部解码；
- 历史目录 `.quality_quarantine` 中的 135 项不属于这 636 条编号集合；
- 60 条只因相机/状态时间偏移、晚帧或采集 warning 未进入严格同步候选池，
  原数据未删除。

权威报告：

```text
agibot/reports/xichong_right_single_grasp_validation.json
```

### 4.3 转成 GR00T/LeRobot 数据

```bash
uv run --no-sync python agibot/scripts/convert_xichong_right_single_grasp.py \
  --workers 8 \
  --quarantine-invalid
```

转换结果：

- 636 episodes；
- 63,470 Parquet rows；
- 1,272 个 H.264 MP4（两路相机）；
- 640×480、YUV420P、10 fps；
- 大小约 2 GB。

原始数据约 29 GB、转换后约 2 GB 是预期现象：转换只保留两路训练图像，丢弃未使用的
左腕图像，并将逐帧 JPEG 编成 H.264。它不是数据缺失。全部视频已完整解码并核对帧对应，
最低 PSNR 为 38.08 dB。该有损编码是 LeRobot/GR00T 常规视频存储方式，在本基线门禁中
没有发现会阻断训练的画质问题。

### 4.4 选择 300 条

选择器先使用 12 个时间分层，再在归一化轨迹特征上做 k-center，以避免只取连续、重复样本：

```bash
uv run --no-sync python agibot/scripts/select_xichong_training_subset.py
```

结果目录 `agibot/gr00t_data/xichong_right_single_grasp_300`：

- 300 episodes、29,052 frames；
- 300 Parquet、600 MP4、共 907 个文件；
- 逻辑大小约 906 MB；
- 本地使用 hardlink 时 `du` 看起来可能更小，这是复用 inode，不是缺文件。

选择依据：

```text
agibot/reports/xichong_right_single_grasp_selection_300.json
```

### 4.5 统计量和训练硬门禁

```bash
uv run --no-sync python gr00t/data/stats.py \
  --dataset-path agibot/gr00t_data/xichong_right_single_grasp_300 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path agibot/configs/xichong_right_single_grasp_config.py

uv run --no-sync python agibot/scripts/validate_xichong_gr00t_training_ready.py \
  --workers 8

.venv/bin/python agibot/scripts/validate_g2_gr00t_shadow_adapter.py
```

开始训练前必须同时满足：

- 数据结构、视频解码、帧数和时间戳通过；
- `meta/stats.json` 与 `meta/relative_stats.json` 已生成；
- 300/300 轨迹任务语义通过：最小抬升 31.110 mm，末段最短连续闭合 17 帧，
  最大稳定半径 3.434 mm；
- AI 视觉审核 300/300 通过；3 条保守候选各加密检查 48 帧后通过；
- processor gate 为 300 episodes、29,052 frames、24,552 windows；
- G2 shadow action shape、坐标和夹爪映射通过；
- SHA256 清单通过。

权威产物：

```text
agibot/reports/xichong_right_single_grasp_300_hard_gate.json
agibot/reports/xichong_right_single_grasp_300_visual_audit.json
agibot/reports/xichong_right_single_grasp_300_g2_shadow_gate.json
agibot/reports/xichong_right_single_grasp_300.sha256
agibot/reviews/xichong_right_single_grasp_300_reviews.json
```

任何一条门禁失败都不得启动付费训练。若修改图像、频率、EEF 表示、夹爪语义、prompt 或
episode 集合，必须重新生成 stats、门禁报告和 SHA 清单。

## 5. 单卡 A100 80 GB 训练

### 5.1 已验证环境

| 项目 | 基线 |
|---|---|
| GPU | 1×A100 80 GB |
| Python | 3.12.13 |
| uv | 0.12.2 |
| PyTorch | 2.9.0+cu128 |
| cuDNN | 9.10.2（venv 中版本优先） |

服务器目录对应关系：

```text
/data/ct/models/GR00T-N1.7-3B
/data/ct/models/Cosmos-Reason2-2B
/data/ct/datasets/xichong_right_single_grasp_300
/data/ct/scripts/xichong_right_single_grasp_config.py
```

### 5.2 固定训练配置

| 参数 | 值 |
|---|---|
| GPU 数 | 1 |
| per-device batch | 16 |
| gradient accumulation | 2 |
| effective batch | 32 |
| workers | 4 |
| learning rate | `1e-4` |
| weight decay | `1e-5` |
| warmup ratio | `0.05` |
| state dropout | `0.2` |
| episode sampling | `0.1` |
| augmentation | color jitter |
| 训练模块 | projector + diffusion |
| 冻结模块 | LLM + visual backbone |
| max steps | 30,000 |
| log interval | 10 |
| save interval | 5,000 |
| retained checkpoints | 最新 2 个 |

虽然每 5k 保存一次，保留策略只留下最后两个 checkpoint，兼顾对照和数据盘空间。

### 5.3 模型下载

先在 Hugging Face 对 GR00T 和所需 backbone 完成 access。下载脚本从临时环境变量读取 token，
不会将其写入文件；交互输入可避免 token 进入 shell history：

```bash
cd /data/ct/Isaac-GR00T
read -rsp 'Hugging Face token: ' HF_TOKEN
export HF_TOKEN
echo
bash agibot/training/download_gr00t_n1d7_models.sh
unset HF_TOKEN
```

下载必须落到 `/data/ct/models`。不要把 token 放进命令历史、脚本或文档。

### 5.4 上传和完整性验证

推荐从本地用 `rsync` 上传 300 条数据、配置和训练脚本；服务器端再核对 907 个文件的
SHA256。不能仅凭目录大小判断上传完成。

### 5.5 启动顺序

服务器仓库/脚本实际部署在 `/data/ct/scripts` 时：

```bash
cd /data/ct

bash scripts/training/preflight_xichong_1xa10080.sh
bash scripts/training/launch_xichong_1xa10080.sh audit
bash scripts/training/launch_xichong_1xa10080.sh smoke
bash scripts/training/launch_xichong_1xa10080.sh baseline
```

只有 preflight、audit、100-step smoke 都 PASS 才运行 baseline。中断后使用 launcher 的
`resume` 模式，不要手工拼接一套新参数。

实时观察：

```bash
watch -n 2 nvidia-smi
tail -f /data/ct/logs/<run-name>.log
watch -n 10 'du -sh /data/ct/outputs/*; df -h /data'
```

基线结果：

```text
run id: xichong_rgrasp_n1d7_e300_f10_h16_1xa10080_s30k_ckpt5k_v1
steps: 30000
elapsed: 24206.8432 s（约 6:43:26）
throughput: 1.239 step/s
reported train_loss: 0.0598410
final 10-step mean: 0.0345
last 500-step mean: 0.033990
retained: checkpoint-25000, checkpoint-30000
```

训练完成报告：

```text
agibot/reports/xichong_rgrasp_n1d7_e300_f10_h16_1xa10080_s30k_ckpt5k_v1_training_summary.json
```

已解决的服务器环境坑：

- 模型配置曾尝试在线访问 gated Cosmos：固定本地 backbone 路径并启用 local-files-only；
- 系统 cuDNN 9.1 曾覆盖 venv 的 9.10.2：确保 venv 的 CUDA/cuDNN library 路径优先。

## 6. checkpoint 打包和离线验证

正式推理包：

```text
agibot/models/xichong_rgrasp_n1d7_checkpoint-30000/model
```

验收结果：18 个文件，17,469,635,129 bytes，3 个权重 shard，
3,144,016,000 个 BF16 参数；服务器与本地文件 SHA256 一致。

离线评估包括：

- 5 条训练轨迹 × 3 seeds；
- 12 条真正 held-out 轨迹 × 3 seeds；
- ReplayPolicy 与源 Parquet 完全一致；
- held-out 模型服务 150 次调用，0 transport error、0 safety reject；
- 暖态调用 latency mean/P90 为 0.273/0.289 s；
- 故障注入、动作维度和非有限值拒绝通过。

H8 离线诊断曾把 held-out XYZ mean/P90 从 23.00/68.45 mm 改善到
9.99/22.98 mm，但它只说明长 chunk 的误差累积现象。当前真机基线已经按官方完整 H16
循环成功，正式运行不得因此改回 H8 或自制 receding-horizon 分段脚本。

位置覆盖报告：

```text
agibot/reports/xichong_generalization_coverage.json
```

300 条训练起点 q01–q99 跨度仅约 0.1536 mm（x）、0.00265 mm（y）、0.1296 mm（z）；
闭合位置覆盖约 72.57×52.70×39.04 mm。因此只能声明
`IN_DISTRIBUTION_GENERALIZATION_ONLY`。桥接层不得用固定 XYZ 编程替代模型，但现场初始
构型、底盘和架子应保持在训练分布附近。

## 7. G2 真机部署和完整推理

### 7.1 正式执行语义

官方参考：

- `gr00t/eval/real_robot/SO100/eval_so100.py`：获取 action 后顺序执行整个 chunk；
- `gr00t/eval/_horizon_contract.py`：默认 `n_action_steps == action_horizon`。

本项目对应流程：

```text
读取最新双相机 + 右臂状态
        ↓
GR00T 输出 16×8（EEF 7 + gripper 1）
        ↓
按原顺序执行全部 16 个 10 Hz action
        ↓
每个 action 由 G2 以 5 个 50 Hz tick 插值跟踪
        ↓
重新观测，再请求下一个完整 chunk
        ↓
检测到闭合目标并完成 ≥3 cm 抬升后结束，保持当前姿态
```

“完整推理”不表示模型一次生成整条 episode；GR00T 一次输出 16 步（1.6 s）动作块，
完整执行后根据新图像继续闭环。这正是当前成功验证的官方模式。

正式 runner：

```text
agibot/scripts/run_g2_groot_full_protected_inference.py
```

它执行官方 decoder 的夹爪绝对值，不做阈值替换、单调闭合 latch、强制张开、模型动作
裁剪或平滑。仅验证 action 必须为 `16×8` 且全部 finite。G2 action bridge 仍保留与模型
无关的硬件合理性检查，不能删除。

### 7.2 角色和端口

| 进程 | 位置 | 端口 |
|---|---|---|
| G2 observation bridge | 机器人 | `127.0.0.1:9100` |
| G2 action bridge | 机器人 | `127.0.0.1:9200` |
| SSH 转发 | 推理机 | `19100→9100`、`19200→9200` |
| GR00T inference server | 推理机 | `127.0.0.1:5564` |
| full inference runner | 推理机 | 连接上述三个服务 |

机器人端目录：

```text
/home/agi/vla_ct/groot_right_arm_shadow
```

一次性部署时至少同步：

- `g2_groot_right_observation_bridge.py`
- `g2_groot_persistent_h1_action_bridge.py`
- `g2_groot_persistent_right_arm_controller.py`
- `g2_groot_gripper_runtime.py`
- `g2_groot_gripper_state_machine.py`
- `recover_g2_groot_pre_wbc_pose.py`
- `g2_groot_trajectory_tracking_probe.py`

### 7.3 开始前现场检查

必须有现场人员完成：

- 清空机器人工作空间并确认急停可用；
- 只启用右臂，左臂和左夹爪保持静止；
- 右夹爪张开；
- 机器人 EEF 接近第 2 节训练起始位姿；
- 头部和右腕图像清晰，工件处于训练分布附近；
- 底盘、架子和工件没有发生训练集未覆盖的大幅换位；
- 网络稳定；此前出现过 70% 丢包和 409–724 ms RTT，若复现必须先停。

### 7.4 启动机器人观察桥

在机器人终端 1：

```bash
cd /home/agi/vla_ct/groot_right_arm_shadow
source /home/agi/app/env.sh
python3 -u g2_groot_right_observation_bridge.py \
  --host 127.0.0.1 \
  --port 9100 \
  --jpeg-quality 92 \
  --camera-timeout-ms 500 \
  --max-camera-skew-ms 100 \
  --max-state-camera-skew-ms 50
```

### 7.5 启动机器人动作桥

在机器人终端 2：

```bash
cd /home/agi/vla_ct/groot_right_arm_shadow
source /home/agi/app/env.sh
python3 -u g2_groot_persistent_h1_action_bridge.py \
  --host 127.0.0.1 \
  --port 9200 \
  --enable-control \
  --enable-gripper \
  --enable-approach-session \
  --enable-protected-close \
  --confirm ENABLE_G2_GROOT_RIGHT_ARM_SINGLE_PROTECTED_CLOSE_SESSION \
  --session-limit-s 1800
```

当前 bridge 的硬件范围是宽松兜底，不是模型阶段门控：workspace min
`[0.448,-0.333,0.987]`，max `[0.754,-0.197,1.217]`；
最大 10 Hz target step 0.25 m/180°，最大命令年龄 5 s，session 总位移/旋转
0.24 m/20°。不要为追求“无任何限制”而删除这些非模型特定的最终硬件检查。

### 7.6 建立 SSH 转发

在推理机终端 1，用 SSH key 或交互输入密码；文档中不保存凭据：

```bash
ssh -N \
  -L 19100:127.0.0.1:9100 \
  -L 19200:127.0.0.1:9200 \
  <robot-user>@<robot-host>
```

### 7.7 启动 GR00T 服务

在推理机终端 2：

```bash
cd /path/to/Isaac-GR00T
.venv/bin/python -u gr00t/eval/run_gr00t_server.py \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path agibot/models/xichong_rgrasp_n1d7_checkpoint-30000/model \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 5564
```

### 7.8 运行完整真机推理

在推理机终端 3：

```bash
cd /path/to/Isaac-GR00T
.venv/bin/python -u agibot/scripts/run_g2_groot_full_protected_inference.py \
  --execute \
  --confirm EXECUTE_G2_GROOT_FULL_RIGHT_ARM_PROTECTED_INFERENCE \
  --max-cycles 0 \
  --report agibot/reports/g2_groot_team_trial_$(date +%Y%m%d_%H%M%S).json
```

`--max-cycles 0` 表示不人为截断模型循环，由任务完成条件结束。执行期间现场人员持续观察；
遇到碰撞风险、失联、非预期大幅跳变或相机冻结，立即急停/中止，而不是等软件自己恢复。

成功判定：

- 模型输出过闭合目标；
- 实际夹爪完成值达到当前闭合判据，或现场确认夹住；
- EEF 相对 pickup 至少抬升 30 mm；
- runner 报告 `PASS`；
- 完成后 action bridge 保持当前抓取姿态，不自动张开。

### 7.9 张开夹爪并回训练起点

确认可以释放物体后，在推理机运行：

```bash
cd /path/to/Isaac-GR00T
.venv/bin/python -u agibot/scripts/recover_g2_groot_training_start.py \
  --execute \
  --expanded-session \
  --confirm RECOVER_G2_GROOT_RIGHT_ARM_FROM_APPROACH_TO_TRAINING_START \
  --port 19200 \
  --report agibot/reports/g2_groot_team_recovery_$(date +%Y%m%d_%H%M%S).json
```

恢复脚本只发一次张开命令，然后用最大约 5 mm/1° 的 waypoint 回起点；最终容差
0.5 mm/0.002 rad，最后向 action bridge 发送 shutdown 并释放控制。复位完成后再依次
Ctrl-C 停止模型服务、观察桥和 SSH tunnel。

## 8. 已完成的真机验收记录

### Trial 1

```text
报告: agibot/reports/g2_groot_complete_grasp_no_auto_open_20260827.json
结果: PASS
完整 chunks: 2
抬升: 75.286 mm
最终夹爪模型值: -0.044658
```

随后复位：

```text
agibot/reports/g2_groot_post_success_release_reset_20260827.json
EEF 终点误差: 0.0718 mm / 0.00710°
夹爪: 已张开
```

### Trial 2

```text
报告: agibot/reports/g2_groot_complete_grasp_run2_20260827.json
结果: PASS
完整 chunks/actions: 4 / 64
抬升: 110.104 mm
最终夹爪模型值: -0.006892
G2 raw feedback: 111.627
夹爪状态: 2
实际夹爪命令数: 24
现场结论: 完整抓取顺利
```

随后复位：

```text
agibot/reports/g2_groot_post_run2_release_reset_20260827.json
waypoints: 24
EEF 终点误差: 0.0663 mm / 0.0120°
夹爪: 仅一次张开命令
GDK: mode=5, error=0
控制权: 已释放
```

## 9. 每轮团队执行清单

### 数据/训练负责人

- [ ] 协议表未改变，或已为变更建立新实验版本。
- [ ] 原始数据报告、300 条选择报告、视觉审核、hard gate、shadow gate 全部 PASS。
- [ ] 数据和模型 SHA256 在目标服务器复核。
- [ ] 付费开卡前 preflight 和 audit 通过。
- [ ] 100-step smoke 无 OOM、NaN、数据加载异常。
- [ ] 正式训练日志、loss、checkpoint 和磁盘余量持续记录。
- [ ] 30k 包完成参数数、shard 数、大小和 SHA 核验。

### 推理负责人

- [ ] 使用本节指定的 30k 模型和正式 full runner。
- [ ] 不使用旧分段、H1/H2/H4、强制开爪或夹爪 latch 脚本。
- [ ] 机器人已回训练起点，右夹爪张开，图像和物体位置合理。
- [ ] observation/action bridge、SSH 转发和模型服务均为独立持久进程。
- [ ] action 是 `16×8` finite；每轮完整执行 16 步。
- [ ] 现场人员持有急停并观察完整过程。
- [ ] 每次生成唯一 inference report。
- [ ] 成功后先保持，再由现场确认执行 release/recovery。
- [ ] recovery 报告确认夹爪张开、EEF 回起点、控制释放。

## 10. 问题复盘：哪些方案失败、为什么

### 10.1 分段小范围执行让动作提前回撤

早期 H1/H2/H4、stage 上限和 receding-approach 验证会在完整策略行为中间重新观测，
改变模型闭环上下文，表现为尚未到闭合位置就回撤。参考官方真机脚本后，改成每次完整执行
16-step chunk，抓取流程随即稳定。正式流程因此只保留第 7 节的 full runner。

### 10.2 夹爪不闭合、闭合后又张开

根因不是模型不会闭合，而是桥接曾加入阈值替换、强制张开、单调 latch 或反馈状态误读。
修复后使用官方 decoder 输出的绝对夹爪值，按训练语义
`-0.785=open, 0=closed` 原样下发；不再用手写阈值改变策略。

### 10.3 重复夹爪命令

Trial 1 复位曾出现 `command_count=585`，因为状态 2 被当成“尚未完成”而反复下发。
修复为消费 active target 后，Trial 2 完整推理只有 24 个不同夹爪目标，复位只发一次 open。

### 10.4 GDK status 3、UUID mismatch 和 MotionPlan 误差

- 零错误码下的短暂 status 3 是接触后保持状态；当前实现允许它，非零错误仍 fatal。
- `MotionPlanResponse uuid mismatch` 与约 0.218 mm 跟踪误差通过单一、持久 GDK owner
  和 50 Hz adaptive WBC 控制路径处理，避免多个会话争夺响应。
- 恢复阶段若连接 reset，先只读确认 bridge 是否退出，再重启 action bridge，不要盲目重发。

### 10.5 恢复脚本命令写错

历史两次失败只发生在 argparse，未发送机器人命令：一次误用 `--action-port`，一次
confirmation 与 `--expanded-session` 不匹配。第 7.9 节命令是唯一已验证形式。

### 10.6 网络抖动

曾测得 70% 丢包、RTT 409–724 ms。网络恢复后使用持久连接、JPEG quality 92 并重新做
warm shadow 才继续真机。若网络再次达到这一量级，停止推理；提高动作频率不能修复丢包。

### 10.7 末段抖动

Trial 2 最后一个 chunk 存在最大约 49.326 mm/100 ms、5.120°/100 ms 尖峰。当前 V1 为了
保持“训练数据/模型输出是什么就执行什么”，没有加入模型特定平滑。若要改进，作为 V2
单独评估：优先补采稳定收回段或重新训练；若加入滤波，必须重做离线、shadow 和真机验收，
不能悄悄修改当前基线。

## 11. 代码完整性和回归测试

当前正式核心文件 SHA256：

```text
cd14a4ec901d589f257eebae6a1e26704a26b68631b0f502de0bfd84d3a1bbea  agibot/scripts/run_g2_groot_full_protected_inference.py
bf1eb9d7db153a186b292520bcdb3506a4b73b40e337128a5f73a7ab677022df  agibot/scripts/recover_g2_groot_training_start.py
58e7aec4b0cf7b0c2301198ac313a8032a1379fa11f2229dd6830184e93e7e17  agibot/tools/g2_gr00t_shadow_adapter.py
9346b6c715ff78bc5c13c16dd9862365dada72e8ea58a76942f01a8c9932b7c2  agibot/robot/g2_groot_h1_bridge_client.py
3b42abc31e20abf292a6d4bb8dd4194e0ed708b7be73f12d4e52732b57cb2fe2  agibot/robot/g2_groot_persistent_right_arm_controller.py
b589539b9f3a1a2d3ab846f0682bfc6a16d2cee866d445c4899e113aadc1991a  agibot/robot/g2_groot_persistent_h1_action_bridge.py
55ff4b8607edb7a17fd8b84b2b098ad52e005d6bbf520adf957c2147f668d336  agibot/robot/g2_groot_gripper_runtime.py
45bbf856c2b6a4d644b5bebd583060abf85d6820f14a5d32d77527baf32c0f56  agibot/robot/g2_groot_gripper_state_machine.py
```

修改桥接或推理代码后运行：

```bash
cd /path/to/Isaac-GR00T

PYTHONPATH=agibot/robot .venv/bin/python -m unittest -q \
  agibot/robot/test_g2_groot_gripper_state_machine.py \
  agibot/robot/test_g2_groot_gripper_runtime.py \
  agibot/scripts/test_g2_groot_full_protected_inference.py \
  agibot/scripts/test_xichong_policy_client.py \
  agibot/scripts/test_team_cli.py
```

若测试文件名随仓库演进，以 `rg --files agibot | rg 'test_.*\.py$'` 找到相关用例并全量运行。
当前已记录结果为 50/50 PASS。

## 12. 团队交付与变更规则

每个新实验必须留存：

- 原始数据来源、采集频率、相机、坐标系和夹爪定义；
- episode 清单、筛选理由、统计量、门禁 JSON 和 SHA256；
- 模型/代码 revision、完整训练参数、硬件环境和日志；
- checkpoint 完整性与离线评估；
- 每次真机 inference/recovery 报告；
- 现场异常、是否急停、实物是否成功抓取；
- 与本基线的明确差异。

以下改变必须升级实验版本，不能覆盖当前基线：

- 修改相机、分辨率、频率、坐标系、Rot6D 顺序或夹爪正负方向；
- 从右臂扩为双臂；
- 更换数据 episode 或统计量；
- 更换基础模型、checkpoint、action horizon 或 prompt；
- 加入动作裁剪、平滑、门控或夹爪状态机；
- 改变底盘/架子/工件位置分布并声称跨位置泛化。

团队 Skill 化时，以本文为流程来源，将固定命令做成参数化脚本；Skill 不应复制一份会独立
漂移的结论。任何自动化仍须保留“数据硬门禁 → smoke → 正式训练 → offline/shadow →
现场确认 → 完整真机推理 → recovery”的顺序。
