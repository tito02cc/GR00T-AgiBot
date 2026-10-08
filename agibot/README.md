# Agibot G2 × GR00T N1.7

G2 原始数采数据清洗、GR00T 格式转换、模型训练与真机推理操作说明。
任务参数和实验结果见各任务目录；10.20.15.194 机器人的运行入口见
[固定推理部署](deployments/g2_194/README.md)。以下命令均从仓库根目录执行。

项目各任务的处理结果、训练数值、桥接演进与实机复盘汇总见
[技术报告](docs/GR00T_TECHNICAL_REPORT.md)。

### 阅读顺序

- 首次使用：第 1–3 节选择任务、准备资源、安装环境，第 5 节配置路径。
- 重做数据和训练：第 4–6 节；新任务同时阅读右臂放置配置母本。
- 运行现有模型：第 2、3、5、7 节，不需要原始训练数据。
- 查文件与排错：第 8 节目录、第 9 节运行检查、第 10 节常见问题。

## 1. 任务

| 任务 | 原始目录（相对 data/） | 转换训练集（相对 gr00t_data/） | 模型包（相对 models/） |
|---|---|---|---|
| 早期右臂抓取 | `xichong_right_single_grasp/` | `xichong_right_single_grasp_300/` | `xichong_rgrasp_n1d7_checkpoint-30000/` |
| xichong 右臂放置 | `xichong_right_place_r0002_job01/` | `xichong_right_place_r0002_train_600/` | `xichong_rplace_r0002_n1d7_checkpoint-30000/` |
| zhewan 右臂放置 | `zhewan_right_place_r0003_job01/` | `zhewan_right_place_r0003_400/train/` | `zhewan_rplace_r0003_n1d7_checkpoint-30000/` |
| xichong 右臂抓取 r0002（10.20.15.194） | 见[任务目录](tasks/g2_194/xichong_right_grasp_r0002/README.md) | `g2_194/xichong_right_grasp_r0002_400/train/` | `xichong_rgrasp_r0002_n1d7_checkpoint-30000/` |

- **当前 10.20.15.194 机器人的两个放置任务：** 使用 [固定推理包](deployments/g2_194/README.md)。
  xichong 保留 `continuous_20260909`；zhewan 保留 `h16_collision_latched_20260911`。
  均有现场成功反馈，均不启用 RTC 或新增平滑滤波。
- **早期抓取：** 对应 10.20.15.60 机器人的历史实测，使用 [抓取案例](examples/xichong_right_single_grasp/README.md)
  和 [Runbook](examples/xichong_right_single_grasp/RUNBOOK.md)，不能拿抓取完成条件控制放置任务。
- **任务过程与问题记录：** [xichong 放置](examples/xichong_right_place_r0002/README.md)、
  [zhewan 放置](tasks/zhewan_right_place_r0003/README.md)、
  [10.20.15.194 机器人记录](docs/robots/10.20.15.194.md)。
- **实机演示：** [两个推理结果视频](inference_videos/README.md)。视频不是训练数据或成功率统计。

上述 IP 对应实测机器人。当前实现已验证右臂；左臂／双臂任务需适配数据字段、相机和 GDK 接口。

## 2. 数据与模型

### 2.1 代码依赖

`agibot/` 使用根目录的 `gr00t/`、`pyproject.toml` 和 `uv.lock`，需在完整仓库中运行。
本地训练和推理补丁见 [UPSTREAM_PATCHES](docs/UPSTREAM_PATCHES.md)，更新上游时需保留并复核这些改动。

### 2.2 各阶段所需资源

| 阶段 | 必需资源 | 无需准备 |
|---|---|---|
| 阅读流程／修改代码 | 仓库、文档、任务配置与名单 | 训练数据、checkpoint |
| 用现有模型推理 | 对应任务的完整推理模型包，以及推理环境和实际机器人 GDK | 原始数据、训练集、optimizer |
| 离线评估 | 对应模型包、相应转换 heldout 数据、评估环境 | 全部原始 episode |
| 从基础模型训练 | 转换后的 train 数据、相应 modality 与训练参数、GR00T 基座及 Cosmos 骨干 | 原始数据（不重做转换时）、旧任务微调模型 |
| 重做清洗与转换 | 原始完整 episode、任务名单与筛查／转换配置 | 旧微调模型 |
| 精确断点续训 | 完整训练 checkpoint 与原训练环境／配置，包括 optimizer 等状态 | 仅推理包不能代替 |

原始 episode 包含图像、数采元数据、状态、动作和时间戳，处理时需保持完整。
转换后的 LeRobot 数据集由 `data/`、`videos/`、`meta/` 等目录组成，三者配套使用。

### 2.3 数据集目录

- xichong 放置：train 为 `gr00t_data/xichong_right_place_r0002_train_600/`；
  heldout 为 `gr00t_data/xichong_right_place_r0002_heldout_100/`。
- zhewan 放置：`gr00t_data/zhewan_right_place_r0003_400/` 内分 `train/` 与 `heldout/`，
  分别 400／58 条。**训练路径指向 train 子目录**，不能把整个父目录或 heldout 当训练输入。
- xichong 抓取 r0002（10.20.15.194）：
  `gr00t_data/g2_194/xichong_right_grasp_r0002_400/` 内分 `train/` 与 `heldout/`，
  分别 400／91 条；当前本机模型入口为 `models/xichong_rgrasp_r0002_n1d7_checkpoint-30000/model/`。
  数据、训练和实机记录见[任务目录](tasks/g2_194/xichong_right_grasp_r0002/README.md)。
- 早期抓取：训练使用 `gr00t_data/xichong_right_single_grasp_300/`，不是把同名前缀下
  full、pre_hard_gate 等历史目录全部合并。原始 636 条中筛选 300 条用于训练。
- 推理模型包名称见第 1 节。每个包的 `model/` 是服务器加载入口，`backbone/` 是配套骨干。
- 从预训练基座开始微调需要原始 GR00T 基座；已有任务 checkpoint 用于该任务推理或续训。

### 2.4 本地文件

数据、权重和运行报告需单独准备，Git clone 不会包含被忽略的文件。
`local_reports/`、`reports/`、`reviews/`、`evaluations/`、`live_shadow/` 保存本地运行产物。
凭据通过环境变量或本地配置管理，不写入版本库；普通目录复制和压缩不会应用 `.gitignore`。

### 2.5 从 Hugging Face 下载

以下仓库公开可读，无需登录、配置 token 或申请读取权限。先按第 3 节安装环境，
再从代码仓库根目录执行下载。示例中的 `token=False` 表示匿名下载。

| 资源 | 仓库 | 大小 |
|---|---|---|
| 转换数据归档 | [GR00T-AgiBot-data](https://huggingface.co/datasets/Minth-Group/GR00T-AgiBot-data) | 约 7.17 GiB |
| xichong 放置 30k 推理包 | [xichong 模型](https://huggingface.co/Minth-Group/GR00T-G2-xichong-right-place-r0002-30k) | 约 16.27 GiB |
| zhewan 放置 30k 推理包 | [zhewan 模型](https://huggingface.co/Minth-Group/GR00T-G2-zhewan-right-place-r0003-30k) | 约 16.27 GiB |
| 新 xichong 抓取 r0002 转换数据，400/91 | [独立数据仓库](https://huggingface.co/datasets/Minth-Group/GR00T-G2-xichong-right-grasp-r0002-data) | 约 2.26 GiB |
| 新 xichong 抓取 r0002 30k 推理包 | [新抓取模型](https://huggingface.co/Minth-Group/GR00T-G2-xichong-right-grasp-r0002-30k) | 约 16.27 GiB |
| 早期 xichong 抓取 30k 推理包 | [早期抓取模型](https://huggingface.co/Minth-Group/GR00T-G2-xichong-right-single-grasp-30k) | 约 16.27 GiB |

只运行一个任务时，只需下载对应模型，不必下载数据或另一个任务的模型。

资源按用途分开保存：GitHub 提供代码、配置、文档和演示视频；Hugging Face 数据仓库
提供前三项历史任务的转换数据归档；新抓取 r0002 使用独立数据仓库；四个模型仓库各自保存对应任务的权重、骨干和配置。
**原始数采 episode 不在这些公开仓库里**，见第 2.6 节。
模型仓库的 `model/`、`backbone/` 不要合并或改名，现有下载脚本和推理入口依赖这组结构。

**转换数据：**

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "Minth-Group/GR00T-AgiBot-data",
    repo_type="dataset",
    local_dir="downloads/groot-data",
    token=False,
)
PY
tar -xzf downloads/groot-data/Isaac-GR00T-converted-data.tar.gz --strip-components=1
```

解压结果为 `agibot/gr00t_data/`。先确认没有需要保留的同名文件，并预留下载包和
解压数据的磁盘空间。归档包含历史转换版本，训练目录按第 2.3 节选择，不混合历史版本或 heldout。
这不是原始数采数据，也不是直接供 `datasets.load_dataset()` 加载的仓库。

新 xichong 抓取 r0002 的转换数据单独下载，目录结构直接是 `train/`、`heldout/`，无需解压历史归档：

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "Minth-Group/GR00T-G2-xichong-right-grasp-r0002-data",
    repo_type="dataset",
    local_dir="agibot/gr00t_data/g2_194/xichong_right_grasp_r0002_400",
    token=False,
)
PY
```

训练仅指向该目录的 `train/`，不要把 `heldout/` 或整个父目录作为训练输入。

**推理模型：选择一个任务下载，并配置本地骨干路径。**

```bash
.venv/bin/python - <<'PY'
import subprocess
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

task = "zhewan_place"  # 从下列四项中选一个
packages = {
    "xichong_place": (
        "Minth-Group/GR00T-G2-xichong-right-place-r0002-30k",
        "xichong_rplace_r0002_n1d7_checkpoint-30000",
    ),
    "zhewan_place": (
        "Minth-Group/GR00T-G2-zhewan-right-place-r0003-30k",
        "zhewan_rplace_r0003_n1d7_checkpoint-30000",
    ),
    "xichong_grasp_r0002": (
        "Minth-Group/GR00T-G2-xichong-right-grasp-r0002-30k",
        "xichong_rgrasp_r0002_n1d7_checkpoint-30000",
    ),
    "xichong_single_grasp": (
        "Minth-Group/GR00T-G2-xichong-right-single-grasp-30k",
        "xichong_rgrasp_n1d7_checkpoint-30000",
    ),
}
repo_id, bundle_name = packages[task]
bundle = Path("agibot/models") / bundle_name
snapshot_download(repo_id, local_dir=str(bundle), token=False)
subprocess.run([sys.executable, str(bundle / "configure_local_paths.py")], check=True)
PY
```

每个推理包包含 `model/`、配套 `backbone/` 和路径配置脚本。模型服务器加载
`model/`；`configure_local_paths.py` 只更新两个活动配置中的本地骨干路径，
不改权重或归一化统计量。模型包搬迁后需再次执行该脚本。
xichong 放置和早期抓取的 percentile、zhewan 放置和新抓取 r0002 的 min/max 配置必须各自保留。

这些包不含 optimizer、scheduler、RNG，不能用于精确恢复原训练状态；
原始数采数据未在上述仓库发布。
公开下载不改变许可：模型遵循资源页列出的 NVIDIA 许可，数据使用与再分发要求见数据集卡。
这里的匿名下载说明只适用于上述发布仓库；重新获取 NVIDIA 上游基座时，仍按第 6.1 节办理其访问要求。

### 2.6 原始数采、内部备份与云端历史路径

以下是项目维护机上最后核对过的相对路径。**原始 episode 未公开**；重做筛查、清洗或转换时须向数据权利方另取。转换数据和四个推理模型已有第 2.5 节的下载入口；维护机上的同名目录只是内部备份，不必重复传输。不要拿早期抓取或放置模型代替新抓取 r0002。

| 资源 | 维护机上的路径（相对本仓库） | 用途 |
|---|---|---|
| 新抓取转换集，400 train + 91 heldout，约 2.26 GiB | `agibot/gr00t_data/g2_194/xichong_right_grasp_r0002_400/` | 内部备份；公开下载见第 2.5 节 |
| 新抓取 30k 推理包，约 17 GiB | `agibot/models/xichong_rgrasp_r0002_n1d7_checkpoint-30000/` | 内部备份；公开下载见第 2.5 节 |
| 新抓取已选原始 episode，491 条、21,896,397,988 字节 | `agibot/data/g2_194/xichong_right_grasp_r0002_job01/` | 重做该批数据清洗/转换；已有转换集训练无需获取 |
| 早期抓取 30k 推理包 | `agibot/models/xichong_rgrasp_n1d7_checkpoint-30000/` | 内部备份；公开下载见第 2.5 节 |

若公开仓库暂时不可用而改走团队内部传输，将示例 `SOURCE_HOST` 改成实际可访问的内部账户/主机，并确认其对上述目录有读取权限。以下命令**在接收机的仓库根目录**运行；路径中的 `Isaac-GR00T` 是维护机现有 checkout 名，不是接收机必须使用的目录名：

```bash
SOURCE_HOST="user@internal-host"
mkdir -p agibot/gr00t_data/g2_194 agibot/models
rsync -aP "${SOURCE_HOST}:/home/admin1/ct/Isaac-GR00T/agibot/gr00t_data/g2_194/xichong_right_grasp_r0002_400/" \
  agibot/gr00t_data/g2_194/xichong_right_grasp_r0002_400/
rsync -aP "${SOURCE_HOST}:/home/admin1/ct/Isaac-GR00T/agibot/models/xichong_rgrasp_r0002_n1d7_checkpoint-30000/" \
  agibot/models/xichong_rgrasp_r0002_n1d7_checkpoint-30000/
.venv/bin/python agibot/tools/configure_model_backbone.py \
  agibot/models/xichong_rgrasp_r0002_n1d7_checkpoint-30000
```

`rsync` 仅是传输示例，不会自动创建源机器访问权限。传完后核对 train/heldout 下的 `data/`、`videos/`、`meta/`，以及模型包的 `model/`、`backbone/`；如需重做清洗，再单独传完整原始 episode。路径修复脚本只修改模型两个活动 JSON 中的本地骨干路径，不修改权重或训练统计量；搬迁模型后需重跑。

云端 A100 数据盘的**最后记录**是 `/root/gpufree-data/GR00T/`：基座在 `models/GR00T-N1.7-3B/`、骨干在 `models/Cosmos-Reason2-2B/`；新抓取转换集在 `datasets/xichong_right_grasp_r0002_400/{train,heldout}/`，30k 训练输出在 `outputs/xichong_rgrasp_r0002_n1d7_e400_f10_h16_1xa10080_s30k_ckpt5k_v1/checkpoint-30000/`。云端 checkpoint 是训练产物，**不是**可直接搬到本地的完整推理包；它需要配套骨干和路径处理。云端当前连接与文件存在性未核实，访问主机、端口和凭据应向维护者获取，不写入公开仓库。zhewan 云端数据及旧 checkpoint 已于 2026-09-20 清理，不应再从该云端路径尝试下载；见[清理记录](tasks/zhewan_right_place_r0003/CLOUD_TRAINING.md)。

## 3. 安装与运行环境

以下命令从完整仓库根目录执行，适用于训练／推理工作站：

```bash
uv python install 3.12
uv sync
agibot/bin/groot-g2 doctor
```

不复制原机器 `.venv`、CUDA 环境或模型缓存作为可移植环境。
GPU 环境、驱动与框架兼容性按 [官方 README](../README_UPSTREAM.md) 和任务训练记录核对。
`doctor` 检查代码与配置；GPU 前向和机器人执行分别在后续步骤验证。

机器人端使用其已安装 GDK 的环境。10.20.15.194 机器人的固定桥接曾在系统 Python 3.10 下编译／导入验证，
由脚本加载 `/home/agi/app/env.sh`。不要在机器人上直接用工作站环境覆盖厂商 GDK。
另一台 G2 的路径、GDK 版本、夹爪语义可能不同，参考 [G2 适配检查](docs/G2_ADAPTATION_GUIDE.md)。

### 3.1 哪些步骤在哪台机器运行

| 环境 | 承担的工作 | 注意事项 |
|---|---|---|
| 数采机／数据准备机 | 原始筛查、完整 episode 选择、转换与审核 | 不需要控制机器人 |
| 训练服务器 | 基座加载、GPU 短测、正式训练与 checkpoint 保存 | 当前案例是单卡 A100 80 GB；无卡模式只做文件／配置准备 |
| 推理工作站 | 官方模型服务、图像／状态输入、完整推理 runner、报告 | 需要 GPU 和对应任务模型；可与机器人分开 |
| G2 机器人 | 厂商 GDK、实时观测、arm/tool 联合执行 | 依赖现场网络与 GDK，不能用工作站环境代替 |

数据准备机、训练机和推理工作站可以合并，也可以分开，路径必须各自填写。
软件依赖以本仓库 `pyproject.toml` 和 `uv.lock` 为准，不另行拼装任意版本。
转换需要 FFmpeg，官方 loader 还使用 torchcodec；当前仓库依赖注释注明
torchcodec 0.8.0 配套 FFmpeg 4–7，不支持 FFmpeg 8。
运行 pytest 检查前，用 `uv sync --extra dev` 安装开发依赖。

## 4. 数据清洗与 GR00T 转换

新右臂放置任务优先使用 [右臂放置配置母本](templates/right_place/README.md)，不要复制
抓取任务“闭爪／抬升即成功”的筛选条件。原始协议见 [DATA_FORMAT](docs/DATA_FORMAT.md)。

推荐顺序：

```text
原始 episode 筛查与图像解码 → 按采集会话选 train/heldout → 明确 episode 名单
→ 复查选中完整 episode → 转换 → 只用 train 计算 stats → 官方 loader／处理器验证
```

每个任务在 `agibot/tasks/<task-id>/` 保存独立 `prepare.json`、筛查配置、`train.txt`、
`heldout.txt`，共用数据处理代码。新数据必须生成自己的名单，不能沿用旧任务的 episode 编号。

复现现有 zhewan 数据准备时，先取得原始数据、核对 `prepare.json` 路径，再执行：

```bash
.venv/bin/python agibot/scripts/prepare_g2_task.py \
  --task-config agibot/tasks/zhewan_right_place_r0003/prepare.json
```

该入口默认拒绝覆盖已有转换数据；已有输出仅复核时使用 `--stage verify`。
xichong 的旧清洗入口为 `scripts/clean_xichong_right_place_dataset.py`，保留作历史复现，
新任务不直接照搬它的任务阈值。详细筛查／名单生成命令见母本说明。

已验证的右臂任务共同语义（归一化仍按各任务配置）：

- 头部＋右腕图像，策略输入 RGB、640×480；10 Hz 动作，H16。
- EEF 为 XYZ＋Rot6D；Rot6D 使用旋转矩阵前两行。
- 官方处理器按配置处理相对 EEF；夹爪为绝对 native radians，约 0 闭合、−0.785 张开。
- 实时 GDK EEF 使用 base_link、米和 XYZW 四元数，不能重复反归一化或重复累加相对动作。
- zhewan 放置和新 xichong 抓取 r0002 使用 min/max；xichong 放置及早期抓取保留各自训练时的 percentile。统计量不能跨任务混用。
- 图像有损编码、数据结构检查或 loss 收敛，都不能单独保证实机成功。

### 4.1 数据处理完成后检查什么

新放置任务的关键产物如下，具体路径由 `prepare.json` 决定：

| 产物 | 用途 |
|---|---|
| `train.txt`、`heldout.txt` | 完整 episode 名单；按采集会话隔离，不能按帧拆分 |
| `report_dir/local_raw_audit.json` | 选中原始数据的结构、同步和数值复核 |
| `report_dir/conversion_summary.json` | 转换后的 episode／帧数汇总 |
| `report_dir/train_conversion_audit.json`、`heldout_conversion_audit.json` | 转换值、图像对应和解码检查 |
| `report_dir/official_loading_and_math.json` | 官方加载器与处理器数值验证 |
| `report_dir/preparation_status.json` | 完整自动检查通过状态为 `AUTOMATED_DATA_CHECKS_PASS` |
| 每个分集的 `meta/source_episode_map.json` | 转换编号到原始 episode、UUID、任务映射的溯源 |
| 每个分集的 `meta/training_preprocessing.json` | 训练处理器应采用的归一化设置 |

`--stage convert` 只转换，不代表完整验证已完成。复核现有转换结果时：

```bash
.venv/bin/python agibot/scripts/prepare_g2_task.py \
  --task-config agibot/tasks/zhewan_right_place_r0003/prepare.json --stage verify
```

验证不会重做 Parquet／视频转换，但会重新计算 train 统计量、同步至 heldout 并更新报告及元数据；
需要保留旧报告时指定新报告目录。
报告不通过时定位失败步骤，不直接修改状态或删除检查。自动检查不能判断全部物理放置结果；
按母本中的 `build_g2_place_review.py` 生成关键帧，检查释放位置、卡挂、掉件和回撤。
不合格 episode 可不入训练名单，无需为了清洗而删除原始记录。

## 5. 路径与任务配置

部分配置保留了原运行环境的绝对路径，首次运行前修改以下项目。

| 配置 | 配置项 |
|---|---|
| `tasks/<task>/prepare.json` | raw/output/report 路径、名单、期望数量、modality |
| 任务训练 `.env` | 数据盘、仓库、基座、骨干、train、清单路径与新的 RUN_ID |
| 固定包 `<task>/profile.json` | 模型相对路径、prompt、初始参考、workspace、端口、机器人 |
| 模型 `model/config.json` | 活动配置中的骨干 `model_name` |
| 模型 `model/processor_config.json` | processor 中的骨干 `model_name`；保留训练归一化设置 |
| 机器人 `robot_bridge.sh` | 厂商 env.sh 路径、GDK／夹爪适配，不能只替换 IP |

将训练 `.env` 改为实际路径后再运行；任务 wrapper 会 source 自己的 env 文件，
单纯在外面 export 一个新路径可能被覆盖。选择新的输出 RUN_ID，不覆盖已有训练。

适配新 GDK 或工位时另建部署版本，保留已有实测版本。
profile 中的初始位姿用于状态核对，不会自动执行复位。

## 6. 模型与训练

### 6.1 基础模型

从原始基座微调需要 GR00T-N1.7-3B 和 Cosmos-Reason2-2B。
先在 Hugging Face 完成 [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) 的许可访问，并为具有访问权限的账户准备只读 token。脚本
[download_gr00t_n1d7_models.sh](training/download_gr00t_n1d7_models.sh)
固定了 [GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) 与 Cosmos 的模型版本，并包含权重 SHA 校验。在**训练服务器**的完整仓库根目录、已有 `.venv` 的环境中，指定实际数据盘路径后运行：

```bash
export CT_ROOT=/path/to/data-disk/GR00T
read -rsp 'Hugging Face read token: ' HF_TOKEN; echo
export HF_TOKEN
bash agibot/training/download_gr00t_n1d7_models.sh
unset HF_TOKEN
```

`CT_ROOT` 下会生成 `models/GR00T-N1.7-3B/`、`models/Cosmos-Reason2-2B/` 和下载缓存；训练 `.env` 中的基座/骨干路径必须指向实际位置。不要将 token 写入 `.env`、README、shell 命令历史或 Git 仓库。若已有经核对的基座与骨干，可复用，无须为每个任务重新下载。

### 6.2 训练入口

| 任务 | 配置文件 | 训练 wrapper |
|---|---|---|
| xichong 放置 | [训练参数](training/xichong_right_place_r0002_1xa10080.env) | `agibot/training/run_xichong_right_place_r0002_1xa10080.sh` |
| zhewan 放置 | [训练参数](tasks/zhewan_right_place_r0003/train_1xa10080.env) | `agibot/tasks/zhewan_right_place_r0003/train.sh` |
| xichong 抓取 r0002 | [训练参数](tasks/g2_194/xichong_right_grasp_r0002/train_1xa10080.env) | `agibot/tasks/g2_194/xichong_right_grasp_r0002/train.sh` |

三项已记录的单卡配置均为 A100 80 GB、30,000 optimizer step、每 5,000 step 保存、
保留最新两份、每 10 step 记录 loss；单次 batch 16、累积 2，有效 batch 32。
新任务可从这组参数开始，根据短测显存、loss 和评估结果调整。

先准备数据／模型并修改上述路径，再按任务执行：

```bash
# xichong
bash agibot/training/run_xichong_right_place_r0002_1xa10080.sh audit
bash agibot/training/run_xichong_right_place_r0002_1xa10080.sh smoke
# 短测通过并决定开始新训练后：
bash agibot/training/run_xichong_right_place_r0002_1xa10080.sh baseline

# zhewan（独立任务，不要同时执行两套）
bash agibot/tasks/zhewan_right_place_r0003/train.sh audit
bash agibot/tasks/zhewan_right_place_r0003/train.sh smoke
# 短测通过并决定开始新训练后：
bash agibot/tasks/zhewan_right_place_r0003/train.sh baseline

# 新 xichong 抓取 r0002（独立任务，不能与上述任务同时运行）
bash agibot/tasks/g2_194/xichong_right_grasp_r0002/train.sh audit
bash agibot/tasks/g2_194/xichong_right_grasp_r0002/train.sh smoke
# 短测通过并决定开始新训练后：
bash agibot/tasks/g2_194/xichong_right_grasp_r0002/train.sh baseline
```

`audit` 检查数据、模型与配置，需要对应文件和清单，不执行训练 step。
`smoke` 会实际占用 GPU；`baseline` 是新正式训练，不是自动从短测接着训练；
断点续训使用 `resume` 并提供完整训练状态。

历史 xichong 放置默认使用 SHA256SUMS；zhewan 和新抓取使用文件名／字节数 inventory。
清单应在源数据验证完成后生成，并与数据集版本一起保存，用于检查后续复制是否完整。
训练记录见
[zhewan CLOUD_TRAINING](tasks/zhewan_right_place_r0003/CLOUD_TRAINING.md) 与
[xichong STATUS](examples/xichong_right_place_r0002/STATUS.md)；新抓取另见
[训练记录](tasks/g2_194/xichong_right_grasp_r0002/CLOUD_TRAINING.md)。

### 6.3 日志、保存和恢复

公共 launcher 的路径约定如下，变量来自所选任务的训练 `.env`：

- 正式训练日志：`CT_ROOT/logs/RUN_ID.baseline.log`；audit、smoke、resume 各有对应日志。
- 正式输出：`CT_ROOT/outputs/RUN_ID/`；当前 30k 案例最终保留 checkpoint-25000、checkpoint-30000。
- 短测默认使用 `RUN_ID_smoke`，不会续接到正式训练。

checkpoint 中的 `trainer_state.json` 记录 step 和 loss 历史；结合退出码、最终 step、
非有限 loss／梯度、OOM 和保存日志判断是否完成，不能只看 checkpoint 目录是否存在。
同 RUN_ID、同模式再次启动会由 `tee` 重写该模式日志；恢复前保留旧日志。

后台训练应在**训练服务器**使用 tmux，示例见任务 CLOUD_TRAINING 文档。
本机 `tmux attach` 看不到云端会话；`no sessions` 不等于训练必然失败，应查日志和进程。
断开 SSH 不等于关闭实例；训练中不能切无卡模式或关机。

`resume` 在相同 RUN_ID 输出下查找 checkpoint，需完整训练状态；新训练才换新 RUN_ID。
保留两份 checkpoint 不代表磁盘只占两份权重：optimizer、输出根目录最终保存、基座、
骨干、缓存和短测也占空间，开始前按实际大小预留保存余量。

### 6.4 从训练进入评估／部署

先核对保存模型中的实际处理器，而不仅是训练命令行。例如：

```bash
.venv/bin/python agibot/training/check_saved_normalization.py \
  --model /path/to/checkpoint-30000 --dataset /path/to/converted/train
```

`/path/to/` 必须替换为实际路径。该检查重载处理器，不做 GPU 模型前向。
zhewan 曾修复“CLI 设 min/max、实际基座 processor 仍用 percentile”的覆盖问题；
迁移仓库时不能遗漏相关补丁和重载检查。

接着使用官方模型服务和对应 heldout 数据检查图像、state、H16 输出、EEF 恢复、夹爪映射
及释放／回撤时序。zhewan 的复现入口与指标解释见
[离线测试](tasks/zhewan_right_place_r0003/OFFLINE_TEST.md)。该页是 9 月 10 日记录，
其中旧实机入口已由第 7 节固定包替代；离线入口的 `inference.json` 也需适配本机路径。

离线同时间索引误差不等于机械臂跟踪误差，不能直接换算成功率。训练完成、离线通过、
现场完整完成是三个不同阶段，分别记录测试结果。

## 7. G2 推理

### 7.1 模型加载

推理使用对应任务模型包的 `model/` 和 `backbone/`，不依赖原始训练数据。
三片 safetensors、索引、processor、statistics 和相关配置不可拆着混用。
模型服务器 `--model-path` 指向包内的 `model/`，不是包根目录。

搬迁后将 `model/config.json` 和 `model/processor_config.json` 内的骨干 `model_name`
改为本机 `backbone/Cosmos-Reason2-2B` 绝对路径。不要修改 `*.server-original.json`
来代替活动配置，不改变 checkpoint 的统计量、归一化和动作语义。
本机推理包不含 optimizer／scheduler／RNG，恢复训练需使用完整训练 checkpoint。
从 Hugging Face 下载的四个推理包均自带 `configure_local_paths.py`；内部复制的旧包
可用 `agibot/tools/configure_model_backbone.py <模型包目录>` 进行同样的路径修复。

原始包内 manifest/README 的某些状态是下载或早期失败时的历史快照；最新真机结果以任务部署记录
和 [固定包说明](deployments/g2_194/README.md) 为准，不将旧阶段字段当作最终验收结论。

### 7.2 10.20.15.194 机器人日常入口

```bash
# 只打印配套命令，不启动服务、不移动机器人
.venv/bin/python agibot/deployments/g2_194/commands.py xichong_right_place_r0002 \
  --report agibot/local_reports/xichong_new_run/inference.json

.venv/bin/python agibot/deployments/g2_194/commands.py zhewan_right_place_r0003 \
  --report agibot/local_reports/zhewan_new_run/inference.json
```

选择一个任务，依输出说明在相应机器／终端运行模型、观测、动作服务与转发。
报告路径每次换新；两个任务使用相同端口，不可同时运行，也不能把旧模型服务误当新任务服务。
固定配置自带初始参考，因此打印和运行这条入口不依赖原始数据或转换训练集。

机器人固定包位置默认为 `/home/agi/vla_ct/releases/g2_194_20260914/`；
另一位置用 `commands.py --robot-root` 指定。位置参数不能代替机器人适配：
脚本仍需匹配实际 GDK、相机、EEF 和夹爪。完整步骤见固定包 README。

### 7.3 10.20.15.194 机器人新抓取 r0002

新抓取有独立任务入口，复用已验证的 arm/tool 联合 GDK 控制实现，**不使用放置任务的释放/回撤完成条件**。先按第 2.6 节取得对应模型包并修复本地骨干路径，再生成启动命令（此命令只打印，不启动服务）：

```bash
.venv/bin/python agibot/tasks/g2_194/xichong_right_grasp_r0002/inference.py commands \
  --report agibot/local_reports/g2_194/xichong_right_grasp_r0002_live/new_run/inference.json
```

完整部署顺序、机器人端任务包装层、端口和执行结果见
[新抓取桥接记录](tasks/g2_194/xichong_right_grasp_r0002/BRIDGE_DEPLOYMENT.md)。模型按完整 H16 块闭环输出接近、闭爪、抬升；软件的“闭爪且抬升稳定”只是一种反馈判据，不能代替现场确认是否真正抓起或进入卡槽。已记录的有限轮次里既有够不到/未入槽，也有现场确认抓起且入槽；不能据此推算稳定成功率。

### 7.4 已知边界

- 先预热模型，再显式激活控制；结束／异常后不自动复位、闭爪或松爪。
- 现场确认工件状态、初始图像／位姿、运动范围和急停值守后才能执行。
- 新任务／新机器即使型号相同，也需核对 GDK 和夹爪开闭语义。
- 软件回归结果与真机记录分别保存，长期成功率尚未统计。
- 历史 `groot-g2 robot action` 仍是 legacy 入口，不用于替代 10.20.15.194 机器人的固定包。
  不同时运行旧 mux、arm child 或独立夹爪 daemon。
- [RTC 实验](rtc/README.md) 和公共 `robot/`、`scripts/` 中的实验版本不是当前固定运行入口。

更多要求见 [G2 适配指南](docs/G2_ADAPTATION_GUIDE.md)、
[运行流程](docs/G2_RUNTIME_FLOW.md) 和 [现场检查](docs/SAFETY_AND_COMMISSIONING.md)。

## 8. 目录说明

| 目录／文件 | 用途 |
|---|---|
| [deployments/g2_194/](deployments/g2_194/README.md) | 两个放置任务的固定推理脚本、依赖、profile 与命令入口 |
| [tasks/](tasks/README.md) | 按任务的数据名单、处理／训练配置与记录 |
| [templates/right_place/](templates/right_place/README.md) | 新右臂放置任务的数据处理母本 |
| [examples/](examples/) | 早期抓取与放置的历史实测说明，不自动替换现入口 |
| [inference_videos/](inference_videos/README.md) | 两个独立演示视频，不是训练集 |
| `data/`、`gr00t_data/`、`models/` | 原始数据、转换数据、模型权重 |
| [tools/configure_model_backbone.py](tools/configure_model_backbone.py) | 内部复制推理包后的本地骨干路径修复 |
| [training/](training/) | 下载、训练启动和检查工具 |
| [scripts/](scripts/)／[tools/](tools/) | 转换、评估、适配与诊断脚本 |
| [robot/](robot/README_G2_GROOT_BRIDGE.md) | 通用／历史机器人代码；日常运行以固定快照为准 |
| [docs/COMPONENT_MAP.md](docs/COMPONENT_MAP.md) | 组件关系与入口索引 |

历史 TOML task profile 与新的 `prepare.json`、固定包 `profile.json` 用途不同，
不能相互替换。旧通用 CLI 的背景见 [TASK_ADAPTATION](docs/TASK_ADAPTATION.md)。

## 9. 运行前检查

1. 环境：完整仓库、Python 依赖、GPU 和视频解码环境可用。
2. 资源：数据分集、模型、modality、归一化设置与所选任务一致。
3. 路径：数据、模型、日志、报告目录正确，新实验使用独立输出目录。
4. 训练：配置检查、短测和保存后重载通过，再启动正式训练。
5. 真机：桥接版本、服务端口、GDK、工位初始状态正确，现场确认后启动。

无需数据和 GDK 的固定包基础检查：

```bash
.venv/bin/python -m pytest -q agibot/deployments/g2_194/test_package.py
```

## 10. 常见问题

| 现象 | 优先检查 |
|---|---|
| 找不到模型／训练集 | 对应资源是否已准备，Git 是否忽略对应目录，配置路径是否正确 |
| 改了环境变量仍访问旧路径 | wrapper 会 source 任务 `.env`，修改真正被加载的文件 |
| 骨干不存在、尝试联网下载 | Cosmos 是否齐全，两个活动配置的 `model_name` 是否仍指向旧路径 |
| 训练归一化不对 | 比较预处理元数据、CLI、实际 processor 和保存后重载值，不混用两个任务 |
| 已有输出目录被拒绝 | 新处理换新输出；复核用 verify，中断训练用 resume，不删除旧结果绕过 |
| 视频解码失败 | FFmpeg／torchcodec 兼容性、视频完整性和相机字段，不跳过错误样本 |
| 推理连不上或行为像另一任务 | 模型、端口、SSH 转发与 profile 是否配套，两任务不能同时占用相同端口 |
| 机械臂与夹爪衔接异常 | 是否使用对应固定联合 arm/tool 桥接，有无旧 mux／独立夹爪进程占用控制，当前 GDK 是否一致 |
| 夹爪方向或开度不对 | 数采定义、native radians 和本机 GDK 映射，不能随意翻转模型输出 |
| 现场异常下压／轨迹异常 | 停止当前测试并留存报告，区分模型目标、观测／坐标、通信与反馈，不关闭保护或自动重试掩盖问题 |

排错时记录任务名、模型目录、桥接版本、配置、命令、日志和本轮报告；分享前移除凭据。
公共脚本改动验证后再固定为新部署版本，不让成功版本自动跟随实验代码变化。
