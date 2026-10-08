# Agibot G2 × GR00T N1.7

本仓库基于 [NVIDIA Isaac GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T)，记录 Agibot G2 从原始数采、数据清洗和转换、单卡 A100 后训练，到离线评估和真机推理的实现。已实测的任务是右臂抓取与放置；左臂、双臂或不同 GDK 版本需要单独适配，不能直接沿用右臂字段与控制映射。

从这里开始：

| 内容 | 入口 |
|---|---|
| 安装、数据/模型下载、训练和推理操作 | [agibot/README.md](agibot/README.md) |
| 技术设计、实验数据与实机复盘 | [GR00T 技术报告](agibot/docs/GR00T_TECHNICAL_REPORT.md) |
| 10.20.15.194 机器人新抓取任务 | [xichong_right_grasp_r0002](agibot/tasks/g2_194/xichong_right_grasp_r0002/README.md) |
| 10.20.15.194 机器人两个放置任务的固定推理入口 | [部署说明](agibot/deployments/g2_194/README.md) |
| 实机演示 | [推理视频](agibot/inference_videos/README.md) |
| NVIDIA 原版说明 | [README_UPSTREAM.md](README_UPSTREAM.md) |

## 仓库中有什么

GitHub 保存代码、任务配置、episode 名单、处理与训练脚本、推理桥接、测试、文档和两段演示视频。`agibot/` **不是独立项目**：运行时还需要根目录的 `gr00t/`、`pyproject.toml` 和 `uv.lock`。克隆仓库不会取得原始 episode、转换后的 Parquet/MP4、基座权重、微调权重或本机运行报告；这些大文件被 `.gitignore` 排除。

上游 `demo_data/` 中的部分 MP4/Parquet 由 Git LFS 管理。若要运行 NVIDIA 上游示例，需另装 `git-lfs` 并在克隆后执行 `git lfs pull`；Agibot 的数据和模型仍按下文从 Hugging Face 或内部存储取得，`git lfs pull` 不会下载它们。

| 任务 | 训练/留出 | 转换数据 | 30k 推理模型 |
|---|---:|---|---|
| 早期 xichong 右臂抓取（10.20.15.60） | 300 / 另有历史评估集 | [公开数据归档](https://huggingface.co/datasets/Minth-Group/GR00T-AgiBot-data) | [公开模型包](https://huggingface.co/Minth-Group/GR00T-G2-xichong-right-single-grasp-30k) |
| xichong 右臂放置 r0002 | 600 / 100 | [公开数据归档](https://huggingface.co/datasets/Minth-Group/GR00T-AgiBot-data) | [公开模型包](https://huggingface.co/Minth-Group/GR00T-G2-xichong-right-place-r0002-30k) |
| zhewan 右臂放置 r0003 | 400 / 58 | [公开数据归档](https://huggingface.co/datasets/Minth-Group/GR00T-AgiBot-data) | [公开模型包](https://huggingface.co/Minth-Group/GR00T-G2-zhewan-right-place-r0003-30k) |
| 新 xichong 右臂抓取 r0002（10.20.15.194） | 400 / 91 | [独立数据仓库](https://huggingface.co/datasets/Minth-Group/GR00T-G2-xichong-right-grasp-r0002-data) | [公开模型包](https://huggingface.co/Minth-Group/GR00T-G2-xichong-right-grasp-r0002-30k) |

历史转换数据归档约 7.17 GiB，**不含**新抓取 r0002；该任务的 400/91 条转换集在独立仓库，约 2.26 GiB。四项任务的 30k 推理模型包分别下载。原始 episode 均未上传到 GitHub/Hugging Face；要重做清洗或转换，必须另取对应任务的完整原始数据。只运行现有模型的推理，不需要下载训练集。

## 最短准备流程

```bash
git clone https://github.com/tito02cc/GR00T-AgiBot.git
cd GR00T-AgiBot
uv python install 3.12
uv sync
agibot/bin/groot-g2 doctor
```

接着按 [操作手册 §2.5](agibot/README.md#25-从-hugging-face-下载) 下载所选任务的**转换数据**或**推理模型包**，并重新配置模型中的本地骨干路径。训练还需要 NVIDIA 的 GR00T-N1.7-3B 基座和 Cosmos-Reason2-2B 骨干，按 [操作手册 §6.1](agibot/README.md#61-基础模型) 获取；Cosmos 的许可访问需在 Hugging Face 完成。原始 episode 的内部位置及云端历史路径见 [操作手册 §2.6](agibot/README.md#26-原始数采内部备份与云端历史路径)。**Git clone 不含任务数据或权重；转换数据也不能代替原始 episode 重新清洗。**

开始训练前，选择任务配置，核对数据分集、模态、归一化、基座和输出路径，依次运行 `audit`、GPU `smoke`、`baseline`。完整命令见 [操作手册 §6](agibot/README.md#6-模型与训练)。真机推理需对应模型包、GPU 推理工作站和经本机 GDK 核对的机器人桥接，见 [操作手册 §7](agibot/README.md#7-g2-推理)。任务记录中的固定 IP/路径是历史环境，不会自动匹配另一台机器人或云服务器。

## 上游与许可

本仓库保留 [NVIDIA/Isaac-GR00T@51d4c89](https://github.com/NVIDIA/Isaac-GR00T/commit/51d4c89) 的 [原始 README](README_UPSTREAM.md)；本地差异见 [补丁说明](agibot/docs/UPSTREAM_PATCHES.md)。代码、模型和数据分别遵循其对应许可；公开可下载不代表可以忽略资源页的使用条款。仓库根目录的 [LICENSE](LICENSE) 与 [CONTRIBUTING.md](CONTRIBUTING.md) 继续适用。
