# Agibot G2 × GR00T N1.7

基于 NVIDIA Isaac GR00T N1.7，提供 Agibot G2 从原始数采数据清洗、格式转换、
模型后训练到真机推理的代码、配置和操作记录。

当前已实测右臂抓取及两个右臂放置任务。相同数采格式和 G2 机器人可复用已有流程；
新任务、其他 GDK 版本及左臂／双臂需调整数据字段和控制接口。

## 使用说明

**完整使用说明：[agibot/README.md](agibot/README.md)。**
包含环境安装、数据处理、训练配置和推理部署步骤。

| 要做的事情 | 文档入口 |
|---|---|
| 安装环境、准备数据与模型 | [操作手册](agibot/README.md) |
| 处理新的右臂放置数据 | [配置母本与筛查命令](agibot/templates/right_place/README.md) |
| 复现 xichong 放置任务 | [任务记录](agibot/examples/xichong_right_place_r0002/README.md) |
| 复现 zhewan 放置任务 | [任务记录](agibot/tasks/zhewan_right_place_r0003/README.md) |
| 使用 10.20.15.194 机器人的两个放置任务固定推理桥接 | [部署说明](agibot/deployments/g2_194/README.md) |
| 查看实机演示 | [推理结果视频](agibot/inference_videos/README.md) |
| 查看早期抓取流程 | [抓取案例](agibot/examples/xichong_right_single_grasp/README.md) |

## 仓库与资源

`agibot/` 依赖仓库根目录的 `gr00t/`、`pyproject.toml` 和 `uv.lock`，不能独立运行。
GitHub 保存代码、配置、文档和实机视频；转换数据及两个放置任务的 30k 推理模型在
Hugging Face 公开提供，下载无需登录或申请读取权限。

| 资源 | Hugging Face |
|---|---|
| 转换后的数据归档（约 7.17 GiB） | [GR00T-AgiBot-data](https://huggingface.co/datasets/Minth-Group/GR00T-AgiBot-data) |
| xichong 右臂放置 30k 模型（约 16.27 GiB） | [xichong 模型](https://huggingface.co/Minth-Group/GR00T-G2-xichong-right-place-r0002-30k) |
| zhewan 右臂放置 30k 模型（约 16.27 GiB） | [zhewan 模型](https://huggingface.co/Minth-Group/GR00T-G2-zhewan-right-place-r0003-30k) |

下载、解压及骨干路径配置见 [操作手册第 2.5 节](agibot/README.md#25-从-hugging-face-下载)。
上述数据和模型已完整上传，可直接公开下载。只做现有任务推理时，下载对应任务模型即可。
原始数采数据、早期抓取模型及精确续训所需的优化器状态不在上述发布范围。
公开可读不改变数据及模型的许可要求，具体见各资源页。

## NVIDIA 上游

本仓库基于 [NVIDIA/Isaac-GR00T@51d4c89](https://github.com/NVIDIA/Isaac-GR00T/commit/51d4c89)。
官方 README 原文保留在 [README_UPSTREAM.md](README_UPSTREAM.md)，
上游项目为 [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)。

升级前阅读 [本地补丁说明](agibot/docs/UPSTREAM_PATCHES.md)，复核数据、训练及推理兼容性。
继续遵循 [LICENSE](LICENSE) 与 [CONTRIBUTING.md](CONTRIBUTING.md)。
