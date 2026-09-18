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
原始数据、转换数据和模型权重需单独准备，目录约定见操作手册。

## NVIDIA 上游

本仓库基于 [NVIDIA/Isaac-GR00T@51d4c89](https://github.com/NVIDIA/Isaac-GR00T/commit/51d4c89)。
官方 README 原文保留在 [README_UPSTREAM.md](README_UPSTREAM.md)，
上游项目为 [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)。

升级前阅读 [本地补丁说明](agibot/docs/UPSTREAM_PATCHES.md)，复核数据、训练及推理兼容性。
继续遵循 [LICENSE](LICENSE) 与 [CONTRIBUTING.md](CONTRIBUTING.md)。
