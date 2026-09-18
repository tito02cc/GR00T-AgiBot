# 新任务配置

## 1. 建立项目目录

从现有模板创建 task profile：

```bash
cp -R agibot/templates/task_profile agibot/profiles/<task-id>
```

task profile 随项目代码提交，记录数据处理、训练和推理使用的统一参数。单台机器人的配置
保存在 `agibot/configs/local/<robot-id>.toml`，由部署环境管理。

## 2. 填写任务信息

在 `task.toml` 中填写 task id、`task.arm`、prompt、episode 数、训练参数和各流程入口。
`task.arm` 按任务选择 `left` 或 `right`，同侧的腕部图像、EEF、夹爪、modality 和 pipeline
字段保持一致。当前仓库已经完成右臂任务验证；左臂任务沿用相同数据结构并配置左臂适配。

团队使用相同的数采程序和 G2，EEF/夹爪表示、10 Hz 和 H16 通常保持不变。

若项目调整相机、控制自由度、坐标系或 action 表示，需要同步更新转换器、GR00T modality、
model decoder、机器人 bridge 和回归测试。

## 3. 复用数据处理

相同数采格式可以直接复用以下逻辑：

- frame 与 `head_color`、`hand_right` 图像对应检查；
- XYZ+XYZW 与 XYZ+Rot6D 转换；
- next-delta 到绝对 EEF 的动作重建；
- 夹爪 state/action 的单位和方向检查；
- Parquet、MP4、stats 和 SHA256 生成。

任务变化时主要调整质量判断，例如抓取、抬升、放置或插接的成功条件，以及 train/held-out
划分。现有 `xichong_*` 脚本提供一套已经运行过的抓取实现，可以复制到
`agibot/tasks/<task-id>/` 后修改任务语义并补充测试。

推理完成条件也由任务 runner 定义。抓取案例使用“夹爪闭合并抬升”，其他任务在 profile
中配置对应的 `run_inference.py` 和 `recover.py`。

## 4. 确定数据和训练规模

先审计全部轨迹，再根据场景覆盖、轨迹差异和 held-out 结果确定训练 episode。模板中的 24
条和 10k steps 用于展示配置字段；抓取案例中的 300 条和 30k steps 是一组可参考的实测
配置。

正式训练前依次完成 config audit、短程 smoke、loss 检查和磁盘容量确认。

## 5. 记录数据与模型版本

项目 `artifacts.lock.json` 记录：

- 基础模型和 backbone revision；
- 数据集 URI、episode/frame/Parquet/MP4 数量；
- 数据 SHA256 manifest；
- checkpoint steps、shard 数和模型 SHA256 manifest。

数据与模型存放在对象存储或数据盘，Git 中提交 lock 和必要的脱敏摘要。

## 6. 核对 GDK 环境

团队机器人型号相同，但 GDK 版本、相机枚举、TF、夹爪反馈状态和控制补偿可能不同。复制
`g2_right_arm.template.toml` 后检查：

- GDK 版本、运行 mode 和 error code；
- `base_link` 到 `arm_r_end_link` 的位姿；
- 两路相机身份、分辨率与同步；
- omnipicker raw open/closed、方向和 settled status；
- 训练起点、workspace 和 WBC compensation；
- observation/action bridge 端口和唯一控制 owner。

检查结果通过 `groot-g2 doctor`、read-only preflight 和 commissioning 报告留档。GDK 接口
或反馈语义发生变化时，同步更新 bridge 适配和测试。

## 7. 项目验收记录

- task profile、代码 revision 和数采版本；
- 原始数据审计、episode 选择和 held-out 清单；
- 转换数据 SHA、stats 和 hard gate；
- 训练配置、loss 和 checkpoint SHA；
- 离线与 shadow 评估；
- GDK/机器人检查、真机 inference 和 recovery 报告。
