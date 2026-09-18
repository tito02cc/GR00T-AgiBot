# Task profile 模板

复制整个目录到 `agibot/profiles/<task-id>/`，填写任务信息、数据规模、训练计划和 pipeline
入口。`task.arm` 按任务选择 `left` 或 `right`，并同步使用同侧图像、state/action、modality
和机器人 bridge。`CHANGE_ME` 标记需要项目确认的字段，示例数字按实际数据和训练计划更新。
