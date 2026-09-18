# GR00T 数据转换（兼容入口）

新的团队入口是 [`README.md`](README.md)，原始数据字段和动作重建定义见
[`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md)。

新的右臂放置任务优先使用[按任务配置流程](tasks/README.md)：

```bash
.venv/bin/python agibot/scripts/prepare_g2_task.py \
  --task-config agibot/tasks/<task-id>/prepare.json
```

它按显式 train/heldout 名单转换、仅拟合训练统计量，并做全帧源视频对应及官方处理器检查。
下面是已配置 task adapter 的历史项目兼容命令；不要直接套用抓取案例的成功条件：

```bash
agibot/bin/groot-g2 prepare \
  --task-config agibot/profiles/<task-id>/task.toml \
  --raw-root /path/to/raw \
  --full-dataset /path/to/gr00t/full \
  --selected-dataset /path/to/gr00t/selected \
  --report-dir /path/to/reports \
  --count <audited-episode-count> \
  --workers 8
```

该入口读取 task profile 中声明的转换、筛选、modality、训练门禁和 shadow adapter。
使用相同数采格式时可以复用案例转换逻辑，并按任务目标调整成功条件。
