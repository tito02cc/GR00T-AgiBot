# 右臂原始 episode 示例

`frame.schema.json` 描述当前已验证右臂任务读取的数采字段。左臂任务使用相同 episode 目录
结构，由对应 task adapter 读取左臂位姿、夹爪和腕部图像字段。完整目录为：

```text
episode_NNNNNN/
├── arrays.npz
├── frames.jsonl
├── meta_info.json
├── quality_report.json
└── images/
```

字段、数组和动作语义见 [`../../docs/DATA_FORMAT.md`](../../docs/DATA_FORMAT.md)。项目通过 task
profile 中配置的 converter 进行检查：

```bash
.venv/bin/python agibot/tasks/<task-id>/convert.py \
  --source /path/to/raw-root \
  --report /path/to/raw-validation.json \
  --validate-only
```

仓库保留 schema，不附带数采图像。团队可选取一条自己的 episode 运行 `--validate-only`，
确认数采版本与当前协议一致。抓取案例的实现位于
`agibot/scripts/convert_xichong_right_single_grasp.py`。
