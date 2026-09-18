# 相对 NVIDIA 基线的必要补丁

本团队包以 NVIDIA Isaac-GR00T 提交 `51d4c89` 为基线。除 `agibot/` 外，当前 fork 还修改了
4 个上游文件；它们是单卡、离线模型和可审计训练流程的一部分，push 时不能漏掉：

| 文件 | 用途 |
|---|---|
| `gr00t/configs/finetune_config.py` | 增加本地 backbone、离线加载、loss 日志频率和 dry-run 参数 |
| `gr00t/experiment/launch_finetune.py` | 传递上述参数，并输出不加载权重的完整训练配置审计 |
| `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` | 让本地 Cosmos/Qwen3 路径也能正确选择 Qwen3 backbone |
| `gr00t/model/gr00t_n1d7/setup.py` | 把实际 backbone 路径传入 N1.7 action model |

这些改动解决的是路径和可审计性，不改变本任务的数据语义、action horizon 或模型结构。
升级 NVIDIA 上游版本时，不要盲目覆盖：逐项检查上游是否已经提供等价能力，重新运行
`groot-g2 train audit`、100-step smoke 和完整回归测试后再移除本地补丁。
