# G2 右臂单工件抓取案例

返回[项目主文档](../../README.md)。

这个案例记录了从 636 条原始 episode 中审计并选择 300 条、单卡 A100 微调 30k steps、
离线/shadow 验证以及 G2 真机完整抓取的全过程。详细参数、结果和问题复盘见
[`RUNBOOK.md`](RUNBOOK.md)。对应实现保留 `xichong_*` 文件名，便于将脚本与案例报告对应。

新任务可以参考：

- 原始 frame 与双相机图像检查；
- next-delta → 绝对 EEF → GR00T relative action 的转换；
- 右 EEF/右夹爪 modality；
- config audit、smoke、baseline 的训练顺序；
- G2 observation/action bridge 和 GDK 排障；
- 完整 H16 推理及夹爪 action 映射。

项目配置时更新 task prompt、episode 数、训练计划、数据/模型地址，以及目标机器人的 GDK、
起始位姿、workspace、夹爪端点和控制补偿。任务终态与本案例不同时，同步调整数据门禁和
inference runner。

新项目目录可从 [`../../templates/task_profile/`](../../templates/task_profile/) 复制。
