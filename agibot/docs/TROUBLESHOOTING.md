# 常见问题

## 模型不到闭合点就回撤

检查是否使用旧 H1/H2/H4、stage 或 receding-approach 脚本。正式路径必须完整执行每个
16-step chunk，再重新观测。

## 到位但夹爪不闭合，或闭合后又张开

先检查 raw open/closed 方向和 official decoder 输出。不要用自定义阈值、强制 open 或
单调 latch 替换模型值。确认 active gripper target 完成后被消费，不会重复发送。

## `MotionPlanResponse uuid mismatch`

通常表示多个 subscriber/owner 竞争响应。确保只有一个持久 GDK owner。正式 waypoint
控制不使用 MotionPlan。

## 小位移存在约 0.2 mm 误差

不要为每个 action 冷启动 MotionPlan/WBC。使用已验证的持久 50 Hz adaptive controller，
并在目标机器人上重新标定初始 compensation。

## 推理看起来很慢

区分模型 latency、observation RTT 和 10 Hz action 执行。严重 Wi-Fi 丢包不能通过提高
控制频率解决；先恢复网络并重新做 warm shadow。

## 转换数据比原始数据小很多

正常。训练只保留头部和右腕，左腕不写入；逐帧 JPEG 编成 H.264。必须用 hard gate 的
帧数、完整解码、source correspondence 和 PSNR 判断是否缺失，不能只比较目录大小。

## 单卡 A100 OOM

先确认是 80 GB A100、venv cuDNN 优先、batch/accumulation 没有偏离基线，并从 audit 和
100-step smoke 重新开始。不要在正式 run 中临时改一部分参数。

## Hugging Face backbone 下载失败

确认账号已经取得两个 gated repository 的访问权限，token 只通过临时 `HF_TOKEN` 环境
变量传递。模型必须固定到 `artifacts.lock.json` 中的 revision。

## `doctor` 拒绝机器人配置

这是预期保护。不要绕过：填写真实 GDK 版本、夹爪 raw 端点/status、workspace 和起始位姿，
完成现场报告后才设置 `calibration_confirmed=true`。
