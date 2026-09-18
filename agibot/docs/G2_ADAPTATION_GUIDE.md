# G2 适配与校准指南

当前实测配置和工具以右臂任务为例。机械臂侧别由 task profile 选择，左臂任务对相应 TF、
腕部相机、夹爪、workspace 和 bridge 执行同样检查。

## 原则

团队使用相同的 Agibot G2，但 GDK 版本、相机枚举、TF、omnipicker 反馈端点、状态码、
WBC 冷启动补偿和网络仍可能不同。每台机器人建立独立 TOML 和 commissioning 报告，
完成检查后设置 `calibration_confirmed=true`。

## 1. 建立本机配置

```bash
mkdir -p agibot/configs/local
cp agibot/configs/g2_right_arm.template.toml \
  agibot/configs/local/<robot-id>.toml
```

`configs/local/` 被 Git 忽略，避免把机器身份和部署细节发布到 GitHub。团队应在受控资产库
保存正式校准文件和报告。

## 2. 只读核对

在没有任何控制 owner 的情况下检查：

- GDK 初始化成功，记录准确版本；
- whole-body mode、右臂/右末端 error code；
- `get_tf_from_base_link("arm_r_end_link")` 返回 XYZ+XYZW；
- 静止时四元数归一化且连续；
- `head_color` 与 `hand_right` 没有互换；
- 两路处理后均为 RGB 640×480；
- 相机 skew ≤100 ms，state-camera skew ≤50 ms；
- observation bridge 只绑定 loopback，报告零 motor command。

## 3. 夹爪校准

空载、现场持有急停：

1. 读取完全张开时 raw position、status、error；
2. 使用独立夹爪 probe 做小范围往返；
3. 经负责人确认后做空载闭合，记录完全闭合 raw；
4. 确认 raw 是否随闭合增大；
5. 确认 status 0/2/3 在当前 GDK 的实际含义；
6. 验证映射端点为 `-0.785=open`、`0=closed`；
7. 检查重复相同 target 不会反复下发硬件命令。

端点、方向、joint name 或 status 语义与当前实现不同时，同步修改 observation bridge、
gripper state machine/runtime、测试和部署报告，再完成 shadow 与真机检查。

## 4. GDK 控制路径

当前基线要求一个持久 GDK owner。历史上两个 MotionPlan subscriber 会产生 UUID mismatch；
MotionPlan 的亚毫米目标也存在跟踪残差，因此正式策略使用持久 50 Hz
`EndEffectorPoseControl`，每个 10 Hz policy target 插值 5 次。

在新机器上依次验证：

1. 没有竞争 controller；
2. 零位保持无明显冷启动跳变；
3. 0.5 mm 往返跟踪；
4. 10 Hz 连续 waypoint、50 Hz deadline；
5. shutdown 后 mode/error/control ownership 正常；
6. 实测并记录初始 translation/rotation compensation。

任何 UUID mismatch、控制权冲突、非零 error 或持续 deadline miss 都应停止适配。

## 5. 初始位姿和 workspace

每个项目从训练数据统计起点分布，并用目标机器人的只读 live capture 对比。底盘和工装
位置也应纳入检查，避免通过放宽 tolerance 掩盖部署差异。抓取案例的实测坐标记录在案例
Runbook 中。

workspace 同时覆盖机械安全范围和训练轨迹范围。TOML 保存校准结果，控制代码中的硬件
范围负责执行检查；两者不一致时通过代码评审一起更新。

## 6. 配置门禁

```bash
agibot/bin/groot-g2 doctor \
  --robot-config agibot/configs/local/<robot-id>.toml
```

该检查保证字段完整和固定协议一致，但无法替代现场测量。完成 live preflight 后，应把
配置、read-only 报告、夹爪报告、0.5 mm 跟踪报告和审批人一并存档。
