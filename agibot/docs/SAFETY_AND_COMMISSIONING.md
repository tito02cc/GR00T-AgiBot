# 安全和真机准入

本文命令对应当前已验证的右臂 bridge。task profile 选择左臂时，使用同等检查要求的左臂
bridge、夹爪配置和工作空间。

## 不可跳过的顺序

```text
软件测试
→ 数据/模型 SHA
→ G2 配置门禁
→ observation-only
→ 夹爪独立校准
→ 右臂低风险跟踪
→ shadow 模型调用
→ 完整 H16 真机推理
→ recovery
```

前一步失败时不得通过放宽阈值进入下一步。

## 机器人端 observation bridge

部署目录建议 `/home/agi/vla_ct/groot_right_arm_shadow`。加载机器人环境后：

```bash
python3 -u g2_groot_right_observation_bridge.py \
  --bind-host 127.0.0.1 \
  --port 9100 \
  --camera-timeout-ms 500 \
  --max-camera-skew-ms 100 \
  --max-state-camera-skew-ms 50 \
  --model-input-jpeg-quality 92
```

## 机器人端 action bridge

只有现场确认后启动：

```bash
python3 -u g2_groot_persistent_h1_action_bridge.py \
  --enable-control \
  --enable-gripper \
  --enable-approach-session \
  --enable-protected-close \
  --confirm ENABLE_G2_GROOT_RIGHT_ARM_SINGLE_PROTECTED_CLOSE_SESSION \
  --session-limit-s 1800
```

该 bridge 必须只绑定 `127.0.0.1:9200`。当前右臂实例不接收左臂、头、腰和底盘字段。

## SSH 转发

使用 SSH key 或交互密码，凭据不得写入仓库：

```bash
ssh -N \
  -L 19100:127.0.0.1:9100 \
  -L 19200:127.0.0.1:9200 \
  <robot-user>@<robot-host>
```

## 现场检查

- 工作空间无人员和障碍，急停已验证；
- 左臂和左夹爪保持静止；
- 右夹爪张开；
- EEF、底盘、架子、工件和图像接近训练分布；
- 网络无明显丢包或数百毫秒抖动；
- observation/action/model 三个服务均为持久进程；
- 现场人员全程观察，不进行无人值守推理。

## 完整推理语义

当前 G2 右 EEF 协议每次获取 16×8 action，依次执行全部 16 个 10 Hz waypoint，再根据新
观测请求下一个 chunk。夹爪采用 official decoder 的绝对值，不加入阈值或强制开爪。
何时判定任务完成由 task profile 的 inference runner 定义；放置、插接等任务配置各自的
完成条件。

硬件合理性检查仍必须保留：有限数值、协议 shape、工作空间、机器人 health、唯一控制
owner、命令新鲜度和 bridge session 生命周期。这些不是任务阶段编程。

## 停止条件

出现以下任一情况立即中止/急停：

- 人员或障碍进入工作空间；
- 图像冻结、相机身份错误或观测过期；
- 非预期大幅跳变或接近机械边界；
- GDK error、UUID mismatch、竞争 controller；
- 网络严重丢包；
- 夹爪方向与命令不一致；
- bridge/report 状态与现场运动不一致。

## 完成与复位

runner PASS 后 action bridge 保持抓取姿态，不会自动张开。由现场确认可以释放后再执行
`groot-g2 recover --execute`。复位报告必须确认：夹爪张开、EEF 回训练起点容差内、GDK
mode/error 正常且控制权释放。
