# G2 机械臂与夹爪联合调用

> 历史研究记录：用户要求恢复原 legacy 运动路径后，本文相关实验脚本已从正式目录
> 归档退出，不再提供当前可运行入口。以下保留的是当时的接口研究与失败证据；
> 当前操作只参考 [G2 推理运行流程](G2_RUNTIME_FLOW.md)。

2026-09-08 17:09 最新结果：补齐参考姿态后的持件原地保持复测仍未通过；
7 个保持命令、约 0.14 秒后下移 1.513 mm，停止后下移约 2.584 mm。夹爪始终闭合，
没有进入抬升或完整推理。参考初始化修复未消除物理偏移，具体根因仍待核实。
完整后端仍未真机验收，不能直接启动完整推理。默认入口仍保留旧后端，
不能把软件测试通过或 GDK 返回 `0` 当作机械臂和夹爪实际执行成功。详见案例 STATUS。

当前正式放置 runner 还要求 `explicit_activate_v1` 延迟激活协议；native H1 bridge
没有实现该协议，不能与该 runner 直接配套使用。本页仅保留候选接口与历史验证记录，
不提供启动完整 native 推理的命令。当前正式流程见 [G2 推理运行流程](G2_RUNTIME_FLOW.md)。

## 接口与数据

使用机器人已安装的 GDK 3.3.8 Python 接口，不改驱动、不自行实现 IK、不直接发布 WBC
消息，也不修改仲裁优先级：

```python
# 初始化一次，再开始轨迹流。具体实现使用下文 NativeTrajectorySender，
# 读取 TF、检查 XYZW 后调用官方 setter；不要把 EEF 姿态当成 link3 姿态。
left = tf.get_tf_from_base_link("arm_l_link3").rotation
right = tf.get_tf_from_base_link("arm_r_link3").rotation
robot.set_reference_frame_poses(
    left.x, left.y, left.z, left.w,
    right.x, right.y, right.z, right.w,
)

robot.trajectory_tracking_control(
    0, {},
    [{
        "right_arm": {
            "control_type": "ABS_POSE",
            "action_data": [x, y, z, qx, qy, qz, qw],
        },
        "right_effector": {
            "control_type": "ABS_JOINT",
            "action_data": [gripper_radians],
        },
    }],
    robot_link="base_link",
    trajectory_reference_time=0.02,
)
```

以上是调用格式，不是可以直接执行的机器人目标。位置单位 m，四元数顺序 XYZW，当前
omnipicker 的关节角为原生弧度，`-0.785` 张开、`0` 闭合；不转成百分比或二值状态。
其他机器人必须重新核对反馈编码和端点。

### 启动参考姿态修复

当前 GDK 的 `ABS_POSE` 除末端目标外，还带有用于 IK 多解选择的参考姿态。
右臂目标是 `arm_r_end_link`，但参考连杆是 `arm_r_link3`；左侧对应
`arm_l_link3`。不能把右臂 EEF 四元数传给参考姿态接口。

2026-09-08 只读检查机器人 `10.20.15.194` 上安装的官方 adapter 确认：
`Robot::Init` 使用固定参考四元数，而非当前 TF；右侧默认 XYZW 为
`[-0.777, -0.033, 0.016, 0.628]`。与当时实测右 link3 姿态的旋转差约 83°。
这是之前桥接遗漏初始化的依据；后续实测确认补上初始化仍会下移，不能把它当作已解决。

`NativeTrajectorySender.initialize_references(robot, tf)` 现在在首次轨迹发送前，
从 `base_link` 读取两侧 link3 的当前四元数，调用官方
`set_reference_frame_poses`。该 setter 只更新此 Robot 实例的参考参数，不发运动命令。
这些参考在一个会话内固定，不随模型输出反复改写。初始化失败、缺少 TF、换 Robot
实例均不能继续发送；状态记录实际参考值。当前只发布右臂与右夹爪，读取左侧参考并不
意味着控制左臂。换 GDK 版本时要重新确认参考连杆语义。

启动流程为：读取状态 → 初始化参考 → 创建反馈 worker → 标定 → 同一 worker 连续控制。
标定和运行的每个发送周期都会先读取夹爪反馈；启动故障、非有限 TF、已过期的周期不会
多发一条命令。标定失败的清理不再因线程未启动而覆盖原始错误。这些修改不改变模型
动作的单位、频率、开闭端点或中间开度。

依据来自当前机器人随 GDK 安装的 `build_dep/python/pybind/cpp/bind_robot.cpp`
292–381 行、`build_dep/cpp/aarch64/include/common/types.h` 的 A2D 结构，以及固件
`tools/model_predict_demo.py` 的单点实时轨迹示例。Python 绑定实际忽略前两个参数
`infer_timestamp`、`robot_states`；桥接自行检查命令时间，不能依赖这两个参数同步状态。

模型仍按 10 Hz 输出 H16。沿用现有 EEF 解码与 50 Hz 插值，同一个 20 ms 请求里包含
EEF 和夹爪：每个模型 waypoint 对应 5 次联合调用，夹爪在这 5 次调用中线性插值到原始
目标。部分开度不会被丢弃，也没有端点切换或开闭等待。只校正浮点数在物理端点处的
微小越界。模型、归一化、Rot6D 解码和动作块执行顺序不变。

## 代码入口

- `agibot/robot/g2_groot_native_trajectory.py`：构造联合请求，无独立 GDK 生命周期。
- `agibot/robot/g2_groot_native_control_worker.py`：复用持久 worker，读取夹爪反馈，
  不进入旧夹爪事务状态机，不启动或重启子进程。
- `agibot/robot/g2_groot_persistent_h1_action_bridge.py`：新增
  `--control-backend native_trajectory`。原默认 `legacy_cartesian` 未改变。
- `agibot/scripts/run_g2_groot_full_place_inference.py`：保留解析 native 物理释放事件的
  辅助逻辑，但正式入口要求 `explicit_activate_v1` 的 standby/activate 生命周期。
  native H1 bridge 缺少此生命周期，不能因辅助逻辑存在就认为正式入口已兼容。

新后端必须在启动标定之前选定。同一个会话不调用 `end_effector_pose_control`、
`move_ee_pos` 或 `move_end_effector_joint`。当前 native 使用
`compensation_policy="none"`：不沿用 legacy 偏置，也不运行 legacy 自适应补偿。
其中的标定过程只检查保持结果，不学习补偿。**启动控制和标定会下发真实命令**，
不是只读检查；这条候选路径的稳定保持仍未验收。

夹着工件启动时，实际关节反馈不一定等于之前的闭合命令。已确认此前是 `0.0` 闭合时，
指定 `--initial-gripper-command 0.0` 保持该命令，不用受工件接触影响的开度代替夹持目标。
这不会修改后续模型输出，也不是强制闭合整场推理的开关。

## 只读检查示例

这是已核对原生弧度反馈机器人的候选配置；在机器人桥接目录执行。不能与旧 mux/arm/
gripper daemon 同时运行。只读 observation bridge可以保留。

仅检查接口和状态，不启动控制：

```bash
python3 g2_groot_persistent_h1_action_bridge.py \
  --control-backend native_trajectory \
  --required-motion-mode 1 \
  --gripper-joint-name idx71_gripper_r_inner_joint1 \
  --gripper-feedback-encoding native_radians
```

不要给这条检查命令添加 `--enable-control` 后直接连接正式放置 runner。native bridge
会先启动标定/持位，而 runner 随后因缺少延迟激活协议拒绝连接，不能实现“模型就绪后
才激活机械臂”。本轮不实现新的 native 激活后端，也不通过删除 runner 的协议检查
绕过这个不兼容。

旧 mux 的 `set_gripper` 专属命令也不能发给本后端。候选 bridge 的联合动作操作是
`execute_h1_gripper`；这只是协议说明，不是已验收的部署入口。后续若继续适配，应先
解决 raw hold 行为与生命周期兼容性，再记录独立真机验收；不能直接重用案例的正式
runner 命令。

## 验证边界

持件测试仅发送 6 次“当前 EEF＋夹爪 `0.0`”请求，在约 0.12 秒时因实际下移 1.94 mm
停止继续发命令，未执行计划中的上移或返回。控制器退出瞬态后总偏移约 4.74 mm、下移
4.14 mm；停止发布不是硬件急停。夹爪反馈始终 `0.0`，图像显示工件仍被夹持。
运动控制日志确认 `Right_Arm|Right_Tool` 联合激活。因此本次不是“消息没有送达”，而是
原始轨迹路径的零位保持未达标；具体原因尚未确定，不能直接套用固定偏移或增大允许误差。
该独立测试没有运行完整 H1 bridge 的启动、监测与任务生命周期，不能等同于完整后端
验收。当前 native 已明确禁用 legacy 补偿，不能再用“完整后端会补偿”解释此次失败。

软件测试检查联合请求字段、单位、5 tick 插值、完整 H16、回执、连接和错误传播。
执行回执表示该 waypoint 的 GDK 发布完成并附带实际反馈，不表示两种执行器已精确到位。
夹爪未实际张开时，runner 不会因此判定放置成功。

真机仍要确认：当前模式下联合请求确实被执行、机械臂跟踪正常、夹爪在机械臂运动时
响应，并且连续完整推理没有原来的长停顿。当前机器人日志已有 `/gdk/model_predict`
订阅连接，但这不是执行成功证据。此前其他机器人出现过仲裁丢弃、返回 `0` 不运动的
情况，不能照搬为本机结论，也不能据此绕过仲裁。若官方接口在本机不驱动夹爪，应记录
反馈交由 GDK 支持方核对，而不是修改驱动或静默切回多个相互竞争的控制路径。

退出时保留夹爪最后命令，不自动张开。GDK 报错或结果不明确时停止进一步发送，不自动
重发动作；这不等于硬件急停，现场急停仍由机器人提供。
