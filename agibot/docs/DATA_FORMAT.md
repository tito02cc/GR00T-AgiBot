# 原始数据与 GR00T 转换协议

以下字段以已经验证的右臂任务为例。机械臂侧别由 task profile 决定；左臂任务保持相同的
时间、位姿、动作和图像语义，并在 converter、schema 和 modality 中使用对应左臂字段。

## 原始 episode 布局

每条成功轨迹必须位于 `episode_NNNNNN/`，至少包含：

```text
episode_000000/
├── arrays.npz
├── frames.jsonl
├── meta_info.json
├── quality_report.json
└── images/
    ├── head_color_000000.jpg
    └── hand_right_000000.jpg
```

`frames.jsonl` 每行对应一个 10 Hz 控制帧。转换器真正依赖的关键字段：

| 字段 | 语义 |
|---|---|
| `frame_index` | 从 0 连续递增 |
| `timestamp_monotonic` | 严格递增 |
| `pose_frame` | `base_link_tf` |
| `action_mode` | `next_delta` |
| `prompt` | 非空任务描述 |
| `right_ee_pose` | XYZ + 四元数 XYZW，共 7D |
| `right_gripper.position` | 当前训练空间夹爪绝对值 |
| `action_right_7d` | 世界/基座系 delta XYZ + delta rotvec + 下一步绝对夹爪 |
| `next_right_ee_pose` | 下一步绝对 XYZ + XYZW |
| `next_right_gripper` | 下一步绝对夹爪值 |
| `images.head_color` | 当前头部 JPEG 相对路径 |
| `images.hand_right` | 当前右腕 JPEG 相对路径 |
| `image_meta` | 相机和状态同步时间 |

JPEG 必须是 RGB、640×480，并与 frame index 一一对应。左腕图像可以存在，但当前公共
右臂 modality 不会写入训练数据。

`arrays.npz` 必须包含：

```text
states                 [T, 16]
actions                [T, 14]
ee_poses               [T, 14]
grippers               [T, 2]
timestamps_monotonic   [T]
```

`meta_info.json` 必须记录 success/failure、data validation、`pose_frame=base_link_tf` 和
`action_mode=next_delta`。`quality_report.json` 必须通过 collector contract；任务末段的
成功语义由各 task adapter 定义，不能沿用其他案例的抓取/抬升条件。

## 动作重建

原始动作不是 GR00T 已经处理好的相对 EEF：

```text
target_xyz = current_xyz + delta_xyz_world
R_target   = Exp(delta_rotvec_world) @ R_current
gripper    = next absolute gripper
```

转换器先重建绝对目标位姿，再保存 XYZ+Rot6D。GR00T processor 根据当前 state 生成训练所需
的相对 EEF action。禁止把原始 delta 直接标成 `RELATIVE`，否则会二次做差。

Rot6D 为旋转矩阵前两行：

```text
r00 r01 r02 r10 r11 r12
```

## 转换输出

输出遵循 LeRobot v2：Parquet 保存状态/动作，`head_color` 与 `hand_right` 编成
640×480、10 fps、H.264 MP4。H.264 是预期有损存储。通用转换验证器默认全量视频解码、
首/中/尾源图像 PSNR 抽查；新增右臂放置流水线启用 `--all-source-frames`，逐帧比较源 JPEG。
同时检查帧数、时间戳、统计量和动作重建。PSNR 不能证明模型效果或物理任务成功。

## 新数据准入

同样格式不代表同样语义。新数据必须重新运行：

1. 原始结构和全部 JPEG 解码；
2. delta→绝对位姿重建；
3. 同步偏差门禁；
4. 多样性选择；
5. stats；
6. 视频/source correspondence；
7. 任务语义检查：抓取任务检查闭合/抬升，放置任务检查初始夹持、释放及回撤，另做视觉审核；
8. official processor round trip；
9. G2 shadow adapter。

统一命令见 `agibot/bin/groot-g2 prepare --help`。
