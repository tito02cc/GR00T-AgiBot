# Xichong right-place r0002: inference start reference

## 数据、模型与运行入口

环境安装、数据处理和训练步骤见 [操作手册](../../README.md)。
本任务原始数据为 `agibot/data/xichong_right_place_r0002_job01/`，训练集为
`agibot/gr00t_data/xichong_right_place_r0002_train_600/`，模型包为
`agibot/models/xichong_rplace_r0002_n1d7_checkpoint-30000/`。
本任务为放置任务，与早期 `single_grasp` 抓取任务使用不同的数据和模型。

日常推理统一使用 [10.20.15.194 机器人两任务固定包](../../deployments/g2_194/README.md)。
以下记录初始参考、历史运行命令和实测结果。

> **Latest hardware results (2026-09-09):** the single-owner
> `g2_groot_continuous_action_bridge.py` completed two successive full runs.
> Each returned `PASS_PLACED_RELEASED_AND_RETRACTED` after three H16 chunks
> (48 rows), with physical gripper opening and 234.892 / 269.941 mm retraction.
> Each row executed five combined GDK ticks, without the old mux's process
> handoff or a recorded bridge fault. The operator confirmed the first run was
> complete; the second followed an explicitly requested close/reset.
> These are two successful hardware runs, not proof of precise workpiece seating
> or long-term reliability. See [`STATUS.md`](STATUS.md) and
> [continuous-control details](../../docs/G2_CONTINUOUS_CONTROL.md).

The operator subsequently confirmed the second run was smooth and selected this
bridge for future use on robot 10.20.15.194. See the [robot-specific record](../../docs/robots/10.20.15.194.md)
for the default implementation, exact deployment path and task-specific boundaries.

The initial pose is derived from the first observation of all 600 selected
training episodes. It is a task-space pose in `base_link`, not a joint-space
home command.

Use converted episode 258 (raw episode 000855) as the executable reference
because it is an actual demonstration and lies nearest the robust center of
the training starts:

```text
right EEF XYZ + quaternion XYZW
[0.5013904572, -0.1765020341, 1.0539140701,
 0.5225321067, -0.0018789451, 0.8526121988, 0.0030175206]

right gripper
-0.0057645165   # closed, holding the workpiece
```

The machine-readable source and statistics are in
`initial_pose_reference.json`.

## Start gate

Full inference may begin only when all of these are true:

1. The right EEF is within 5 mm and 2 degrees of the reference.
2. Right-gripper feedback is within 0.03 of the reference and the workpiece is
   physically held. This task starts closed and ends by opening the gripper.
3. The head and right-wrist views contain the placement workstation and match
   the demonstrated camera orientation; both policy images are RGB 640x480
   after aspect-preserving letterbox.
4. GDK mode is the IP-specific configured mode, motion/arm/end errors are zero,
   no estop is active, and one persistent bridge is the only custom owner.
5. Camera-pair skew is at most 100 ms and state/camera skew is at most 50 ms.
6. Use the matched single-owner sender/controller/service. Activation performs
   2 s of feedback calibration from zero correction; millimetre-scale startup
   convergence is allowed, not interpreted as an exact raw zero-hold. Check the
   final calibration result. A matching EEF does not prove a matching joint
   configuration or diagnose the historical legacy collision warnings.
7. The GR00T N1.7 server is loaded with the r0002 checkpoint and its official
   modality configuration; the placement runner executes every H16 chunk via
   the persistent GDK bridge at 10 Hz / 50 Hz.

Do not use the older grasp/lift runner for this task: it assumes an initially
open gripper and declares success after closure and lift. Use
`agibot/scripts/run_g2_groot_full_place_inference.py`. It requires a closed
gripper holding the workpiece at startup and executes every decoded H16 EEF row.
The single-owner service sends right-arm EEF and native-radian gripper commands
in the same official `Robot.trajectory_tracking_control` request. Each 10 Hz
waypoint uses five 50 Hz ticks, including intermediate gripper openings. It does
not stop an arm process, wait for a tool endpoint, and restart the arm. The
controller retains feedback correction for EEF tracking; the GDK payload is not
an uncorrected copy of the model pose. The runner ends after confirmed physical
opening and retraction of at least 130 mm from the measured release pose. The
130 mm completion threshold is below
the minimum 140 mm demonstrated final retraction, and is checked only after a
complete H16 chunk; it does not clip, replace, smooth, or skip model actions.

## Runtime commands

The current runner uses the bridge's `explicit_activate_v1` protocol; deploy
the matched versions together. Starting the service leaves it in read-only standby.
The runner checks model connection/modality and calls official `get_action`
using a live observation while the arm is inactive, discards that warmup action,
then calls `policy.reset` (not a robot reset). Only then does it activate the arm
and obtain a fresh observation for the actual H16 loop.

Activation initializes the live link3 reference orientations and calibrates
the combined backend from zero feedback correction, then keeps one Robot/TF
instance and one control worker alive. `--initial-gripper-command 0.0` holds the
already-gripped workpiece during startup; subsequent model targets control the
gripper, including partial openings. Do not activate manually before the runner.
Official GDK faults still stop publication; there is no automatic legacy fallback.

2026-09-14: The two task-specific bridge packages for robot 10.20.15.194 are now pinned. For daily
inference use the [pinned package commands](../../deployments/g2_194/README.md),
which select this task's successful `continuous_20260909` bridge and explicit
parameters. Commands below are historical reproduction notes, not the pinned entry.

Start the checkpoint server on the inference workstation:

```bash
cd /path/to/Isaac-GR00T
.venv/bin/python -u gr00t/eval/run_gr00t_server.py \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path agibot/models/xichong_rplace_r0002_n1d7_checkpoint-30000/model \
  --modality-config-path agibot/configs/xichong_right_place_r0002_config.py \
  --device cuda:0 --host 127.0.0.1 --port 5564
```

On the G2, start the read-only observation bridge if it is not already running:

```bash
source /home/agi/app/env.sh
cd /home/agi/vla_ct/bridges/10.20.15.194
python3 -u g2_groot_right_observation_bridge.py \
  --bind-host 127.0.0.1 --port 9100 \
  --gripper-joint-name idx71_gripper_r_inner_joint1 \
  --gripper-feedback-encoding native_radians \
  --model-input-jpeg-quality 92 \
  --camera-timeout-ms 500 \
  --max-camera-skew-ms 100 --max-state-camera-skew-ms 50
```

In another G2 terminal, start the single-owner action service used in this run:

```bash
source /home/agi/app/env.sh
export PYTHONPATH="/home/agi/vla_ct/bridges/10.20.15.194:${PYTHONPATH:-}"
cd /home/agi/vla_ct/bridges/10.20.15.194/candidates/continuous_20260909
python3 -u g2_groot_continuous_action_bridge.py \
  --enable-control \
  --initial-gripper-command 0.0 \
  --gripper-joint-name idx71_gripper_r_inner_joint1 \
  --required-motion-mode 1 \
  --calibration-duration-s 2.0 \
  --workspace-min 0.4172 -0.2392 0.9750 \
  --workspace-max 0.8403 -0.1031 1.2674 \
  --bind-host 127.0.0.1 --port 9200 \
  --session-limit-s 1800
```

The parent bridge directory supplies the shared modules through `PYTHONPATH`;
the candidate directory supplies the matched continuous sender/controller/service.
The GDK environment and paths above belong to `10.20.15.194`. Do not run the old
mux, an arm child, or a separate tool daemon alongside this service. The service
has no standalone `set_gripper` operation: prepare the loaded initial state
separately before the policy session; do not automatically release a held item.

On the workstation, establish the forwards if they are not already running:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 19100:127.0.0.1:9100 \
  -L 19200:127.0.0.1:9200 \
  agi@10.20.15.194
```

With the model and both forwards ready, run the complete policy loop:

```bash
cd /path/to/Isaac-GR00T
.venv/bin/python -u agibot/scripts/run_g2_groot_full_place_inference.py \
  --execute \
  --confirm EXECUTE_G2_GROOT_FULL_RIGHT_ARM_PLACE_INFERENCE \
  --observation-port 19100 --action-port 19200 --model-port 5564 \
  --max-cycles 0 \
  --report "agibot/reports/xichong_rplace_continuous_$(date +%Y%m%d_%H%M%S).json"
```

`--max-cycles 0` means there is no artificial policy-cycle cap. The operator
or hardware stop remains authoritative. The action bridge's workspace is the
demonstrated task envelope plus 5 mm, not a phase gate.

Normal shutdown or failure cleanup stops control publishing without an extra
gripper-open command. Model-requested release still occurs inside the task;
returning the arm or manually opening a held workpiece is a separate action.
Stopping publication is not an instantaneous hardware stop.

## Result and legacy fallback

Both 2026-09-09 reports record 48 completed rows, five ticks per row,
`native_simultaneous` execution and no bridge fault:

| Run | Evidence | Final jaw / retraction |
|---|---|---|
| First | [`full_7tou1x/inference.json`](../../local_reports/continuous_control_20260909/full_7tou1x/inference.json) | `-0.780735 rad` / `0.234892 m` |
| Second | [`repeat_2/inference.json`](../../local_reports/continuous_control_20260909/repeat_2/inference.json) | `-0.785 rad` / `0.269941 m` |

The operator confirmed the first run was complete and requested closing/resetting
before the second. The second run was announced and delayed 5 s for the requested
recording; that was operator preparation, not an added pause at gripper release
or an automatic countdown in the runner command above. Program PASS is not an
inspection of seating accuracy, grip quality, long-term reliability, or the
absence of all transient motion.

The original `g2_groot_place_action_mux.py` plus legacy arm child and gripper
daemon remain an explicit rollback option, not the current startup command and
not an automatic fallback. Their endpoint handoff has a measured approximately
10 s pause; partial openings were deferred until an endpoint. The old raw-native
hold experiments also remain historical failures, distinct from this calibrated
single-owner run. For rollback history see [`STATUS.md`](STATUS.md) and
[the legacy runtime flow](../../docs/G2_RUNTIME_FLOW.md); stop the candidate
before intentionally starting another control owner.
