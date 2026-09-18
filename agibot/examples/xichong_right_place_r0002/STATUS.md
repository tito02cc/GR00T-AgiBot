# Xichong right-place r0002: current status

Last updated: 2026-09-09

Operator acceptance: after the second run, the operator explicitly confirmed
that inference was smooth and selected this bridge for future use on
`10.20.15.194`. The robot-specific default and reuse instructions are recorded
in [the 194 bridge record](../../docs/robots/10.20.15.194.md). This supersedes
the old endpoint-handoff default for this machine, not for every robot/GDK.

## Latest hardware results — 2026-09-09: two full runs PASS

The isolated single-owner combined GDK service completed two successive full
model-driven placement runs on `10.20.15.194`. Both returned
`PASS_PLACED_RELEASED_AND_RETRACTED`. This is hardware evidence, not a simulated
pipeline; these were separate runs with an operator-requested close/reset between them.

| Run / report start | Evidence | Final jaw / measured retraction |
|---|---|---|
| First / 14:21:24 CST | [`full_7tou1x/inference.json`](../../local_reports/continuous_control_20260909/full_7tou1x/inference.json) | `-0.780735 rad` / `0.234892051 m` |
| Second / 14:24:41 CST | [`repeat_2/inference.json`](../../local_reports/continuous_control_20260909/repeat_2/inference.json) | `-0.785 rad` / `0.269941170 m` |

- Each run completed three H16 chunks: 48 rows, each with a successful receipt
  and five combined arm/tool ticks. All 96 rows report `native_simultaneous`.
- The same official `Robot.trajectory_tracking_control` request carries EEF
  and native-radian gripper targets. There was no legacy mux process handoff,
  endpoint-triggered arm restart, or recorded bridge fault in either run.
- Physical release feedback was `-0.735982 rad` / `-0.724306 rad` for the two
  runs. Both measured retractions exceeded the runner's `0.13 m` threshold.
- Final bridge status was ready, queue depth 0, `fatal_error=null`, with healthy
  release/completion status. These values describe the recorded runs, not the
  robot's current state after subsequent operator actions.

The operator confirmed that the first run was complete, then requested closing
the gripper, returning to the initial position and repeating. The return's
controller verification reported `0.000026717 m` position error with jaw `0.0`.
Before the second run, the operator received the requested recording notice and
a 5 s countdown. Neither return nor countdown is an automatic phase of the
policy runner. The second robot-side run directory was `repeat_BHx0A3`.
Afterward the operator confirmed the second run was smooth.

These results confirm two complete arm/release/retract executions. **They do not
prove that the workpiece seated precisely, establish repeated-run reliability,
or diagnose the earlier GDK collision/IK behaviour.** Physical placement quality
still needs scene inspection; other starts, payloads, robots and GDK versions
need their own validation. The old uncompensated native failures remain valid
historical results, not failures that have been retroactively reclassified.

## Current entry point: calibrated single-owner service

Use `g2_groot_continuous_action_bridge.py` with its matching continuous sender
and controller, and `run_g2_groot_full_place_inference.py`. Robot-side files are
in `/home/agi/vla_ct/bridges/10.20.15.194/candidates/continuous_20260909/`, with
the parent bridge directory on `PYTHONPATH` for shared modules. Use gripper name
`idx71_gripper_r_inner_joint1` and initial closed command `0.0`. Exact commands
are in [`README.md`](README.md).

The bridge starts in standby. The runner warms the official policy, discards
warmup actions, resets policy state, then explicitly activates the bridge and
obtains a fresh observation for H16 execution. Calibration lasts 2 s and learns
feedback correction from zero, rather than replaying a legacy fixed bias.
Millimetre-scale startup convergence is allowed: the first run's maximum calibration
transient was `1.9991 mm`, converging to `0.0573 mm` position error and about
`0.0095 degrees` rotation error. That is not a claim of zero raw-API startup
motion. Its post-calibration task-reference error was `2.2899 mm / 0.5227 degrees`.

During policy execution, intermediate gripper targets are interpolated alongside
EEF motion. Inference/communication gaps use the same owner's hold ticks, not
tool handoff/restart. Shutdown stops publishing without an extra open command.
Legacy remains available only as an explicit rollback; there is no automatic
fallback, and it must not run alongside the candidate.
Details: [continuous-control findings](../../docs/G2_CONTINUOUS_CONTROL.md).

## History — restored legacy arm and corrected endpoint handoff (2026-09-08)

All sections through "Historical combined API backend" below describe earlier
revisions and observations. Their then-current blockers and deployment statements
are superseded by the 2026-09-09 result above; they remain for diagnosis/rollback.

The original legacy Cartesian controller and paired H1 bridge were restored
from the pre-native backup. Original adaptive compensation and 10 Hz / 50 Hz
execution are restored. The additional joint-hold monitor, zero-seed defaults,
and native backend experiments are archived and are not the active deployment.
Official controller errors still stop publishing; shutdown leaves the gripper
at its last command.

The mux now executes the EEF of the row that triggers a tool change exactly
once, waits for its successful execution receipt, stops the arm child, confirms
the physical tool endpoint, restarts the arm child, and continues with the next
row. It no longer moves the tool before executing that row or replays the row
after restart. This fixes ordering but retains the measured seconds-long
ownership handoff. It does not meet the request for continuous arm/tool control.

The restored controller, H1 bridge, and mux were deployed to `10.20.15.194`.
The previous deployment is recoverable from robot-side
`backups/restore-legacy-20260908-GEFA5u/`; the local experimental files are in
`agibot/local_reports/restore_baseline_20260908_DAkhCn/experimental_local/`.
Focused verification passed 141 tests, including a simulated four-H16 TCP
pipeline. Robot-side compilation and standby startup/exit passed with zero
action commands. There is no post-restoration full hardware rollout result.
The previous observed pose is historical, not a fresh 2026-09-09 measurement.

Before the successful 2026-09-09 run, read-only review of the installed GDK declarations confirmed
that the legacy EEF request has no tool field, while the official trajectory
request can carry right-arm pose and right-tool angle together. At that time,
the next technical issue was the loaded-hold displacement of that combined
API, not model training or a Python sleep. See
[continuous-control findings](../../docs/G2_CONTINUOUS_CONTROL.md).
No hardware commands or firmware changes were made during this review.

## Historical software experiment — after 19:39 CST (superseded)

Default controller seeds are now zero translation and identity rotation, not
the historical fixed bias inherited from earlier testing. Explicit validated
seeds remain supported. Calibration/motion gains and all limits are unchanged.
Initial seed values/sources and separate phase/cumulative joint diagnostics
are recorded. This revision has not been exercised under hardware control.

The 18:45 joint-2 change decomposes into 1.831827 degrees during calibration
and another 0.175129 degrees afterward; the cumulative 2.006956 degrees caused
the bridge limit, not a new GDK error. This does not establish either safe
settling or continuing divergence. Do not silently raise the limit or retry.

Read-only official GDK getters reported right-arm additional payload mass
0 kg, with zero center of mass. The workpiece mass and existing tool-model
calibration need confirmation; no payload setting was changed. The separate
collision-detection configuration getter returned disabled, sensitivity 3,
checkout 5 ms; this does not mean self-collision prediction is disabled, nor
does it explain the earlier `collision imminent`. No collision settings changed.

## Historical hardware retest — 2026-09-08 18:45 CST: FAIL

The updated legacy controller was tested at the current loaded pose, with an
unchanged EEF target: 2 s calibration then a planned 2 s hold. Calibration
completed at 0.303 mm EEF error, but right joint 2 accumulated 2.006956 degrees
from takeover and triggered the joint monitor after 109 publications (~2.18 s).
The hold did not complete. Mean/max dispatch intervals were 20.001/21.152 ms;
no deadline error or new GDK collision warning caused this stop.

Freezing idle compensation has **not eliminated joint drift**. The arm still
uses the original EEF API; this result alone does not establish an SDK defect.
The tool remained closed with error 0 and the workpiece visible. No return,
model action, tool command, or retry followed. Only observation remains active.
Settled EEF XYZ is `[0.476363615,-0.236307986,0.785478248]` m, not the task start.
Evidence: local `agibot/local_reports/xichong_right_place_runtime/`
`runtime_retest_20260908_1844/`. Do not proceed to a full rollout on this result.

## Historical software repair — 18:38 CST (superseded)

The default remains the original legacy EEF controller and placement mux, not
the failed native candidate. The current local repair defers arm activation
until the official GR00T policy has been warmed up: the mux starts in standby,
the runner discards the warmup result, resets policy state, explicitly activates
the arm, and obtains a fresh observation for the first executed H16.

Legacy calibration/motion retain adaptive compensation; idle holding freezes
the existing compensation (`adapt_during_hold=False`). Source-timestamped joint
monitoring covers calibration, idle, and repeated waypoints. It does not limit
normal moving joint changes or cover every near-static noisy policy action.
Shutdown preserves the last gripper command rather than opening automatically.
See [runtime flow](../../docs/G2_RUNTIME_FLOW.md) for the exact boundaries.

The repair/deployment itself sent **no hardware motion or gripper commands**;
the subsequent authorized hold test is recorded above. Five robot-side
scripts were deployed to `10.20.15.194` at 18:38 CST. Python 3.10 compilation,
read-only legacy preflight, and 26 source-timestamped joint samples passed.
The local focused suites passed 246 tests, including legacy/native simulated
four-H16 TCP pipelines and full-size status receipts. The change
mitigates unintended idle behavior; it is not a proven cure for GDK IK or the
unresolved collision prediction. No new end-to-end hardware PASS is claimed.

The previous robot scripts are recoverable from
`/home/agi/vla_ct/bridges/10.20.15.194/backups/runtime-flow-20260908-YOdplM/`.
The 18:38 read-only EEF was `[0.476440967,-0.236294628,0.785458762]` m,
not the stored task initial pose; the pose changed since the earlier attempt
without this repair sending motion. Right-jaw feedback is `0.0` rad, error 0.
Only the observation bridge remains running. These readings do not establish
loaded-control stability or clearance for a rollout.

Official logs locate joint drift before the first policy action: joint 2 was
`-1.6467387 rad` at 18:00:30 and `-1.7059644 rad` at 18:00:40.337, before the
runner started at 18:00:40.9738 and its 1.58585 s first inference. EEF-only
holding can therefore change the joint configuration without model commands.
The legacy `EndEffectorPoseControl` does not carry joint/link3 references;
`set_reference_frame_poses` does not change that legacy path. The precise GDK
collision root cause and affected links have not been established.

## Hardware history — observations before the successful single-owner run

Historical physical attempt (2026-09-08 18:00 CST): after the user reported adjusting
the arm posture, one full legacy rollout was attempted. Startup calibration and
initial checks passed. H0-H2 received queued ACKs; H3 was rejected following
GDK `motion control error=2: collision imminent`. No complete H16, release,
gripper handoff, or retract completed. All action children exited; jaw feedback
remained 0.0 and images showed the workpiece held. No retry or reset followed.
The repeated hardware warning remains unresolved; preserve the original arm
interface and diagnose the collision-related configuration before another rollout.
The then-current EEF was `[0.502680307,-0.176833352,1.055963603]` m (18:01:17 sample),
not the exact 17:42 returned state recorded below.

Historical return (2026-09-08 17:42 CST): the user-confirmed loaded return passed
using the original legacy EEF controller. Post-exit error was 0.0899 mm /
0.0347 degrees; jaw feedback stayed 0.0 and the workpiece remained held. No
gripper command or new inference rollout was sent. A later read-only check
reported motion mode 1 and motion error 0; this does not resolve the collision
prediction from the 17:27 policy run below. Full-path readiness is not established.

Earlier full legacy attempt (2026-09-08 17:27 CST): startup calibration and
model/bridge preflight passed; the first H16 was interrupted by official GDK
`motion control error=2: collision imminent`. Five waypoint requests were
accepted; the sixth was rejected after the worker fault. No full-chunk receipt,
release, gripper ownership handoff, or retraction completed. Jaw feedback stayed
0.0 and the workpiece remained held. No automatic retry/reset was sent. The
control children have exited. This is not a native-backend test; the earlier
legacy return success does not certify the whole policy trajectory.

Historical state (2026-09-08 17:21 CST): the user-requested return to the
task initial pose **passed using the original legacy EEF controller**. After
control exited, error was 0.144 mm / 0.044 degrees; jaw feedback remained 0.0
and images showed the workpiece held. No gripper command or inference was sent.
Only the observation bridge remains active. Subsequent work should preserve the
working legacy arm path and focus on arm/tool handoff; the failed native path
below is not the operational baseline.

Native retest (2026-09-08 17:09 CST): **FAIL after reference initialization**.
All 7 requests held the same initial EEF and closed jaw; after 0.140445 s the
EEF had moved down 1.513 mm, ending further dispatch before any upward target.
Later feedback settled around 2.822 mm total offset / 2.584 mm down / 0.688°.
The tool remained closed and the workpiece remained held. No model rollout or
reset was sent. Fixing the reference initialization did not eliminate the raw
API startup drift; root cause remains unconfirmed. Only the observation bridge
remains active. Do not start full native inference or assume the old start pose.

The following repair verification describes the state **before this retest**:

Earlier software repair (2026-09-08, after the failed probe): native startup now
initializes the official link3 IK references from live TF using
`Robot.set_reference_frame_poses`. The prior bridge left the SDK's fixed
references unchanged (the right reference differed from the measured link3
orientation by about 83 degrees). Startup gripper checks now precede every
publication, including calibration; expired ticks and invalid final calibration
feedback are rejected before a false completion. These are software repairs,
not evidence that the earlier physical drift has disappeared. No post-repair
motion test or full rollout has been performed. The workpiece must remain held
until the next supervised test is prepared.

Post-repair verification: 208 focused software tests passed, including the
production startup ordering and four-H16 native TCP pipeline with simulated
feedback. Four updated scripts were deployed to `10.20.15.194`; Python 3.10
compilation, read-only preflight, and an official reference-setter-only call
passed (`trajectory_calls=0`, closed-jaw feedback, no tool errors). The previous
scripts are in robot-side `backups/native-startup-20260908-qQFgPn/`. Only the
existing observation bridge remains running; no controller was left active.

Native-API hardware check (2026-09-08 16:13 CST): a loaded closed-jaw,
uncompensated zero-hold probe failed after 6 requests / 0.12 s due to uncommanded
downward drift. The tool stayed closed at 0 rad; images confirmed the workpiece
remained held. Settled offset was about 4.74 mm (4.14 mm downward, 2.18 degrees).
No upward/return waypoint, jaw-open command, or model rollout was sent. The arm
was not restored automatically. GDK logs confirmed receipt and activation of
`Right_Arm|Right_Tool`, so native message delivery is established, not tracking
accuracy or moving-gripper performance. The calibrated production backend has
not been tested on hardware. Do not start a full native rollout on this basis.

The data, GR00T conversion, training run, and 30k checkpoint have passed their
offline checks. On hardware, arm motion, physical gripper release, and the
model-directed retract have each succeeded in staged/resumed runs.

As of 2026-09-08, the task had **not passed one uninterrupted end-to-end hardware run** from
the training-derived initial pose through placement, release, and retract. The
latest continuous attempt was stopped during its first H16 action chunk by the
GDK collision predictor (`motion control error=2: collision imminent`), before
the release phase. Do not describe this task as commissioned until the
acceptance criteria below are met.

On 2026-09-07, two stationary-arm, empty-jaw hardware tests completed the tool
release and arm ownership handoff. Those tests do not demonstrate a complete
model-driven placement. The 2026-09-08 software repairs below have not yet been
validated in a fresh complete hardware placement run.

## Historical continuity repairs (2026-09-08)

- Wait for outstanding arm execution receipts before relinquishing the arm;
  preserve receipts and reject duplicate IDs across child restarts.
- Execute opening and closing endpoints from current tool feedback; explicitly
  report deferred partial gripper targets in the endpoint handoff protocol.
- Complete tool travel before restarting the arm; report no-motion timeouts
  with actual position, without inventing a completed closure.
- Resume 10 Hz submissions after a blocking handoff without a catch-up burst.
- Isolate client disconnects so child services survive a lost response; never
  automatically repeat an ambiguous motor command.
- Require every H16 execution receipt and physical release evidence for task
  completion; preserve failed-cycle targets and original error diagnostics.

Software verification: 108 focused tests pass, including a synthetic four-H16
TCP pipeline with 64 exact EEF actions and receipts through close/open handoffs.
The GDK process switch still entails a measured approximately 10-second pause;
this is an automatically resumed session, not simultaneous arm/tool streaming.

## Historical combined API backend (2026-09-08)

An opt-in `native_trajectory` backend now sends right-arm EEF and right-tool
native joint angle in the same `Robot.trajectory_tracking_control()` request.
It retains 10 Hz model waypoints and interpolates both channels at 50 Hz, with
one persistent GDK owner and no subprocess handoff. The placement runner also
recognizes physical release feedback from this backend.

This is implemented software, **not yet hardware-validated as a complete backend**.
The raw-API zero-hold hardware failure is recorded above. The legacy mux
remains available; do not run both at once. See
[`G2_NATIVE_TRAJECTORY.md`](../../docs/G2_NATIVE_TRAJECTORY.md) for the exact
interface, candidate startup command, and verification boundary. Neither GDK
drivers nor arbitration configuration were changed.

Before the later startup repair, the focused suite had 156 passing tests, including the native
four-H16 TCP path through the production interpolation controller and simulated
feedback. The five bridge/controller files were deployed; robot Python 3.10
compilation and native-backend read-only startup passed. No physical command
was sent during this update.

## Validated assets

- Raw dataset: 2011 episodes; 1791 eligible after cleaning.
- Training split: 600 episodes / 45,343 frames.
- Held-out split: 100 episodes / 7,434 frames, with no overlap with training.
- Conversion audit: 700/700 selected episodes valid, LeRobot v2.1, 10 Hz,
  right EEF XYZ + Rot6D and absolute right-gripper action.
- Training: GR00T N1.7 3B, one A100 80 GB, 30,000 steps; checkpoints every
  5,000 steps, retaining 25k and 30k.
- 30k checkpoint: transferred locally and checked for file integrity and CUDA
  loading.

## Active robot architecture

Both latest verified runs used this path:

```text
GR00T placement runner -> continuous action bridge
                         -> one Robot / one 50 Hz control worker
                         -> combined right-arm EEF + right-tool target
```

The read-only observation bridge supplies head/wrist images, EEF and gripper
feedback separately; it sends no control commands. The old placement mux with
arm/tool child processes is a manually selected rollback path only. Its measured
approximately 10 s endpoint handoff is not part of the new path.

## Remaining validation

Inspect the physical placement and final scene, then repeat the same recorded
startup and full-run procedure to measure success rate and final motion quality.
Do not equate two program PASS results with long-term reliability, production
commissioning or generalisation.
Keep any new failure reports, including controller/GDK diagnostics; do not
suppress errors or bypass robot protection to preserve a PASS label.

If rollback is required, stop the single-owner service first and explicitly
select the matched legacy mux/controller revision documented in
[the legacy runtime flow](../../docs/G2_RUNTIME_FLOW.md). Do not mix backends
or let a failed run automatically switch control owners or replay actions.
