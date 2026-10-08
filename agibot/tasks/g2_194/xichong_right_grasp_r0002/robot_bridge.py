#!/usr/bin/env python3
"""Task launcher around the pinned 194 GDK runtime; default is read-only standby."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('standby', 'control', 'observation'), nargs='?', default='standby')
    parser.add_argument('--confirm-scene', action='store_true', help='Onsite confirmation of current scene and candidate workspace; never disables protection')
    args = parser.parse_args()
    profile = json.loads(Path(__file__).with_name('inference.json').read_text())
    runtime = Path(profile['robot_runtime'])
    if args.mode == 'control' and not args.confirm_scene:
        parser.error('Control requires onsite scene/workspace confirmation; no automatic reset or gripper motion')
    if not runtime.is_dir():
        parser.error(f'Missing pinned runtime: {runtime}')
    sys.path.insert(0, str(runtime))
    if args.mode == 'observation':
        os.execv(sys.executable, [sys.executable, '-u', str(Path(__file__).with_name('observation_bridge.py')),
                 '--bind-host', '127.0.0.1', '--port', str(profile['robot_observation_port']),
                 '--gripper-joint-name', profile['gripper_joint_name'],
                 '--gripper-feedback-encoding', 'native_radians', '--model-input-jpeg-quality', '92',
                 '--camera-timeout-ms', '500', '--max-camera-skew-ms', '100', '--max-state-camera-skew-ms', '50'])
    import agibot_gdk as gdk
    import g2_groot_continuous_action_bridge as bridge
    from g2_groot_native_collision_latch import NativeCollisionLatch
    # Add only the physical feedback already present in full receipts. No change
    # to GDK calls, sender interpolation, target mapping, or collision handling.
    bridge.COMPACT_RESULT_FIELDS += ('gripper_status',)

    class TaskService(bridge.ContinuousActionService):
        def status(self, **kwargs):
            # The original standby has no worker and caches startup feedback.
            # Refresh read-only feedback before this task's pre-activation check;
            # the running control worker still owns polling after activation.
            if self.activation_state == 'standby':
                self._poll_feedback()
            return super().status(**kwargs)

        def info(self):
            return dict(super().info(), task=profile['task'], session_limit_s=0,
                        task_workspace_min=profile['workspace_min'], task_workspace_max=profile['workspace_max'],
                        grasp_feedback_receipts=True)

        def activate(self, payload, owner):
            self._poll_feedback()
            if float(self.last_observation.raw_position) > profile['completion']['open_max']:
                raise RuntimeError('This grasp task starts open; activation will not open a closed gripper')
            return super().activate(payload, owner)

    if gdk.gdk_init() != gdk.GDKRes.kSuccess:
        raise RuntimeError('GDK init failed')
    service = None
    try:
        robot, tf = gdk.Robot(), gdk.TF()
        time.sleep(2)
        service = TaskService(robot, tf, initial_gripper_command=profile['initial_gripper'],
                             workspace_min=profile['workspace_min'], workspace_max=profile['workspace_max'],
                             enable_control=args.mode == 'control', required_motion_mode=1,
                             calibration_duration_s=2.0, gripper_joint_name=profile['gripper_joint_name'],
                             native_collision_latch=NativeCollisionLatch(profile['required_control_mode']),
                             freeze_compensation_after_calibration=True)
        # No total session cutoff. Existing 60s connection timeout, fault latch,
        # owner disconnect and operator shutdown continue to stop publication.
        bridge.serve(service, '127.0.0.1', profile['robot_action_port'], float('inf'))
    finally:
        if service is not None:
            service.stop()
        gdk.gdk_release()


if __name__ == '__main__':
    main()
