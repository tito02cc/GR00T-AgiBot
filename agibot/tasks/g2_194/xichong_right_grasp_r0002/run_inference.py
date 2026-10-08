#!/usr/bin/env python3
"""Full synchronous grasp policy using the pinned, successful G2 H16 transport."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime
import argparse
import json
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RUNTIME = ROOT / 'agibot/deployments/g2_194/workstation_runtime'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUNTIME))
from agibot.scripts import run_g2_groot_full_place_inference as shared
from agibot.scripts.run_g2_groot_full_protected_inference import (
    calibrate_bridge_clock, execute_action_chunk, validate_complete_chunk,
)
from agibot.tools.g2_gr00t_shadow_adapter import build_policy_observation, decode_action_chunk
from agibot.tools.g2_groot_live_observation_client import G2LiveObservationClient
from gr00t.policy.server_client import PolicyClient

CONFIRMATION = 'EXECUTE_G2_GROOT_XICHONG_R0002_GRASP'


class GraspProgress:
    """Observe physical receipt feedback only; never alter model targets."""
    def __init__(self, config):
        self.config = config
        self.anchor = None
        self.window = deque(maxlen=config['stable_frames'])
        self.last_ns = 0

    def observe(self, rows):
        for row in rows:
            receipt = row.get('completion') or {}
            if row.get('status') != 'COMPLETED' or receipt.get('ok') is not True:
                raise RuntimeError('Missing successful completion evidence')
            pose = np.asarray(receipt['live_pose_at_100ms'], dtype=float)
            grip = receipt['gripper_status']
            observation = grip['last_observation']
            position = float(observation['raw_position'])
            stamp = int(receipt['execution_finished_monotonic_ns'])
            if (pose.shape != (7,) or not np.isfinite(pose).all() or not np.isfinite(position)
                    or stamp <= self.last_ns or grip.get('feedback_encoding') != 'native_radians'
                    or grip.get('fault') or observation.get('motor_err_code') or observation.get('whole_end_error')):
                raise RuntimeError('Invalid physical grasp feedback')
            self.last_ns = stamp
            if position <= self.config['open_max']:
                self.anchor = None
                self.window.clear()
            if position >= self.config['held_min']:
                if self.anchor is None:
                    self.anchor = pose.copy()
                self.window.append((pose.copy(), stamp))
            else:
                self.window.clear()

    def status(self, status):
        live = np.asarray(status['live_pose'], dtype=float)
        position = float(status['right_gripper']['last_observation']['raw_position'])
        lift = float(live[2]-self.anchor[2]) if self.anchor is not None else None
        stable = False
        radius = None
        if len(self.window) == self.window.maxlen:
            radius = float(max(np.linalg.norm(p[:3]-live[:3]) for p, _ in self.window))
            stable = radius <= self.config['stability_radius_m']
        healthy = (status.get('ready') is True and not status.get('fatal_error')
                   and not status.get('live_pose_stale') and not status.get('gripper_feedback_stale')
                   and status.get('queue_depth') == 0 and np.isfinite(live).all() and np.isfinite(position))
        passed = (healthy and stable and lift is not None and lift >= self.config['minimum_lift_m']
                  and position >= self.config['held_min'])
        return {'feedback_sequence_complete': bool(passed), 'lift_m': lift,
                'physical_gripper_native': position, 'terminal_radius_m': radius,
                'physical_hold_samples': len(self.window),
                'physical_object_grasp_verified': False}


def validate_contract(info, profile, *, activated=False):
    required = {'schema': shared.BRIDGE_SCHEMA, 'task': profile['task'],
                'grasp_feedback_receipts': True, 'right_arm_enabled': True,
                'right_gripper_protected_closure_enabled': True,
                'right_gripper_physical_zero_closure_enabled': True,
                'right_gripper_command_range': [-0.785, 0.0],
                'left_arm_enabled': False, 'head_waist_chassis_enabled': False,
                'shutdown_gripper_action': 'hold', 'chunk_submission': 'atomic_h16_native_v1',
                'gripper_command_mode': 'native_trajectory_tracking', 'gripper_policy_mode': 'per_row_absolute',
                'model_waypoint_hz': 10.0, 'control_hz': 50.0, 'interpolation_ticks': 5,
                'session_limit_s': 0, 'task_workspace_min': profile['workspace_min'],
                'task_workspace_max': profile['workspace_max']}
    for key, value in required.items():
        if info.get(key) != value:
            raise RuntimeError(f'Wrong task/native bridge contract: {key}')
    shared.validate_execution_options(info, SimpleNamespace(require_native_collision_latch=True,
        required_control_mode=profile['required_control_mode'], freeze_compensation_after_calibration=True),
        activated=activated)


def pre_activation(status, profile):
    pose = np.asarray(status['live_pose'], dtype=float)
    grip = status['right_gripper']
    observation = grip['last_observation']
    position = float(observation['raw_position'])
    if (pose.shape != (7,) or not np.isfinite(pose).all() or not np.isfinite(position)
            or status.get('fatal_error') or status.get('live_pose_stale') or status.get('gripper_feedback_stale')
            or grip.get('fault') or observation.get('motor_err_code') or observation.get('whole_end_error')):
        raise RuntimeError('Robot feedback is not healthy')
    if (pose[:3] < profile['workspace_min']).any() or (pose[:3] > profile['workspace_max']).any():
        raise RuntimeError('Current pose outside this task workspace; no automatic reset')
    if grip.get('feedback_encoding') != 'native_radians' or position > profile['completion']['open_max']:
        raise RuntimeError('Grasp task requires native open-gripper start; not opening automatically')
    return {'initial_pose': pose.tolist(), 'initial_gripper': position,
            'training_reference_distance_m': float(np.linalg.norm(pose[:3]-np.array(profile['initial_pose_reference'][:3]))),
            'reference_is_diagnostic_not_a_fixed_target': True}


def policy_targets(action):
    targets = decode_action_chunk(action).astype(float)
    raw = np.asarray(action['right_gripper'])[0, :, 0]
    if not np.isfinite(raw).all() or np.any(raw < -.785-1e-6) or np.any(raw > 1e-6):
        raise RuntimeError('Policy gripper outside native range; no silent clipping')
    return targets


def smooth_open_gripper_hover(targets, current_pose, current_gripper, config):
    """Reduce small pose reversals while preserving H16 endpoints and gripper rows."""
    original = np.asarray(targets, dtype=np.float64)
    pose = np.asarray(current_pose, dtype=np.float64)
    if (original.shape != (16, 8) or pose.shape != (7,)
            or not np.isfinite(original).all() or not np.isfinite(pose).all()
            or not np.isfinite(current_gripper)):
        raise ValueError('hover smoothing requires finite H16 and current feedback')
    result = original.copy()
    xyz = original[:, :3]
    max_step = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).max())
    net = float(np.linalg.norm(xyz[-1] - xyz[0]))
    first_error = float(np.linalg.norm(xyz[0] - pose[:3]))
    detail = {'applied': False, 'reason': 'disabled', 'max_xyz_adjustment_m': 0.0,
              'max_rotation_adjustment_deg': 0.0, 'max_step_m': max_step,
              'net_displacement_m': net, 'first_waypoint_error_m': first_error,
              'max_gripper_target': float(original[:, 7].max())}
    if not config['enabled']:
        return result, detail
    open_max = float(config['open_max'])
    if current_gripper > open_max or detail['max_gripper_target'] > open_max:
        detail['reason'] = 'gripper_not_open_throughout_chunk'
        return result, detail
    if max_step > config['max_step_m']:
        detail['reason'] = 'waypoint_step_above_hover_limit'
        return result, detail
    if net > config['max_net_m']:
        detail['reason'] = 'net_motion_above_hover_limit'
        return result, detail
    if first_error > config['max_initial_error_m']:
        detail['reason'] = 'first_waypoint_far_from_feedback'
        return result, detail
    weight = float(config['neighbor_weight'])
    if not 0 < weight < 0.25:
        raise ValueError('neighbor_weight must be between 0 and 0.25')
    result[1:-1, :3] = (weight * xyz[:-2] + (1 - 2 * weight) * xyz[1:-1]
                         + weight * xyz[2:])
    maximum_xyz = float(np.linalg.norm(result[:, :3] - xyz, axis=1).max())
    if maximum_xyz > config['max_xyz_adjustment_m']:
        detail['reason'] = 'xyz_adjustment_too_large'
        return original.copy(), detail
    quaternions = original[:, 3:7]
    for index in range(1, 15):
        center = quaternions[index]
        left, right = quaternions[index - 1].copy(), quaternions[index + 1].copy()
        if np.dot(left, center) < 0:
            left *= -1
        if np.dot(right, center) < 0:
            right *= -1
        blended = weight * left + (1 - 2 * weight) * center + weight * right
        result[index, 3:7] = blended / np.linalg.norm(blended)
    dots = np.abs(np.sum(result[:, 3:7] * quaternions, axis=1))
    maximum_rotation = float(np.degrees(2 * np.arccos(np.clip(dots, -1.0, 1.0))).max())
    if maximum_rotation > config['max_rotation_adjustment_deg']:
        detail['reason'] = 'rotation_adjustment_too_large'
        return original.copy(), detail
    detail.update(applied=True, reason='open_gripper_hover',
                  max_xyz_adjustment_m=maximum_xyz,
                  max_rotation_adjustment_deg=maximum_rotation)
    return result, detail


def timed_call(function, *args):
    started = time.monotonic()
    result = function(*args)
    return result, time.monotonic() - started


def run(profile, report, save):
    progress = GraspProgress(profile['completion'])
    with ExitStack() as stack:
        obs = stack.enter_context(G2LiveObservationClient('127.0.0.1', profile['observation_port'], compute_payload_hashes=False))
        model = stack.enter_context(PolicyClient(host='127.0.0.1', port=profile['model_port'], timeout_ms=60000))
        if obs.get_info().get('control_api_exposed') is not False:
            raise RuntimeError('Observation service is not read-only')
        shared.inspect_standby_bridge('127.0.0.1', profile['action_port'])
        report['model_preparation'] = {}
        shared.prepare_model(model, obs, profile['prompt'], report['model_preparation'])
        client = stack.enter_context(shared.PlacementBridgeSession('127.0.0.1', profile['action_port']))
        info = client.request({'op': 'info'})
        validate_contract(info, profile)
        report['preflight'] = pre_activation(client.request({'op': 'status'}), profile)
        report['transport'] = shared.configure_transport_optimization(client, info, True)
        print('MODEL_READY: activating one combined arm/gripper controller', flush=True)
        response = client.request({'op': 'activate', 'confirm': shared.ACTIVATION_CONFIRMATION})
        if response.get('ok') is not True:
            raise RuntimeError(f'Activation rejected: {response}')
        info, status = client.request({'op': 'info'}), client.request({'op': 'status'})
        shared.validate_activation_state(info, status, 'active')
        validate_contract(info, profile, activated=True)
        offset, rtt = calibrate_bridge_clock(client)
        if rtt > 500_000_000:
            raise RuntimeError('Bridge clock RTT exceeds 0.5s')
        report['clock_rtt_ns'] = rtt
        io_pool = stack.enter_context(ThreadPoolExecutor(max_workers=2, thread_name_prefix='g2-observation-status'))
        cycle = 0
        previous_chunk_completed_at = None
        previous_last_command_id = None
        while True:  # No max cycles, total duration or policy-step cap.
            observation_started = time.monotonic()
            snapshot_future = io_pool.submit(timed_call, obs.get_snapshot)
            status_request = {'op': 'status'}
            if previous_last_command_id is not None:
                # The previous H16 receipts were already validated.  Keep
                # fresh pose/gripper/fault fields, omit those redundant rows.
                status_request['after_command_id'] = previous_last_command_id
            status_future = io_pool.submit(timed_call, client.request, status_request)
            snapshot, snapshot_s = snapshot_future.result()
            status, status_s = status_future.result()
            observation_status_s = time.monotonic() - observation_started
            timing = shared.validate_snapshot(snapshot, status)
            if not status.get('ready') or status.get('fatal_error'):
                raise RuntimeError('Bridge became unhealthy before next inference')
            pose = np.asarray(snapshot.metadata['right_eef_xyz_quaternion_xyzw'])
            started = time.monotonic()
            action, _ = model.get_action(build_policy_observation(snapshot.head_color_rgb,
                snapshot.hand_right_rgb, pose, snapshot.metadata['right_gripper']['training_position'], profile['prompt']))
            raw_targets = policy_targets(action)
            targets, smoothing = smooth_open_gripper_hover(raw_targets, pose,
                float(status['right_gripper']['last_observation']['raw_position']),
                profile['open_gripper_hover_smoothing'])
            row = {'cycle': cycle, 'inference_s': time.monotonic()-started, 'timing': timing,
                   'observation_status_s': observation_status_s,
                   'snapshot_request_decode_s': snapshot_s, 'action_status_request_s': status_s,
                   'action_status_receipts_returned': len(status.get('recent_results', [])),
                   'action_status_payload_bytes_estimate': len(json.dumps(status, separators=(',', ':'))),
                   'robot_capture_s': float(snapshot.metadata['capture_duration_ms']) / 1000
                       if 'capture_duration_ms' in snapshot.metadata else None,
                   'image_payload_bytes': sum(int(image['payload_bytes'])
                       for image in snapshot.metadata.get('images', [])),
                   'raw_targets': raw_targets.tolist(), 'targets': targets.tolist(),
                   'hover_smoothing': smoothing,
                   'chunk_metrics': validate_complete_chunk(pose, targets),
                   'executions': [], 'status': 'EXECUTING'}
            if previous_chunk_completed_at is not None:
                row['replan_gap_s'] = time.monotonic() - previous_chunk_completed_at
            report['cycles'].append(row)
            save()
            completion = {}
            execute_action_chunk(client, targets, f'grasp-r0002-c{cycle}', offset,
                                 execution_rows=row['executions'], native_ack_pacing=True,
                                 native_chunk_submission=True, completion_status_out=completion)
            previous_chunk_completed_at = time.monotonic()
            end = shared.chunk_end_status(client, completion)
            progress.observe(row['executions'])
            previous_last_command_id = row['executions'][-1]['command_id']
            row.update(status='COMPLETED', task_progress=progress.status(end))
            print(json.dumps({'cycle': cycle, 'inference_s': row['inference_s'], **row['task_progress']}), flush=True)
            save()
            if row['task_progress']['feedback_sequence_complete']:
                report.update(status='FEEDBACK_GRASP_LIFT_STABLE', final_bridge_status=end)
                break
            cycle += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--confirm', default='')
    parser.add_argument('--confirm-scene', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not args.execute or args.confirm != CONFIRMATION or not args.confirm_scene:
        parser.error(f'Execution requires onsite confirmation and --execute --confirm {CONFIRMATION} --confirm-scene')
    if args.report.exists():
        parser.error('Use a new report path')
    if not Path(shared.__file__).resolve().is_relative_to(RUNTIME):
        raise RuntimeError('Runner must import the pinned workstation runtime')
    profile = json.loads(Path(__file__).with_name('inference.json').read_text())
    report = {'task': profile['task'], 'started_at': datetime.now().astimezone().isoformat(),
              'status': 'STARTED', 'max_cycles': 0, 'cycles': [], 'profile': profile}
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.report.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(args.report)

    def interrupted(*_):
        raise KeyboardInterrupt('Operator/process stop')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        run(profile, report, save)
    except KeyboardInterrupt:
        report['status'] = 'OPERATOR_STOPPED'
    except Exception as error:
        report.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['finished_at'] = datetime.now().astimezone().isoformat()
        save()
    return 0 if report['status'] == 'FEEDBACK_GRASP_LIFT_STABLE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
