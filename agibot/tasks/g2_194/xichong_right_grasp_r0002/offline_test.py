#!/usr/bin/env python3
"""Recorded-observation grasp evaluation. Loopback model only; no robot I/O."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from agibot.scripts.evaluate_g2_place_offline import (
    check_decoded, check_live_observation_adapter, source_rows, stats,
)
from agibot.scripts.evaluate_xichong_checkpoints import (
    extract_column, prepare_observation, rotation_errors_deg,
)
from agibot.scripts.run_g2_groot_full_protected_inference import validate_model_modality_config
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.server_client import PolicyClient

DATASET = ROOT / 'agibot/gr00t_data/g2_194/xichong_right_grasp_r0002_400'
RAW = ROOT / 'agibot/data/g2_194/xichong_right_grasp_r0002_job01'


def first_held(values, threshold):
    indices = np.flatnonzero(np.asarray(values) >= threshold)
    return int(indices[0]) if len(indices) else None


def reopen_indices(values, held_min, open_max):
    """Hysteresis diagnostic, not a robot gripper controller."""
    held = False
    result = []
    for i, value in enumerate(values):
        if value >= held_min:
            held = True
        elif held and value <= open_max:
            result.append(i)
            held = False
    return result


def lift_from_close(eef, close, hold):
    if close is None:
        return None
    return float((np.median(eef[-hold:, 2]) - eef[close, 2]) * 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=5586)
    parser.add_argument('--train-episodes', type=int, default=8)
    parser.add_argument('--heldout-ids', type=int, nargs='+')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Use a new output directory; do not overwrite earlier predictions')
    args.output_dir.mkdir(parents=True)
    client = PolicyClient(host='127.0.0.1', port=args.port, timeout_ms=120000)
    try:
        modality = client.get_modality_config()
        validate_model_modality_config(modality)
        maps = {s: source_rows(DATASET / s) for s in ('train', 'heldout')}
        identities = {s: {r['identity']['episode_uuid'] for r in rows.values()}
                      for s, rows in maps.items()}
        assert not identities['train'] & identities['heldout']
        probe = {'right_eef': np.tile([.5, -.2, .9, 1, 0, 0, 0, 1, 0], (1, 16, 1)),
                 'right_gripper': np.linspace(-.785, 0, 16).reshape(1, 16, 1)}
        check_decoded(probe)
        reports, latency = {}, []
        warmup = None
        for split in ('train', 'heldout'):
            loader = LeRobotEpisodeLoader(DATASET / split, modality)
            ids = (np.linspace(0, len(loader)-1, args.train_episodes, dtype=int).tolist()
                   if split == 'train' else args.heldout_ids or list(range(len(loader))))
            rows, xyz_all, first_all, angle_all, jaw_all, jumps_all, gt_jumps_all = [], [], [], [], [], [], []
            for episode in ids:
                source = maps[split][episode]['source_episode']
                quality = json.loads((RAW / source / 'quality_report.json').read_text())
                contract = quality['semantic_terminal']
                assert contract['phase'] == 'right_grasp'
                limits = contract['thresholds']
                closed, opened = limits['initial_gripper_held_min'], limits['terminal_gripper_open_max']
                hold = limits['terminal_hold_saved_frames']
                trajectory = loader[episode]
                eef = extract_column(trajectory, 'action.right_eef')
                jaw = extract_column(trajectory, 'action.right_gripper')[:, 0]
                predictions, chunks, within, gt_within, down = [], [], [], [], []
                client.reset()
                for step in range(0, len(trajectory), 16):
                    obs = prepare_observation(trajectory, step, loader, EmbodimentTag.NEW_EMBODIMENT)
                    if warmup is None:
                        check_live_observation_adapter(obs)
                        t = time.perf_counter()
                        action, _ = client.get_action(obs)
                        warmup = time.perf_counter() - t
                        check_decoded(action)
                        client.reset()
                    t = time.perf_counter()
                    action, _ = client.get_action(obs)
                    duration = time.perf_counter() - t
                    latency.append(duration)
                    targets = check_decoded(action)
                    assert np.isfinite(targets).all()
                    count = min(16, len(trajectory) - step)
                    p = np.c_[action['right_eef'][0, :count], action['right_gripper'][0, :count]]
                    predictions.append(p)
                    within.extend(np.linalg.norm(np.diff(p[:, :3], axis=0), axis=1)*1000)
                    gt_within.extend(np.linalg.norm(np.diff(eef[step:step+count, :3], axis=0), axis=1)*1000)
                    down.extend(np.maximum(-np.diff(p[:, 2]), 0)*1000)
                    chunks.append({'step': step, 'valid_rows': count, 'latency_s': duration,
                                   'full_decoded_targets': targets.tolist()})
                predicted = np.concatenate(predictions)
                xyz = np.linalg.norm(predicted[:, :3] - eef[:, :3], axis=1)*1000
                angles = rotation_errors_deg(eef[:, 3:], predicted[:, 3:9])
                jaw_error = np.abs(predicted[:, 9] - jaw)
                gt_close, pred_close = first_held(jaw, closed), first_held(predicted[:, 9], closed)
                reopening = reopen_indices(predicted[:, 9], closed, opened)
                row = {
                    'episode': episode, 'source': source, 'frames': len(eef), 'calls': len(chunks),
                    'thresholds_from_this_episode': limits,
                    'xyz_error_mm': stats(xyz), 'first_action_xyz_error_mm': stats(xyz[::16]),
                    'rotation_error_deg': stats(angles), 'gripper_abs_error_rad': stats(jaw_error),
                    'gt_first_held_step': gt_close, 'pred_first_held_step': pred_close,
                    'held_timing_error_s': (pred_close-gt_close)/10 if pred_close is not None and gt_close is not None else None,
                    'held_position_reference_distance_mm': float(np.linalg.norm(predicted[pred_close, :3]-eef[gt_close, :3])*1000)
                        if pred_close is not None and gt_close is not None else None,
                    'gt_lift_from_first_held_mm': lift_from_close(eef, gt_close, hold),
                    'pred_stitched_lift_from_first_held_mm': lift_from_close(predicted, pred_close, hold),
                    'pred_final_hold_fraction': float(np.mean(predicted[-hold:, 9] >= closed)),
                    'pred_reopen_steps': reopening,
                    'reopen_at_observation_reset': [i for i in reopening if i % 16 == 0],
                    'reopen_within_chunk': [i for i in reopening if i % 16 != 0],
                    'raw_gripper_min_max': [float(predicted[:, 9].min()), float(predicted[:, 9].max())],
                    'raw_gripper_out_of_range_rows': int(np.count_nonzero((predicted[:, 9] < -.785-1e-6) | (predicted[:, 9] > 1e-6))),
                    'within_chunk_xyz_step_mm': stats(within), 'gt_matching_xyz_step_mm': stats(gt_within),
                    'within_chunk_downward_step_mm': stats(down),
                }
                detail = dict(row, gt_eef=eef.tolist(), gt_gripper=jaw.tolist(),
                              pred_eef_gripper=predicted.tolist(), chunks=chunks)
                (args.output_dir / f'{split}_{episode:04d}.json').write_text(json.dumps(detail, indent=2)+'\n')
                rows.append(row)
                xyz_all.extend(xyz); first_all.extend(xyz[::16]); angle_all.extend(angles)
                jaw_all.extend(jaw_error); jumps_all.extend(within); gt_jumps_all.extend(gt_within)
                print(f'{split} {episode}: calls={len(chunks)} XYZ={xyz.mean():.2f}mm held GT/pred={gt_close}/{pred_close}', flush=True)
            reports[split] = {
                'episodes': len(rows), 'frames': len(xyz_all), 'xyz_error_mm': stats(xyz_all),
                'first_action_xyz_error_mm': stats(first_all), 'rotation_error_deg': stats(angle_all),
                'gripper_abs_error_rad': stats(jaw_all), 'within_chunk_xyz_step_mm': stats(jumps_all),
                'gt_matching_xyz_step_mm': stats(gt_jumps_all),
                'held_timing_signed_s': stats([r['held_timing_error_s'] for r in rows if r['held_timing_error_s'] is not None]),
                'held_timing_abs_s': stats([abs(r['held_timing_error_s']) for r in rows if r['held_timing_error_s'] is not None]),
                'held_position_reference_distance_mm': stats([r['held_position_reference_distance_mm'] for r in rows if r['held_position_reference_distance_mm'] is not None]),
                'missing_held_episodes': sum(r['pred_first_held_step'] is None for r in rows),
                'reopen_within_chunk_episodes': sum(bool(r['reopen_within_chunk']) for r in rows),
                'reopen_at_reset_episodes': sum(bool(r['reopen_at_observation_reset']) for r in rows),
                'raw_gripper_out_of_range_rows': sum(r['raw_gripper_out_of_range_rows'] for r in rows),
                'episode_metrics': rows,
            }
        result = {
            'status': 'OFFLINE_EXECUTION_PASS_EFFECT_REQUIRES_REVIEW', 'splits': reports,
            'warmup_seconds': warmup, 'loopback_latency_seconds': stats(latency), 'inference_calls': len(latency),
            'full_h16_decode': 'PASS', 'live_observation_adapter': 'PASS', 'native_gripper_mapping_probe': 'PASS',
            'gripper_diagnostic': 'held_min from this batch collection contract; NOT proof of physical grasp or fully closed jaws',
            'seed': 'Official server stochastic default; no override',
            'scope': 'Open-loop recorded observations every H16; no generated feedback, no robot I/O. Stitched lift and reopen are diagnostics, not closed-loop task success. Partial terminal chunks scored only on real frames.',
        }
        (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print('OFFLINE_EXECUTION_COMPLETE', flush=True)
    finally:
        client.socket.close(linger=0)
        client.context.term()


if __name__ == '__main__':
    main()
