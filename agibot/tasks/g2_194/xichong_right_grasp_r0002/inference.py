#!/usr/bin/env python3
"""Print this task's deployment commands, or start its local model server."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[4]
TASK = Path(__file__).resolve().parent


def commands(profile, report):
    python = str(ROOT / '.venv/bin/python')
    remote = profile['robot_task_dir'] + '/robot_bridge.sh'
    server = [python, '-u', str(ROOT/'gr00t/eval/run_gr00t_server.py'), '--embodiment-tag', 'NEW_EMBODIMENT',
              '--model-path', str(ROOT/profile['model']), '--device', 'cuda:0', '--host', '127.0.0.1',
              '--port', str(profile['model_port'])]
    return {
        'model_server_on_workstation': server,
        'observation_on_robot': ['bash', remote, 'observation'],
        'standby_on_robot_read_only': ['bash', remote, 'standby'],
        'control_on_robot_after_scene_confirmation_instead_of_standby': ['bash', remote, 'control', '--confirm-scene'],
        'forward_on_workstation': ['ssh', '-N', '-o', 'ExitOnForwardFailure=yes',
            '-L', f"{profile['observation_port']}:127.0.0.1:{profile['robot_observation_port']}",
            '-L', f"{profile['action_port']}:127.0.0.1:{profile['robot_action_port']}", 'agi@'+profile['robot_ip']],
        'optional_read_only_preflight_capture': [python, str(ROOT/'agibot/scripts/capture_g2_observation.py'),
            '--host', '127.0.0.1', '--port', str(profile['observation_port']),
            '--output-dir', str(report.parent/'preflight')],
        'optional_scene_scale_diagnostic_no_motion_gate': [python, str(TASK/'check_scene_geometry.py'),
            '--image', str(report.parent/'preflight/head_color.jpg')],
        'live_only_after_onsite_ready_and_recording_notice': [python, '-u', str(TASK/'run_inference.py'),
            '--execute', '--confirm', 'EXECUTE_G2_GROOT_XICHONG_R0002_GRASP', '--confirm-scene', '--report', str(report)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('commands', 'server'))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    p = json.loads((TASK/'inference.json').read_text())
    config = json.loads((ROOT/p['model']/'processor_config.json').read_text())
    if config['processor_kwargs']['use_percentiles'] is not p['use_percentiles']:
        raise ValueError('Saved model normalization does not match task')
    if args.mode == 'commands' and (args.report is None or args.report.exists()):
        parser.error('commands requires a new --report path')
    items = commands(p, args.report.resolve() if args.report else Path('unused.json'))
    if args.mode == 'commands':
        print('# PRINT ONLY. No services or motion started. Standby/control are alternatives.')
        print('# Robot protection and scene must be confirmed; no automatic reset, opening or mode change.')
        for label, command in items.items():
            print(f'\n# {label}\n{shlex.join(command)}')
    else:
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', NO_ALBUMENTATIONS_UPDATE='1')
        command = items['model_server_on_workstation']
        os.chdir(ROOT)
        os.execv(command[0], command)


if __name__ == '__main__':
    main()
