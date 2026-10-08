#!/usr/bin/env python3
"""Download this task's 30k inference bundle; preserve cloud resume checkpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from agibot.training.check_cloud_inputs import check_model

REMOTE = '/root/gpufree-data/GR00T/outputs/xichong_rgrasp_r0002_n1d7_e400_f10_h16_1xa10080_s30k_ckpt5k_v1/checkpoint-30000'
BUNDLE = ROOT / 'agibot/models/xichong_rgrasp_r0002_n1d7_checkpoint-30000'
REPORT = ROOT / 'agibot/local_reports/g2_194/xichong_right_grasp_r0002_prepare_20260920/download'
BACKBONE_SOURCE = ROOT / 'agibot/models/zhewan_rplace_r0003_n1d7_checkpoint-30000/backbone/Cosmos-Reason2-2B'
DATASET = ROOT / 'agibot/gr00t_data/g2_194/xichong_right_grasp_r0002_400/train'


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def localize(value, backbone):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == 'model_name' and child == '/root/gpufree-data/GR00T/models/Cosmos-Reason2-2B':
                value[key] = str(backbone)
            else:
                localize(child, backbone)
    elif isinstance(value, list):
        for child in value:
            localize(child, backbone)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh-control', required=True)
    parser.add_argument('--host', default='root@120.209.70.195')
    parser.add_argument('--port', type=int, default=30263)
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    model = BUNDLE / 'model'
    model.mkdir(parents=True, exist_ok=True)
    state = {'status': 'STARTING', 'source_checkpoint': REMOTE,
             'started_at': datetime.now().astimezone().isoformat(), 'bundle': str(BUNDLE)}
    status_path = REPORT / 'status.json'
    save_json(status_path, state)
    ssh = ['ssh', '-S', args.ssh_control, '-p', str(args.port), '-o', 'BatchMode=yes',
           '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4']
    inventory_code = f'''import json
from pathlib import Path
p=Path({REMOTE!r})
files={{str(f.relative_to(p)):f.stat().st_size for f in p.rglob('*') if f.is_file() and
       ((f.parent==p and f.suffix in ('.json','.safetensors')) or 'experiment_cfg' in f.relative_to(p).parts)}}
assert json.loads((p/'trainer_state.json').read_text())['global_step']==30000
print(json.dumps(files))
'''
    try:
        expected = json.loads(subprocess.run(ssh + [args.host, 'python3 -'], input=inventory_code,
                              text=True, capture_output=True, check=True).stdout)
        save_json(REPORT / 'source_inventory.json', expected)
        weights = sorted(k for k in expected if k.endswith('.safetensors'))
        assert len(weights) == 3 and all(Path(k).name == k for k in weights)
        state.update(status='DOWNLOADING', expected_bytes=sum(expected.values()))
        save_json(status_path, state)
        transport = shlex.join(ssh)
        source = f'{args.host}:{REMOTE}'
        # Unmodified files, rsync transfer verification, no full-file SHA pass.
        with (REPORT / 'metadata.log').open('a') as log:
            subprocess.run(['rsync', '-rt', '--partial', '-e', transport,
                            '--include=/*.json', '--include=/experiment_cfg/***', '--exclude=*',
                            source + '/', str(model) + '/'], stdout=log, stderr=subprocess.STDOUT, check=True)

        def transfer(name):
            with (REPORT / (name + '.log')).open('a') as log:
                subprocess.run(['rsync', '-rt', '--partial', '--info=progress2', '-e', transport,
                                source + '/' + name, str(model) + '/'],
                               stdout=log, stderr=subprocess.STDOUT, check=True)

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(transfer, weights))
        for name, size in expected.items():
            if (model / name).stat().st_size != size:
                raise ValueError(f'Incomplete downloaded file: {name}')
        model_check = check_model(model)
        backbone = BUNDLE / 'backbone/Cosmos-Reason2-2B'
        backbone.parent.mkdir(exist_ok=True)
        if not backbone.exists():
            subprocess.run(['cp', '-a', '--reflink=auto', str(BACKBONE_SOURCE), str(backbone)], check=True)
        backbone_check = check_model(backbone)
        for name in ('config.json', 'processor_config.json'):
            path = model / name
            original = path.with_name(path.stem + '.server-original.json')
            shutil.copy2(path, original)
            value = json.loads(original.read_text())
            if name == 'processor_config.json':
                assert value['processor_kwargs']['use_percentiles'] is False
            localize(value, backbone)
            save_json(path, value)
        manifest = dict(state, status='FILES_DOWNLOADED_PROCESSOR_CHECK_PENDING',
                        source_files_and_bytes=expected, model_check=model_check,
                        backbone_check=backbone_check, backbone_source=str(BACKBONE_SOURCE),
                        sha_scan_performed=False, full_model_forward='NOT_RUN', robot_test='NOT_RUN',
                        excluded_resume_state=['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'training_args.bin'],
                        local_changes='Only active model_name paths; original JSON configurations preserved')
        save_json(BUNDLE / 'inference_bundle_manifest.json', manifest)
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', NO_ALBUMENTATIONS_UPDATE='1')
        with (REPORT / 'processor_reload.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT / 'agibot/training/check_saved_normalization.py'),
                            '--model', str(model), '--dataset', str(DATASET)],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest.update(status='DOWNLOAD_COMPLETE_PROCESSOR_RELOAD_PASS',
                        completed_at=datetime.now().astimezone().isoformat())
        save_json(BUNDLE / 'inference_bundle_manifest.json', manifest)
        save_json(status_path, manifest)
        shutil.copy2(model / 'trainer_state.json', REPORT / 'trainer_state.json')
        print(json.dumps(manifest, indent=2), flush=True)
    except Exception as exc:
        state.update(status='FAILED_RESUMABLE', error=str(exc),
                     updated_at=datetime.now().astimezone().isoformat())
        save_json(status_path, state)
        raise


if __name__ == '__main__':
    main()
