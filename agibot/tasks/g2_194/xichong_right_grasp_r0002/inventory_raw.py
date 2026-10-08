#!/usr/bin/env python3
"""Inventory this new task without historical quality gates or sample selection.

Source files are read-only. Collector labels and numeric distributions describe
the batch; they are not independently verified physical success judgments.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np


def read_json(path):
    return json.loads(path.read_text())


def inspect_episode(ep):
    row = {"episode": ep.name, "integrity_errors": []}
    try:
        files = list(p for p in ep.rglob('*') if p.is_file())
        if any(p.is_symlink() for p in ep.rglob('*')):
            row['integrity_errors'].append('symlink_requires_review')
        row.update(file_count=len(files), bytes=sum(p.stat().st_size for p in files))
        meta, quality = read_json(ep/'meta_info.json'), read_json(ep/'quality_report.json')
        frames = [json.loads(line) for line in (ep/'frames.jsonl').read_text().splitlines() if line.strip()]
        n = len(frames)
        if n < 2:
            raise ValueError('fewer than two saved frames')
        row.update(frames=n, uuid=meta.get('episode_uuid'), created_at=meta.get('created_at'),
                   robot_id=meta.get('robot_id'), task_id=meta.get('task_id'),
                   selected=meta.get('selected'), annotations=meta.get('annotations'),
                   prompt=json.loads(meta['text'])['description'],
                   collector_contract=quality.get('contract'),
                   collector_terminal=quality.get('semantic_terminal', quality.get('xichong_terminal')),
                   collector_timing=quality.get('timing'),
                   collector_duplicate_images=quality.get('images',{}).get('duplicate_previous_counts'))
        with np.load(ep/'arrays.npz', allow_pickle=False) as arrays:
            expected={'states':(n,16),'actions':(n,14),'ee_poses':(n,14),'grippers':(n,2),'timestamps_monotonic':(n,)}
            for key, shape in expected.items():
                value=arrays[key]
                if value.shape != shape or not np.isfinite(value).all():
                    row['integrity_errors'].append('npz_shape_or_nonfinite:'+key)
        ts=np.asarray([f['timestamp_monotonic'] for f in frames],dtype=float)
        pose=np.asarray([f['right_ee_pose'] for f in frames],dtype=float)
        grip=np.asarray([f['right_gripper']['position'] for f in frames],dtype=float)
        action=np.asarray([f['action_right_7d'] for f in frames],dtype=float)
        if pose.shape!=(n,7) or action.shape!=(n,7) or not all(np.isfinite(a).all() for a in [ts,pose,grip,action]):
            raise ValueError('invalid frame numeric shape or nonfinite values')
        if not np.all(np.diff(ts)>0): row['integrity_errors'].append('nonmonotonic_timestamps')
        if [f['frame_index'] for f in frames]!=list(range(n)):row['integrity_errors'].append('noncontiguous_frame_index')
        row['metrics']={
            'duration_s':float(ts[-1]-ts[0]),'dt_min_s':float(np.diff(ts).min()),
            'dt_max_s':float(np.diff(ts).max()),'dt_median_s':float(np.median(np.diff(ts))),
            'start_xyz':pose[0,:3].tolist(),'end_xyz':pose[-1,:3].tolist(),
            'xyz_min':pose[:,:3].min(axis=0).tolist(),'xyz_max':pose[:,:3].max(axis=0).tolist(),
            'path_m':float(np.linalg.norm(np.diff(pose[:,:3],axis=0),axis=1).sum()),
            'max_action_translation_m':float(np.linalg.norm(action[:,:3],axis=1).max()),
            'max_action_rotation_rad':float(np.linalg.norm(action[:,3:6],axis=1).max()),
            'gripper_start':float(grip[0]),'gripper_end':float(grip[-1]),
            'gripper_min':float(grip.min()),'gripper_max':float(grip.max()),
        }
        camera_stats={}
        for camera in ['head_color','hand_right','hand_left']:
            offsets=[]
            for i,f in enumerate(frames):
                expected=f'images/{camera}_{i:06d}.jpg'
                if f.get('images',{}).get(camera)!=expected or not (ep/expected).is_file() or (ep/expected).stat().st_size==0:
                    row['integrity_errors'].append(f'image_mapping_missing_empty:{camera}:{i}')
                midpoint=f.get('image_meta',{}).get(camera,{}).get('software_midpoint_monotonic')
                if midpoint is not None:offsets.append(abs(float(midpoint)-ts[i])*1000)
            camera_stats[camera]={'images':len(list((ep/'images').glob(camera+'_*.jpg'))),
                                  'max_software_state_offset_ms':max(offsets) if offsets else None}
        row['cameras']=camera_stats
    except Exception as error:
        row['integrity_errors'].append(type(error).__name__+': '+str(error))
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    source=args.raw_root.resolve(); output=args.output.resolve()
    if output==source or source in output.parents:raise ValueError('output must be outside source')
    episodes=sorted(ep for ep in source.iterdir() if ep.is_dir() and re.fullmatch(r'episode_\d{6}',ep.name))
    rows=[]
    for i,ep in enumerate(episodes,1):
        rows.append(inspect_episode(ep))
        if i%50==0 or i==len(episodes):print(f'inventoried={i}/{len(episodes)}',flush=True)
    summary={'episodes':len(rows),'bytes':sum(r.get('bytes',0) for r in rows),
             'files':sum(r.get('file_count',0) for r in rows),'frames':sum(r.get('frames',0) for r in rows),
             'integrity_flagged_episodes':[r['episode'] for r in rows if r['integrity_errors']],
             'by_date':dict(Counter(str(r.get('created_at',''))[:10] for r in rows))}
    distribution={}
    for key in ['duration_s','dt_max_s','path_m','max_action_translation_m','max_action_rotation_rad','gripper_start','gripper_end']:
        values=[r['metrics'][key] for r in rows if 'metrics' in r]
        if values:distribution[key]=dict(zip(['min','p05','p50','p95','max'],np.quantile(values,[0,.05,.5,.95,1]).tolist()))
    report={'created_at':datetime.now(timezone.utc).isoformat(),'raw_root':str(source),
            'scope':'Inventory only. No historical task thresholds, semantic exclusions, image decode, or visual success claim.',
            'summary':summary,'distributions':distribution,'episodes':rows}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'summary':summary,'distributions':distribution},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
