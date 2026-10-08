#!/usr/bin/env python3
"""Check this batch against its own collector contract; source remains read-only.

2e-5 is a float32 consistency tolerance, not a historical motion/task gate.
Timing and large-step values are reported, not rejected using another task's limits.
Image decoding is an independent required stage before selecting transfer data.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ATOL = 2e-5


def compare_numeric(actual, expected, quaternion_starts=()):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return False
    delta = np.abs(a - b)
    for start in quaternion_starts:
        qa, qb = a[:, start:start+4], b[:, start:start+4]
        sign = np.where((qa*qb).sum(axis=1)<0, -1., 1.)
        delta[:, start:start+4] = np.abs(qa-sign[:,None]*qb)
    return bool(np.all(delta<=ATOL))


def task_metrics(pose, grip, next_grip, thresholds):
    """Derive semantic checks solely from this episode's recorded task contract."""
    n=len(grip);hold=int(thresholds['terminal_hold_saved_frames'])
    if hold<1 or n<hold: raise ValueError('insufficient terminal hold samples')
    held=grip>=float(thresholds['initial_gripper_held_min'])
    anchor=n
    while anchor>0 and held[anchor-1]:anchor-=1
    reasons=[]
    if grip[0]>float(thresholds['terminal_gripper_open_max']):reasons.append('initial_not_open')
    if not held[-hold:].all():reasons.append('terminal_not_held')
    if next_grip[-1]<float(thresholds['initial_gripper_held_min']):reasons.append('last_next_not_held')
    lift=None
    if not 0<anchor<n-1:reasons.append('no_interior_grasp_anchor')
    else:
        lift=float(pose[-1,2]-pose[anchor,2])
        if lift+ATOL<float(thresholds['grasp_min_lift_m_per_arm']):reasons.append('lift_below_this_task_contract')
    radius=float(np.linalg.norm(pose[-hold:,:3]-pose[-hold:,:3].mean(axis=0),axis=1).max())
    if radius>float(thresholds['terminal_stability_max_m_per_arm'])+ATOL:reasons.append('terminal_motion_exceeds_this_task_contract')
    return {'grasp_index':anchor if anchor<n else None,'lift_z_m':lift,'terminal_radius_m':radius},reasons


def inspect(root, inventory_row, source):
    row={'episode':inventory_row['episode'],'created_at':inventory_row['created_at'],
         'uuid':inventory_row['uuid'],'bytes':inventory_row['bytes'],'file_count':inventory_row['file_count'],
         'frames':inventory_row['frames'],'invalid_reasons':list(inventory_row['integrity_errors']),
         'review_reasons':[],'notes':[]}
    try:
        ep=root/row['episode']
        frames=[json.loads(line) for line in (ep/'frames.jsonl').read_text().splitlines() if line.strip()]
        n=len(frames)
        meta=json.loads((ep/'meta_info.json').read_text()); q=json.loads((ep/'quality_report.json').read_text())
        terminal=q['semantic_terminal'];thresholds=terminal['thresholds']
        if terminal.get('phase')!='right_grasp' or thresholds.get('active_arms')!=['right']:
            raise ValueError('unexpected task phase or active arms')
        if meta.get('task_id')!=source['source_task_id']:row['invalid_reasons'].append('source_task_identity')
        if meta.get('robot_id')!=source['robot_id']:row['invalid_reasons'].append('robot_identity')
        prompt=json.loads(meta['text'])['description']
        for key,expected in [('episode_uuid',meta['episode_uuid']),('prompt',prompt),('action_mode','next_delta'),('pose_frame','base_link_tf'),('arm_mode','right')]:
            if any(f.get(key)!=expected for f in frames):row['invalid_reasons'].append('frame_contract:'+key)
        if len(frames)!=inventory_row['frames']:row['invalid_reasons'].append('source_changed_since_inventory')
        right=np.asarray([f['right_ee_pose'] for f in frames],dtype=float)
        left=np.asarray([f['left_ee_pose'] for f in frames],dtype=float)
        grip=np.asarray([f['right_gripper']['position'] for f in frames],dtype=float)
        lg=np.asarray([f['left_gripper']['position'] for f in frames],dtype=float)
        target=np.asarray([f['next_right_ee_pose'] for f in frames],dtype=float)
        next_grip=np.asarray([f['next_right_gripper'] for f in frames],dtype=float)
        action=np.asarray([f['action_right_7d'] for f in frames],dtype=float)
        la=np.asarray([f['action_left_7d'] for f in frames],dtype=float)
        action14=np.asarray([f['action_14d'] for f in frames],dtype=float)
        ts=np.asarray([f['timestamp_monotonic'] for f in frames],dtype=float)
        for label,a in [('pose',right),('left_pose',left),('target',target),('grip',grip),('actions',action),('timestamps',ts)]:
            if not np.isfinite(a).all():raise ValueError('nonfinite '+label)
        for label,pose in [('current',right),('left',left),('target',target)]:
            if not np.all(np.abs(np.linalg.norm(pose[:,3:],axis=1)-1)<=ATOL):row['invalid_reasons'].append('quaternion:'+label)
        references={'states':np.column_stack((left,lg,right,grip)),
                    'ee_poses':np.column_stack((left,right)), 'grippers':np.column_stack((lg,grip)),
                    'actions':action14,'timestamps_monotonic':ts}
        with np.load(ep/'arrays.npz',allow_pickle=False) as arrays:
            for key,expected in references.items():
                if not compare_numeric(arrays[key],expected,{'states':(3,11),'ee_poses':(3,10)}.get(key,())):
                    row['invalid_reasons'].append('npz_frame_mismatch:'+key)
        for label,a,b,qs in [('action14',action14,np.column_stack((la,action)),()),
                             ('delta_xyz',right[:,:3]+action[:,:3],target[:,:3],()),
                             ('action_grip',action[:,6],next_grip,()),
                             ('next_current_pose',target[:-1],right[1:],(3,)),
                             ('next_current_grip',next_grip[:-1],grip[1:],())]:
            if not compare_numeric(a,b,qs):row['invalid_reasons'].append(label)
        angle=((Rotation.from_rotvec(action[:,3:6])*Rotation.from_quat(right[:,3:])).inv()*Rotation.from_quat(target[:,3:])).magnitude()
        if angle.max()>ATOL:row['invalid_reasons'].append('delta_rotation')
        for key in ['contract','semantic_terminal']:
            if q.get(key,{}).get('ok') is not True:row['review_reasons'].append('collector:'+key)
        if meta.get('annotations',{}).get('success')!='y' or meta.get('quality',{}).get('data_validate') is not True:
            row['review_reasons'].append('collector_annotation_or_validation')
        if q['timing'].get('late_frame_gate_ok') is False:row['review_reasons'].append('collector_timing_gate')
        # Warnings permitted by this collection's contract remain visible, not auto-rejected.
        row['notes'].extend(q['contract'].get('warnings',[]))
        metrics,reasons=task_metrics(right,grip,next_grip,thresholds)
        row['metrics']={**inventory_row['metrics'],**metrics}
        row['review_reasons'].extend(reasons)
        row['contract_thresholds']=thresholds
        if q['timing']['frame_count']<thresholds['min_capture_frames']:row['review_reasons'].append('short_capture_for_this_task')
        if not np.all(np.diff(ts)>0):row['invalid_reasons'].append('nonmonotonic_timestamps')
        anchor=metrics['grasp_index']
        if not row['invalid_reasons'] and anchor is not None and 0<anchor<n-1:
            query=np.r_[np.linspace(ts[0],ts[anchor],20),np.linspace(ts[anchor],ts[-1],13)[1:]]
            row['phase_trajectory']={
                'xyz':np.column_stack([np.interp(query,ts,right[:,axis]) for axis in range(3)]).tolist(),
                'quaternion':Slerp(ts,Rotation.from_quat(right[:,3:]))(query).as_quat().tolist(),
                'gripper':np.interp(query,ts,grip).tolist(),
            }
    except Exception as exc:
        row['invalid_reasons'].append(type(exc).__name__+': '+str(exc))
    row['status']='invalid' if row['invalid_reasons'] else 'review' if row['review_reasons'] else 'candidate'
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--source-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    source=json.loads(args.source_config.read_text());inventory=json.loads(args.inventory.read_text())
    root=Path(inventory['raw_root']).resolve()
    if str(root)!=source['remote_raw_root']:raise ValueError('Wrong task source')
    if args.output.resolve().is_relative_to(root):raise ValueError('Report must be outside raw source')
    rows=[]
    for i,r in enumerate(inventory['episodes'],1):
        rows.append(inspect(root,r,source))
        if i%50==0 or i==len(inventory['episodes']):print(f'numeric_checked={i}/{len(inventory["episodes"])}',flush=True)
    uuids=Counter(r['uuid'] for r in rows)
    for row in rows:
        if uuids[row['uuid']]>1:
            row['invalid_reasons'].append('duplicate_episode_uuid');row['status']='invalid'
    result={'raw_root':str(root),'scope':'This batch collector contract plus numeric consistency; image decoding required separately; not visual success certification',
            'counts':dict(Counter(r['status'] for r in rows)),
            'reason_counts':dict(Counter(reason for r in rows for reason in r['invalid_reasons']+r['review_reasons'])),
            'episodes':rows}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
