#!/usr/bin/env python3
"""Compare transfer sizes/trajectory coverage, without fixing a training count.

Requires this batch's numeric report and completed full image decoding. A whole
acquisition session is withheld. Feature scales are estimated from training-pool
data only; no motion thresholds or sampling scales from previous tasks are used.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def sessions_from(rows):
    groups=[];previous=None
    for row in sorted(rows,key=lambda r:(r['created_at'],r['episode'])):
        now=datetime.fromisoformat(row['created_at'])
        if previous is None or now.date()!=previous.date() or (now-previous).total_seconds()>900:groups.append([])
        groups[-1].append(row);previous=now
    return groups


def normalized_features(pool):
    xyz=np.asarray([r['phase_trajectory']['xyz'] for r in pool])
    quaternion=np.asarray([r['phase_trajectory']['quaternion'] for r in pool])
    rotation=Rotation.from_quat(quaternion.reshape(-1,4)).as_matrix()[:,:2,:].reshape(len(pool),-1)
    blocks=[xyz.reshape(len(pool),-1),rotation,
            np.asarray([r['phase_trajectory']['gripper'] for r in pool]),
            np.asarray([[r['metrics']['duration_s'],r['metrics']['path_m']] for r in pool])]
    features=[];scales=[]
    for block in blocks:
        centered=block-np.median(block,axis=0)
        # One robust scale per feature group avoids magnifying tiny sensor noise.
        scale=float(np.quantile(np.abs(centered),.75))
        scales.append(scale)
        if scale>1e-8:features.append(centered/(scale*np.sqrt(block.shape[1])))
    if not features:raise ValueError('No measurable trajectory variation')
    return np.concatenate(features,axis=1),xyz,scales


def farthest_order(features,count):
    seed=int(np.argmin(np.square(features-np.median(features,axis=0)).sum(axis=1)))
    selected=[];nearest=np.full(len(features),np.inf)
    for _ in range(count):
        idx=seed if not selected else int(np.argmax(nearest))
        selected.append(idx)
        nearest=np.minimum(nearest,np.square(features-features[idx]).sum(axis=1))
        nearest[selected]=-np.inf
    return selected


def coverage(xyz,chosen):
    query=xyz.reshape(len(xyz),-1);ref=xyz[chosen].reshape(len(chosen),-1)
    distances=np.maximum((query*query).sum(axis=1)[:,None]+(ref*ref).sum(axis=1)[None,:]-2*query@ref.T,0)
    nearest=np.sqrt(distances.min(axis=1)/xyz.shape[1])*1000
    reserve=np.ones(len(xyz),dtype=bool);reserve[chosen]=False
    vals=nearest[reserve]
    return {k:float(v) for k,v in zip(['p50','p90','p95','max'],np.quantile(vals,[.5,.9,.95,1]))} if len(vals) else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--numeric-report',type=Path,required=True);p.add_argument('--image-report',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--counts',type=int,nargs='+',default=[200,300,400,500])
    args=p.parse_args();numeric=json.loads(args.numeric_report.read_text());images=json.loads(args.image_report.read_text())
    if numeric['raw_root']!=images['source_root']:raise ValueError('Source mismatch')
    if not {'head_color','hand_right'}.issubset(images['cameras']):raise ValueError('Missing input camera decode')
    rows=numeric['episodes'];counts=sorted(set(args.counts))
    if not counts or counts[0]<1:raise ValueError('Invalid candidate counts')
    if len({r['episode'] for r in rows})!=len(rows):raise ValueError('Duplicate numeric rows')
    if {r['episode'] for r in rows}!={r['episode'] for r in images['episodes']}:raise ValueError('Incomplete image report')
    if args.output_dir.resolve().is_relative_to(Path(numeric['raw_root']).resolve()):raise ValueError('Output inside source')
    decoded={r['episode'] for r in images['episodes'] if r['status']=='pass'}
    eligible={r['episode'] for r in rows if r['status']=='candidate' and r['episode'] in decoded}
    sessions=sessions_from(rows)
    suitable=[g for g in sessions if 0<len(eligible.intersection(r['episode'] for r in g))<=len(eligible)-counts[0]]
    if not suitable:raise ValueError('No independent chronological session remains for heldout')
    target=.1*len(eligible)
    held_session=min(suitable,key=lambda g:abs(len(eligible.intersection(r['episode'] for r in g))-target))
    embargo={r['episode'] for r in held_session}
    heldout=[r for r in held_session if r['episode'] in eligible]
    pool=sorted([r for r in rows if r['episode'] in eligible and r['episode'] not in embargo],key=lambda r:r['episode'])
    feasible=[c for c in counts if c<=len(pool)]
    f,xyz,scales=normalized_features(pool);order=farthest_order(f,max(feasible))
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    report={'scope':'Candidate transfer plans, not confirmed visual success or final training approval',
            'eligible':len(eligible),'excluded_or_review':len(rows)-len(eligible),
            'heldout_count':len(heldout),'heldout_session':[held_session[0]['episode'],held_session[-1]['episode']],
            'session_rule':'New calendar day or >15 minute acquisition gap; entire session embargoed from training',
            'heldout_selection':'Whole session closest to 10% of eligible data, chosen before coverage scoring',
            'feature_scales_training_pool_only':scales,'alternatives':{}}
    def manifest(name,rs):
        text=''.join(r['episode']+'\n' for r in sorted(rs,key=lambda r:r['episode']))
        dest=output/name
        if dest.exists() and dest.read_text()!=text:raise ValueError('Refusing to overwrite different manifest '+name)
        dest.write_text(text)
    manifest('heldout.txt',heldout)
    for count in feasible:
        idx=order[:count];train=[pool[i] for i in idx]
        manifest(f'train_{count}.txt',train);manifest(f'transfer_{count}.txt',train+heldout)
        report['alternatives'][str(count)]={'train':count,'heldout':len(heldout),
            'bytes':sum(r['bytes'] for r in train+heldout),'frames':sum(r['frames'] for r in train),
            'unselected_pool_nearest_xyz_rms_mm':coverage(xyz,idx)}
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
