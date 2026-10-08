#!/usr/bin/env python3
"""Resume complete raw episode transfer; never select training data or delete sources."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--ssh-control-path',type=Path,required=True)
    selection=parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--episode-list',type=Path)
    selection.add_argument('--all-episodes',action='store_true')
    args=parser.parse_args()
    config=json.loads(Path(__file__).with_name('source.json').read_text())
    inventory=json.loads(args.inventory.read_text())
    if inventory['raw_root']!=config['remote_raw_root']:
        raise ValueError('Inventory source mismatch')
    all_rows={r['episode']:r for r in inventory['episodes']}
    names=args.episode_list.read_text().splitlines() if args.episode_list else list(all_rows)
    if len(set(names))!=len(names) or not names or any(not re.fullmatch(r'episode_\d{6}',n) for n in names):
        raise ValueError('Invalid inventory episode names')
    if not set(names)<=set(all_rows):raise ValueError('Manifest contains unknown episodes')
    selected_rows=[all_rows[name] for name in names]
    if inventory['summary']['integrity_flagged_episodes']:
        raise ValueError('Review flagged episode integrity before transfer')
    root=args.repo_root.resolve()
    target=(root/config['local_raw_root']).resolve()
    if not target.is_relative_to(root/'agibot/data/g2_194'):
        raise ValueError('Destination must be inside robot-specific raw data directory')
    target.mkdir(parents=True,exist_ok=True)
    report=args.inventory.resolve().parent
    files_list=report/('pull_'+args.episode_list.stem+'.txt' if args.episode_list else 'transfer_all_episodes.txt')
    text=''.join(name+'/\n' for name in sorted(names))
    if files_list.exists() and files_list.read_text()!=text:
        raise ValueError('Frozen transfer list differs')
    files_list.write_text(text)
    status={'state':'transferring','started_at':time.time(),'source':config['remote_raw_root'],
            'destination':str(target),'episodes':len(names),'expected_bytes':sum(r['bytes'] for r in selected_rows),
            'selection':str(args.episode_list) if args.episode_list else 'Explicit all-episode transfer'}
    def save():
        temp=report/'pull.status.json.tmp'
        temp.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
        temp.replace(report/'pull.status.json')
    save()
    # Reuse an authenticated task-specific SSH connection; no credentials stored here.
    ssh=f'ssh -S {args.ssh_control_path.resolve()} -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4'
    common=['rsync','-rt','--recursive','--files-from='+str(files_list),'-e',ssh]
    source=f"{config['robot_user']}@{config['robot_host']}:{config['remote_raw_root']}/"
    try:
        subprocess.run(common+['--partial','--partial-dir=.rsync-partial','--info=progress2,name0','--stats',source,str(target)+'/'],check=True)
        status['state']='verifying_file_sizes';save()
        verification=subprocess.run(common+['--dry-run','--size-only','--itemize-changes',source,str(target)+'/'],check=True,text=True,capture_output=True)
        (report/'rsync_size_verification.txt').write_text(verification.stdout)
        if verification.stdout.strip():
            raise RuntimeError('Remote/local file differences remain')
        for row in selected_rows:
            files=[p for p in (target/row['episode']).rglob('*') if p.is_file()]
            if len(files)!=row['file_count'] or sum(p.stat().st_size for p in files)!=row['bytes']:
                raise RuntimeError('Inventory mismatch for '+row['episode'])
        status.update(state='complete',completed_at=time.time(),verification='rsync path/size dry-run plus per-episode file counts/bytes; no SHA scan')
        save();print('RAW_TRANSFER_COMPLETE',flush=True)
    except Exception as exc:
        status.update(state='failed',error=type(exc).__name__+': '+str(exc));save()
        raise


if __name__=='__main__':main()
