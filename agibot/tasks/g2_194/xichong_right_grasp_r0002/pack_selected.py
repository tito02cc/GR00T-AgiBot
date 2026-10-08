#!/usr/bin/env python3
"""Build resumable lossless tar.zst shards of complete frozen episodes on robot."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def write_json(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    inventory=json.loads(args.inventory.read_text());source=Path(inventory['raw_root']).resolve()
    dest=args.output_dir.resolve()
    if dest==source or source in dest.parents:raise ValueError('Do not pack inside source')
    rows={r['episode']:r for r in inventory['episodes']}
    names=args.manifest.read_text().splitlines()
    if len(names)!=491 or len(set(names))!=491 or any(not re.fullmatch(r'episode_\d{6}',n) or n not in rows for n in names):
        raise ValueError('Expected frozen 400+91 episode list')
    dest.mkdir(parents=True,exist_ok=True)
    groups=[];current=[];size=0
    for name in names:
        if current and size+rows[name]['bytes']>512*1024**2:groups.append(current);current=[];size=0
        current.append(name);size+=rows[name]['bytes']
    if current:groups.append(current)
    shards=[]
    for i,group in enumerate(groups):
        files={}
        for name in group:
            paths=sorted(p for p in (source/name).rglob('*') if p.is_file())
            if any(p.is_symlink() for p in (source/name).rglob('*')):raise ValueError('Symlink in '+name)
            if len(paths)!=rows[name]['file_count'] or sum(p.stat().st_size for p in paths)!=rows[name]['bytes']:
                raise ValueError('Source changed since inventory: '+name)
            for path in paths:files[str(path.relative_to(source))]=path.stat().st_size
        shards.append({'name':f'episodes_{i:03d}.tar.zst','episodes':group,'raw_bytes':sum(files.values()),'files':files})
    manifest={'source':str(source),'episodes':names,'raw_bytes':sum(s['raw_bytes'] for s in shards),
              'format':'tar + zstd level 1, lossless; complete episodes, unchanged files','shards':shards}
    index=dest/'pack_manifest.json'
    if index.exists() and json.loads(index.read_text())!=manifest:raise ValueError('Existing archive manifest differs')
    write_json(index,manifest)
    print('PACK_MANIFEST_READY',len(shards),'shards',manifest['raw_bytes'],'raw bytes',flush=True)
    state={'state':'packing','total_shards':len(shards),'completed_shards':0,'compressed_bytes':0}
    for shard in shards:
        final=dest/shard['name'];receipt=dest/(shard['name']+'.ready.json')
        if receipt.exists() and final.is_file() and json.loads(receipt.read_text())['archive_bytes']==final.stat().st_size:
            print('REUSE',shard['name'],flush=True)
        else:
            files_list=dest/(shard['name']+'.files.txt')
            files_list.write_text(''.join(name+'\n' for name in shard['files']))
            temp=dest/(shard['name']+'.partial')
            subprocess.run(['tar','--format=pax','-I','zstd -1 -T2','-cf',str(temp),'-C',str(source),
                            '--verbatim-files-from','--no-recursion','-T',str(files_list)],check=True)
            temp.replace(final)
            write_json(receipt,{'archive_bytes':final.stat().st_size,'raw_bytes':shard['raw_bytes'],'episodes':shard['episodes']})
        state['completed_shards']+=1;state['compressed_bytes']+=final.stat().st_size
        write_json(dest/'pack.status.json',state)
        print('SHARD_READY',shard['name'],final.stat().st_size,flush=True)
    state['state']='complete';write_json(dest/'pack.status.json',state);print('PACK_COMPLETE',flush=True)


if __name__=='__main__':main()
