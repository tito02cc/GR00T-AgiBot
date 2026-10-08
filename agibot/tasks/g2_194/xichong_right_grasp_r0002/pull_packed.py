#!/usr/bin/env python3
"""Fetch immutable shards, resume downloads, and verify extracted paths and sizes."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import time


def extract_verified(archive,target,expected):
    """Accept only regular files in the exact producer inventory; reject traversal."""
    seen=set();target=target.resolve()
    process=subprocess.Popen(['zstd','-d','-c',str(archive)],stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout,mode='r|') as tar:
            for member in tar:
                name=member.name;path=PurePosixPath(name)
                if (not member.isfile() or path.is_absolute() or '..' in path.parts or
                    name not in expected or name in seen or member.size!=expected[name]):
                    raise ValueError('Unexpected archive member: '+name)
                destination=target/name
                if destination.is_symlink() or not destination.resolve().is_relative_to(target):
                    raise ValueError('Unsafe extraction destination')
                tar.extract(member,path=target,filter='data')
                if destination.stat().st_size!=expected[name]:raise ValueError('Extracted file size mismatch')
                seen.add(name)
        # Drain the decompressor so the zstd stream checksum is checked too.
        while process.stdout.read(1024*1024):pass
        if process.wait()!=0:raise RuntimeError('zstd decompression failed')
        if seen!=set(expected):raise ValueError('Missing archive members')
    except BaseException:
        process.kill();process.wait();raise
    finally:process.stdout.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ssh-control-path',type=Path,required=True)
    p.add_argument('--remote-dir',required=True);p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--target',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True)
    args=p.parse_args();cache=args.cache_dir.resolve();cache.mkdir(parents=True,exist_ok=True)
    target=args.target.resolve();target.mkdir(parents=True,exist_ok=True)
    inventory=json.loads(args.inventory.read_text());known={r['episode']:r for r in inventory['episodes']}
    ssh=['ssh','-S',str(args.ssh_control_path.resolve()),'-o','BatchMode=yes','-o','ConnectTimeout=15',
         '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=4','agi@10.20.15.194']
    def remote_json(name,wait=True):
        for attempt in range(720 if wait else 1):
            result=subprocess.run(ssh+['cat '+shlex.quote(args.remote_dir+'/'+name)],capture_output=True,text=True,timeout=45)
            if result.returncode==0:return json.loads(result.stdout)
            if attempt%12==0:print('WAIT_FOR_PACK',name,flush=True)
            if not wait:raise RuntimeError('Remote metadata unavailable')
            time.sleep(5)
        raise RuntimeError('Timed out waiting for archive readiness')
    def save(path,data):
        temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
    manifest=remote_json('pack_manifest.json');save(cache/'pack_manifest.json',manifest)
    frozen=Path(__file__).with_name('transfer.txt').read_text().splitlines()
    if manifest['episodes']!=frozen or manifest['source']!=inventory['raw_root']:
        raise ValueError('Archive manifest differs from frozen task selection')
    state={'state':'transferring','started_at':time.time(),'episodes':len(frozen),
           'raw_bytes':manifest['raw_bytes'],'total_shards':len(manifest['shards']),'completed_shards':0,
           'completed_raw_bytes':0,'downloaded_archive_bytes':0,'destination':str(target)}
    status=cache.parent/'packed_pull.status.json';save(status,state)
    try:
        for shard in manifest['shards']:
            name=shard['name'];archive=cache/name;done=cache/(name+'.extracted.json')
            if not done.exists() and all((target/f).is_file() and not (target/f).is_symlink()
                                        and (target/f).stat().st_size==size for f,size in shard['files'].items()):
                save(done,{'archive_bytes':0,'files':len(shard['files']),'raw_bytes':shard['raw_bytes'],
                           'source':'reused existing raw files matching this source inventory', 'completed_at':time.time()})
                print('REUSE_LOCAL_RAW',name,flush=True)
            if done.exists():
                if not all((target/f).is_file() and (target/f).stat().st_size==size for f,size in shard['files'].items()):
                    raise ValueError('Previously extracted shard no longer matches')
                print('REUSE_EXTRACTED',name,flush=True)
            else:
                receipt=remote_json(name+'.ready.json')
                if receipt['episodes']!=shard['episodes']:raise ValueError('Shard receipt selection mismatch')
                state['current_shard']=name;state['current_archive_bytes']=receipt['archive_bytes'];save(status,state)
                cmd=['rsync','-t','--partial','--append-verify','--info=progress2,name0','--stats','-e',shlex.join(ssh[:-1]),
                     'agi@10.20.15.194:'+args.remote_dir+'/'+name,str(archive)]
                for attempt in range(3):
                    if subprocess.run(cmd).returncode==0:break
                    if attempt==2:raise RuntimeError('Archive transfer failed after retries')
                    time.sleep(10)
                if archive.stat().st_size!=receipt['archive_bytes']:raise ValueError('Archive size mismatch')
                state['state']='extracting';save(status,state)
                extract_verified(archive,target,shard['files'])
                save(done,{'archive_bytes':archive.stat().st_size,'files':len(shard['files']),
                           'raw_bytes':shard['raw_bytes'],'completed_at':time.time()})
            state.update(state='transferring',completed_shards=state['completed_shards']+1,
                         completed_raw_bytes=state['completed_raw_bytes']+shard['raw_bytes'],
                         downloaded_archive_bytes=state['downloaded_archive_bytes']+json.loads(done.read_text())['archive_bytes'])
            save(status,state);print('SHARD_COMPLETE',name,state['completed_shards'],'/',state['total_shards'],flush=True)
        for episode in frozen:
            paths=[p for p in (target/episode).rglob('*') if p.is_file() and '.rsync-partial' not in p.parts]
            if len(paths)!=known[episode]['file_count'] or sum(p.stat().st_size for p in paths)!=known[episode]['bytes']:
                raise ValueError('Final episode inventory mismatch: '+episode)
        state.update(state='complete',completed_at=time.time(),verification='zstd stream integrity, exact member paths/sizes, per-episode counts/bytes')
        save(status,state);print('SELECTED_RAW_TRANSFER_COMPLETE',flush=True)
    except BaseException as error:
        state.update(state='failed',error=type(error).__name__+': '+str(error));save(status,state);raise


if __name__=='__main__':main()
