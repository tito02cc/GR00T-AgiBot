import importlib.util
import io
from pathlib import Path
import subprocess
import tarfile
import pytest

spec=importlib.util.spec_from_file_location('packed_pull',Path(__file__).with_name('pull_packed.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def archive(tmp_path,name='episode_000001/frames.jsonl',payload=b'test\n',symlink=False):
    raw=tmp_path/'sample.tar'
    with tarfile.open(raw,'w') as t:
        entry=tarfile.TarInfo(name)
        if symlink:entry.type=tarfile.SYMTYPE;entry.linkname='/tmp/outside'
        else:entry.size=len(payload)
        t.addfile(entry,None if symlink else io.BytesIO(payload))
    packed=tmp_path/'sample.tar.zst'
    subprocess.run(['zstd','-q','-1',str(raw),'-o',str(packed)],check=True)
    return packed


def test_extract_exact_original_bytes(tmp_path):
    packed=archive(tmp_path);dest=tmp_path/'raw';dest.mkdir()
    module.extract_verified(packed,dest,{'episode_000001/frames.jsonl':5})
    assert (dest/'episode_000001/frames.jsonl').read_bytes()==b'test\n'


@pytest.mark.parametrize('name,symlink',[('../outside',False),('episode_000001/link',True)])
def test_unsafe_members_rejected(tmp_path,name,symlink):
    packed=archive(tmp_path,name=name,symlink=symlink);dest=tmp_path/'raw';dest.mkdir()
    with pytest.raises(ValueError):module.extract_verified(packed,dest,{name:0 if symlink else 5})


def test_size_mismatch_rejected(tmp_path):
    packed=archive(tmp_path);dest=tmp_path/'raw';dest.mkdir()
    with pytest.raises(ValueError):module.extract_verified(packed,dest,{'episode_000001/frames.jsonl':6})
