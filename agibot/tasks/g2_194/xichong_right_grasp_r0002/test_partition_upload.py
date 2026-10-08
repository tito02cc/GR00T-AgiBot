import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("partition_upload", Path(__file__).with_name("partition_upload.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_complete_disjoint_and_balanced():
    files = {f"train/{i}.mp4": size for i, size in enumerate([9, 8, 7, 6, 3, 1, 1])}
    parts, sizes = module.partition(files)
    assert not set(parts[0]) & set(parts[1])
    assert set(parts[0]) | set(parts[1]) == set(files)
    assert sum(sizes) == sum(files.values())
    assert abs(sizes[0] - sizes[1]) <= 1


@pytest.mark.parametrize("name", ["/absolute/file", "train/../bad", "train/a\nb"])
def test_unsafe_paths_rejected(name):
    with pytest.raises(ValueError):
        module.partition({name: 1})
