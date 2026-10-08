#!/usr/bin/env python3
"""Partition exact inventories into disjoint rsync lists; never alter dataset files."""
import argparse
import json
from pathlib import Path, PurePosixPath


def partition(files):
    buckets, sizes = [[], []], [0, 0]
    for name, size in sorted(files.items(), key=lambda pair: (-pair[1], pair[0])):
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\n" in name or size < 0:
            raise ValueError(f"Unsafe inventory entry: {name}")
        index = min(range(2), key=lambda i: sizes[i])
        buckets[index].append(name)
        sizes[index] += size
    return [sorted(bucket) for bucket in buckets], sizes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir", type=Path, required=True)
    args = p.parse_args()
    files = {}
    for split in ("train", "heldout"):
        data = json.loads((args.report_dir / f"xichong_right_grasp_r0002_{split}.inventory.json").read_text())
        files.update({f"{split}/{name}": size for name, size in data.items()})
    buckets, sizes = partition(files)
    for index, bucket in enumerate(buckets):
        (args.report_dir / f"upload_part_{index}.txt").write_text("\n".join(bucket) + "\n")
    print(json.dumps({"partition_files": list(map(len, buckets)), "partition_bytes": sizes}))


if __name__ == "__main__":
    main()
