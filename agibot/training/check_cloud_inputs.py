#!/usr/bin/env python3
"""Small CPU-only inventory/header checks. No hashes, tensor loading or training."""

import argparse
import json
from pathlib import Path
import struct


def inventory(root):
    return {
        str(p.relative_to(root)): p.stat().st_size for p in sorted(root.rglob("*")) if p.is_file()
    }


def check_inventory(root, manifest):
    expected = json.loads(manifest.read_text())
    actual = inventory(root)
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    changed = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])
    if missing or extra or changed:
        raise ValueError({"missing": missing, "extra": extra, "size_mismatch": changed})
    return {"files": len(actual), "bytes": sum(actual.values()), "hash_checked": False}


def check_model(root):
    config = json.loads((root / "config.json").read_text())
    index_path = root / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"] if index_path.exists() else None
    shards = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]
    seen = {}
    for name in shards:
        if Path(name).name != name:
            raise ValueError(f"Invalid shard filename: {name}")
        path = root / name
        size = path.stat().st_size
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < header_size < min(size - 8, 32 * 1024**2):
                raise ValueError(f"Invalid safetensors header: {path}")
            header = json.loads(stream.read(header_size))
        end = 0
        tensors = [(key, value) for key, value in header.items() if key != "__metadata__"]
        for key, value in sorted(tensors, key=lambda pair: pair[1]["data_offsets"]):
            start, stop = value["data_offsets"]
            if start != end or stop < start or key in seen:
                raise ValueError(f"Invalid tensor offsets or duplicate: {path}: {key}")
            end = stop
            seen[key] = name
        if end + 8 + header_size != size:
            raise ValueError(f"Shard length does not match header: {path}")
    if weight_map is not None and seen != weight_map:
        raise ValueError("Model index and tensor headers disagree")
    return {
        "model_type": config.get("model_type"),
        "shards": len(shards),
        "tensors": len(seen),
        "header_and_size": "PASS",
        "tensor_payload_or_gpu_load": "NOT_CHECKED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["inventory", "check", "model"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"Missing root: {args.root}")
    if args.mode == "model":
        result = check_model(args.root)
    else:
        if args.inventory is None:
            parser.error("--inventory required")
        if args.inventory.resolve().is_relative_to(args.root.resolve()):
            parser.error("Inventory must be stored outside the dataset")
        if args.mode == "inventory":
            result = inventory(args.root)
            args.inventory.parent.mkdir(parents=True, exist_ok=True)
            args.inventory.write_text(json.dumps(result, indent=2) + "\n")
            result = {"files": len(result), "bytes": sum(result.values())}
        else:
            result = check_inventory(args.root, args.inventory)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
