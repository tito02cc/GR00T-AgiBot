#!/usr/bin/env python3
"""Save an existing read-only observation bridge's JPEGs and metadata."""

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from g2_groot_live_observation_client import _request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # Packet validation only; no content hashing, resizing, or control API.
    metadata, blobs = _request(args.host, args.port, "snapshot", 10.0)
    records = metadata.get("images", [])
    if len(records) != 2 or [item["name"] for item in records] != ["head_color", "hand_right"]:
        raise RuntimeError("expected head and right-wrist images")
    if len(blobs) != 2 or any(item["encoding"] not in ("JPEG", "PNG") for item in records):
        raise RuntimeError("expected two encoded images")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for record, blob in zip(records, blobs):
        extension = ".jpg" if record["encoding"] == "JPEG" else ".png"
        target = args.output_dir / (record["name"] + extension)
        target.write_bytes(blob)
        print(target.resolve())
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
