#!/usr/bin/env python3
"""Print the release-phase jaw trace from a hardware bridge-test report."""

import json
import sys

report = json.load(open(sys.argv[1]))
for chunk in report.get("chunks", []):
    for row in chunk.get("rows", []):
        release = row.get("release")
        if not release:
            continue
        samples = release["samples"]
        print(f"release row {row['row']}  samples={len(samples)}  "
              f"commands_sent={release['commands_sent']}  "
              f"elapsed_s={release['elapsed_s']:.3f}")
        previous = None
        for sample in samples:
            position = sample["raw_position"]
            delta = "" if previous is None else f"  delta={position - previous:+.5f}"
            print(
                f"  t={sample['elapsed_s']:6.3f}s  pos={position:+.6f}  "
                f"status={sample['motor_status']}  "
                f"effort={sample['effort']:6.2f}{delta}"
            )
            previous = position
