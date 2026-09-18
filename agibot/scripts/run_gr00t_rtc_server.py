#!/usr/bin/env python3
"""Separate opt-in RTC model service. Does not import GDK or move a robot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    parser.add_argument("--port", type=int, default=5565)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("invalid port")
    from agibot.rtc.policy import RtcGr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.server_client import PolicyServer

    # Modality/stats come from the actual checkpoint, not a modified horizon.
    policy = RtcGr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=str(args.model_path),
        device=args.device,
        strict=True,
    )
    print(json.dumps({"event": "rtc_config", **policy.get_rtc_config()}), flush=True)
    with PolicyServer(policy=policy, host=args.host, port=args.port) as server:
        server.register_endpoint("get_rtc_config", policy.get_rtc_config, requires_input=False)
        try:
            server.run()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
