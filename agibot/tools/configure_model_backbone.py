#!/usr/bin/env python3
"""Point a copied GR00T inference bundle at its bundled Cosmos backbone.

Only the active model_name values in model/config.json and
model/processor_config.json are changed. Weights and normalization values stay as-is.
"""

import argparse
import json
from pathlib import Path


def configure(bundle: Path, check_only: bool = False) -> list[tuple[Path, str, str]]:
    bundle = bundle.expanduser().resolve()
    model = bundle / "model"
    backbone = bundle / "backbone" / "Cosmos-Reason2-2B"
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        if not (backbone / name).is_file():
            raise FileNotFoundError(backbone / name)

    updates = []
    for name, nested in (("config.json", False), ("processor_config.json", True)):
        path = model / name
        document = json.loads(path.read_text(encoding="utf-8"))
        values = document["processor_kwargs"] if nested else document
        old = values["model_name"]
        if not isinstance(old, str):
            raise TypeError(f"{path}: model_name must be a string")
        new = str(backbone)
        updates.append((path, old, new, document))

    if not check_only:
        for path, old, new, document in updates:
            if old == new:
                continue
            values = document["processor_kwargs"] if path.name == "processor_config.json" else document
            values["model_name"] = new
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)

    return [(path, old, new) for path, old, new, _ in updates]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="directory containing model/ and backbone/")
    parser.add_argument("--check", action="store_true", help="read and validate without changing files")
    args = parser.parse_args()
    for path, old, new in configure(args.bundle, check_only=args.check):
        print(f"{path}: {old} -> {new}" if old != new else f"{path}: already configured")


if __name__ == "__main__":
    main()
