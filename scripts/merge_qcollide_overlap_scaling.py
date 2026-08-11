#!/usr/bin/env python3
"""Merge Q-COLLIDE overlap scaling components into the quantum main table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.overlap_scaling import merge_overlap_scaling_components


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    artifacts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input_directory.glob("*.json"))
    ]
    merged = merge_overlap_scaling_components(artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "gates": merged["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
