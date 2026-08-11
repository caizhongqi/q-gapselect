#!/usr/bin/env python3
"""Merge control-validated Cora collision-atlas v2 components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.graph_controlled_results import merge_controlled_graph_components


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifacts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input_directory.glob("*.json"))
    ]
    merged = merge_controlled_graph_components(
        artifacts,
        configured_architectures=config["architectures"],
        configured_seeds=config["model_seeds"],
    )
    merged["experiment_config"] = str(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "gates": merged["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
