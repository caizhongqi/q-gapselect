#!/usr/bin/env python3
"""Merge CIFAR checkpoint-performance replay components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.cifar_performance_results import (
    merge_cifar_performance_components,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    files = sorted(args.input_directory.rglob("*.json"))
    if not files:
        raise RuntimeError(f"no component JSON files found under {args.input_directory}")
    artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    merged = merge_cifar_performance_components(
        artifacts,
        configured_architectures=[str(value) for value in config["architectures"]],
        configured_seeds=[int(value) for value in config["model_seeds"]],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "components": merged["component_count"],
                "rows": len(merged["rows"]),
                "output": str(args.output),
                "gates": merged["gates"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
