#!/usr/bin/env python3
"""Merge independent CIFAR topology components into one frozen diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.cifar_results import (
    compact_cifar_topology_summary,
    merge_cifar_topology_components,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    files = sorted(args.input_directory.rglob("*.json"))
    if not files:
        raise RuntimeError(f"no component JSON files found under {args.input_directory}")
    components = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    artifact = merge_cifar_topology_components(components, config=config)
    summary = compact_cifar_topology_summary(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "components": len(components),
                "rows": len(artifact["rows"]),
                "output": str(args.output),
                "summary": str(args.summary),
                "gates": artifact["gates"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
