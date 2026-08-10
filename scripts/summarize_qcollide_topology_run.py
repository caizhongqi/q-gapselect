#!/usr/bin/env python3
"""Summarize graph-level collision topology from a merged visual artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.topology_results import build_collision_topology_run_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = json.loads(args.input.read_text(encoding="utf-8"))
    summary = build_collision_topology_run_summary(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["capacity_associations"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
