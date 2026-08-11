#!/usr/bin/env python3
"""Build the CIFAR training-time collision-risk monitoring utility table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.training_risk_monitor import build_training_risk_monitor_table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    topology = json.loads(args.topology.read_text(encoding="utf-8"))
    performance = json.loads(args.performance.read_text(encoding="utf-8"))
    artifact = build_training_risk_monitor_table(topology, performance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "summary": artifact["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
