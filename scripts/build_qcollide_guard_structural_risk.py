#!/usr/bin/env python3
"""Build the Guard structural residual-risk utility table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.guard_structural_risk import build_guard_structural_risk_table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    artifact = build_guard_structural_risk_table(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "summary": artifact["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
