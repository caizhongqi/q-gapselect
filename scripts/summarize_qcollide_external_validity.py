#!/usr/bin/env python3
"""Build a cross-dataset Q-COLLIDE capacity meta-analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.capacity_meta import build_capacity_meta_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qcollide_capacity_meta_summary.json"),
    )
    args = parser.parse_args()
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    result = build_capacity_meta_summary(documents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "cells": result["cell_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
