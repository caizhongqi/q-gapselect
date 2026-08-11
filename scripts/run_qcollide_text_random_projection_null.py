#!/usr/bin/env python3
"""Run one matched-rank random-subspace text falsification component."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.text_null_controls import run_text_random_projection_null_component


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture-seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact = run_text_random_projection_null_component(
        config,
        model_name=args.model,
        fixture_seed=args.fixture_seed,
    )
    artifact["experiment_config"] = str(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "model": args.model}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
