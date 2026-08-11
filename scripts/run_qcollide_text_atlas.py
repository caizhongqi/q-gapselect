#!/usr/bin/env python3
"""Run one pretrained text-encoder Q-COLLIDE atlas component."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.text_atlas import run_text_atlas_component


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qcollide_text_atlas_phase1.json"),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture-seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact = run_text_atlas_component(
        config,
        model_name=args.model,
        fixture_seed=args.fixture_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model": args.model,
                "fixture_seed": args.fixture_seed,
                "evaluation_accuracy": artifact["model_diagnostics"]["evaluation_accuracy"],
                "rank_count": len(artifact["rows"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
