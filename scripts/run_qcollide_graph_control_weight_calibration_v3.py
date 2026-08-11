#!/usr/bin/env python3
"""Run one frozen Cora auxiliary-control weight calibration cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgapselect.qcollide.graph_controlled_atlas import run_controlled_graph_atlas_component


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--model-seed", required=True, type=int)
    parser.add_argument("--control-loss-weight", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    allowed_weights = tuple(float(value) for value in config["control_loss_weights"])
    if not any(abs(args.control_loss_weight - value) <= 1e-12 for value in allowed_weights):
        raise ValueError("control-loss-weight is outside the frozen calibration grid")
    component_config = dict(config)
    component_config["control_loss_weight"] = float(args.control_loss_weight)
    component_config.pop("control_loss_weights", None)
    artifact = run_controlled_graph_atlas_component(
        component_config,
        architecture=args.architecture,
        model_seed=args.model_seed,
    )
    artifact["calibration"] = {
        "protocol": "independent_weight_calibration_v3",
        "control_loss_weight": float(args.control_loss_weight),
        "main_model_seeds_used_for_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "architecture": args.architecture,
                "model_seed": args.model_seed,
                "control_loss_weight": args.control_loss_weight,
                "gates": artifact["gates"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
