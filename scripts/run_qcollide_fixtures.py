#!/usr/bin/env python3
"""Run a deterministic, lightweight F0--F4 Q-COLLIDE calibration panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qgapselect.qcollide import (
    evaluate_instance,
    geometry_to_packing,
    packed_claw,
    pair_oracle_negative,
    random_range,
    weighted_prefix_claw,
)


def _instances(config: dict[str, Any]):
    seed = int(config["master_seed"])
    yield pair_oracle_negative(**config["f0"], seed=seed + 0)
    yield packed_claw(**config["f1"], seed=seed + 1)
    yield random_range(**config["f2"], seed=seed + 2)
    f3 = dict(config["f3"])
    f3["cumulative_bits"] = tuple(f3["cumulative_bits"])
    f3["stage_costs"] = tuple(float(x) for x in f3["stage_costs"])
    yield weighted_prefix_claw(**f3, seed=seed + 3)
    yield geometry_to_packing(**config["f4"], seed=seed + 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/qcollide_fixtures.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qcollide_fixture_diagnostic.json"),
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    records = [evaluate_instance(instance) for instance in _instances(config)]
    payload = {
        "schema_version": 1,
        "experiment": "qcollide_f0_f4_calibration_v1",
        "master_seed": config["master_seed"],
        "claims": {
            "quantum_numbers_are": "analytic endpoint-query proxies",
            "not_claimed": [
                "statevector execution",
                "compiled circuit advantage",
                "hardware runtime advantage",
                "free QRAM",
            ],
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
