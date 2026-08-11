#!/usr/bin/env python3
"""Merge real Guard query-budget utility components."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input_directory.glob("*.json"))
    ]
    if not records:
        raise RuntimeError("no Guard utility component artifacts found")
    if {record["artifact_type"] for record in records} != {
        "qcollide_guard_query_budget_utility_component"
    }:
        raise RuntimeError("unexpected Guard utility artifact type")

    by_fraction = defaultdict(list)
    for record in records:
        for row in record["discovery_curves"]:
            by_fraction[float(row["query_fraction"])].append(row)

    table = []
    for fraction, rows in sorted(by_fraction.items()):
        if len(rows) != len(records):
            raise RuntimeError(f"incomplete query-fraction cell {fraction}")
        random_utility = [float(row["random_utility_fraction"]) for row in rows]
        nearest_utility = [float(row["nearest_control_utility_fraction"]) for row in rows]
        diversified_utility = [
            float(row["collision_diversified_utility_fraction"]) for row in rows
        ]
        table.append(
            {
                "query_fraction": fraction,
                "seed_count": len(rows),
                "mean_random_utility": statistics.mean(random_utility),
                "mean_nearest_control_utility": statistics.mean(nearest_utility),
                "mean_collision_diversified_utility": statistics.mean(diversified_utility),
                "mean_diversified_minus_nearest": statistics.mean(
                    diversified - nearest
                    for diversified, nearest in zip(
                        diversified_utility,
                        nearest_utility,
                        strict=True,
                    )
                ),
                "diversified_at_least_nearest_seed_fraction": statistics.mean(
                    diversified >= nearest
                    for diversified, nearest in zip(
                        diversified_utility,
                        nearest_utility,
                        strict=True,
                    )
                ),
                "diversified_strictly_better_seed_fraction": statistics.mean(
                    diversified > nearest
                    for diversified, nearest in zip(
                        diversified_utility,
                        nearest_utility,
                        strict=True,
                    )
                ),
            }
        )

    payload = {
        "artifact_type": "qcollide_guard_query_budget_utility_main_table",
        "schema_version": 1,
        "guard_model": records[0]["guard_model"],
        "behavior_model": records[0]["behavior_model"],
        "dataset": records[0]["dataset"],
        "seeds": sorted(int(record["seed"]) for record in records),
        "unique_fixture_count": len({record["fixture_sha256"] for record in records}),
        "mean_full_matching_capacity": statistics.mean(
            float(record["collision_graph"]["matching_capacity"]) for record in records
        ),
        "table": table,
        "records": records,
        "claim_boundary": {
            "fixture_seeds_are_resampling_replications": True,
            "independent_pretrained_model_replications": False,
            "same_query_budget_for_methods": True,
            "nearest_and_diversified_share_prequery_control_information": True,
            "unqueried_collision_labels_visible": False,
            "quantum_runtime_claimed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "rows": len(table)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
