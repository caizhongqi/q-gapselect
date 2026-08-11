"""Structural residual-risk summaries for real Guard × downstream-model fixtures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def build_guard_structural_risk_table(
    artifact: Mapping[str, object],
) -> dict[str, object]:
    """Summarize independent residual failures for guard-model selection.

    This table is descriptive rather than causal.  It intentionally reports raw
    injection acceptance alongside collision capacity so that topology is not
    presented as a replacement for conventional failure-rate metrics.
    """

    if str(artifact.get("artifact_type")) != "qcollide_guard_application_main_table":
        raise ValueError("unexpected Guard application artifact type")
    records = list(artifact["records"])
    if not records:
        raise ValueError("Guard application artifact contains no records")

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["guard_model"]), []).append(record)

    rows: list[dict[str, object]] = []
    for guard_model, group in sorted(grouped.items()):
        benign = [float(item["domain"]["benign_acceptance_observed"]) for item in group]
        injection = [
            float(item["domain"]["injection_acceptance_observed"]) for item in group
        ]
        accepted_injections = [
            float(item["domain"]["accepted_injection_prompts"]) for item in group
        ]
        capacities = [
            float(item["collision_graph"]["matching_capacity"]) for item in group
        ]
        capacity_fractions = [
            float(item["collision_graph"]["capacity_fraction"]) for item in group
        ]
        edge_counts = [float(item["collision_graph"]["edge_count"]) for item in group]
        redundancy = [
            edges / capacity if capacity > 0.0 else 0.0
            for edges, capacity in zip(edge_counts, capacities, strict=True)
        ]
        benign_mean, benign_std = _mean_std(benign)
        injection_mean, injection_std = _mean_std(injection)
        accepted_mean, accepted_std = _mean_std(accepted_injections)
        capacity_mean, capacity_std = _mean_std(capacities)
        fraction_mean, fraction_std = _mean_std(capacity_fractions)
        edge_mean, edge_std = _mean_std(edge_counts)
        redundancy_positive = [
            value
            for value, capacity in zip(redundancy, capacities, strict=True)
            if capacity > 0.0
        ]
        redundancy_mean, redundancy_std = (
            _mean_std(redundancy_positive) if redundancy_positive else (None, None)
        )
        rows.append(
            {
                "guard_model": guard_model,
                "seed_count": len(group),
                "seeds": sorted(int(item["seed"]) for item in group),
                "mean_benign_acceptance": benign_mean,
                "std_benign_acceptance": benign_std,
                "mean_injection_acceptance": injection_mean,
                "std_injection_acceptance": injection_std,
                "mean_accepted_injection_count": accepted_mean,
                "std_accepted_injection_count": accepted_std,
                "mean_independent_failure_capacity": capacity_mean,
                "std_independent_failure_capacity": capacity_std,
                "mean_independent_fraction_of_accepted_failures": fraction_mean,
                "std_independent_fraction_of_accepted_failures": fraction_std,
                "mean_collision_edge_count": edge_mean,
                "std_collision_edge_count": edge_std,
                "mean_edges_per_independent_failure_positive_capacity": redundancy_mean,
                "std_edges_per_independent_failure_positive_capacity": redundancy_std,
                "nonzero_capacity_seed_fraction": float(
                    np.mean(np.asarray(capacities, dtype=float) > 0.0)
                ),
            }
        )

    benign_means = [float(row["mean_benign_acceptance"]) for row in rows]
    return {
        "artifact_type": "qcollide_guard_structural_risk_utility_table",
        "schema_version": 1,
        "source_artifact_type": str(artifact["artifact_type"]),
        "rows": rows,
        "summary": {
            "guard_count": len(rows),
            "maximum_pairwise_mean_benign_acceptance_gap": (
                max(benign_means) - min(benign_means) if benign_means else 0.0
            ),
            "all_fixtures_use_public_existing_prompts": bool(
                artifact.get("claim_boundary", {}).get("public_existing_prompts_only")
            ),
        },
        "claim_boundary": {
            "practical_use": "guard_model_selection_and_residual_failure_triage",
            "topology_replaces_failure_rate_metrics": False,
            "incremental_prediction_beyond_injection_acceptance_proved": False,
            "causal_guard_architecture_effect_proved": False,
            "independent_failure_capacity_is_exact_on_observed_collision_graph": True,
            "public_existing_prompts_only": True,
            "generated_jailbreak_content": False,
        },
    }


__all__ = ["build_guard_structural_risk_table"]
