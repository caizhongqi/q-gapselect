"""Same-interface analytic evaluation for frozen Q-COLLIDE fixtures."""

from __future__ import annotations

from dataclasses import asdict
from math import isfinite
from typing import Any

from .contracts import CollisionInstance
from .costs import (
    classical_packed_cost,
    forbidden_claw_cost_for_pair_oracle,
    pair_grover_cost,
    prefix_rms_cost,
    prefix_survival_rates,
    product_johnson_cost,
    weighted_qcollide_cost,
)
from .graph import packing_statistics


def evaluate_instance(instance: CollisionInstance) -> dict[str, Any]:
    """Evaluate graph structure and admissible analytic cost models.

    The result explicitly labels every quantum number as an analytic query-cost
    proxy. No statevector, circuit, hardware, QRAM, or wall-clock advantage is
    claimed by this function.
    """

    stats = packing_statistics(instance)
    result: dict[str, Any] = {
        "name": instance.name,
        "fixture": instance.metadata.get("fixture"),
        "fingerprint": instance.metadata.get("fingerprint"),
        "endpoint_local": instance.endpoint_local,
        "n_left": instance.n_left,
        "n_right": instance.n_right,
        "packing": asdict(stats),
        "cost_semantics": "analytic_endpoint_query_proxy",
    }

    if not instance.endpoint_local:
        result["pair_grover"] = pair_grover_cost(
            instance.n_left,
            instance.n_right,
            max(1, stats.edge_count),
        )
        result["structured_claw_admissible"] = False
        result["structured_claw_cost"] = forbidden_claw_cost_for_pair_oracle(
            endpoint_local=False
        )
        return result

    result["structured_claw_admissible"] = True
    if stats.matching_size == 0:
        result["classical_packed"] = None
        result["product_johnson"] = None
        return result

    if instance.n_left != instance.n_right:
        raise ValueError("current packed-cost calibration requires balanced domains")
    classical = classical_packed_cost(instance.n_left, stats.matching_size)
    quantum = product_johnson_cost(instance.n_left, stats.matching_size)
    result["classical_packed"] = classical
    result["product_johnson"] = asdict(quantum)

    if instance.left[0].prefixes:
        survival = prefix_survival_rates(
            (record.prefixes for record in instance.left),
            (record.prefixes for record in instance.right),
        )
        rms = prefix_rms_cost(instance.left[0].stage_costs, survival)
        result["prefix_survival_rates"] = survival
        result["rms_endpoint_cost"] = rms
        result["max_endpoint_cost"] = sum(instance.left[0].stage_costs)
        result["weighted_qcollide"] = weighted_qcollide_cost(
            instance.n_left,
            stats.matching_size,
            rms,
        )
        result["variable_time_speedup_proxy"] = (
            result["max_endpoint_cost"] / rms if isfinite(rms) and rms > 0.0 else None
        )
    return result
