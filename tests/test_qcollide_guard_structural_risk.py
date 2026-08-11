from __future__ import annotations

from qgapselect.qcollide.guard_structural_risk import build_guard_structural_risk_table


def _record(
    guard: str,
    seed: int,
    *,
    benign: float,
    injection: float,
    accepted: int,
    k: int,
    edges: int,
):
    return {
        "guard_model": guard,
        "seed": seed,
        "domain": {
            "benign_acceptance_observed": benign,
            "injection_acceptance_observed": injection,
            "accepted_injection_prompts": accepted,
        },
        "collision_graph": {
            "matching_capacity": k,
            "capacity_fraction": k / accepted if accepted else 0.0,
            "edge_count": edges,
        },
    }


def test_structural_risk_table_keeps_rate_and_topology_separate() -> None:
    source = {
        "artifact_type": "qcollide_guard_application_main_table",
        "records": [
            _record(
                "guard-a",
                1,
                benign=0.95,
                injection=0.20,
                accepted=20,
                k=5,
                edges=30,
            ),
            _record(
                "guard-a",
                2,
                benign=0.96,
                injection=0.18,
                accepted=18,
                k=6,
                edges=24,
            ),
            _record(
                "guard-b",
                1,
                benign=0.97,
                injection=0.02,
                accepted=2,
                k=0,
                edges=0,
            ),
            _record(
                "guard-b",
                2,
                benign=0.96,
                injection=0.01,
                accepted=1,
                k=0,
                edges=0,
            ),
        ],
        "claim_boundary": {"public_existing_prompts_only": True},
    }
    result = build_guard_structural_risk_table(source)
    assert result["summary"]["guard_count"] == 2
    by_guard = {row["guard_model"]: row for row in result["rows"]}
    assert by_guard["guard-a"]["mean_independent_failure_capacity"] == 5.5
    assert by_guard["guard-b"]["nonzero_capacity_seed_fraction"] == 0.0
    assert result["claim_boundary"]["topology_replaces_failure_rate_metrics"] is False
    assert (
        result["claim_boundary"][
            "incremental_prediction_beyond_injection_acceptance_proved"
        ]
        is False
    )
