import numpy as np

from qgapselect.qcollide.topology_results import build_collision_topology_run_summary


def _row(architecture: str, rank: int, seed: int, packing: float) -> dict[str, object]:
    edge_count = max(1, int(round(10 * packing)))
    return {
        "architecture": architecture,
        "model_seed": seed,
        "visible_rank": rank,
        "intervention": "baseline",
        "closure_rank": 0,
        "mean_openness": packing,
        "packing_fraction": packing,
        "collision_beta0_active": max(1, int(round(4 * packing))),
        "collision_beta1_active": max(0, edge_count - 2),
        "edge_count": edge_count,
        "collision_normalized_component_entropy": packing,
        "collision_effective_component_count": 1.0 + packing,
        "collision_largest_component_edge_fraction": 1.0 - 0.5 * packing,
        "collision_independence_ratio": packing / edge_count,
        "collision_displacement_entropy_rank": 1.0 + 3.0 * packing,
        "collision_displacement_stable_rank": 1.0 + 2.0 * packing,
    }


def _artifact() -> dict[str, object]:
    rows = []
    for architecture, offset in (("cnn", 0.0), ("tiny_vit", 0.1)):
        for rank, packing in ((1, 0.8), (2, 0.5), (4, 0.2)):
            for seed in (0, 1):
                rows.append(_row(architecture, rank, seed, packing + offset))
    return {
        "claim_scope": {
            "dataset": "test-images",
            "full_collision_topology_emitted": True,
        },
        "rows": rows,
    }


def test_summary_recovers_cells_architecture_contrasts_and_associations() -> None:
    summary = build_collision_topology_run_summary(_artifact())
    assert summary["cell_count"] == 6
    assert len(summary["architecture_contrasts"]) == 3
    assert np.isclose(summary["capacity_associations"]["openness"], 1.0)
    assert summary["claim_scope"]["training_phase_transition_claimed"] is False
    assert all(
        row["capacity_fraction_nonincreasing_fraction"] == 1.0
        for row in summary["monotonicity"]
    )


def test_uninstrumented_artifact_is_rejected() -> None:
    artifact = _artifact()
    artifact["claim_scope"]["full_collision_topology_emitted"] = False
    with np.testing.assert_raises(ValueError):
        build_collision_topology_run_summary(artifact)
