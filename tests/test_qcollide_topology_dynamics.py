import numpy as np

from qgapselect.qcollide.topology import collision_topology_metrics
from qgapselect.qcollide.topology_dynamics import (
    TopologySnapshot,
    analyze_topology_trajectory,
    detect_topology_change_point,
    topology_snapshot,
)


def test_piecewise_trajectory_recovers_the_known_breakpoint() -> None:
    snapshots = []
    for step in range(10):
        value = 0.02 * step if step < 5 else 0.1 + 0.25 * (step - 4)
        snapshots.append(
            TopologySnapshot(
                step=float(step),
                capacity_fraction=value,
                normalized_component_edge_entropy=0.0,
                displacement_entropy_rank=1.0,
                beta1_density=0.0,
            )
        )
    change = detect_topology_change_point(snapshots)
    assert change.split_index == 5
    assert np.isclose(change.transition_step, 4.5)
    assert change.relative_sse_improvement > 0.99
    assert change.slope_after > change.slope_before


def test_snapshot_and_multimetric_analysis_preserve_claim_boundary() -> None:
    metrics = collision_topology_metrics(
        ((0,), (1,)),
        2,
        edge_vectors=np.eye(2),
    )
    snapshots = [topology_snapshot(float(step), metrics) for step in range(4)]
    output = analyze_topology_trajectory(snapshots)
    assert set(output) == {
        "beta1_density",
        "capacity_fraction",
        "displacement_entropy_rank",
        "normalized_component_edge_entropy",
    }
    assert all(change.relative_sse_improvement == 0.0 for change in output.values())


def test_duplicate_steps_and_missing_dimension_are_rejected_or_skipped() -> None:
    duplicate = [
        TopologySnapshot(0.0, 0.0, 0.0, None, 0.0),
        TopologySnapshot(0.0, 0.1, 0.0, None, 0.0),
        TopologySnapshot(1.0, 0.2, 0.0, None, 0.0),
        TopologySnapshot(2.0, 0.3, 0.0, None, 0.0),
    ]
    with np.testing.assert_raises(ValueError):
        detect_topology_change_point(duplicate)
    valid = [
        TopologySnapshot(float(step), 0.1 * step, 0.0, None, 0.0)
        for step in range(4)
    ]
    assert "displacement_entropy_rank" not in analyze_topology_trajectory(valid)
