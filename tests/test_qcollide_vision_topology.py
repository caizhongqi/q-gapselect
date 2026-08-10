import numpy as np
import pytest

pytest.importorskip("torch")

from qgapselect.qcollide.vision_attack import evaluate_packing
from qgapselect.qcollide.vision_types import (
    AnchorPanel,
    AttackCandidate,
    ProjectionCalibration,
)


def test_visual_packing_emits_cycles_entropy_capacity_dimension_and_persistence() -> None:
    anchors = AnchorPanel(
        images=np.zeros((2, 2, 2), dtype=np.float32),
        labels=np.array([0, 0]),
        hidden=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        logits=np.zeros((2, 2)),
        hidden_jacobians=np.zeros((2, 3, 4)),
        targets=np.array([1, 1]),
    )
    attacks = tuple(
        AttackCandidate(
            source_index=index,
            source_label=0,
            target_label=1,
            image=np.zeros((2, 2), dtype=np.float32),
            hidden=np.array([0.0, 2.0 + index, 0.0]),
            logits=np.array([0.0, 1.0]),
            input_l2=1.0,
            source_control_distance=0.0,
            source_payload_distance=1.0,
            openness=1.0,
            tunnel_dimension=3,
        )
        for index in range(2)
    )
    calibration = ProjectionCalibration(
        projection=np.array([[1.0, 0.0, 0.0]]),
        standardized_projection=np.array([[1.0, 0.0, 0.0]]),
        control_epsilon=0.01,
        payload_delta=0.5,
        control_acceptance=0.95,
        payload_exceedance=0.05,
    )
    result = evaluate_packing(attacks, anchors, calibration)
    assert result.topology is not None
    assert result.filtration is not None
    assert result.edge_count == 4
    assert result.matching_size == 2
    assert result.topology.beta0_active == 1
    assert result.topology.beta1_active == 1
    assert result.topology.normalized_component_edge_entropy == 0.0
    assert result.topology.displacement_rank_95 == 1
    assert np.isclose(result.topology.displacement_entropy_rank, 1.0)
    assert result.filtration.summary.capacity_monotone
    assert np.isclose(result.filtration.summary.capacity_robustness_ratio, 1.0)
    assert result.filtration.basin_persistence.essential_interval_count == 1
    assert np.isclose(
        result.filtration.basin_persistence.normalized_total_lifetime,
        0.5,
    )
