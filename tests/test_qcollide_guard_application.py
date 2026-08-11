import numpy as np

from qgapselect.qcollide.guard_application import (
    GuardCollisionThresholds,
    build_guard_collision_graph,
    calibrate_acceptance_threshold,
    calibrate_collision_thresholds,
    nearest_control_discovery,
    packed_classical_query_proxy,
    packed_quantum_query_proxy,
    query_budget_grid,
    random_pair_discovery,
)


def test_acceptance_threshold_matches_requested_benign_quantile() -> None:
    scores = np.linspace(0.0, 1.0, 101)
    threshold = calibrate_acceptance_threshold(scores, benign_acceptance=0.9)
    assert np.isclose(threshold, 0.9)


def test_collision_graph_uses_all_three_predicates() -> None:
    left_guard = np.array([[1.0, 0.0], [0.0, 1.0]])
    right_guard = np.array([[1.0, 0.0], [0.0, 1.0]])
    left_payload = np.array([[1.0, 0.0], [0.0, 1.0]])
    right_payload = np.array([[0.0, 1.0], [1.0, 0.0]])
    left_logits = np.array([[1.0, 0.0], [0.0, 1.0]])
    right_logits = np.array([[0.0, 1.0], [1.0, 0.0]])
    graph = build_guard_collision_graph(
        left_guard_embeddings=left_guard,
        right_guard_embeddings=right_guard,
        left_behavior_embeddings=left_payload,
        right_behavior_embeddings=right_payload,
        left_behavior_logits=left_logits,
        right_behavior_logits=right_logits,
        thresholds=GuardCollisionThresholds(
            acceptance_threshold=0.5,
            control_epsilon=0.01,
            payload_delta=0.9,
            behavior_gamma=0.9,
        ),
    )
    assert graph.edge_count == 2
    assert graph.matching_size == 2


def test_nearest_control_baseline_dominates_when_true_edges_are_nearest() -> None:
    left_guard = np.array([[1.0, 0.0], [0.0, 1.0]])
    right_guard = np.array([[1.0, 0.0], [0.0, 1.0]])
    graph = build_guard_collision_graph(
        left_guard_embeddings=left_guard,
        right_guard_embeddings=right_guard,
        left_behavior_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]]),
        right_behavior_embeddings=np.array([[0.0, 1.0], [1.0, 0.0]]),
        left_behavior_logits=np.array([[1.0, 0.0], [0.0, 1.0]]),
        right_behavior_logits=np.array([[0.0, 1.0], [1.0, 0.0]]),
        thresholds=GuardCollisionThresholds(0.5, 0.01, 0.9, 0.9),
    )
    assert nearest_control_discovery(graph, query_budget=2) == 2
    random_value = random_pair_discovery(graph, query_budget=2, seed=4)
    assert 0 <= random_value <= 2


def test_collision_thresholds_use_benign_nearest_neighbours() -> None:
    benign = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ]
    )
    thresholds = calibrate_collision_thresholds(
        benign_guard_scores=np.array([0.01, 0.02, 0.03, 0.04]),
        benign_guard_embeddings=benign,
        benign_behavior_embeddings=np.roll(benign, 1, axis=1),
        benign_behavior_logits=np.roll(benign, 2, axis=1),
        benign_acceptance=0.75,
        control_quantile=0.5,
        payload_quantile=0.5,
        behavior_quantile=0.5,
    )
    assert 0.0 < thresholds.acceptance_threshold < 0.05
    assert thresholds.control_epsilon >= 0.0
    assert thresholds.payload_delta >= 0.0
    assert thresholds.behavior_gamma >= 0.0


def test_query_proxies_have_quantum_cubic_root_advantage() -> None:
    classical = packed_classical_query_proxy(256, 256, 16)
    quantum = packed_quantum_query_proxy(256, 256, 16)
    assert quantum < classical


def test_query_budget_grid_ends_at_all_pairs() -> None:
    grid = query_budget_grid(20, 30)
    assert grid[-1] == 600
    assert all(left < right for left, right in zip(grid, grid[1:], strict=False))
