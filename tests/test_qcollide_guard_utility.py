import numpy as np

from qgapselect.qcollide.guard_application import GuardCollisionGraph, nearest_control_discovery
from qgapselect.qcollide.guard_utility import collision_diversified_control_discovery


def _graph(adjacency, distances):
    values = np.asarray(distances, dtype=float)
    return GuardCollisionGraph(
        adjacency=tuple(tuple(row) for row in adjacency),
        n_right=values.shape[1],
        control_distances=values,
        payload_distances=np.zeros_like(values),
        behavior_distances=np.zeros_like(values),
    )


def test_collision_diversification_can_recover_independent_modes_missed_by_nearest():
    graph = _graph(
        ((0,), (1,), (2,)),
        (
            (0.01, 0.02, 0.03),
            (0.06, 0.04, 0.07),
            (0.08, 0.09, 0.05),
        ),
    )
    assert nearest_control_discovery(graph, query_budget=3) == 1
    assert collision_diversified_control_discovery(graph, query_budget=3) == 3


def test_collision_diversified_policy_respects_zero_and_full_budgets():
    graph = _graph(
        ((0, 1), (0,)),
        (
            (0.1, 0.2),
            (0.3, 0.4),
        ),
    )
    assert collision_diversified_control_discovery(graph, query_budget=0) == 0
    assert collision_diversified_control_discovery(graph, query_budget=100) == graph.matching_size


def test_collision_diversified_policy_never_exceeds_exact_capacity():
    graph = _graph(
        ((0,), (), (1,)),
        (
            (0.1, 0.2),
            (0.3, 0.4),
            (0.5, 0.6),
        ),
    )
    for budget in range(7):
        discovered = collision_diversified_control_discovery(graph, query_budget=budget)
        assert 0 <= discovered <= graph.matching_size
