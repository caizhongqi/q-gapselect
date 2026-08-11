"""Decision-oriented query-budget utilities for real Guard collision graphs."""

from __future__ import annotations

import numpy as np

from .graph import maximum_bipartite_matching
from .guard_application import GuardCollisionGraph


def collision_diversified_control_discovery(
    graph: GuardCollisionGraph,
    *,
    query_budget: int,
) -> int:
    """Discover independent collisions with the same pre-query information as nearest-control.

    The policy knows only the pairwise control-distance matrix before querying.
    After each queried pair it may use that pair's observed collision/non-collision
    outcome. Pairs incident to endpoints that have already produced a collision
    are deprioritized, then endpoint query load is balanced, and control distance
    breaks remaining ties. Unqueried collision labels are never inspected.
    """

    if query_budget < 0:
        raise ValueError("query_budget must be non-negative")
    total = graph.n_left * graph.n_right
    budget = min(query_budget, total)
    if budget == 0 or total == 0:
        return 0

    distances = np.asarray(graph.control_distances, dtype=float)
    if distances.shape != (graph.n_left, graph.n_right):
        raise ValueError("control-distance matrix does not match graph dimensions")

    queried = np.zeros((graph.n_left, graph.n_right), dtype=bool)
    left_queries = np.zeros(graph.n_left, dtype=np.int64)
    right_queries = np.zeros(graph.n_right, dtype=np.int64)
    left_positive = np.zeros(graph.n_left, dtype=bool)
    right_positive = np.zeros(graph.n_right, dtype=bool)
    found = [[] for _ in range(graph.n_left)]
    edge_sets = [set(row) for row in graph.adjacency]

    left_index = np.repeat(np.arange(graph.n_left), graph.n_right)
    right_index = np.tile(np.arange(graph.n_right), graph.n_left)
    distance_flat = distances.reshape(-1)

    for _ in range(budget):
        available = ~queried.reshape(-1)
        candidate_flat = np.flatnonzero(available)
        candidate_left = left_index[candidate_flat]
        candidate_right = right_index[candidate_flat]
        positive_endpoint_penalty = (
            left_positive[candidate_left].astype(np.int8)
            + right_positive[candidate_right].astype(np.int8)
        )
        query_load = left_queries[candidate_left] + right_queries[candidate_right]
        order = np.lexsort(
            (
                candidate_flat,
                distance_flat[candidate_flat],
                query_load,
                positive_endpoint_penalty,
            )
        )
        flat = int(candidate_flat[int(order[0])])
        left = int(left_index[flat])
        right = int(right_index[flat])
        queried[left, right] = True
        left_queries[left] += 1
        right_queries[right] += 1
        if right in edge_sets[left]:
            found[left].append(right)
            left_positive[left] = True
            right_positive[right] = True

    adjacency = tuple(tuple(row) for row in found)
    return maximum_bipartite_matching(adjacency, graph.n_right)


__all__ = ["collision_diversified_control_discovery"]
