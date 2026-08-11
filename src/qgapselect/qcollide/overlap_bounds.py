"""Provable capacity envelopes for overlapping bipartite collision graphs."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from .overlap_capacity import OverlapGraph


@dataclass(frozen=True)
class CapacityEnvelope:
    edge_count: int
    maximum_degree: int
    lower_bound: int
    upper_bound: int
    true_matching_size: int
    lower_ratio: float
    upper_ratio: float


def bipartite_capacity_envelope(graph: OverlapGraph) -> CapacityEnvelope:
    """Return deterministic matching-capacity bounds from edge count and max degree.

    A bipartite graph of maximum degree Delta admits a proper edge colouring
    with Delta colours. Each colour class is a matching, so one class contains
    at least ceil(|E| / Delta) edges. Therefore

        ceil(|E| / Delta) <= nu(G) <= min(|A|, |B|, |E|).

    The lower bound is exact for a star and the upper bound is exact for a
    disjoint matching. This envelope is intentionally coarse but applies to
    arbitrary bipartite overlap without assuming independent witnesses.
    """

    edges = graph.edge_count
    degree = graph.maximum_degree
    truth = graph.matching_size
    if edges == 0:
        lower = upper = 0
    else:
        if degree <= 0:
            raise RuntimeError("nonempty graph must have positive maximum degree")
        lower = ceil(edges / degree)
        upper = min(graph.n_left, graph.n_right, edges)
    return CapacityEnvelope(
        edge_count=edges,
        maximum_degree=degree,
        lower_bound=lower,
        upper_bound=upper,
        true_matching_size=truth,
        lower_ratio=(lower / truth) if truth else 1.0,
        upper_ratio=(upper / truth) if truth else 1.0,
    )


__all__ = ["CapacityEnvelope", "bipartite_capacity_envelope"]
