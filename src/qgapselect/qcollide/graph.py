"""Collision-edge construction and exact bipartite packing statistics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import fsum

import numpy as np

from .contracts import CollisionCriteria, CollisionInstance, EndpointRecord


def _euclidean(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> float:
    if len(lhs) != len(rhs):
        raise ValueError("vector dimensions must agree")
    if not lhs:
        return 0.0
    return float(np.linalg.norm(np.asarray(lhs, dtype=float) - np.asarray(rhs, dtype=float)))


def endpoint_pair_is_valid(
    left: EndpointRecord,
    right: EndpointRecord,
    criteria: CollisionCriteria,
) -> bool:
    """Evaluate the complete endpoint-local conditional-collision predicate."""

    if left.side != "A" or right.side != "B":
        raise ValueError("expected an A endpoint and a B endpoint")
    if not left.valid or not right.valid:
        return False
    if left.signature != right.signature:
        return False
    control_distance = _euclidean(left.control, right.control)
    payload_distance = _euclidean(left.payload, right.payload)
    behavior_distance = abs(left.behavior - right.behavior)
    return (
        control_distance <= criteria.control_epsilon
        and payload_distance >= criteria.payload_delta
        and behavior_distance >= criteria.behavior_gamma
    )


def build_adjacency(instance: CollisionInstance) -> tuple[tuple[int, ...], ...]:
    """Build a deterministic adjacency list without granting free search information."""

    adjacency: list[list[int]] = [[] for _ in instance.left]
    if not instance.endpoint_local:
        for i, j in sorted(instance.explicit_pair_marks):
            adjacency[i].append(j)
        return tuple(tuple(row) for row in adjacency)

    buckets: dict[tuple[int, ...], list[EndpointRecord]] = {}
    for record in instance.right:
        buckets.setdefault(record.signature, []).append(record)
    for left in instance.left:
        for right in buckets.get(left.signature, []):
            if endpoint_pair_is_valid(left, right, instance.criteria):
                adjacency[left.index].append(right.index)
    return tuple(tuple(row) for row in adjacency)


def maximum_bipartite_matching(adjacency: tuple[tuple[int, ...], ...], n_right: int) -> int:
    """Return the exact maximum matching size using Hopcroft--Karp."""

    n_left = len(adjacency)
    pair_left = [-1] * n_left
    pair_right = [-1] * n_right
    distance = [0] * n_left
    infinity = n_left + n_right + 1

    def bfs() -> bool:
        queue: deque[int] = deque()
        found = False
        for u in range(n_left):
            if pair_left[u] == -1:
                distance[u] = 0
                queue.append(u)
            else:
                distance[u] = infinity
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                mate = pair_right[v]
                if mate == -1:
                    found = True
                elif distance[mate] == infinity:
                    distance[mate] = distance[u] + 1
                    queue.append(mate)
        return found

    def dfs(u: int) -> bool:
        for v in adjacency[u]:
            mate = pair_right[v]
            if mate == -1 or (distance[mate] == distance[u] + 1 and dfs(mate)):
                pair_left[u] = v
                pair_right[v] = u
                return True
        distance[u] = infinity
        return False

    matching = 0
    while bfs():
        for u in range(n_left):
            if pair_left[u] == -1 and dfs(u):
                matching += 1
    return matching


def _gini(values: list[int]) -> float:
    if not values or sum(values) == 0:
        return 0.0
    ordered = sorted(float(value) for value in values)
    n = len(ordered)
    numerator = fsum((2 * i - n - 1) * value for i, value in enumerate(ordered, start=1))
    denominator = n * fsum(ordered)
    return float(numerator / denominator)


@dataclass(frozen=True)
class PackingStatistics:
    edge_count: int
    matching_size: int
    independence_ratio: float
    max_left_degree: int
    max_right_degree: int
    degree_gini: float


def packing_statistics(instance: CollisionInstance) -> PackingStatistics:
    adjacency = build_adjacency(instance)
    left_degrees = [len(row) for row in adjacency]
    right_degrees = [0] * instance.n_right
    for row in adjacency:
        for v in row:
            right_degrees[v] += 1
    edge_count = sum(left_degrees)
    matching_size = maximum_bipartite_matching(adjacency, instance.n_right)
    all_degrees = left_degrees + right_degrees
    return PackingStatistics(
        edge_count=edge_count,
        matching_size=matching_size,
        independence_ratio=(matching_size / edge_count) if edge_count else 0.0,
        max_left_degree=max(left_degrees, default=0),
        max_right_degree=max(right_degrees, default=0),
        degree_gini=_gini(all_degrees),
    )
