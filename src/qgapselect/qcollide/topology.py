"""Finite collision-complex metrics for conditional neural collisions.

The active collision graph is treated as a one-dimensional CW complex. Its
Betti numbers therefore have an exact graph-theoretic interpretation:
``beta_0`` counts edge-bearing connected components and ``beta_1`` counts
independent cycles. Capacity is kept separate as the maximum bipartite
matching. Hidden-space displacement spectra supply a geometric, not
combinatorial, dimension descriptor.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from math import exp, log

import numpy as np

from .graph import maximum_bipartite_matching


@dataclass(frozen=True)
class CollisionTopologyMetrics:
    n_left: int
    n_right: int
    edge_count: int
    active_left_count: int
    active_right_count: int
    active_vertex_fraction: float
    beta0_active: int
    beta1_active: int
    component_edge_sizes: tuple[int, ...]
    component_vertex_sizes: tuple[int, ...]
    component_edge_entropy: float
    normalized_component_edge_entropy: float
    effective_component_count: float
    largest_component_edge_fraction: float
    matching_size: int
    capacity_fraction: float
    independence_ratio: float
    displacement_rank_95: int | None
    displacement_entropy_rank: float | None
    displacement_stable_rank: float | None
    displacement_total_energy: float | None


@dataclass(frozen=True)
class CollisionFiltrationPoint:
    control_epsilon: float
    metrics: CollisionTopologyMetrics


def _canonical_adjacency(
    adjacency: Iterable[Iterable[int]],
    n_right: int,
) -> tuple[tuple[int, ...], ...]:
    if n_right < 0:
        raise ValueError("n_right must be non-negative")
    output: list[tuple[int, ...]] = []
    for left, raw in enumerate(adjacency):
        row = tuple(int(value) for value in raw)
        if len(set(row)) != len(row):
            raise ValueError(f"adjacency row {left} contains duplicate edges")
        if any(value < 0 or value >= n_right for value in row):
            raise ValueError(f"adjacency row {left} contains an out-of-range right index")
        output.append(tuple(sorted(row)))
    return tuple(output)


def _active_components(
    adjacency: tuple[tuple[int, ...], ...],
    n_right: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
    reverse: list[list[int]] = [[] for _ in range(n_right)]
    active_left = {left for left, row in enumerate(adjacency) if row}
    active_right: set[int] = set()
    for left, row in enumerate(adjacency):
        for right in row:
            reverse[right].append(left)
            active_right.add(right)

    visited_left: set[int] = set()
    visited_right: set[int] = set()
    component_edges: list[int] = []
    component_vertices: list[int] = []
    for start in sorted(active_left):
        if start in visited_left:
            continue
        queue: deque[tuple[str, int]] = deque([("left", start)])
        left_members: set[int] = set()
        right_members: set[int] = set()
        while queue:
            side, index = queue.popleft()
            if side == "left":
                if index in visited_left:
                    continue
                visited_left.add(index)
                left_members.add(index)
                for right in adjacency[index]:
                    if right not in visited_right:
                        queue.append(("right", right))
            else:
                if index in visited_right:
                    continue
                visited_right.add(index)
                right_members.add(index)
                for left in reverse[index]:
                    if left not in visited_left:
                        queue.append(("left", left))
        edge_count = sum(len(adjacency[left]) for left in left_members)
        component_edges.append(edge_count)
        component_vertices.append(len(left_members) + len(right_members))

    order = sorted(
        range(len(component_edges)),
        key=lambda index: (-component_edges[index], -component_vertices[index]),
    )
    return (
        tuple(component_edges[index] for index in order),
        tuple(component_vertices[index] for index in order),
        len(active_left),
        len(active_right),
    )


def _displacement_spectrum(
    edge_vectors: np.ndarray | None,
    edge_count: int,
    energy_threshold: float,
) -> tuple[int | None, float | None, float | None, float | None]:
    if edge_vectors is None:
        return None, None, None, None
    if not 0.0 < energy_threshold <= 1.0:
        raise ValueError("energy_threshold must lie in (0,1]")
    matrix = np.asarray(edge_vectors, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("edge_vectors must be a matrix")
    if matrix.shape[0] != edge_count:
        raise ValueError("edge_vectors must contain one row per adjacency edge")
    if edge_count == 0 or matrix.shape[1] == 0:
        return 0, 0.0, 0.0, 0.0
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    energy = singular_values * singular_values
    total = float(energy.sum())
    if total <= 0.0:
        return 0, 0.0, 0.0, 0.0
    probabilities = energy / total
    cumulative = np.cumsum(probabilities)
    rank_95 = int(np.searchsorted(cumulative, energy_threshold, side="left") + 1)
    positive = probabilities[probabilities > 0.0]
    entropy_rank = exp(-float(np.sum(positive * np.log(positive))))
    stable_rank = total / float(energy.max())
    return rank_95, float(entropy_rank), float(stable_rank), total


def collision_topology_metrics(
    adjacency: Iterable[Iterable[int]],
    n_right: int,
    *,
    edge_vectors: np.ndarray | None = None,
    matching_size: int | None = None,
    displacement_energy_threshold: float = 0.95,
) -> CollisionTopologyMetrics:
    """Compute finite collision-complex invariants.

    The active complex excludes isolated candidates because they carry no
    collision relation. ``edge_vectors`` must follow row-major adjacency order:
    all vectors for left vertex zero, followed by left vertex one, and so on.
    """

    canonical = _canonical_adjacency(adjacency, n_right)
    n_left = len(canonical)
    edge_count = sum(len(row) for row in canonical)
    component_edges, component_vertices, active_left, active_right = _active_components(
        canonical,
        n_right,
    )
    beta0 = len(component_edges)
    active_vertices = active_left + active_right
    beta1 = edge_count - active_vertices + beta0
    if beta1 < 0:
        raise RuntimeError("computed a negative graph cycle rank")

    if edge_count:
        masses = np.asarray(component_edges, dtype=float) / edge_count
        positive = masses[masses > 0.0]
        entropy = -float(np.sum(positive * np.log(positive)))
        normalized_entropy = entropy / log(beta0) if beta0 > 1 else 0.0
        effective_components = exp(entropy)
        largest_fraction = float(masses.max())
    else:
        entropy = 0.0
        normalized_entropy = 0.0
        effective_components = 0.0
        largest_fraction = 0.0

    observed_matching = (
        maximum_bipartite_matching(canonical, n_right)
        if matching_size is None
        else int(matching_size)
    )
    if observed_matching < 0 or observed_matching > min(n_left, n_right):
        raise ValueError("matching_size is outside the bipartite-domain bound")
    capacity_denominator = min(n_left, n_right)
    rank_95, entropy_rank, stable_rank, total_energy = _displacement_spectrum(
        edge_vectors,
        edge_count,
        displacement_energy_threshold,
    )
    all_vertices = n_left + n_right
    return CollisionTopologyMetrics(
        n_left=n_left,
        n_right=n_right,
        edge_count=edge_count,
        active_left_count=active_left,
        active_right_count=active_right,
        active_vertex_fraction=(active_vertices / all_vertices) if all_vertices else 0.0,
        beta0_active=beta0,
        beta1_active=beta1,
        component_edge_sizes=component_edges,
        component_vertex_sizes=component_vertices,
        component_edge_entropy=entropy,
        normalized_component_edge_entropy=normalized_entropy,
        effective_component_count=float(effective_components),
        largest_component_edge_fraction=largest_fraction,
        matching_size=observed_matching,
        capacity_fraction=(observed_matching / capacity_denominator)
        if capacity_denominator
        else 0.0,
        independence_ratio=(observed_matching / edge_count) if edge_count else 0.0,
        displacement_rank_95=rank_95,
        displacement_entropy_rank=entropy_rank,
        displacement_stable_rank=stable_rank,
        displacement_total_energy=total_energy,
    )


def collision_filtration(
    control_distances: np.ndarray,
    payload_distances: np.ndarray,
    behavior_distances: np.ndarray,
    control_epsilons: Iterable[float],
    *,
    payload_delta: float,
    behavior_gamma: float,
    valid_pairs: np.ndarray | None = None,
    edge_displacements: np.ndarray | None = None,
) -> tuple[CollisionFiltrationPoint, ...]:
    """Build a monotone control-threshold filtration of collision graphs."""

    control = np.asarray(control_distances, dtype=float)
    payload = np.asarray(payload_distances, dtype=float)
    behavior = np.asarray(behavior_distances, dtype=float)
    if control.ndim != 2 or payload.shape != control.shape or behavior.shape != control.shape:
        raise ValueError("distance arrays must be aligned two-dimensional matrices")
    if payload_delta < 0.0 or behavior_gamma < 0.0:
        raise ValueError("payload_delta and behavior_gamma must be non-negative")
    if valid_pairs is None:
        valid = np.ones(control.shape, dtype=bool)
    else:
        valid = np.asarray(valid_pairs, dtype=bool)
        if valid.shape != control.shape:
            raise ValueError("valid_pairs must align with distance matrices")
    displacement = (
        None
        if edge_displacements is None
        else np.asarray(edge_displacements, dtype=float)
    )
    if displacement is not None and (
        displacement.ndim != 3 or displacement.shape[:2] != control.shape
    ):
        raise ValueError("edge_displacements must have shape (n_left,n_right,dimension)")

    thresholds = tuple(float(value) for value in control_epsilons)
    if not thresholds or any(value < 0.0 for value in thresholds):
        raise ValueError("control_epsilons must be a non-empty non-negative sequence")
    if any(
        left > right
        for left, right in zip(thresholds, thresholds[1:], strict=False)
    ):
        raise ValueError("control_epsilons must be non-decreasing")

    eligible = valid & (payload >= payload_delta) & (behavior >= behavior_gamma)
    points: list[CollisionFiltrationPoint] = []
    for epsilon in thresholds:
        mask = eligible & (control <= epsilon)
        adjacency: list[tuple[int, ...]] = []
        vectors: list[np.ndarray] = []
        for left in range(control.shape[0]):
            rights = tuple(int(value) for value in np.flatnonzero(mask[left]))
            adjacency.append(rights)
            if displacement is not None:
                vectors.extend(displacement[left, right] for right in rights)
        vector_matrix = None
        if displacement is not None:
            dimension = displacement.shape[2]
            vector_matrix = (
                np.stack(vectors, axis=0)
                if vectors
                else np.empty((0, dimension), dtype=float)
            )
        points.append(
            CollisionFiltrationPoint(
                control_epsilon=epsilon,
                metrics=collision_topology_metrics(
                    tuple(adjacency),
                    control.shape[1],
                    edge_vectors=vector_matrix,
                ),
            )
        )
    return tuple(points)


__all__ = [
    "CollisionFiltrationPoint",
    "CollisionTopologyMetrics",
    "collision_filtration",
    "collision_topology_metrics",
]
