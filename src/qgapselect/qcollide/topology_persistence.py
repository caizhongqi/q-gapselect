"""Threshold-filtration and truncated basin-persistence summaries.

The module deliberately studies the zero-dimensional persistence of edge-bearing
collision basins. It does not claim a full persistent-homology reconstruction of
the ambient neural representation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import exp

import numpy as np

from .topology import CollisionFiltrationPoint, collision_filtration


@dataclass(frozen=True)
class CollisionBasinInterval:
    birth_epsilon: float
    death_epsilon: float
    lifetime: float
    essential_at_window_end: bool


@dataclass(frozen=True)
class CollisionBasinPersistence:
    epsilon_max: float
    interval_count: int
    finite_interval_count: int
    essential_interval_count: int
    zero_lifetime_count: int
    total_lifetime: float
    normalized_total_lifetime: float
    mean_lifetime: float
    maximum_lifetime: float
    persistence_entropy: float
    effective_persistent_basin_count: float
    half_window_persistent_basin_count: int
    intervals: tuple[CollisionBasinInterval, ...]


@dataclass(frozen=True)
class CollisionFiltrationSummary:
    epsilon_min: float
    epsilon_max: float
    nominal_epsilon: float
    point_count: int
    edge_count_monotone: bool
    capacity_monotone: bool
    cycle_rank_monotone: bool
    capacity_auc: float
    basin_density_auc: float
    cycle_density_auc: float
    component_entropy_auc: float
    displacement_entropy_rank_auc: float | None
    capacity_onset_epsilon: float | None
    half_max_capacity_epsilon: float | None
    cycle_onset_epsilon: float | None
    peak_beta0: int
    peak_beta0_epsilon: float
    maximum_capacity_fraction: float
    nominal_capacity_fraction: float
    capacity_robustness_ratio: float


@dataclass(frozen=True)
class CollisionFiltrationProfile:
    points: tuple[CollisionFiltrationPoint, ...]
    summary: CollisionFiltrationSummary
    basin_persistence: CollisionBasinPersistence


def _validate_pair_arrays(
    control_distances: np.ndarray,
    payload_distances: np.ndarray,
    behavior_distances: np.ndarray,
    valid_pairs: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    control = np.asarray(control_distances, dtype=float)
    payload = np.asarray(payload_distances, dtype=float)
    behavior = np.asarray(behavior_distances, dtype=float)
    if control.ndim != 2 or payload.shape != control.shape or behavior.shape != control.shape:
        raise ValueError("distance arrays must be aligned two-dimensional matrices")
    if valid_pairs is None:
        valid = np.ones(control.shape, dtype=bool)
    else:
        valid = np.asarray(valid_pairs, dtype=bool)
        if valid.shape != control.shape:
            raise ValueError("valid_pairs must align with distance arrays")
    return control, payload, behavior, valid


def collision_basin_persistence(
    control_distances: np.ndarray,
    payload_distances: np.ndarray,
    behavior_distances: np.ndarray,
    *,
    epsilon_max: float,
    payload_delta: float,
    behavior_gamma: float,
    valid_pairs: np.ndarray | None = None,
) -> CollisionBasinPersistence:
    """Compute truncated zero-dimensional persistence of collision basins.

    Edges enter at their control distance. A collision basin is born when an
    edge first activates two previously unseen endpoints. When two active
    basins merge, the younger basin dies (elder rule). Components surviving at
    ``epsilon_max`` are truncated rather than assigned infinite death time.
    """

    if epsilon_max <= 0.0 or not np.isfinite(epsilon_max):
        raise ValueError("epsilon_max must be finite and positive")
    if payload_delta < 0.0 or behavior_gamma < 0.0:
        raise ValueError("payload_delta and behavior_gamma must be non-negative")
    control, payload, behavior, valid = _validate_pair_arrays(
        control_distances,
        payload_distances,
        behavior_distances,
        valid_pairs,
    )
    eligible = (
        valid
        & np.isfinite(control)
        & (control <= epsilon_max)
        & (payload >= payload_delta)
        & (behavior >= behavior_gamma)
    )
    n_left, n_right = control.shape
    vertex_count = n_left + n_right
    parent = list(range(vertex_count))
    active = [False] * vertex_count
    birth: list[float | None] = [None] * vertex_count

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    edges = sorted(
        (
            float(control[left, right]),
            int(left),
            int(n_left + right),
        )
        for left, right in zip(*np.nonzero(eligible), strict=True)
    )
    finite: list[CollisionBasinInterval] = []
    for weight, left_vertex, right_vertex in edges:
        left_active = active[left_vertex]
        right_active = active[right_vertex]
        if not left_active and not right_active:
            root = min(left_vertex, right_vertex)
            child = max(left_vertex, right_vertex)
            active[root] = True
            active[child] = True
            parent[child] = root
            birth[root] = weight
            continue
        if left_active and not right_active:
            root = find(left_vertex)
            active[right_vertex] = True
            parent[right_vertex] = root
            continue
        if right_active and not left_active:
            root = find(right_vertex)
            active[left_vertex] = True
            parent[left_vertex] = root
            continue

        left_root = find(left_vertex)
        right_root = find(right_vertex)
        if left_root == right_root:
            continue
        left_birth = birth[left_root]
        right_birth = birth[right_root]
        if left_birth is None or right_birth is None:
            raise RuntimeError("active collision component lacks a birth threshold")
        if (left_birth, left_root) <= (right_birth, right_root):
            survivor, deceased = left_root, right_root
            deceased_birth = right_birth
        else:
            survivor, deceased = right_root, left_root
            deceased_birth = left_birth
        lifetime = max(0.0, weight - deceased_birth)
        finite.append(
            CollisionBasinInterval(
                birth_epsilon=float(deceased_birth),
                death_epsilon=float(weight),
                lifetime=float(lifetime),
                essential_at_window_end=False,
            )
        )
        parent[deceased] = survivor
        birth[survivor] = min(left_birth, right_birth)

    roots: dict[int, float] = {}
    for vertex, is_active in enumerate(active):
        if not is_active:
            continue
        root = find(vertex)
        root_birth = birth[root]
        if root_birth is None:
            raise RuntimeError("active collision component lacks a root birth threshold")
        roots[root] = float(root_birth)
    essential = [
        CollisionBasinInterval(
            birth_epsilon=root_birth,
            death_epsilon=float(epsilon_max),
            lifetime=float(max(0.0, epsilon_max - root_birth)),
            essential_at_window_end=True,
        )
        for _, root_birth in sorted(roots.items())
    ]
    intervals = tuple(
        sorted(
            (*finite, *essential),
            key=lambda interval: (
                interval.birth_epsilon,
                interval.death_epsilon,
                interval.essential_at_window_end,
            ),
        )
    )
    lifetimes = np.asarray([interval.lifetime for interval in intervals], dtype=float)
    positive = lifetimes[lifetimes > 0.0]
    total = float(positive.sum()) if positive.size else 0.0
    if total > 0.0:
        probabilities = positive / total
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        effective = exp(entropy)
    else:
        entropy = 0.0
        effective = 0.0
    domain_scale = min(n_left, n_right)
    return CollisionBasinPersistence(
        epsilon_max=float(epsilon_max),
        interval_count=len(intervals),
        finite_interval_count=len(finite),
        essential_interval_count=len(essential),
        zero_lifetime_count=int(np.count_nonzero(lifetimes <= 0.0)),
        total_lifetime=total,
        normalized_total_lifetime=(total / (epsilon_max * domain_scale))
        if domain_scale
        else 0.0,
        mean_lifetime=float(lifetimes.mean()) if lifetimes.size else 0.0,
        maximum_lifetime=float(lifetimes.max()) if lifetimes.size else 0.0,
        persistence_entropy=entropy,
        effective_persistent_basin_count=float(effective),
        half_window_persistent_basin_count=int(
            np.count_nonzero(lifetimes >= 0.5 * epsilon_max)
        ),
        intervals=intervals,
    )


def _normalized_auc(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 1 or x[-1] <= x[0]:
        return float(y[-1])
    widths = np.diff(x)
    areas = widths * 0.5 * (y[:-1] + y[1:])
    return float(areas.sum() / (x[-1] - x[0]))


def _first_threshold(
    thresholds: np.ndarray,
    values: np.ndarray,
    predicate,
) -> float | None:
    indices = np.flatnonzero([predicate(float(value)) for value in values])
    return float(thresholds[int(indices[0])]) if len(indices) else None


def collision_filtration_summary(
    points: Iterable[CollisionFiltrationPoint],
    *,
    nominal_epsilon: float,
) -> CollisionFiltrationSummary:
    """Summarize threshold robustness without selecting a favourable epsilon."""

    ordered = tuple(points)
    if not ordered:
        raise ValueError("filtration points must be non-empty")
    thresholds = np.asarray([point.control_epsilon for point in ordered], dtype=float)
    if np.any(np.diff(thresholds) < 0.0):
        raise ValueError("filtration thresholds must be non-decreasing")
    if nominal_epsilon <= 0.0 or not np.isfinite(nominal_epsilon):
        raise ValueError("nominal_epsilon must be finite and positive")
    capacities = np.asarray(
        [point.metrics.capacity_fraction for point in ordered],
        dtype=float,
    )
    edge_counts = np.asarray([point.metrics.edge_count for point in ordered], dtype=float)
    beta0 = np.asarray([point.metrics.beta0_active for point in ordered], dtype=float)
    beta1 = np.asarray([point.metrics.beta1_active for point in ordered], dtype=float)
    domain = max(1, min(ordered[0].metrics.n_left, ordered[0].metrics.n_right))
    basin_density = beta0 / domain
    cycle_density = np.divide(
        beta1,
        edge_counts,
        out=np.zeros_like(beta1),
        where=edge_counts > 0.0,
    )
    entropy = np.asarray(
        [point.metrics.normalized_component_edge_entropy for point in ordered],
        dtype=float,
    )
    displacement_values = [point.metrics.displacement_entropy_rank for point in ordered]
    displacement_auc = None
    if all(value is not None for value in displacement_values):
        displacement_auc = _normalized_auc(
            thresholds,
            np.asarray(displacement_values, dtype=float),
        )
    peak_index = int(np.argmax(beta0))
    max_capacity = float(capacities.max())
    nominal_index = int(np.argmin(np.abs(thresholds - nominal_epsilon)))
    capacity_auc = _normalized_auc(thresholds, capacities)
    return CollisionFiltrationSummary(
        epsilon_min=float(thresholds[0]),
        epsilon_max=float(thresholds[-1]),
        nominal_epsilon=float(nominal_epsilon),
        point_count=len(ordered),
        edge_count_monotone=bool(np.all(np.diff(edge_counts) >= -1e-12)),
        capacity_monotone=bool(np.all(np.diff(capacities) >= -1e-12)),
        cycle_rank_monotone=bool(np.all(np.diff(beta1) >= -1e-12)),
        capacity_auc=capacity_auc,
        basin_density_auc=_normalized_auc(thresholds, basin_density),
        cycle_density_auc=_normalized_auc(thresholds, cycle_density),
        component_entropy_auc=_normalized_auc(thresholds, entropy),
        displacement_entropy_rank_auc=displacement_auc,
        capacity_onset_epsilon=_first_threshold(
            thresholds,
            capacities,
            lambda value: value > 0.0,
        ),
        half_max_capacity_epsilon=_first_threshold(
            thresholds,
            capacities,
            lambda value: value >= 0.5 * max_capacity and max_capacity > 0.0,
        ),
        cycle_onset_epsilon=_first_threshold(
            thresholds,
            beta1,
            lambda value: value > 0.0,
        ),
        peak_beta0=int(beta0[peak_index]),
        peak_beta0_epsilon=float(thresholds[peak_index]),
        maximum_capacity_fraction=max_capacity,
        nominal_capacity_fraction=float(capacities[nominal_index]),
        capacity_robustness_ratio=(capacity_auc / max_capacity)
        if max_capacity > 0.0
        else 0.0,
    )


def collision_filtration_profile(
    control_distances: np.ndarray,
    payload_distances: np.ndarray,
    behavior_distances: np.ndarray,
    control_epsilons: Iterable[float],
    *,
    nominal_epsilon: float,
    payload_delta: float,
    behavior_gamma: float,
    valid_pairs: np.ndarray | None = None,
    edge_displacements: np.ndarray | None = None,
) -> CollisionFiltrationProfile:
    """Construct the finite filtration, summary, and truncated H0 barcode."""

    thresholds = tuple(float(value) for value in control_epsilons)
    points = collision_filtration(
        control_distances,
        payload_distances,
        behavior_distances,
        thresholds,
        payload_delta=payload_delta,
        behavior_gamma=behavior_gamma,
        valid_pairs=valid_pairs,
        edge_displacements=edge_displacements,
    )
    persistence = collision_basin_persistence(
        control_distances,
        payload_distances,
        behavior_distances,
        epsilon_max=max(thresholds),
        payload_delta=payload_delta,
        behavior_gamma=behavior_gamma,
        valid_pairs=valid_pairs,
    )
    return CollisionFiltrationProfile(
        points=points,
        summary=collision_filtration_summary(
            points,
            nominal_epsilon=nominal_epsilon,
        ),
        basin_persistence=persistence,
    )


__all__ = [
    "CollisionBasinInterval",
    "CollisionBasinPersistence",
    "CollisionFiltrationProfile",
    "CollisionFiltrationSummary",
    "collision_basin_persistence",
    "collision_filtration_profile",
    "collision_filtration_summary",
]
