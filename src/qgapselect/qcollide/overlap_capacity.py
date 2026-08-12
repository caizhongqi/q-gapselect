"""Overlap-aware collision graph fixtures and capacity estimators.

The estimators in this module are deliberately diagnostic. They measure how
vertex overlap biases simple survival-based capacity sketches. Only the packed
matching case has an exact inversion formula. General overlap results are
reported as empirical estimators unless a theorem explicitly states otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

import numpy as np

from .graph import maximum_bipartite_matching


@dataclass(frozen=True)
class OverlapGraph:
    family: str
    adjacency: tuple[tuple[int, ...], ...]
    n_right: int

    @property
    def n_left(self) -> int:
        return len(self.adjacency)

    @property
    def edge_count(self) -> int:
        return sum(len(row) for row in self.adjacency)

    @property
    def matching_size(self) -> int:
        return maximum_bipartite_matching(self.adjacency, self.n_right)

    @property
    def left_degrees(self) -> tuple[int, ...]:
        return tuple(len(row) for row in self.adjacency)

    @property
    def right_degrees(self) -> tuple[int, ...]:
        values = [0] * self.n_right
        for row in self.adjacency:
            for vertex in row:
                values[vertex] += 1
        return tuple(values)

    @property
    def maximum_degree(self) -> int:
        return max((*self.left_degrees, *self.right_degrees), default=0)


def _validate_domain(n: int, matching_size: int) -> None:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 1 <= matching_size <= n:
        raise ValueError("matching_size must lie in [1,n]")


def make_matching_graph(n: int, matching_size: int, seed: int) -> OverlapGraph:
    """Disjoint packed matching; exact survival inversion applies."""

    _validate_domain(n, matching_size)
    rng = np.random.default_rng(seed)
    left = rng.choice(n, size=matching_size, replace=False)
    right = rng.choice(n, size=matching_size, replace=False)
    rows = [[] for _ in range(n)]
    for u, v in zip(left, right, strict=True):
        rows[int(u)].append(int(v))
    return OverlapGraph("matching", tuple(tuple(row) for row in rows), n)


def make_sparse_overlap_graph(
    n: int,
    matching_size: int,
    maximum_degree: int,
    seed: int,
) -> OverlapGraph:
    """Plant a matching, then add bounded-degree overlap edges."""

    _validate_domain(n, matching_size)
    if maximum_degree < 2:
        raise ValueError("maximum_degree must be at least two")
    rng = np.random.default_rng(seed)
    base = make_matching_graph(n, matching_size, seed)
    rows = [set(row) for row in base.adjacency]
    right_degree = list(base.right_degrees)
    attempts = 0
    target_edges = min(n * maximum_degree, max(base.edge_count, 3 * matching_size))
    while sum(len(row) for row in rows) < target_edges and attempts < 50 * target_edges:
        attempts += 1
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        if len(rows[u]) >= maximum_degree or right_degree[v] >= maximum_degree:
            continue
        if v in rows[u]:
            continue
        rows[u].add(v)
        right_degree[v] += 1
    return OverlapGraph(
        "sparse_overlap",
        tuple(tuple(sorted(row)) for row in rows),
        n,
    )


def make_hub_graph(n: int, spokes: int, seed: int) -> OverlapGraph:
    """Many edges but matching capacity one: a hard edge-count negative control."""

    if n <= 0 or not 1 <= spokes <= n:
        raise ValueError("require n > 0 and 1 <= spokes <= n")
    rng = np.random.default_rng(seed)
    hub = int(rng.integers(0, n))
    right = rng.choice(n, size=spokes, replace=False)
    rows = [[] for _ in range(n)]
    rows[hub].extend(int(value) for value in right)
    return OverlapGraph("hub", tuple(tuple(row) for row in rows), n)


def make_dense_graph(n: int, density: float, seed: int) -> OverlapGraph:
    """Dense random bipartite graph for stress testing overlap estimators."""

    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < density <= 1.0:
        raise ValueError("density must lie in (0,1]")
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < density
    rows = tuple(tuple(int(v) for v in np.flatnonzero(mask[u])) for u in range(n))
    return OverlapGraph("dense", rows, n)


def induced_graph(
    graph: OverlapGraph,
    keep_left: np.ndarray,
    keep_right: np.ndarray,
) -> OverlapGraph:
    left_mask = np.asarray(keep_left, dtype=bool)
    right_mask = np.asarray(keep_right, dtype=bool)
    if left_mask.shape != (graph.n_left,) or right_mask.shape != (graph.n_right,):
        raise ValueError("retention masks have incorrect shape")
    rows: list[tuple[int, ...]] = []
    for u, row in enumerate(graph.adjacency):
        if not left_mask[u]:
            rows.append(())
            continue
        rows.append(tuple(v for v in row if right_mask[v]))
    return OverlapGraph(graph.family, tuple(rows), graph.n_right)


def packed_survival_probability(matching_size: int, q: float) -> float:
    if matching_size < 0:
        raise ValueError("matching_size must be non-negative")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0,1]")
    return 1.0 - (1.0 - q * q) ** matching_size


def invert_packed_survival(survival: float, q: float) -> float:
    if not 0.0 <= survival < 1.0:
        raise ValueError("survival must lie in [0,1)")
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0,1)")
    if survival == 0.0:
        return 0.0
    return log(1.0 - survival) / log(1.0 - q * q)


def degree_aware_probabilities(
    graph: OverlapGraph,
    target_q: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Downweight high-degree vertices while preserving packed-case q."""

    if not 0.0 < target_q <= 1.0:
        raise ValueError("target_q must lie in (0,1]")
    left_degree = np.asarray(graph.left_degrees, dtype=float)
    right_degree = np.asarray(graph.right_degrees, dtype=float)
    left = np.minimum(1.0, target_q / np.sqrt(np.maximum(left_degree, 1.0)))
    right = np.minimum(1.0, target_q / np.sqrt(np.maximum(right_degree, 1.0)))
    return left, right


@dataclass(frozen=True)
class CapacityEstimate:
    family: str
    true_matching_size: int
    edge_count: int
    maximum_degree: int
    trials: int
    survival_probability: float
    packed_inversion_estimate: float
    mean_sampled_matching: float
    scaled_matching_estimate: float
    absolute_error_packed_inversion: float
    absolute_error_scaled_matching: float


def estimate_overlap_capacity(
    graph: OverlapGraph,
    *,
    target_q: float,
    trials: int,
    seed: int,
    degree_aware: bool,
) -> CapacityEstimate:
    """Monte-Carlo diagnostic for overlap-aware capacity sketches.

    ``packed_inversion_estimate`` intentionally applies the exact disjoint
    matching inversion to all graph families. Its bias on hub/dense graphs is a
    negative control demonstrating why overlap correction is required.

    ``scaled_matching_estimate`` rescales the observed induced matching by the
    mean retention probability of the vertices used by each side. It is an
    empirical estimator, not a proved general-graph theorem.
    """

    if trials <= 0:
        raise ValueError("trials must be positive")
    rng = np.random.default_rng(seed)
    if degree_aware:
        p_left, p_right = degree_aware_probabilities(graph, target_q)
    else:
        p_left = np.full(graph.n_left, target_q, dtype=float)
        p_right = np.full(graph.n_right, target_q, dtype=float)

    positive = 0
    sampled: list[int] = []
    for _ in range(trials):
        keep_left = rng.random(graph.n_left) < p_left
        keep_right = rng.random(graph.n_right) < p_right
        value = induced_graph(graph, keep_left, keep_right).matching_size
        sampled.append(value)
        positive += int(value > 0)

    survival = positive / trials
    clipped_survival = min(survival, 1.0 - 0.5 / trials)
    packed_estimate = invert_packed_survival(clipped_survival, target_q)
    mean_sampled = float(np.mean(sampled))
    effective_pair_probability = float(np.mean(p_left) * np.mean(p_right))
    scaled = mean_sampled / max(effective_pair_probability, 1e-12)
    truth = graph.matching_size
    return CapacityEstimate(
        family=graph.family,
        true_matching_size=truth,
        edge_count=graph.edge_count,
        maximum_degree=graph.maximum_degree,
        trials=trials,
        survival_probability=float(survival),
        packed_inversion_estimate=float(packed_estimate),
        mean_sampled_matching=mean_sampled,
        scaled_matching_estimate=float(scaled),
        absolute_error_packed_inversion=abs(float(packed_estimate) - truth),
        absolute_error_scaled_matching=abs(float(scaled) - truth),
    )


def critical_retention_probability(matching_size: int) -> float:
    """Choose a non-degenerate q at the packed survival transition.

    For K >= 2 this is 1/sqrt(K). For K=1 we cap the effective denominator at
    two so that q stays strictly below one and the logarithmic inversion remains
    defined. The asymptotic scaling is unchanged.
    """

    if matching_size <= 0:
        return 1.0 / sqrt(2.0)
    return 1.0 / sqrt(max(matching_size, 2))


__all__ = [
    "CapacityEstimate",
    "OverlapGraph",
    "critical_retention_probability",
    "degree_aware_probabilities",
    "estimate_overlap_capacity",
    "induced_graph",
    "invert_packed_survival",
    "make_dense_graph",
    "make_hub_graph",
    "make_matching_graph",
    "make_sparse_overlap_graph",
    "packed_survival_probability",
]
