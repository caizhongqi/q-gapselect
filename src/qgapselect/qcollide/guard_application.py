"""Real guard/LLM functional-collision application utilities.

This module evaluates *existing prompts* from public guardrail datasets. It does
not generate jailbreak strings or optimize harmful content. The application
question is structural: can an input guard map distinct prompts to nearly the
same accepted control representation while a downstream language model assigns
substantially different next-token behaviour?
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

from .graph import maximum_bipartite_matching


@dataclass(frozen=True)
class GuardCollisionThresholds:
    acceptance_threshold: float
    control_epsilon: float
    payload_delta: float
    behavior_gamma: float


@dataclass(frozen=True)
class GuardCollisionGraph:
    adjacency: tuple[tuple[int, ...], ...]
    n_right: int
    control_distances: np.ndarray
    payload_distances: np.ndarray
    behavior_distances: np.ndarray

    @property
    def n_left(self) -> int:
        return len(self.adjacency)

    @property
    def edge_count(self) -> int:
        return sum(len(row) for row in self.adjacency)

    @property
    def matching_size(self) -> int:
        return maximum_bipartite_matching(self.adjacency, self.n_right)


def normalize_rows(values: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("values must be a matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, epsilon)


def cosine_distance_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lhs = normalize_rows(left)
    rhs = normalize_rows(right)
    return np.clip(1.0 - lhs @ rhs.T, 0.0, 2.0)


def euclidean_distance_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lhs = np.asarray(left, dtype=float)
    rhs = np.asarray(right, dtype=float)
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[1] != rhs.shape[1]:
        raise ValueError("left and right must be aligned matrices")
    lhs_square = np.sum(lhs * lhs, axis=1)[:, None]
    rhs_square = np.sum(rhs * rhs, axis=1)[None, :]
    square = np.maximum(lhs_square + rhs_square - 2.0 * lhs @ rhs.T, 0.0)
    return np.sqrt(square)


def calibrate_acceptance_threshold(
    benign_scores: np.ndarray,
    *,
    benign_acceptance: float,
) -> float:
    scores = np.asarray(benign_scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("benign_scores must be a nonempty vector")
    if not 0.0 < benign_acceptance < 1.0:
        raise ValueError("benign_acceptance must lie in (0,1)")
    return float(np.quantile(scores, benign_acceptance))


def _nearest_other_distances(values: np.ndarray) -> np.ndarray:
    distances = cosine_distance_matrix(values, values)
    np.fill_diagonal(distances, np.inf)
    return np.min(distances, axis=1)


def calibrate_collision_thresholds(
    *,
    benign_guard_scores: np.ndarray,
    benign_guard_embeddings: np.ndarray,
    benign_behavior_embeddings: np.ndarray,
    benign_behavior_logits: np.ndarray,
    benign_acceptance: float = 0.95,
    control_quantile: float = 0.75,
    payload_quantile: float = 0.75,
    behavior_quantile: float = 0.75,
) -> GuardCollisionThresholds:
    """Calibrate all thresholds using benign prompts only.

    The acceptance threshold is the benign-score quantile that yields the
    requested benign acceptance. Collision thresholds use nearest-neighbour
    benign distributions so no injection labels influence the geometric cut.
    """

    for name, quantile in (
        ("control_quantile", control_quantile),
        ("payload_quantile", payload_quantile),
        ("behavior_quantile", behavior_quantile),
    ):
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"{name} must lie in (0,1)")
    acceptance = calibrate_acceptance_threshold(
        benign_guard_scores,
        benign_acceptance=benign_acceptance,
    )
    control_nearest = _nearest_other_distances(benign_guard_embeddings)
    payload_nearest = _nearest_other_distances(benign_behavior_embeddings)
    behavior_nearest = _nearest_other_distances(benign_behavior_logits)
    return GuardCollisionThresholds(
        acceptance_threshold=acceptance,
        control_epsilon=float(np.quantile(control_nearest, control_quantile)),
        payload_delta=float(np.quantile(payload_nearest, payload_quantile)),
        behavior_gamma=float(np.quantile(behavior_nearest, behavior_quantile)),
    )


def build_guard_collision_graph(
    *,
    left_guard_embeddings: np.ndarray,
    right_guard_embeddings: np.ndarray,
    left_behavior_embeddings: np.ndarray,
    right_behavior_embeddings: np.ndarray,
    left_behavior_logits: np.ndarray,
    right_behavior_logits: np.ndarray,
    thresholds: GuardCollisionThresholds,
) -> GuardCollisionGraph:
    control = cosine_distance_matrix(left_guard_embeddings, right_guard_embeddings)
    payload = cosine_distance_matrix(left_behavior_embeddings, right_behavior_embeddings)
    behavior = cosine_distance_matrix(left_behavior_logits, right_behavior_logits)
    mask = (
        (control <= thresholds.control_epsilon)
        & (payload >= thresholds.payload_delta)
        & (behavior >= thresholds.behavior_gamma)
    )
    adjacency = tuple(
        tuple(int(value) for value in np.flatnonzero(mask[index]))
        for index in range(mask.shape[0])
    )
    return GuardCollisionGraph(
        adjacency=adjacency,
        n_right=mask.shape[1],
        control_distances=control,
        payload_distances=payload,
        behavior_distances=behavior,
    )


def random_pair_discovery(
    graph: GuardCollisionGraph,
    *,
    query_budget: int,
    seed: int,
) -> int:
    if query_budget < 0:
        raise ValueError("query_budget must be non-negative")
    total = graph.n_left * graph.n_right
    budget = min(query_budget, total)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(total, size=budget, replace=False) if budget else np.array([], dtype=int)
    rows = [[] for _ in range(graph.n_left)]
    edge_sets = [set(row) for row in graph.adjacency]
    for flat in chosen:
        u = int(flat // graph.n_right)
        v = int(flat % graph.n_right)
        if v in edge_sets[u]:
            rows[u].append(v)
    return maximum_bipartite_matching(tuple(tuple(row) for row in rows), graph.n_right)


def nearest_control_discovery(
    graph: GuardCollisionGraph,
    *,
    query_budget: int,
) -> int:
    """Strong classical baseline: validate closest guard-space pairs first."""

    if query_budget < 0:
        raise ValueError("query_budget must be non-negative")
    total = graph.n_left * graph.n_right
    budget = min(query_budget, total)
    if budget == 0:
        return 0
    flat_order = np.argsort(graph.control_distances, axis=None)[:budget]
    rows = [[] for _ in range(graph.n_left)]
    edge_sets = [set(row) for row in graph.adjacency]
    for flat in flat_order:
        u = int(flat // graph.n_right)
        v = int(flat % graph.n_right)
        if v in edge_sets[u]:
            rows[u].append(v)
    return maximum_bipartite_matching(tuple(tuple(row) for row in rows), graph.n_right)


def packed_classical_query_proxy(n_left: int, n_right: int, matching_size: int) -> float:
    if min(n_left, n_right) <= 0:
        return 0.0
    if matching_size <= 0:
        return float(n_left * n_right)
    domain = float(n_left * n_right)
    return float(np.sqrt(domain / matching_size))


def packed_quantum_query_proxy(n_left: int, n_right: int, matching_size: int) -> float:
    """Endpoint-query scaling proxy for multi-solution claw discovery.

    The expression is labelled a proxy because the production guard/LLM
    experiment does not execute a coherent quantum circuit.
    """

    if min(n_left, n_right) <= 0:
        return 0.0
    if matching_size <= 0:
        return float((n_left * n_right) ** (1.0 / 3.0))
    return float(((n_left * n_right) / matching_size) ** (1.0 / 3.0))


def query_budget_grid(n_left: int, n_right: int) -> tuple[int, ...]:
    total = n_left * n_right
    if total <= 0:
        return (0,)
    fractions = (0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 1.0)
    return tuple(sorted({min(total, max(1, ceil(total * value))) for value in fractions}))


__all__ = [
    "GuardCollisionGraph",
    "GuardCollisionThresholds",
    "build_guard_collision_graph",
    "calibrate_acceptance_threshold",
    "calibrate_collision_thresholds",
    "cosine_distance_matrix",
    "euclidean_distance_matrix",
    "nearest_control_discovery",
    "normalize_rows",
    "packed_classical_query_proxy",
    "packed_quantum_query_proxy",
    "query_budget_grid",
    "random_pair_discovery",
]
