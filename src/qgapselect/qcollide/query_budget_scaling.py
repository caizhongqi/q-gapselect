"""Same-oracle query-budget experiments for collision-capacity estimation.

This module deliberately separates empirical classical query counts from the
packed-model quantum analytic contour.  It does not simulate quantum hardware
and it does not promote the overlap contour to a theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

from .graph import maximum_bipartite_matching


@dataclass(frozen=True)
class ControlledCollisionFixture:
    family: str
    n: int
    matching_size: int
    active_left: tuple[int, ...]
    active_right: tuple[int, ...]
    overlap_degree: int = 1
    hub_spokes: int = 1

    @property
    def edge_count(self) -> int:
        k = self.matching_size
        if self.family == "matching":
            return k
        if self.family == "bounded_overlap":
            return k * min(self.overlap_degree, k)
        if self.family == "multi_hub":
            return k * self.hub_spokes
        if self.family == "biclique":
            return k * k
        raise ValueError(f"unknown fixture family {self.family!r}")

    @property
    def maximum_degree(self) -> int:
        if self.family == "matching":
            return 1
        if self.family == "bounded_overlap":
            return min(self.overlap_degree, self.matching_size)
        if self.family == "multi_hub":
            return self.hub_spokes
        if self.family == "biclique":
            return self.matching_size
        raise ValueError(f"unknown fixture family {self.family!r}")


def _validate_fixture_domain(n: int, matching_size: int) -> None:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 1 <= matching_size <= n:
        raise ValueError("matching_size must lie in [1,n]")


def make_controlled_fixture(
    family: str,
    *,
    n: int,
    matching_size: int,
    seed: int,
    overlap_degree: int = 4,
    hub_spokes: int = 8,
) -> ControlledCollisionFixture:
    """Create an exchangeable fixture whose exact matching capacity is K."""

    _validate_fixture_domain(n, matching_size)
    rng = np.random.default_rng(seed)
    active_left = tuple(int(v) for v in rng.choice(n, matching_size, replace=False))

    if family == "multi_hub":
        spokes = min(max(2, hub_spokes), max(1, n // matching_size))
        right_count = matching_size * spokes
        active_right = tuple(int(v) for v in rng.choice(n, right_count, replace=False))
        return ControlledCollisionFixture(
            family=family,
            n=n,
            matching_size=matching_size,
            active_left=active_left,
            active_right=active_right,
            hub_spokes=spokes,
        )

    active_right = tuple(int(v) for v in rng.choice(n, matching_size, replace=False))
    if family == "matching":
        return ControlledCollisionFixture(family, n, matching_size, active_left, active_right)
    if family == "bounded_overlap":
        if overlap_degree < 2:
            raise ValueError("overlap_degree must be at least two")
        return ControlledCollisionFixture(
            family,
            n,
            matching_size,
            active_left,
            active_right,
            overlap_degree=min(overlap_degree, matching_size),
        )
    if family == "biclique":
        return ControlledCollisionFixture(family, n, matching_size, active_left, active_right)
    raise ValueError(f"unknown fixture family {family!r}")


def _matching_sample_count(
    n: int,
    matching_size: int,
    queried_per_side: int,
    rng: np.random.Generator,
) -> int:
    """Sample the exact induced matching count without materializing large masks."""

    left_active = int(
        rng.hypergeometric(matching_size, n - matching_size, queried_per_side)
    )
    right_active = int(
        rng.hypergeometric(matching_size, n - matching_size, queried_per_side)
    )
    if right_active == 0 or left_active == 0:
        return 0
    return int(
        rng.hypergeometric(left_active, matching_size - left_active, right_active)
    )


def _query_masks(
    n: int,
    queried_per_side: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros(n, dtype=bool)
    right = np.zeros(n, dtype=bool)
    left[rng.choice(n, queried_per_side, replace=False)] = True
    right[rng.choice(n, queried_per_side, replace=False)] = True
    return left, right


def _bounded_overlap_sample_capacity(
    fixture: ControlledCollisionFixture,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> int:
    k = fixture.matching_size
    kept_right = right_mask[np.asarray(fixture.active_right, dtype=int)]
    rows: list[tuple[int, ...]] = []
    for index, vertex in enumerate(fixture.active_left):
        if not left_mask[vertex]:
            rows.append(())
            continue
        rows.append(
            tuple(
                (index + offset) % k
                for offset in range(fixture.overlap_degree)
                if kept_right[(index + offset) % k]
            )
        )
    return maximum_bipartite_matching(tuple(rows), k)


def sampled_matching_capacity(
    fixture: ControlledCollisionFixture,
    *,
    queried_per_side: int,
    rng: np.random.Generator,
) -> int:
    """Return exact matching capacity in the uniformly queried induced subgraph."""

    if not 1 <= queried_per_side <= fixture.n:
        raise ValueError("queried_per_side must lie in [1,n]")
    if fixture.family == "matching":
        return _matching_sample_count(
            fixture.n,
            fixture.matching_size,
            queried_per_side,
            rng,
        )

    left_mask, right_mask = _query_masks(fixture.n, queried_per_side, rng)
    active_left = np.asarray(fixture.active_left, dtype=int)
    if fixture.family == "bounded_overlap":
        return _bounded_overlap_sample_capacity(fixture, left_mask, right_mask)
    if fixture.family == "multi_hub":
        groups = np.asarray(fixture.active_right, dtype=int).reshape(
            fixture.matching_size,
            fixture.hub_spokes,
        )
        return int(
            sum(
                bool(left_mask[active_left[index]]) and bool(np.any(right_mask[groups[index]]))
                for index in range(fixture.matching_size)
            )
        )
    if fixture.family == "biclique":
        active_right = np.asarray(fixture.active_right, dtype=int)
        return min(
            int(np.count_nonzero(left_mask[active_left])),
            int(np.count_nonzero(right_mask[active_right])),
        )
    raise ValueError(f"unknown fixture family {fixture.family!r}")


def sampled_edge_count(
    fixture: ControlledCollisionFixture,
    *,
    queried_per_side: int,
    rng: np.random.Generator,
) -> int:
    """Return sampled collision-edge count for the edge-count negative control."""

    if fixture.family == "matching":
        return _matching_sample_count(
            fixture.n,
            fixture.matching_size,
            queried_per_side,
            rng,
        )
    left_mask, right_mask = _query_masks(fixture.n, queried_per_side, rng)
    active_left = np.asarray(fixture.active_left, dtype=int)
    if fixture.family == "bounded_overlap":
        active_right = np.asarray(fixture.active_right, dtype=int)
        total = 0
        for index, vertex in enumerate(active_left):
            if not left_mask[vertex]:
                continue
            total += sum(
                bool(right_mask[active_right[(index + offset) % fixture.matching_size]])
                for offset in range(fixture.overlap_degree)
            )
        return int(total)
    if fixture.family == "multi_hub":
        groups = np.asarray(fixture.active_right, dtype=int).reshape(
            fixture.matching_size,
            fixture.hub_spokes,
        )
        return int(
            sum(
                int(left_mask[active_left[index]]) * int(np.count_nonzero(right_mask[groups[index]]))
                for index in range(fixture.matching_size)
            )
        )
    if fixture.family == "biclique":
        active_right = np.asarray(fixture.active_right, dtype=int)
        return int(
            np.count_nonzero(left_mask[active_left])
            * np.count_nonzero(right_mask[active_right])
        )
    raise ValueError(f"unknown fixture family {fixture.family!r}")


def qccs_packed_query_proxy(
    n: int,
    matching_size: int,
    *,
    relative_epsilon: float,
) -> dict[str, float]:
    """Return the packed-model analytic contour; this is not a runtime measurement."""

    _validate_fixture_domain(n, matching_size)
    if not 0.0 < relative_epsilon < 1.0:
        raise ValueError("relative_epsilon must lie in (0,1)")
    effective_pair_scale = (n * n) / matching_size
    structural_scale = effective_pair_scale ** (1.0 / 3.0)
    return {
        "effective_pair_scale": float(effective_pair_scale),
        "structural_query_scale": float(structural_scale),
        "epsilon_adjusted_query_proxy": float(structural_scale / relative_epsilon),
    }


def evaluate_query_budget_component(
    *,
    panel: str,
    n: int,
    matching_densities: tuple[float, ...],
    budget_fractions: tuple[float, ...],
    graph_seeds: tuple[int, ...],
    sampling_trials: int,
    additive_epsilon: float,
    relative_epsilon: float,
    success_probability: float,
    seed: int,
) -> dict[str, object]:
    """Evaluate actual classical endpoint-record queries on one N slice."""

    if panel not in {"matching", "overlap"}:
        raise ValueError("panel must be 'matching' or 'overlap'")
    if sampling_trials <= 0 or not graph_seeds:
        raise ValueError("sampling trials and graph_seeds must be non-empty")
    if not 0.0 < additive_epsilon < 1.0:
        raise ValueError("additive_epsilon must lie in (0,1)")
    if not 0.0 < relative_epsilon < 1.0:
        raise ValueError("relative_epsilon must lie in (0,1)")
    if not 0.5 < success_probability < 1.0:
        raise ValueError("success_probability must lie in (0.5,1)")

    families = ("matching",) if panel == "matching" else (
        "bounded_overlap",
        "multi_hub",
        "biclique",
    )
    cells: list[dict[str, object]] = []
    for density_index, density in enumerate(matching_densities):
        if not 0.0 < density <= 1.0:
            raise ValueError("matching densities must lie in (0,1]")
        matching_size = max(1, min(n, int(round(n * density))))
        for family_index, family in enumerate(families):
            observations: dict[int, dict[str, list[float]]] = {}
            for graph_seed in graph_seeds:
                fixture = make_controlled_fixture(
                    family,
                    n=n,
                    matching_size=matching_size,
                    seed=seed + 100003 * density_index + 1009 * family_index + graph_seed,
                )
                if fixture.matching_size != matching_size:
                    raise RuntimeError("controlled fixture changed the requested capacity")
                for budget_index, fraction in enumerate(budget_fractions):
                    if not 0.0 < fraction <= 1.0:
                        raise ValueError("budget fractions must lie in (0,1]")
                    queried = min(n, max(1, int(round(n * fraction))))
                    bucket = observations.setdefault(
                        queried,
                        {
                            "matching_additive": [],
                            "matching_relative": [],
                            "edge_additive": [],
                            "edge_relative": [],
                        },
                    )
                    for trial in range(sampling_trials):
                        base_seed = (
                            seed
                            + graph_seed * 1000003
                            + density_index * 10007
                            + family_index * 1009
                            + budget_index * 97
                            + trial
                        )
                        q = queried / n
                        match_rng = np.random.default_rng(base_seed)
                        edge_rng = np.random.default_rng(base_seed + 7919)
                        sampled_matching = sampled_matching_capacity(
                            fixture,
                            queried_per_side=queried,
                            rng=match_rng,
                        )
                        sampled_edges = sampled_edge_count(
                            fixture,
                            queried_per_side=queried,
                            rng=edge_rng,
                        )
                        matching_estimate = min(
                            float(n),
                            float(sampled_matching) / max(q * q, 1e-12),
                        )
                        edge_estimate = min(
                            float(n),
                            float(sampled_edges) / max(q * q, 1e-12),
                        )
                        bucket["matching_additive"].append(
                            abs(matching_estimate - matching_size) / n
                        )
                        bucket["matching_relative"].append(
                            abs(matching_estimate - matching_size) / matching_size
                        )
                        bucket["edge_additive"].append(
                            abs(edge_estimate - matching_size) / n
                        )
                        bucket["edge_relative"].append(
                            abs(edge_estimate - matching_size) / matching_size
                        )

            budget_rows: list[dict[str, object]] = []
            for queried, metrics in sorted(observations.items()):
                matching_add = np.asarray(metrics["matching_additive"], dtype=float)
                matching_rel = np.asarray(metrics["matching_relative"], dtype=float)
                edge_add = np.asarray(metrics["edge_additive"], dtype=float)
                edge_rel = np.asarray(metrics["edge_relative"], dtype=float)
                budget_rows.append(
                    {
                        "queried_per_side": queried,
                        "endpoint_query_count": 2 * queried,
                        "endpoint_query_fraction": queried / n,
                        "observation_count": int(len(matching_add)),
                        "matching_scaled_mean_additive_error": float(np.mean(matching_add)),
                        "matching_scaled_mean_relative_error": float(np.mean(matching_rel)),
                        "matching_scaled_additive_success_rate": float(
                            np.mean(matching_add <= additive_epsilon)
                        ),
                        "matching_scaled_relative_success_rate": float(
                            np.mean(matching_rel <= relative_epsilon)
                        ),
                        "edge_proxy_mean_additive_error": float(np.mean(edge_add)),
                        "edge_proxy_mean_relative_error": float(np.mean(edge_rel)),
                        "edge_proxy_additive_success_rate": float(
                            np.mean(edge_add <= additive_epsilon)
                        ),
                        "edge_proxy_relative_success_rate": float(
                            np.mean(edge_rel <= relative_epsilon)
                        ),
                    }
                )

            def first_success(metric: str) -> int | None:
                for row in budget_rows:
                    if float(row[metric]) >= success_probability:
                        return int(row["endpoint_query_count"])
                return None

            proxy = qccs_packed_query_proxy(
                n,
                matching_size,
                relative_epsilon=relative_epsilon,
            )
            cells.append(
                {
                    "panel": panel,
                    "family": family,
                    "n": n,
                    "matching_size": matching_size,
                    "matching_density": matching_size / n,
                    "edge_count": make_controlled_fixture(
                        family,
                        n=n,
                        matching_size=matching_size,
                        seed=seed + density_index + family_index,
                    ).edge_count,
                    "maximum_degree": make_controlled_fixture(
                        family,
                        n=n,
                        matching_size=matching_size,
                        seed=seed + density_index + family_index,
                    ).maximum_degree,
                    "budget_rows": budget_rows,
                    "minimum_matching_scaled_queries_additive": first_success(
                        "matching_scaled_additive_success_rate"
                    ),
                    "minimum_matching_scaled_queries_relative": first_success(
                        "matching_scaled_relative_success_rate"
                    ),
                    "minimum_edge_proxy_queries_additive": first_success(
                        "edge_proxy_additive_success_rate"
                    ),
                    "minimum_edge_proxy_queries_relative": first_success(
                        "edge_proxy_relative_success_rate"
                    ),
                    "qccs_packed_query_proxy": proxy,
                }
            )

    return {
        "artifact_type": "qcollide_query_budget_scaling_component_v2",
        "schema_version": 2,
        "panel": panel,
        "n": n,
        "matching_densities": list(matching_densities),
        "budget_fractions": list(budget_fractions),
        "graph_seeds": list(graph_seeds),
        "sampling_trials": sampling_trials,
        "additive_epsilon": additive_epsilon,
        "relative_epsilon": relative_epsilon,
        "target_success_probability": success_probability,
        "cells": cells,
        "claim_boundary": {
            "endpoint_record_queries_are_measured": True,
            "classical_query_counts_are_empirical": True,
            "qccs_packed_query_proxy_is_runtime": False,
            "qccs_packed_query_proxy_is_a_proved_overlap_bound": False,
            "qccs_overlap_advantage_claimed": False,
            "matching_capacity_is_controlled_exactly": True,
        },
    }


def ceil_query_proxy(value: float) -> int:
    return int(ceil(value))


__all__ = [
    "ControlledCollisionFixture",
    "ceil_query_proxy",
    "evaluate_query_budget_component",
    "make_controlled_fixture",
    "qccs_packed_query_proxy",
    "sampled_edge_count",
    "sampled_matching_capacity",
]
