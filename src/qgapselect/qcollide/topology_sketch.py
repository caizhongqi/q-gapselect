"""Restricted quantum profiles for prefix-addressable collision-basin sketches.

The profiles in this module are analytic query-complexity statements. They do
not implement a coherent circuit and do not apply to arbitrary connected
components unless a public prefix or signature makes the basin partition
addressable before the collision graph is materialized.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isclose, sqrt


@dataclass(frozen=True)
class CollisionCapacityFactorization:
    normalization_size: int
    basin_count: int
    matching_capacity: int
    maximum_basin_multiplicity: int
    basin_density: float
    within_basin_multiplicity: float
    capacity_fraction: float
    factorized_capacity_fraction: float
    occupancy_lower_capacity_fraction: float
    occupancy_upper_capacity_fraction: float
    unit_multiplicity_exact: bool


@dataclass(frozen=True)
class PrefixAddressableBasin:
    basin_id: str
    left_size: int
    right_size: int
    matching_capacity: int
    quantum_decision_cost: float
    classical_decision_cost: float

    @property
    def occupied(self) -> bool:
        return self.matching_capacity > 0


@dataclass(frozen=True)
class QuantumBasinSketchProfile:
    basin_count: int
    occupied_basin_count: int
    total_matching_capacity: int
    maximum_basin_multiplicity: int
    mean_within_basin_multiplicity: float
    squared_quantum_decision_norm: float
    one_representative_quantum_scale: float | None
    all_representatives_quantum_upper: float
    all_representatives_integral_upper: float
    classical_full_scan_cost: float
    equal_cost_enumeration_lower_scale: float | None
    basin_count_capacity_lower: int
    basin_count_capacity_upper: int
    capacity_approximation_factor: int


def collision_capacity_factorization(
    component_matching_sizes: Iterable[int],
    *,
    normalization_size: int,
) -> CollisionCapacityFactorization:
    """Return the exact basin-density times multiplicity factorization."""

    if normalization_size <= 0:
        raise ValueError("normalization_size must be positive")
    sizes = tuple(int(value) for value in component_matching_sizes)
    if any(value <= 0 for value in sizes):
        raise ValueError("component matching sizes must be positive")
    basin_count = len(sizes)
    capacity = sum(sizes)
    maximum = max(sizes, default=0)
    basin_density = basin_count / normalization_size
    multiplicity = capacity / basin_count if basin_count else 0.0
    capacity_fraction = capacity / normalization_size
    upper = min(1.0, maximum * basin_density) if basin_count else 0.0
    return CollisionCapacityFactorization(
        normalization_size=normalization_size,
        basin_count=basin_count,
        matching_capacity=capacity,
        maximum_basin_multiplicity=maximum,
        basin_density=float(basin_density),
        within_basin_multiplicity=float(multiplicity),
        capacity_fraction=float(capacity_fraction),
        factorized_capacity_fraction=float(basin_density * multiplicity),
        occupancy_lower_capacity_fraction=float(basin_density),
        occupancy_upper_capacity_fraction=float(upper),
        unit_multiplicity_exact=bool(maximum <= 1),
    )


def claw_decision_query_proxy(
    left_size: int,
    right_size: int,
    endpoint_cost: float,
    *,
    promised_matching_capacity: int = 1,
) -> float:
    """Return the packed-claw query proxy for one public basin."""

    if left_size <= 0 or right_size <= 0:
        raise ValueError("basin domain sizes must be positive")
    if endpoint_cost <= 0.0:
        raise ValueError("endpoint_cost must be positive")
    maximum = min(left_size, right_size)
    if not 1 <= promised_matching_capacity <= maximum:
        raise ValueError("promised_matching_capacity is outside the basin bound")
    pair_mass = left_size * right_size / promised_matching_capacity
    return float(endpoint_cost * pair_mass ** (1.0 / 3.0))


def quantum_basin_sketch_profile(
    basins: Iterable[PrefixAddressableBasin],
) -> QuantumBasinSketchProfile:
    """Profile representative enumeration over public collision basins.

    Let ``H_j`` be the coherent cost of deciding whether basin ``j`` is
    occupied and returning one representative on success. Variable-time search
    gives a one-representative scale ``sqrt(sum H_j^2 / m)`` when ``m`` basins
    are occupied. Repeating after coherent exclusion yields the explicit upper
    sum ``sum_{r=1}^m sqrt(sum H_j^2 / r)`` and the integral upper bound
    ``2 sqrt(m sum H_j^2)``.

    In the equal-cost model, finding all occupied basin labels has the standard
    enumeration lower scale ``H sqrt(m L)``. The heterogeneous matching lower
    bound remains open and is not inferred by this function.
    """

    components = tuple(basins)
    if not components:
        raise ValueError("basins must be non-empty")
    seen: set[str] = set()
    for index, basin in enumerate(components):
        if not basin.basin_id or basin.basin_id in seen:
            raise ValueError(f"basin {index} has an empty or duplicate identifier")
        seen.add(basin.basin_id)
        if basin.left_size <= 0 or basin.right_size <= 0:
            raise ValueError(f"basin {index} domain sizes must be positive")
        if not 0 <= basin.matching_capacity <= min(basin.left_size, basin.right_size):
            raise ValueError(f"basin {index} matching capacity is invalid")
        if basin.quantum_decision_cost <= 0.0:
            raise ValueError(f"basin {index} quantum cost must be positive")
        if basin.classical_decision_cost <= 0.0:
            raise ValueError(f"basin {index} classical cost must be positive")

    occupied = tuple(basin for basin in components if basin.occupied)
    occupied_count = len(occupied)
    total_capacity = sum(basin.matching_capacity for basin in occupied)
    maximum_multiplicity = max(
        (basin.matching_capacity for basin in occupied),
        default=0,
    )
    squared_norm = sum(basin.quantum_decision_cost**2 for basin in components)
    one_scale = sqrt(squared_norm / occupied_count) if occupied_count else None
    enumeration_upper = sum(
        sqrt(squared_norm / remaining)
        for remaining in range(occupied_count, 0, -1)
    )
    integral_upper = 2.0 * sqrt(occupied_count * squared_norm)
    quantum_costs = [basin.quantum_decision_cost for basin in components]
    equal_cost = all(
        isclose(cost, quantum_costs[0], rel_tol=1e-12, abs_tol=1e-12)
        for cost in quantum_costs[1:]
    )
    equal_lower = None
    if equal_cost and occupied_count:
        equal_lower = quantum_costs[0] * sqrt(occupied_count * len(components))
    return QuantumBasinSketchProfile(
        basin_count=len(components),
        occupied_basin_count=occupied_count,
        total_matching_capacity=total_capacity,
        maximum_basin_multiplicity=maximum_multiplicity,
        mean_within_basin_multiplicity=(
            total_capacity / occupied_count if occupied_count else 0.0
        ),
        squared_quantum_decision_norm=float(squared_norm),
        one_representative_quantum_scale=(
            None if one_scale is None else float(one_scale)
        ),
        all_representatives_quantum_upper=float(enumeration_upper),
        all_representatives_integral_upper=float(integral_upper),
        classical_full_scan_cost=float(
            sum(basin.classical_decision_cost for basin in components)
        ),
        equal_cost_enumeration_lower_scale=(
            None if equal_lower is None else float(equal_lower)
        ),
        basin_count_capacity_lower=occupied_count,
        basin_count_capacity_upper=maximum_multiplicity * occupied_count,
        capacity_approximation_factor=max(1, maximum_multiplicity),
    )


__all__ = [
    "CollisionCapacityFactorization",
    "PrefixAddressableBasin",
    "QuantumBasinSketchProfile",
    "claw_decision_query_proxy",
    "collision_capacity_factorization",
    "quantum_basin_sketch_profile",
]
