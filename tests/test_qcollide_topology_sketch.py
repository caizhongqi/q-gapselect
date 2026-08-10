import numpy as np

from qgapselect.qcollide.topology_sketch import (
    PrefixAddressableBasin,
    claw_decision_query_proxy,
    collision_capacity_factorization,
    quantum_basin_sketch_profile,
)


def test_capacity_factorization_is_exact() -> None:
    profile = collision_capacity_factorization(
        (1, 1, 2),
        normalization_size=10,
    )
    assert profile.basin_count == 3
    assert profile.matching_capacity == 4
    assert profile.maximum_basin_multiplicity == 2
    assert np.isclose(profile.basin_density, 0.3)
    assert np.isclose(profile.within_basin_multiplicity, 4 / 3)
    assert np.isclose(profile.capacity_fraction, 0.4)
    assert np.isclose(profile.factorized_capacity_fraction, 0.4)
    assert np.isclose(profile.occupancy_upper_capacity_fraction, 0.6)


def test_equal_cost_basin_enumeration_profile() -> None:
    basins = (
        PrefixAddressableBasin("a", 8, 8, 1, 2.0, 8.0),
        PrefixAddressableBasin("b", 8, 8, 2, 2.0, 8.0),
        PrefixAddressableBasin("c", 8, 8, 0, 2.0, 8.0),
        PrefixAddressableBasin("d", 8, 8, 0, 2.0, 8.0),
    )
    profile = quantum_basin_sketch_profile(basins)
    assert profile.basin_count == 4
    assert profile.occupied_basin_count == 2
    assert profile.total_matching_capacity == 3
    assert np.isclose(profile.squared_quantum_decision_norm, 16.0)
    assert np.isclose(profile.one_representative_quantum_scale, np.sqrt(8.0))
    assert np.isclose(
        profile.all_representatives_quantum_upper,
        np.sqrt(8.0) + 4.0,
    )
    assert np.isclose(profile.equal_cost_enumeration_lower_scale, np.sqrt(32.0))
    assert profile.basin_count_capacity_lower == 2
    assert profile.basin_count_capacity_upper == 4
    assert profile.capacity_approximation_factor == 2


def test_claw_decision_proxy_respects_matching_density() -> None:
    sparse = claw_decision_query_proxy(64, 64, 3.0, promised_matching_capacity=1)
    dense = claw_decision_query_proxy(64, 64, 3.0, promised_matching_capacity=8)
    assert dense < sparse
