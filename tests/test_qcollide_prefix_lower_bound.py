import numpy as np

from qgapselect.qcollide.prefix_lower_bound import (
    homogeneous_packed_claw_lower_bound,
    prefix_stage_restriction_profile,
    random_range_claw_distributional_bound,
    random_range_prefix_restriction_profile,
)


def test_homogeneous_bound_has_expected_exponents() -> None:
    base = homogeneous_packed_claw_lower_bound(
        left_domain=512,
        right_domain=512,
        packed_solutions=8,
        endpoint_cost=3.0,
    )
    doubled_domains = homogeneous_packed_claw_lower_bound(
        left_domain=1024,
        right_domain=1024,
        packed_solutions=8,
        endpoint_cost=3.0,
    )
    doubled_solutions = homogeneous_packed_claw_lower_bound(
        left_domain=512,
        right_domain=512,
        packed_solutions=64,
        endpoint_cost=3.0,
    )
    assert np.isclose(doubled_domains / base, 2.0 ** (2.0 / 3.0))
    assert np.isclose(doubled_solutions / base, 0.5)


def test_stage_profile_selects_the_strongest_restriction() -> None:
    profile = prefix_stage_restriction_profile(
        incremental_costs=(1.0, 3.0, 12.0),
        left_survivors=(1024, 256, 64),
        right_survivors=(1024, 256, 64),
        packed_solutions=(16, 8, 4),
    )
    observed = [row.lower_bound for row in profile.stages]
    assert profile.maximum_lower_bound == max(observed)
    assert profile.maximizing_stage == int(np.argmax(observed)) + 1


def test_random_range_bound_recovers_packed_claw_exponents() -> None:
    domain = 4096
    packed_solutions = 64
    range_size = domain * domain // packed_solutions
    bound = random_range_claw_distributional_bound(
        left_domain=domain,
        right_domain=domain,
        range_size=range_size,
        endpoint_cost=1.0,
    )
    assert bound.premises_satisfied
    assert np.isclose(bound.expected_cross_claws, packed_solutions)
    assert bound.expected_isolated_cross_claws > 1.0
    assert np.isclose(bound.quantum_omega_scale, 64.0)
    assert np.isclose(bound.classical_omega_scale, 512.0)


def test_random_range_prefix_profile_selects_expensive_survivor_tail() -> None:
    profile = random_range_prefix_restriction_profile(
        incremental_costs=(1.0, 4.0, 16.0),
        left_survivors=(4096, 1024, 256),
        right_survivors=(4096, 1024, 256),
        range_sizes=(262144, 16384, 1024),
    )
    scales = [row.bound.quantum_omega_scale for row in profile.stages]
    assert profile.maximizing_stage == 3
    assert np.isclose(profile.maximum_quantum_omega_scale, max(scales))
    assert all(row.bound.premises_satisfied for row in profile.stages)


def test_random_range_bound_rejects_vacuous_cross_claw_regime() -> None:
    bound = random_range_claw_distributional_bound(
        left_domain=4,
        right_domain=4,
        range_size=2**30,
        endpoint_cost=7.0,
    )
    assert not bound.premises_satisfied
    assert not bound.domain_condition_satisfied
    assert bound.quantum_omega_scale == 0.0
    assert bound.classical_omega_scale == 0.0
