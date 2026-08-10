import numpy as np

from qgapselect.qcollide.prefix_lower_bound import (
    homogeneous_packed_claw_lower_bound,
    prefix_stage_restriction_profile,
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
