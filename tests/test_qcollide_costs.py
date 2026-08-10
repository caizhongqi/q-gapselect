from math import isclose

from qgapselect.qcollide import (
    packed_claw,
    prefix_rms_cost,
    prefix_survival_rates,
    product_johnson_cost,
    weighted_prefix_claw,
)


def test_product_johnson_boundary_scaling() -> None:
    sparse = product_johnson_cost(n=4096, matching_size=1)
    dense = product_johnson_cost(n=4096, matching_size=4096)
    assert sparse.setup_size in {255, 256}
    assert dense.setup_size in {15, 16}
    assert dense.total_cost < sparse.total_cost


def test_prefix_rms_matches_second_moment_formula() -> None:
    costs = (1.0, 3.0, 8.0)
    survival = (1.0, 0.5, 0.25, 0.125)
    observed = prefix_rms_cost(costs, survival)
    expected_square = 1.0 * 1.0 + 0.5 * (16.0 - 1.0) + 0.25 * (144.0 - 16.0)
    assert isclose(observed * observed, expected_square, rel_tol=1e-12)


def test_weighted_prefix_survival_is_monotone() -> None:
    instance = weighted_prefix_claw(
        n=32,
        matching_size=4,
        cumulative_bits=(2, 5, 12),
        stage_costs=(1.0, 2.0, 8.0),
        seed=7,
    )
    rates = prefix_survival_rates(
        (record.prefixes for record in instance.left),
        (record.prefixes for record in instance.right),
    )
    assert rates[0] == 1.0
    assert all(a >= b for a, b in zip(rates, rates[1:]))
    assert rates[-1] == 4 / (32 * 32)


def test_plain_packed_fixture_remains_available() -> None:
    assert packed_claw(n=8, matching_size=2, seed=1).endpoint_local
