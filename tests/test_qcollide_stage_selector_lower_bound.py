from math import sqrt

import numpy as np
import pytest

from qgapselect.qcollide.stage_selector_lower_bound import (
    conditional_stage_selector_profile,
    disjoint_single_claw_stage_selector_profile,
)


def test_single_claw_selector_uses_cumulative_costs_and_l2_aggregation() -> None:
    profile = disjoint_single_claw_stage_selector_profile(
        incremental_costs=(1.0, 1.0, 1.0),
        balanced_domains=(64, 64, 64),
    )
    expected = sqrt(16.0**2 + 32.0**2 + 48.0**2)
    assert np.isclose(profile.l2_lower_bound, expected)
    assert np.isclose(profile.maximum_stage_lower_bound, 48.0)
    assert profile.aggregation_gain_over_maximum > 1.0
    assert profile.outer_values_certified
    assert [row.balanced_domain for row in profile.components] == [64, 64, 64]


def test_equal_stage_hardness_recovers_sqrt_number_of_stages_gain() -> None:
    profile = conditional_stage_selector_profile(
        incremental_costs=(1.0, 1.0, 1.0, 1.0),
        outer_adversary_values=(8.0, 4.0, 8.0 / 3.0, 2.0),
        outer_values_certified=True,
    )
    assert all(
        np.isclose(row.composed_stage_hardness, 8.0)
        for row in profile.components
    )
    assert np.isclose(profile.l2_lower_bound, 16.0)
    assert np.isclose(profile.aggregation_gain_over_maximum, 2.0)


def test_single_stage_selector_reduces_to_ordinary_composed_bound() -> None:
    profile = disjoint_single_claw_stage_selector_profile(
        incremental_costs=(3.0,),
        balanced_domains=(512,),
    )
    assert np.isclose(
        profile.l2_lower_bound,
        profile.maximum_stage_lower_bound,
    )
    assert np.isclose(profile.aggregation_gain_over_maximum, 1.0)


def test_generic_profile_keeps_uncertified_outer_values_explicit() -> None:
    profile = conditional_stage_selector_profile(
        incremental_costs=(1.0, 3.0),
        outer_adversary_values=(2.0, 5.0),
    )
    assert not profile.outer_values_certified
    assert np.isclose(profile.l2_lower_bound, sqrt(2.0**2 + 20.0**2))


@pytest.mark.parametrize(
    ("costs", "values"),
    [
        ((), ()),
        ((1.0,), ()),
        ((0.0,), (1.0,)),
        ((1.0,), (0.0,)),
    ],
)
def test_selector_profile_rejects_invalid_inputs(
    costs: tuple[float, ...],
    values: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        conditional_stage_selector_profile(
            incremental_costs=costs,
            outer_adversary_values=values,
        )
