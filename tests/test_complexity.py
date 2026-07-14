from __future__ import annotations

import pytest

from qgapselect.complexity import (
    candidate_layer_bound,
    candidate_layer_profile,
    dyadic_certification_level,
    equal_gap_reference,
    partition_direct_sum,
    topk_gap_profile,
)
from qgapselect.experiments import equal_gap_instance, heterogeneous_gap_instance


def test_equal_gap_profile_has_exact_boundary_gaps() -> None:
    instance = equal_gap_instance(12, 4, 0.125, seed=7)
    profile = topk_gap_profile(instance.means, instance.k)

    assert profile.minimum_mean_gap == pytest.approx(0.125)
    assert profile.mean_gaps == pytest.approx((0.125,) * 12)
    assert profile.angular_gaps == pytest.approx(
        (profile.minimum_angular_gap,) * 12
    )
    assert set(profile.selected_indices).isdisjoint(profile.rejected_indices)
    assert len(profile.smaller_side_indices) == 4
    assert not profile.smaller_side_is_complement


def test_large_k_uses_rejected_complement_as_smaller_output() -> None:
    instance = equal_gap_instance(10, 8, 0.25)
    profile = topk_gap_profile(instance.means, instance.k)

    assert profile.smaller_side_is_complement
    assert profile.smaller_side_indices == profile.rejected_indices
    assert profile.smaller_side_size == 2


def test_topk_profile_rejects_boundary_ties_and_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="strict"):
        topk_gap_profile((0.5, 0.5, 0.2), 1)
    with pytest.raises(ValueError, match="k must"):
        topk_gap_profile((0.7, 0.3), 2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        topk_gap_profile((1.1, 0.3), 1)


@pytest.mark.parametrize(
    ("gap", "expected"),
    ((1.0, 1), (0.5, 2), (0.125, 4), (0.1, 4)),
)
def test_dyadic_certification_level(gap: float, expected: int) -> None:
    assert dyadic_certification_level(gap) == expected


def test_layer_profile_is_auditable_and_explicitly_conjectural() -> None:
    instance = heterogeneous_gap_instance(16, 8, 0.03125, spread=8.0)
    layer = candidate_layer_profile(instance.means, instance.k)

    assert layer.theorem_status == "conjectural_query_proxy"
    assert layer.value == pytest.approx(sum(term.value for term in layer.terms))
    assert len(layer.representations) == 2
    assert len(set(layer.certification_levels)) > 1
    assert all(term.active_count > 0 for term in layer.terms)
    assert all(term.value > 0.0 for term in layer.terms)


def test_k1_equal_gap_candidate_scales_as_sqrt_n() -> None:
    small = equal_gap_instance(16, 1, 0.125)
    large = equal_gap_instance(64, 1, 0.125)

    ratio = candidate_layer_bound(large.means, large.k) / candidate_layer_bound(
        small.means, small.k
    )
    assert ratio == pytest.approx(2.0)


def test_half_output_exhibits_near_linear_information_scaling() -> None:
    small = equal_gap_instance(32, 16, 0.125)
    large = equal_gap_instance(64, 32, 0.125)

    ratio = candidate_layer_bound(large.means, large.k) / candidate_layer_bound(
        small.means, small.k
    )
    assert 1.8 < ratio < 2.1


def test_equal_gap_reference_recovers_k1_and_half_limits() -> None:
    assert equal_gap_reference(64, 1, 0.25) == pytest.approx(
        (63.0**0.5) / 0.25
    )
    assert equal_gap_reference(64, 32, 0.25) == pytest.approx(128.0)


def test_partition_direct_sum_is_additive_by_definition() -> None:
    group = equal_gap_instance(8, 1, 0.125)
    one = candidate_layer_bound(group.means, group.k)

    assert partition_direct_sum(((group.means, group.k),) * 5) == pytest.approx(
        5.0 * one
    )


def test_candidate_is_invariant_under_reward_complement_and_set_complement() -> None:
    means = (0.91, 0.78, 0.53, 0.49, 0.31, 0.08)
    complemented = tuple(1.0 - mean for mean in means)

    original = candidate_layer_profile(means, 2)
    dual = candidate_layer_profile(complemented, len(means) - 2)

    assert dual.value == pytest.approx(original.value)
    assert dual.alternative_value == pytest.approx(original.alternative_value)
    assert {original.representation, dual.representation} == {
        "selected",
        "rejected_complement",
    }


def test_heterogeneous_gaps_can_make_the_larger_representation_cheaper() -> None:
    means = (0.51, 0.51, 0.51, 0.50) + (0.10,) * 8
    layer = candidate_layer_profile(means, 3)

    assert layer.representation == "rejected_complement"
    assert len(layer.output_indices) == 9
    assert len(layer.output_indices) > 3
    assert layer.value < layer.alternative_value


def test_angular_gap_not_mean_gap_controls_endpoint_discrimination() -> None:
    near_zero = topk_gap_profile((0.0002, 0.0001), 1)
    interior = topk_gap_profile((0.5001, 0.5), 1)

    assert near_zero.minimum_mean_gap == pytest.approx(
        interior.minimum_mean_gap
    )
    assert near_zero.minimum_angular_gap > 10 * interior.minimum_angular_gap


def test_candidate_charge_is_monotone_on_equal_gap_instances() -> None:
    narrow = equal_gap_instance(32, 8, 0.05)
    wide = equal_gap_instance(32, 8, 0.20)

    assert candidate_layer_bound(wide.means, wide.k) <= candidate_layer_bound(
        narrow.means, narrow.k
    )
