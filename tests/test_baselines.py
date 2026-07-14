from __future__ import annotations

import math

import pytest

from qgapselect.baselines import (
    baseline_estimates,
    classical_information_proxy,
    classical_uniform_hoeffding_upper,
    classical_uniform_proxy,
    partition_baseline_estimates,
    repeated_qbai_proxy,
    uniform_ae_kmin_proxy,
)
from qgapselect.complexity import topk_gap_profile
from qgapselect.experiments import equal_gap_instance


@pytest.mark.parametrize(("n", "k"), ((16, 1), (16, 8), (16, 15)))
def test_equal_gap_baseline_formulas(n: int, k: int) -> None:
    gap = 0.125
    instance = equal_gap_instance(n, k, gap)
    output_size = min(k, n - k)
    angular_gap = topk_gap_profile(instance.means, k).minimum_angular_gap

    assert uniform_ae_kmin_proxy(instance.means, k) == pytest.approx(
        math.sqrt(n * (output_size + 1)) / angular_gap
    )
    assert repeated_qbai_proxy(instance.means, k) == pytest.approx(
        output_size * math.sqrt(n - output_size) / angular_gap
    )
    assert classical_uniform_proxy(instance.means, k) == pytest.approx(
        n / gap**2
    )


def test_finite_uniform_bound_is_integer_and_confidence_monotone() -> None:
    instance = equal_gap_instance(12, 4, 0.2)
    loose = classical_uniform_hoeffding_upper(instance.means, instance.k, delta=0.1)
    strict = classical_uniform_hoeffding_upper(instance.means, instance.k, delta=0.01)

    assert isinstance(loose, int)
    assert strict > loose > 0


def test_baseline_rows_never_label_proxy_as_measured_query_count() -> None:
    instance = equal_gap_instance(16, 8, 0.125)
    estimates = baseline_estimates(instance.means, instance.k)

    assert {estimate.method for estimate in estimates} == {
        "candidate_layer",
        "prior_uniform_ae_kmin",
        "repeated_qbai",
        "classical_information",
        "classical_uniform",
    }
    assert all("proxy" in estimate.claim_status for estimate in estimates)
    assert all("proxy" in estimate.unit for estimate in estimates)


def test_partition_baselines_add_independent_required_groups() -> None:
    group = equal_gap_instance(8, 1, 0.125)
    one = {
        estimate.method: estimate.value
        for estimate in partition_baseline_estimates(((group.means, group.k),))
    }
    four = {
        estimate.method: estimate.value
        for estimate in partition_baseline_estimates(((group.means, group.k),) * 4)
    }

    assert four.keys() == one.keys()
    for method in one:
        assert four[method] == pytest.approx(4.0 * one[method])


def test_information_baseline_has_correct_bernoulli_endpoint_order() -> None:
    coarse = classical_information_proxy((0.0004, 0.0002), 1)
    fine = classical_information_proxy((0.0002, 0.0001), 1)
    coarse_quantum = uniform_ae_kmin_proxy((0.0004, 0.0002), 1)
    fine_quantum = uniform_ae_kmin_proxy((0.0002, 0.0001), 1)

    assert fine / coarse == pytest.approx(2.0, rel=0.02)
    assert fine_quantum / coarse_quantum == pytest.approx(2.0**0.5, rel=0.02)
