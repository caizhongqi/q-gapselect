from __future__ import annotations

import pytest

from qgapselect.estimators import (
    AnalyticIterativeAmplitudeEstimator,
    mean_estimation_charge_proxy,
)
from qgapselect.models import IAEConfig
from qgapselect.oracles import CanonicalBernoulliOracleSimulator


@pytest.mark.parametrize("mean", (0.0, 1.0))
def test_deterministic_amplitudes_stay_inside_reported_confidence_hull(mean: float) -> None:
    config = IAEConfig(
        target_angular_precision=0.05,
        confidence=0.05,
        shots_per_round=32,
        max_rounds=4,
        max_grover_power=15,
        grid_points=1025,
    )
    oracle = CanonicalBernoulliOracleSimulator((mean,), seed=9)
    result = AnalyticIterativeAmplitudeEstimator(config).estimate(oracle, 0)

    assert result.interval.lower <= mean <= result.interval.upper
    assert result.interval.lower <= result.estimate <= result.interval.upper
    assert result.numerical_warning is None
    assert result.executed_query_counts["classical_total"] == 0

    expected_forward = sum(
        observation.shots * (observation.grover_power + 1)
        for observation in result.observations
    )
    expected_inverse = sum(
        observation.shots * observation.grover_power
        for observation in result.observations
    )
    assert result.executed_query_counts["forward"] == expected_forward
    assert result.executed_query_counts["inverse"] == expected_inverse
    assert result.executed_query_counts["coherent_total"] == (
        expected_forward + expected_inverse
    )


def test_analytic_iae_is_reproducible_given_identical_seed_and_configuration() -> None:
    config = IAEConfig(
        target_angular_precision=0.1,
        confidence=0.1,
        shots_per_round=16,
        max_rounds=3,
        grid_points=513,
    )
    first = AnalyticIterativeAmplitudeEstimator(config).estimate(
        CanonicalBernoulliOracleSimulator((0.37,), seed=112), 0
    )
    second = AnalyticIterativeAmplitudeEstimator(config).estimate(
        CanonicalBernoulliOracleSimulator((0.37,), seed=112), 0
    )

    assert first.observations == second.observations
    assert first.estimate == second.estimate
    assert first.interval == second.interval
    assert first.executed_query_counts == second.executed_query_counts
    assert first.interval.lower <= first.estimate <= first.interval.upper


@pytest.mark.parametrize("seed", range(20))
def test_interior_estimate_never_leaves_its_reported_confidence_hull(seed: int) -> None:
    config = IAEConfig(
        target_angular_precision=0.03,
        confidence=0.2,
        shots_per_round=8,
        max_rounds=5,
        max_grover_power=31,
        grid_points=513,
    )
    result = AnalyticIterativeAmplitudeEstimator(config).estimate(
        CanonicalBernoulliOracleSimulator((0.37,), seed=seed), 0
    )

    assert result.interval.lower <= result.estimate <= result.interval.upper


def test_fixed_seed_coverage_regression_for_an_interior_amplitude() -> None:
    config = IAEConfig(
        target_angular_precision=0.03,
        confidence=0.1,
        shots_per_round=24,
        max_rounds=5,
        max_grover_power=31,
        grid_points=1025,
    )
    results = [
        AnalyticIterativeAmplitudeEstimator(config).estimate(
            CanonicalBernoulliOracleSimulator((0.37,), seed=seed), 0
        )
        for seed in range(30)
    ]

    covered = sum(result.interval.lower <= 0.37 <= result.interval.upper for result in results)
    assert covered >= 28


def test_variance_adaptive_expression_is_only_a_validated_numeric_helper() -> None:
    value = mean_estimation_charge_proxy(0.25, 0.04, logarithmic_factor=2.0)
    assert value == pytest.approx(2.0 * (0.5 / 0.04 + 1.0 / 0.2))

    with pytest.raises(ValueError, match="variance"):
        mean_estimation_charge_proxy(0.3, 0.1)
    with pytest.raises(ValueError, match="epsilon"):
        mean_estimation_charge_proxy(0.1, 0.0)
