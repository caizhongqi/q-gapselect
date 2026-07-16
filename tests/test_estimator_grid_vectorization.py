from __future__ import annotations

import ast
import inspect
import math
import random
import textwrap

import pytest

from qgapselect.estimators import (
    AnalyticIterativeAmplitudeEstimator,
    _fit_grid_scalar_reference,
)
from qgapselect.models import GroverObservation, IAEConfig
from qgapselect.oracles import CanonicalBernoulliOracleSimulator


def _assert_fits_match(
    observations: list[GroverObservation],
    config: IAEConfig,
) -> None:
    scalar = _fit_grid_scalar_reference(observations, config)
    vectorised = AnalyticIterativeAmplitudeEstimator._fit_grid(observations, config)

    assert vectorised[3] == scalar[3]
    assert vectorised[0] == pytest.approx(scalar[0], rel=0.0, abs=1e-15)
    assert vectorised[1].lower == pytest.approx(
        scalar[1].lower, rel=0.0, abs=1e-15
    )
    assert vectorised[1].upper == pytest.approx(
        scalar[1].upper, rel=0.0, abs=1e-15
    )
    assert vectorised[2].lower == pytest.approx(
        scalar[2].lower, rel=0.0, abs=1e-15
    )
    assert vectorised[2].upper == pytest.approx(
        scalar[2].upper, rel=0.0, abs=1e-15
    )


@pytest.mark.parametrize("seed", range(64))
def test_vectorised_fit_matches_scalar_specification_across_random_traces(
    seed: int,
) -> None:
    rng = random.Random(seed)
    max_rounds = rng.randint(1, 8)
    shots = rng.choice((8, 16, 32, 64, 96, 128))
    config = IAEConfig(
        target_angular_precision=0.02,
        confidence=rng.choice((0.01, 0.05, 0.1, 0.2, 0.5)),
        shots_per_round=shots,
        max_rounds=max_rounds,
        max_grover_power=127,
        grid_points=rng.choice((257, 513, 1025, 2049)),
    )
    observation_count = rng.randint(1, max_rounds)
    observations = [
        GroverObservation(
            grover_power=min(2**round_index - 1, config.max_grover_power),
            successes=rng.randint(0, shots),
            shots=shots,
        )
        for round_index in range(observation_count)
    ]

    _assert_fits_match(observations, config)


@pytest.mark.parametrize("successes", (0, 64))
def test_vectorised_fit_matches_scalar_specification_at_probability_endpoints(
    successes: int,
) -> None:
    config = IAEConfig(
        target_angular_precision=0.02,
        confidence=0.05,
        shots_per_round=64,
        max_rounds=5,
        max_grover_power=31,
        grid_points=4097,
    )
    observations = [
        GroverObservation(power, successes, 64) for power in (0, 1, 3, 7, 15)
    ]

    _assert_fits_match(observations, config)


def test_vectorised_fit_preserves_first_grid_point_mle_tie_break() -> None:
    config = IAEConfig(grid_points=257)

    scalar = _fit_grid_scalar_reference([], config)
    vectorised = AnalyticIterativeAmplitudeEstimator._fit_grid([], config)

    _assert_fits_match([], config)
    assert scalar[0] == vectorised[0] == 0.0
    assert vectorised[2].lower == 0.0
    assert vectorised[2].upper == math.pi / 2.0


def test_vectorised_fit_preserves_empty_confidence_set_semantics() -> None:
    config = IAEConfig(
        target_angular_precision=0.02,
        confidence=0.9,
        shots_per_round=1000,
        max_rounds=2,
        grid_points=257,
    )
    contradictory = [
        GroverObservation(grover_power=0, successes=0, shots=1000),
        GroverObservation(grover_power=0, successes=1000, shots=1000),
    ]

    _assert_fits_match(contradictory, config)
    estimate, interval, angular_interval, warning = (
        AnalyticIterativeAmplitudeEstimator._fit_grid(contradictory, config)
    )
    assert estimate == 0.5
    assert (interval.lower, interval.upper) == (0.0, 1.0)
    assert (angular_interval.lower, angular_interval.upper) == (0.0, math.pi / 2.0)
    assert warning == "empty numerical confidence set; returned the vacuous interval"


def test_default_fit_has_no_python_loop_over_grid_points() -> None:
    tree = ast.parse(
        textwrap.dedent(
            inspect.getsource(AnalyticIterativeAmplitudeEstimator._fit_grid)
        )
    )

    assert not any(isinstance(node, (ast.For, ast.AsyncFor)) for node in ast.walk(tree))


@pytest.mark.parametrize(("mean", "seed"), ((0.0, 3), (0.37, 11), (1.0, 19)))
def test_vectorisation_does_not_change_estimator_query_ledger(
    mean: float,
    seed: int,
) -> None:
    class ScalarReferenceEstimator(AnalyticIterativeAmplitudeEstimator):
        _fit_grid = staticmethod(_fit_grid_scalar_reference)

    config = IAEConfig(
        target_angular_precision=0.03,
        confidence=0.1,
        shots_per_round=32,
        max_rounds=6,
        max_grover_power=31,
        grid_points=2049,
    )
    vectorised = AnalyticIterativeAmplitudeEstimator(config).estimate(
        CanonicalBernoulliOracleSimulator((mean,), seed=seed), 0
    )
    scalar = ScalarReferenceEstimator(config).estimate(
        CanonicalBernoulliOracleSimulator((mean,), seed=seed), 0
    )

    assert vectorised.observations == scalar.observations
    assert vectorised.executed_query_counts == scalar.executed_query_counts
    assert vectorised.estimate == pytest.approx(scalar.estimate, rel=0.0, abs=1e-15)
    assert vectorised.interval.lower == pytest.approx(
        scalar.interval.lower, rel=0.0, abs=1e-15
    )
    assert vectorised.interval.upper == pytest.approx(
        scalar.interval.upper, rel=0.0, abs=1e-15
    )
