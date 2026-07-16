"""Executable analytic simulators for amplitude-estimation circuits."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .models import (
    AmplitudeEstimate,
    AngularConfidenceInterval,
    ConfidenceInterval,
    GroverObservation,
    IAEConfig,
)
from .oracles import CanonicalBernoulliOracleSimulator, QueryLedger


def _bernoulli_log_likelihood(successes: int, shots: int, probability: float) -> float:
    probability = min(max(probability, 1e-15), 1.0 - 1e-15)
    return successes * math.log(probability) + (shots - successes) * math.log1p(
        -probability
    )


def _fit_grid_scalar_reference(
    observations: list[GroverObservation],
    config: IAEConfig,
) -> tuple[
    float,
    ConfidenceInterval,
    AngularConfidenceInterval,
    str | None,
]:
    """Scalar specification for the vectorised confidence-grid fit.

    This is intentionally kept private and off the execution path.  It makes
    the numerical semantics of :meth:`AnalyticIterativeAmplitudeEstimator._fit_grid`
    executable in regression tests without retaining a Python loop over grid
    points in production experiments.
    """

    grid_size = config.grid_points
    theta_step = (math.pi / 2.0) / (grid_size - 1)
    radius = math.sqrt(
        math.log(2.0 * config.max_rounds / config.confidence)
        / (2.0 * config.shots_per_round)
    )

    minimum_consistent_theta: float | None = None
    maximum_consistent_theta: float | None = None
    maximum_consistent_likelihood = -math.inf
    mle_theta = math.pi / 4.0

    for grid_index in range(grid_size):
        theta = grid_index * theta_step
        consistent = True
        log_likelihood = 0.0
        for observation in observations:
            multiplier = observation.frequency
            probability = math.sin(multiplier * theta) ** 2
            log_likelihood += _bernoulli_log_likelihood(
                observation.successes,
                observation.shots,
                probability,
            )
            numerical_padding = multiplier * theta_step / 2.0
            if (
                abs(probability - observation.empirical_probability)
                > radius + numerical_padding
            ):
                consistent = False

        if consistent:
            if minimum_consistent_theta is None:
                minimum_consistent_theta = theta
            maximum_consistent_theta = theta
            if log_likelihood > maximum_consistent_likelihood:
                maximum_consistent_likelihood = log_likelihood
                mle_theta = theta

    if minimum_consistent_theta is None or maximum_consistent_theta is None:
        return (
            0.5,
            ConfidenceInterval(0.0, 1.0),
            AngularConfidenceInterval(0.0, math.pi / 2.0),
            "empty numerical confidence set; returned the vacuous interval",
        )

    estimate = math.sin(mle_theta) ** 2
    theta_lower = max(0.0, minimum_consistent_theta - theta_step / 2.0)
    theta_upper = min(math.pi / 2.0, maximum_consistent_theta + theta_step / 2.0)
    interval = ConfidenceInterval(
        lower=max(0.0, math.sin(theta_lower) ** 2),
        upper=min(1.0, math.sin(theta_upper) ** 2),
    )
    angular_interval = AngularConfidenceInterval(theta_lower, theta_upper)
    return estimate, interval, angular_interval, None


class AnalyticIterativeAmplitudeEstimator:
    r"""Simulate an iterative amplitude-estimation experiment analytically.

    At round ``r`` the backend measures

    .. math::

       Q^{m_r}A|0\rangle,\qquad m_r\in\{0,1,3,7,\ldots\},

    whose success probability is
    :math:`\sin^2((2m_r+1)\theta)`.  It samples this Bernoulli distribution
    directly; it does not allocate a state vector with ``2**num_qubits``
    entries.

    The returned interval is a conservative hull of a simultaneous-Hoeffding
    grid confidence set.  Grid discretisation is padded using the Lipschitz
    constant ``2m+1``.  This gives a useful executable reference under the
    canonical simulator, but it is not claimed to be the final optimal IAE
    primitive in the Q-GapSelect complexity theorem.
    """

    interval_method = "simultaneous_hoeffding_grid_hull"

    def __init__(self, config: IAEConfig | None = None) -> None:
        self.config = config if config is not None else IAEConfig()

    def estimate(
        self,
        oracle: CanonicalBernoulliOracleSimulator,
        arm: int,
        *,
        confidence: float | None = None,
        target_angular_precision: float | None = None,
        tag: str | None = None,
    ) -> AmplitudeEstimate:
        """Estimate one arm without directly reading its hidden mean."""

        local = self.config
        if confidence is not None:
            local = replace(local, confidence=float(confidence))
        if target_angular_precision is not None:
            local = replace(
                local,
                target_angular_precision=float(target_angular_precision),
            )

        before = oracle.query_snapshot()
        observations: list[GroverObservation] = []
        interval = ConfidenceInterval(0.0, 1.0)
        angular_interval = AngularConfidenceInterval(0.0, math.pi / 2.0)
        estimate = 0.5
        warning: str | None = None

        for round_index in range(local.max_rounds):
            grover_power = min(2**round_index - 1, local.max_grover_power)
            successes = oracle.run_grover_experiment(
                arm,
                grover_power,
                local.shots_per_round,
                tag=tag,
            )
            observations.append(
                GroverObservation(
                    grover_power=grover_power,
                    successes=successes,
                    shots=local.shots_per_round,
                )
            )
            estimate, interval, angular_interval, warning = self._fit_grid(
                observations, local
            )
            if angular_interval.width <= 2.0 * local.target_angular_precision:
                break

        after = oracle.query_snapshot()
        return AmplitudeEstimate(
            arm=int(arm),
            estimate=estimate,
            interval=interval,
            angular_interval=angular_interval,
            observations=tuple(observations),
            executed_query_counts=QueryLedger.difference(after, before),
            interval_method=self.interval_method,
            numerical_warning=warning,
        )

    @staticmethod
    def _fit_grid(
        observations: list[GroverObservation],
        config: IAEConfig,
    ) -> tuple[
        float,
        ConfidenceInterval,
        AngularConfidenceInterval,
        str | None,
    ]:
        grid_size = config.grid_points
        theta_step = (math.pi / 2.0) / (grid_size - 1)
        # The same radius is safe for each executed round after a union bound.
        radius = math.sqrt(
            math.log(2.0 * config.max_rounds / config.confidence)
            / (2.0 * config.shots_per_round)
        )

        # Shape: (rounds, grid points).  The only Python-sized dimension left
        # is the short observation list used to construct the input vectors;
        # all grid evaluation, confidence-set intersection and likelihood
        # aggregation are NumPy operations.
        theta_grid = np.arange(grid_size, dtype=np.float64) * theta_step
        frequencies = np.asarray(
            [observation.frequency for observation in observations],
            dtype=np.float64,
        )
        successes = np.asarray(
            [observation.successes for observation in observations],
            dtype=np.float64,
        )
        shots = np.asarray(
            [observation.shots for observation in observations],
            dtype=np.float64,
        )

        phases = frequencies[:, np.newaxis] * theta_grid[np.newaxis, :]
        probabilities = np.sin(phases) ** 2
        empirical_probabilities = (successes / shots)[:, np.newaxis]
        # |d sin^2(q theta) / d theta| <= q.  Padding by one half-grid
        # cell makes the numerical set an outer hull.
        numerical_padding = (frequencies * theta_step / 2.0)[:, np.newaxis]
        consistent_mask = np.all(
            np.abs(probabilities - empirical_probabilities)
            <= radius + numerical_padding,
            axis=0,
        )

        consistent_indices = np.flatnonzero(consistent_mask)
        if consistent_indices.size == 0:
            # Numerical or model inconsistency must make the interval wider,
            # never silently create a confident estimate.
            return (
                0.5,
                ConfidenceInterval(0.0, 1.0),
                AngularConfidenceInterval(0.0, math.pi / 2.0),
                "empty numerical confidence set; returned the vacuous interval",
            )

        clipped_probabilities = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
        log_likelihoods = np.sum(
            successes[:, np.newaxis] * np.log(clipped_probabilities)
            + (shots - successes)[:, np.newaxis]
            * np.log1p(-clipped_probabilities),
            axis=0,
        )
        consistent_likelihoods = log_likelihoods[consistent_indices]
        # np.argmax returns the first maximum.  Because consistent_indices is
        # increasing, this exactly retains the scalar MLE tie-break toward the
        # smallest theta grid point.
        mle_index = int(consistent_indices[int(np.argmax(consistent_likelihoods))])
        minimum_consistent_theta = float(theta_grid[int(consistent_indices[0])])
        maximum_consistent_theta = float(theta_grid[int(consistent_indices[-1])])
        mle_theta = float(theta_grid[mle_index])

        estimate = math.sin(mle_theta) ** 2
        theta_lower = max(0.0, minimum_consistent_theta - theta_step / 2.0)
        theta_upper = min(math.pi / 2.0, maximum_consistent_theta + theta_step / 2.0)
        interval = ConfidenceInterval(
            lower=max(0.0, math.sin(theta_lower) ** 2),
            upper=min(1.0, math.sin(theta_upper) ** 2),
        )
        angular_interval = AngularConfidenceInterval(theta_lower, theta_upper)
        return estimate, interval, angular_interval, None


# A shorter public name matching the literature while retaining the explicit
# implementation name in traces and documentation.
IterativeAmplitudeEstimator = AnalyticIterativeAmplitudeEstimator


def mean_estimation_charge_proxy(
    variance: float,
    epsilon: float,
    *,
    logarithmic_factor: float = 1.0,
) -> float:
    r"""Evaluate a prior-style variance-adaptive mean-estimation proxy.

    The returned value

    .. math::

       (\sqrt{\sigma^2}/\epsilon + 1/\sqrt{\epsilon})L

    is useful for comparator plots.  It is neither an executed query count nor
    the angular Q-GapSelect layer conjecture.
    """

    if not 0.0 <= variance <= 0.25:
        raise ValueError("Bernoulli variance must lie in [0, 1/4]")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie in (0, 1)")
    if logarithmic_factor <= 0.0:
        raise ValueError("logarithmic_factor must be positive")
    return logarithmic_factor * (
        math.sqrt(variance) / epsilon + 1.0 / math.sqrt(epsilon)
    )
