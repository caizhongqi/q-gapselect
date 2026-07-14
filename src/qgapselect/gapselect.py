"""Executable Q-GapSelect research driver.

The current backend performs analytic IAE separately for every active arm.  It
implements and tests the confidence-elimination logic, but it is intentionally
*not* presented as the conjectured coherent batch-extraction algorithm.  The
proposed layer expression is emitted in a separate theory-accounting object.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from .estimators import AnalyticIterativeAmplitudeEstimator
from .models import (
    ArmEstimate,
    CandidateLayerCharge,
    CandidateTheoryAccounting,
    GapSelectConfig,
    GapSelectResult,
    GapSelectRound,
    IAEConfig,
    TerminationStatus,
)
from .oracles import CanonicalBernoulliOracleSimulator, QueryLedger


class QGapSelect:
    """Top-k confidence elimination over a canonical Bernoulli oracle.

    This executable class is a research harness with two separate outputs:

    * ``executed_query_counts`` are exact logical calls made by the all-active
      analytic-IAE reference backend;
    * ``candidate_theory_accounting`` evaluates the proposed layer expression
      from the observed elimination trace and is explicitly unproved.

    A future coherent ``QBoundary``/``QBatchExtract`` implementation may replace
    the reference backend without changing the result schema.
    """

    def __init__(self, config: GapSelectConfig | None = None) -> None:
        self.config = config if config is not None else GapSelectConfig()

    def run(
        self,
        oracle: CanonicalBernoulliOracleSimulator,
        k: int,
    ) -> GapSelectResult:
        n_arms = oracle.n_arms
        if not 1 <= k <= n_arms:
            raise ValueError("k must be in {1, ..., number of arms}")

        before_run = oracle.query_snapshot()
        accepted: set[int] = set()
        rejected: set[int] = set()
        active: set[int] = set(range(n_arms))
        latest_estimates: dict[int, ArmEstimate] = {}
        traces: list[GapSelectRound] = []
        orientation_charges: dict[str, list[CandidateLayerCharge]] = {
            "selected": [],
            "rejected_complement": [],
        }
        warnings: list[str] = [
            "the executable backend estimates every active arm independently",
            "candidate layer charges are conjectural and are not executed queries",
        ]

        if k == n_arms:
            accepted.update(active)
            active.clear()
            return self._build_result(
                oracle=oracle,
                before_run=before_run,
                k=k,
                accepted=accepted,
                active=active,
                latest_estimates=latest_estimates,
                traces=traces,
                orientation_charges=orientation_charges,
                status=TerminationStatus.INTERVAL_RESOLVED,
                warnings=warnings,
                angular_scale_origin=self.config.initial_angular_epsilon,
            )

        status = TerminationStatus.MAX_ROUNDS
        for round_index in range(1, self.config.max_rounds + 1):
            if len(accepted) == k:
                status = TerminationStatus.INTERVAL_RESOLVED
                break
            remaining_quota = k - len(accepted)
            if remaining_quota == len(active):
                accepted.update(active)
                active.clear()
                status = TerminationStatus.INTERVAL_RESOLVED
                break

            angular_epsilon = self.config.initial_angular_epsilon * (
                self.config.epsilon_decay ** (round_index - 1)
            )
            active_before = tuple(sorted(active))
            before_round = oracle.query_snapshot()
            round_estimates: dict[int, ArmEstimate] = {}

            # Allocate delta across all arms and dyadic outer rounds.  IAE then
            # allocates its local confidence across its own circuit schedule.
            local_confidence = (
                6.0
                * self.config.confidence
                / (math.pi**2 * round_index**2 * n_arms)
            )
            iae = AnalyticIterativeAmplitudeEstimator(
                IAEConfig(
                    target_angular_precision=max(angular_epsilon / 4.0, 1e-8),
                    confidence=local_confidence,
                    shots_per_round=self.config.shots_per_iae_round,
                    max_rounds=self.config.iae_max_rounds,
                    max_grover_power=self.config.iae_max_grover_power,
                    grid_points=self.config.iae_grid_points,
                )
            )

            for arm in active_before:
                estimate = iae.estimate(
                    oracle,
                    arm,
                    tag=f"gapselect_round_{round_index}",
                )
                arm_estimate = ArmEstimate(
                    arm=arm,
                    mean=estimate.estimate,
                    interval=estimate.interval,
                    angular_interval=estimate.angular_interval,
                )
                round_estimates[arm] = arm_estimate
                latest_estimates[arm] = arm_estimate
                if estimate.numerical_warning is not None:
                    warnings.append(f"arm {arm}: {estimate.numerical_warning}")

            newly_accepted, newly_rejected = self._classify(
                round_estimates,
                remaining_quota,
            )

            # Defensive caps: classification is strict and should already obey
            # these constraints, but a numerical regression must not overfill a
            # Top-k answer.
            if len(newly_accepted) > remaining_quota:
                newly_accepted = set(
                    sorted(
                        newly_accepted,
                        key=lambda arm: (-round_estimates[arm].interval.lower, arm),
                    )[:remaining_quota]
                )
                warnings.append(
                    "acceptance cap activated after numerical classification"
                )

            certified_accepted = set(newly_accepted)
            certified_rejected = set(newly_rejected)
            accepted.update(certified_accepted)
            rejected.update(certified_rejected)
            active.difference_update(newly_accepted | newly_rejected)

            if len(accepted) == k:
                # Remaining arms are outside the selected set once all k arms
                # have been certified in.
                certified_rejected.update(active)
                rejected.update(active)
                active.clear()
                status = TerminationStatus.INTERVAL_RESOLVED
            elif len(accepted) + len(active) == k:
                certified_accepted.update(active)
                accepted.update(active)
                active.clear()
                status = TerminationStatus.INTERVAL_RESOLVED

            round_candidate_charges = tuple(
                CandidateLayerCharge(
                    round_index=round_index,
                    angular_epsilon=angular_epsilon,
                    active_count=len(active_before),
                    representation=representation,
                    newly_extracted_outputs=extracted_count,
                    candidate_charge=(
                        math.sqrt(len(active_before) * (extracted_count + 1))
                        / angular_epsilon
                    ),
                )
                for representation, extracted_count in (
                    ("selected", len(certified_accepted)),
                    ("rejected_complement", len(certified_rejected)),
                )
            )
            for charge in round_candidate_charges:
                orientation_charges[charge.representation].append(charge)

            after_round = oracle.query_snapshot()
            traces.append(
                GapSelectRound(
                    round_index=round_index,
                    angular_epsilon=angular_epsilon,
                    active_before=active_before,
                    accepted=tuple(sorted(certified_accepted)),
                    rejected=tuple(sorted(certified_rejected)),
                    estimates=tuple(round_estimates[arm] for arm in active_before),
                    executed_query_counts=QueryLedger.difference(
                        after_round, before_round
                    ),
                    candidate_layer_charges=round_candidate_charges,
                )
            )

            if status is TerminationStatus.INTERVAL_RESOLVED:
                break

        return self._build_result(
            oracle=oracle,
            before_run=before_run,
            k=k,
            accepted=accepted,
            active=active,
            latest_estimates=latest_estimates,
            traces=traces,
            orientation_charges=orientation_charges,
            status=status,
            warnings=warnings,
            angular_scale_origin=self.config.initial_angular_epsilon,
        )

    # Scikit-style convenience alias without making fitting stateful.
    fit = run

    @staticmethod
    def _classify(
        estimates: dict[int, ArmEstimate],
        quota: int,
    ) -> tuple[set[int], set[int]]:
        """Conservatively classify arms from simultaneous confidence intervals."""

        arms = tuple(estimates)
        if quota <= 0:
            return set(), set(arms)
        if quota >= len(arms):
            return set(arms), set()

        accepted: set[int] = set()
        rejected: set[int] = set()
        for arm in arms:
            interval = estimates[arm].interval
            # If fewer than ``quota`` other arms could be as large as this
            # arm's lower endpoint, it is certainly on the selected side.
            possible_above = sum(
                estimates[other].interval.upper >= interval.lower
                for other in arms
                if other != arm
            )
            if possible_above < quota:
                accepted.add(arm)
                continue

            # If at least ``quota`` arms have lower endpoints strictly above
            # this upper endpoint, this arm is certainly outside Top-k.
            certainly_above = sum(
                estimates[other].interval.lower > interval.upper
                for other in arms
                if other != arm
            )
            if certainly_above >= quota:
                rejected.add(arm)
        return accepted, rejected

    @staticmethod
    def _build_result(
        *,
        oracle: CanonicalBernoulliOracleSimulator,
        before_run,
        k: int,
        accepted: set[int],
        active: set[int],
        latest_estimates: dict[int, ArmEstimate],
        traces: list[GapSelectRound],
        orientation_charges: dict[str, list[CandidateLayerCharge]],
        status: TerminationStatus,
        warnings: list[str],
        angular_scale_origin: float,
    ) -> GapSelectResult:
        # When the interval procedure times out, return a deterministic empirical
        # completion but keep it visibly separate from accepted_by_intervals.
        selected = set(accepted)
        if len(selected) < k:
            candidates = sorted(
                active,
                key=lambda arm: (
                    -latest_estimates.get(
                        arm,
                        ArmEstimate(
                            arm=arm,
                            mean=0.5,
                            interval=_vacuous_interval(),
                            angular_interval=_vacuous_angular_interval(),
                        ),
                    ).mean,
                    arm,
                ),
            )
            selected.update(candidates[: k - len(selected)])
            warnings.append(
                "max-round output contains empirical completions not resolved "
                "by intervals"
            )

        after_run = oracle.query_snapshot()
        totals = {
            representation: sum(charge.candidate_charge for charge in charges)
            for representation, charges in orientation_charges.items()
        }
        extracted = {
            representation: sum(
                charge.newly_extracted_outputs for charge in charges
            )
            for representation, charges in orientation_charges.items()
        }
        required = {
            "selected": k,
            "rejected_complement": oracle.n_arms - k,
        }
        completion = {
            representation: (
                extracted[representation] == output_size
                or (not traces and status is TerminationStatus.INTERVAL_RESOLVED)
            )
            for representation, output_size in required.items()
        }
        if status is TerminationStatus.INTERVAL_RESOLVED and not all(
            completion.values()
        ):
            raise RuntimeError(
                "an interval-resolved trace must complete both output certificates"
            )

        if status is TerminationStatus.INTERVAL_RESOLVED:
            chosen_representation: str | None = min(
                totals,
                key=lambda representation: (
                    totals[representation],
                    required[representation],
                    representation != "selected",
                ),
            )
            alternative_representation: str | None = next(
                representation
                for representation in totals
                if representation != chosen_representation
            )
            chosen_charges = tuple(orientation_charges[chosen_representation])
            chosen_total: float | None = totals[chosen_representation]
            alternative_charges = tuple(
                orientation_charges[alternative_representation]
            )
            alternative_total: float | None = totals[alternative_representation]
            comparison_status = "complete_certificate_trace_proxy"
        else:
            chosen_representation = None
            alternative_representation = None
            chosen_charges = ()
            chosen_total = None
            alternative_charges = ()
            alternative_total = None
            comparison_status = "incomplete_trace_not_comparable"
        theory = CandidateTheoryAccounting(
            expression=(
                "min_b sum_r sqrt(N_r * (M_{r,b} + 1)) / epsilon_r; "
                "b in {selected, rejected_complement}"
            ),
            angular_scale_origin=angular_scale_origin,
            chosen_representation=chosen_representation,
            charges=chosen_charges,
            total_candidate_charge=chosen_total,
            alternative_representation=alternative_representation,
            alternative_charges=alternative_charges,
            alternative_candidate_charge=alternative_total,
            orientation_completion=completion,
            orientation_partial_charges=totals,
            comparison_status=comparison_status,
        )
        return GapSelectResult(
            selected=tuple(sorted(selected)),
            accepted_by_intervals=tuple(sorted(accepted)),
            unresolved_at_stop=tuple(sorted(active)),
            status=status,
            rounds=tuple(traces),
            executed_query_counts=QueryLedger.difference(after_run, before_run),
            candidate_theory_accounting=theory,
            warnings=tuple(dict.fromkeys(warnings)),
        )


def _vacuous_interval():
    # Local import avoids adding a confidence-interval symbol to the public
    # algorithm namespace merely for fallback sorting.
    from .models import ConfidenceInterval

    return ConfidenceInterval(0.0, 1.0)


def _vacuous_angular_interval():
    from .models import AngularConfidenceInterval

    return AngularConfidenceInterval(0.0, math.pi / 2.0)


def top_k_from_scores(scores: Iterable[float], k: int) -> tuple[int, ...]:
    """Deterministic classical scorer used only by benchmark/evaluation code."""

    values = tuple(float(score) for score in scores)
    if not 1 <= k <= len(values):
        raise ValueError("k must be in {1, ..., number of scores}")
    return tuple(sorted(sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]))
