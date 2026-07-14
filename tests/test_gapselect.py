from __future__ import annotations

import math
import random

from qgapselect.gapselect import QGapSelect
from qgapselect.models import (
    AngularConfidenceInterval,
    ArmEstimate,
    ConfidenceInterval,
    GapSelectConfig,
    TerminationStatus,
)
from qgapselect.oracles import CanonicalBernoulliOracleSimulator


def _estimate(arm: int, lower: float, upper: float) -> ArmEstimate:
    return ArmEstimate(
        arm=arm,
        mean=(lower + upper) / 2.0,
        interval=ConfidenceInterval(lower, upper),
        angular_interval=AngularConfidenceInterval(
            math.asin(math.sqrt(lower)),
            math.asin(math.sqrt(upper)),
        ),
    )


def test_interval_classifier_certifies_both_sides_of_a_clear_topk_boundary() -> None:
    estimates = {
        0: _estimate(0, 0.86, 0.94),
        1: _estimate(1, 0.74, 0.82),
        2: _estimate(2, 0.18, 0.26),
        3: _estimate(3, 0.06, 0.14),
    }

    accepted, rejected = QGapSelect._classify(estimates, quota=2)
    assert accepted == {0, 1}
    assert rejected == {2, 3}


def test_interval_classifier_does_not_decide_overlapping_arms() -> None:
    estimates = {arm: _estimate(arm, 0.0, 1.0) for arm in range(5)}
    accepted, rejected = QGapSelect._classify(estimates, quota=2)

    assert not accepted
    assert not rejected


def test_classifier_is_sound_for_arbitrary_intervals_containing_true_means() -> None:
    rng = random.Random(20260714)
    for n_arms in range(3, 12):
        for k in range(1, n_arms):
            means = sorted((rng.random() for _ in range(n_arms)), reverse=True)
            truth = set(range(k))
            estimates = {
                arm: _estimate(
                    arm,
                    max(0.0, mean - rng.random() * 0.35),
                    min(1.0, mean + rng.random() * 0.35),
                )
                for arm, mean in enumerate(means)
            }
            accepted, rejected = QGapSelect._classify(estimates, quota=k)

            assert accepted <= truth
            assert rejected.isdisjoint(truth)


def test_selecting_every_arm_is_resolved_without_reward_queries() -> None:
    oracle = CanonicalBernoulliOracleSimulator((0.1, 0.2, 0.3), seed=4)
    result = QGapSelect().run(oracle, 3)

    assert result.selected == (0, 1, 2)
    assert result.accepted_by_intervals == (0, 1, 2)
    assert result.status is TerminationStatus.INTERVAL_RESOLVED
    assert result.executed_query_counts["total"] == 0
    assert not result.rounds


def test_easy_deterministic_instance_is_resolved_by_intervals() -> None:
    config = GapSelectConfig(
        confidence=0.1,
        initial_angular_epsilon=0.25,
        max_rounds=2,
        shots_per_iae_round=64,
        iae_max_rounds=1,
        iae_grid_points=513,
    )
    result = QGapSelect(config).run(
        CanonicalBernoulliOracleSimulator((1.0, 0.0, 0.0), seed=13), 1
    )

    assert result.selected == (0,)
    assert result.accepted_by_intervals == (0,)
    assert result.status is TerminationStatus.INTERVAL_RESOLVED
    assert result.executed_query_counts["coherent_total"] > 0
    assert result.candidate_theory_accounting.orientation_completion == {
        "selected": True,
        "rejected_complement": True,
    }
    assert sum(
        charge.newly_extracted_outputs
        for charge in result.rounds[-1].candidate_layer_charges
    ) == 3


def test_timeout_completion_is_not_mislabelled_as_interval_certification() -> None:
    config = GapSelectConfig(
        confidence=0.001,
        initial_angular_epsilon=0.25,
        max_rounds=1,
        shots_per_iae_round=1,
        iae_max_rounds=1,
        iae_grid_points=257,
    )
    result = QGapSelect(config).run(
        CanonicalBernoulliOracleSimulator((0.5, 0.5, 0.5), seed=2), 1
    )

    assert len(result.selected) == 1
    assert not result.accepted_by_intervals
    assert result.status is TerminationStatus.MAX_ROUNDS
    assert any("empirical completions" in warning for warning in result.warnings)
    assert result.paper_claim_status == "research_implementation_no_complexity_theorem"
    theory = result.candidate_theory_accounting
    assert theory.comparison_status == "incomplete_trace_not_comparable"
    assert theory.chosen_representation is None
    assert theory.total_candidate_charge is None
    assert not any(theory.orientation_completion.values())
    assert set(theory.orientation_partial_charges) == {
        "selected",
        "rejected_complement",
    }


def test_multiseed_reference_recovery_is_recorded_as_a_regression_not_a_theorem() -> None:
    config = GapSelectConfig(
        confidence=0.1,
        initial_angular_epsilon=0.25,
        max_rounds=3,
        shots_per_iae_round=32,
        iae_max_rounds=3,
        iae_grid_points=1025,
    )
    truth = (0, 1)
    results = [
        QGapSelect(config).run(
            CanonicalBernoulliOracleSimulator((0.9, 0.8, 0.2, 0.1), seed=seed),
            2,
        )
        for seed in range(20)
    ]

    assert sum(result.selected == truth for result in results) >= 18
    assert all(
        result.paper_claim_status == "research_implementation_no_complexity_theorem"
        for result in results
    )
