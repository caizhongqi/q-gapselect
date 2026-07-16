from __future__ import annotations

import json
import math

import pytest

from qgapselect.exact_count_fixtures import generate_exact_count_fixture
from qgapselect.frozen_coherent_oracle import build_frozen_empirical_coherent_oracle
from qgapselect.matched_quantum_baselines import (
    CLAIM_SCOPE,
    REFERENCE_STATUS,
    CoarsePartitionBAICompositionBaseline,
    KnownTimeVariableTimeReference,
    KOnlyIndependentAdaptiveBaseline,
    MatchedBaselineConfig,
    RepeatedSingleOutputBaseline,
    UnknownTimeVariableTimeReference,
    aggregate_fixed_cap_evaluations,
    evaluate_fixed_query_cap,
)
from qgapselect.models import IAEConfig


def _fixture():
    return generate_exact_count_fixture(
        {"a": 0.99, "b": 0.90, "c": 0.10, "d": 0.01},
        table_size=100,
        seed=13,
    )


def _config(*, shots: int = 512, max_levels: int = 5) -> MatchedBaselineConfig:
    return MatchedBaselineConfig(
        initial_angular_precision=math.pi / 8.0,
        precision_decay=0.5,
        max_levels=max_levels,
        iae=IAEConfig(
            target_angular_precision=0.1,
            confidence=0.05,
            shots_per_round=shots,
            max_rounds=5,
            max_grover_power=15,
            grid_points=1025,
        ),
    )


def _oracle(seed: int = 7):
    return build_frozen_empirical_coherent_oracle(_fixture(), measurement_seed=seed)


@pytest.mark.parametrize(
    "method",
    [
        KOnlyIndependentAdaptiveBaseline(_config()),
        CoarsePartitionBAICompositionBaseline(
            _config(), block_size=2, partition_seed=3
        ),
        RepeatedSingleOutputBaseline(_config()),
        UnknownTimeVariableTimeReference(_config()),
        KnownTimeVariableTimeReference((5, 5, 5, 5), _config()),
    ],
)
def test_all_references_certify_easy_frozen_topk_under_one_interface(method) -> None:
    result = method.run(
        _oracle(19),
        2,
        query_cap=1_000_000,
        failure_budget=0.05,
    )

    assert result.certified
    assert result.selected == (0, 1)
    assert result.status.startswith("certified")
    assert not result.timeout
    assert result.budget_valid
    assert result.oracle_queries == sum(result.per_arm_queries.values())
    assert result.claim_scope == CLAIM_SCOPE
    assert result.reference_status == REFERENCE_STATUS
    assert not result.official_reproduction
    assert not result.hardware_claimable


def test_hard_cap_stops_before_overshooting_experiment() -> None:
    method = KOnlyIndependentAdaptiveBaseline(_config())

    # One m=0 experiment costs exactly 512 logical oracle calls.  The second
    # experiment is rejected before it can overshoot a 700-query cap.
    result = method.run(
        _oracle(),
        2,
        query_cap=700,
        failure_budget=0.05,
    )

    assert result.timeout
    assert not result.certified
    assert result.status == "query_cap_exhausted"
    assert result.oracle_queries == 512
    assert result.oracle_queries <= result.query_cap
    assert result.stages[0].query_counts["total"] == 512


def test_zero_affordable_experiments_leave_a_zero_fresh_ledger() -> None:
    result = KOnlyIndependentAdaptiveBaseline(_config()).run(
        _oracle(),
        2,
        query_cap=511,
        failure_budget=0.05,
    )

    assert result.timeout
    assert result.oracle_queries == 0
    assert all(value == 0 for value in result.per_arm_queries.values())


def test_nonfresh_oracle_is_rejected_for_fairness() -> None:
    oracle = _oracle()
    oracle.run_grover_experiment(0, 0, 1)

    with pytest.raises(ValueError, match="fresh zero ledger"):
        KOnlyIndependentAdaptiveBaseline(_config()).run(
            oracle,
            2,
            query_cap=1000,
            failure_budget=0.05,
        )


def test_fixed_cap_adapter_scores_all_attempts_without_both_success_filter() -> None:
    method = KOnlyIndependentAdaptiveBaseline(_config())
    rows = tuple(
        evaluate_fixed_query_cap(
            method,
            lambda seed=seed: _oracle(seed),
            k=2,
            query_cap=700,
            failure_budget=0.05,
            expected_top_k=(0, 1),
        )
        for seed in (1, 2, 3)
    )
    aggregate = aggregate_fixed_cap_evaluations(rows)

    assert aggregate.attempts == 3
    assert aggregate.timeout_count == 3
    assert aggregate.uncertified_count == 3
    assert aggregate.certified_exact_count == 0
    assert aggregate.certified_exact_rate == 0.0
    assert aggregate.budget_violation_count == 0
    assert aggregate.mean_queries == 512


def test_fixed_cap_adapter_reports_certified_exact_not_heuristic_exact() -> None:
    evaluation = evaluate_fixed_query_cap(
        KOnlyIndependentAdaptiveBaseline(_config()),
        lambda: _oracle(31),
        k=2,
        query_cap=1_000_000,
        failure_budget=0.05,
        expected_top_k=(1, 0),
    )

    assert evaluation.output_exact
    assert evaluation.certified_exact
    assert evaluation.budget_valid
    assert json.loads(json.dumps(evaluation.as_dict()))["certified_exact"] is True


def test_variable_time_fields_are_explicit_proxies_from_executed_arm_ledgers() -> None:
    result = UnknownTimeVariableTimeReference(_config()).run(
        _oracle(),
        2,
        query_cap=1_000_000,
        failure_budget=0.05,
    )
    expected_rms = math.sqrt(sum(value**2 for value in result.per_arm_queries.values()))

    assert result.variable_time_rms_proxy == pytest.approx(expected_rms)
    assert result.variable_time_rms_proxy <= result.serial_query_total
    assert "unknown_stopping_times" in result.information_regime


def test_known_time_control_declares_stronger_information_and_fails_closed() -> None:
    fixture = generate_exact_count_fixture(
        {"a": 0.55, "b": 0.52, "c": 0.48, "d": 0.45},
        table_size=100,
        seed=9,
    )
    method = KnownTimeVariableTimeReference((1, 1, 1, 1), _config(max_levels=3))
    result = method.run(
        build_frozen_empirical_coherent_oracle(fixture, measurement_seed=4),
        2,
        query_cap=1_000_000,
        failure_budget=0.05,
    )

    assert not result.certified
    assert not result.timeout
    assert result.status == "known_stopping_schedule_exhausted"
    assert "public_per_arm_stop_levels" in result.information_regime


def test_k_only_schedule_exhaustion_is_not_mislabeled_as_known_time() -> None:
    fixture = generate_exact_count_fixture(
        {"a": 0.51, "b": 0.50, "c": 0.49},
        table_size=100,
        seed=29,
    )
    result = KOnlyIndependentAdaptiveBaseline(_config(max_levels=1)).run(
        build_frozen_empirical_coherent_oracle(fixture, measurement_seed=71),
        1,
        query_cap=1_000_000,
        failure_budget=0.05,
    )

    assert not result.certified
    assert result.status == "precision_schedule_exhausted"


def test_composition_stages_share_one_ledger_and_cover_the_total() -> None:
    result = CoarsePartitionBAICompositionBaseline(
        _config(), block_size=2, partition_seed=8
    ).run(
        _oracle(),
        2,
        query_cap=1_000_000,
        failure_budget=0.05,
    )

    assert result.certified
    assert len(result.stages) == 3
    assert sum(stage.query_counts["total"] for stage in result.stages) == (
        result.oracle_queries
    )
    assert "public_partition" in result.information_regime


@pytest.mark.parametrize(
    ("query_cap", "failure_budget"),
    [(-1, 0.05), (True, 0.05), (100, 0.0), (100, 1.0)],
)
def test_invalid_common_budget_inputs_are_rejected(
    query_cap: object, failure_budget: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        KOnlyIndependentAdaptiveBaseline(_config()).run(
            _oracle(),
            2,
            query_cap=query_cap,  # type: ignore[arg-type]
            failure_budget=failure_budget,  # type: ignore[arg-type]
        )


def test_known_time_schedule_must_align_with_oracle() -> None:
    with pytest.raises(ValueError, match="align"):
        KnownTimeVariableTimeReference((2, 2), _config()).run(
            _oracle(),
            2,
            query_cap=10_000,
            failure_budget=0.05,
        )
