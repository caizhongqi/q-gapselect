from __future__ import annotations

import pytest

from qgapselect.attack_metrics import (
    TaskOutcomeStatus,
    aggregate_attack_metrics,
)
from qgapselect.llm_attack import (
    EvaluatedGeneration,
    FunctionalityState,
    GenerationRecord,
    GenerationStatus,
    QueryBudget,
    SecurityState,
    Seed,
    ValidatorResult,
)


def _complete(
    record_id: str,
    *,
    task: str = "task",
    variant: str | None,
    seed: int = 1,
    query_index: int,
    functionality: FunctionalityState,
    security: SecurityState,
) -> EvaluatedGeneration:
    generation = GenerationRecord(
        record_id=record_id,
        model_id="victim",
        task_id=task,
        variant_id=variant,
        seed=Seed(seed),
        query_index=query_index,
        query_cost=0 if variant is None else 1,
        status=GenerationStatus.COMPLETE,
        output_ref=f"test://{record_id}",
    )
    return EvaluatedGeneration(
        generation,
        ValidatorResult(
            record_id=record_id,
            validator_id="validator",
            functionality=functionality,
            security=security,
        ),
    )


def _unfinished(
    record_id: str,
    *,
    status: GenerationStatus,
    query_index: int,
) -> EvaluatedGeneration:
    return EvaluatedGeneration(
        GenerationRecord(
            record_id=record_id,
            model_id="victim",
            task_id="task",
            variant_id="a",
            seed=Seed(1),
            query_index=query_index,
            query_cost=1,
            status=status,
        ),
        None,
    )


def _clean(
    *, security: SecurityState = SecurityState.SAFE
) -> EvaluatedGeneration:
    return _complete(
        "clean",
        variant=None,
        query_index=0,
        functionality=FunctionalityState.FUNCTIONAL,
        security=security,
    )


def test_budget_prefix_truncates_before_later_success() -> None:
    records = [
        _clean(),
        _complete(
            "safe-first",
            variant="a",
            query_index=1,
            functionality=FunctionalityState.FUNCTIONAL,
            security=SecurityState.SAFE,
        ),
        _complete(
            "vulnerable-second",
            variant="a",
            query_index=2,
            functionality=FunctionalityState.FUNCTIONAL,
            security=SecurityState.VULNERABLE,
        ),
    ]

    at_one = aggregate_attack_metrics(
        records,
        budget=QueryBudget(1),
        expected_task_model_units=(("victim", "task"),),
        selected_variants={"task": ("a",)},
    )
    at_two = aggregate_attack_metrics(
        records,
        budget=QueryBudget(2),
        expected_task_model_units=(("victim", "task"),),
        selected_variants={"task": ("a",)},
    )

    assert at_one.fv_asr_at_q.value == 0.0
    assert at_one.task_outcomes[0].attack_queries_included == 1
    assert at_two.fv_asr_at_q.value == 1.0
    assert at_two.paired_counterfactual_asr_at_q.value == 1.0
    assert at_two.query_to_first_success.values == (2,)


def test_no_vulnerability_produces_zero_asr_and_delta() -> None:
    metrics = aggregate_attack_metrics(
        [
            _clean(),
            _complete(
                "safe",
                variant="a",
                query_index=1,
                functionality=FunctionalityState.FUNCTIONAL,
                security=SecurityState.SAFE,
            ),
        ],
        budget=QueryBudget(1),
        expected_task_model_units=(("victim", "task"),),
        selected_variants={"task": ("a",)},
    )

    assert metrics.asr_at_q.value == 0.0
    assert metrics.fv_asr_at_q.value == 0.0
    assert metrics.delta_fv_asr_at_q == 0.0
    assert metrics.functionality_retention.value == 1.0
    assert metrics.task_outcomes[0].status is TaskOutcomeStatus.FAILURE


def test_base_vulnerability_is_subtracted_and_not_counterfactual() -> None:
    metrics = aggregate_attack_metrics(
        [
            _clean(security=SecurityState.VULNERABLE),
            _complete(
                "still-vulnerable",
                variant="a",
                query_index=1,
                functionality=FunctionalityState.FUNCTIONAL,
                security=SecurityState.VULNERABLE,
            ),
        ],
        budget=QueryBudget(1),
        expected_task_model_units=(("victim", "task"),),
        selected_variants={"task": ("a",)},
    )

    assert metrics.fv_asr_at_q.value == 1.0
    assert metrics.diagnostic_clean_any_seed_fv_asr.value == 1.0
    assert metrics.delta_fv_asr_at_q == 0.0
    assert metrics.paired_counterfactual_asr_at_q.value == 0.0


def test_repeated_successes_still_count_as_one_task_unit() -> None:
    metrics = aggregate_attack_metrics(
        [
            _clean(),
            _complete(
                "success-1",
                variant="a",
                query_index=1,
                functionality=FunctionalityState.FUNCTIONAL,
                security=SecurityState.VULNERABLE,
            ),
            _complete(
                "success-2",
                variant="a",
                query_index=2,
                functionality=FunctionalityState.FUNCTIONAL,
                security=SecurityState.VULNERABLE,
            ),
        ],
        budget=QueryBudget(2),
        expected_task_model_units=(("victim", "task"),),
        selected_variants={"task": ("a",)},
    )

    assert metrics.task_model_unit_count == 1
    assert metrics.fv_asr_at_q.as_dict() == {
        "numerator": 1,
        "denominator": 1,
        "value": 1.0,
    }
    assert metrics.query_to_first_success.values == (1,)


def test_timeout_and_indeterminate_are_reported_without_dropping_denominator() -> None:
    timeout = aggregate_attack_metrics(
        [_clean(), _unfinished("timeout", status=GenerationStatus.TIMEOUT, query_index=1)],
        budget=QueryBudget(1),
        expected_task_model_units=(("victim", "task"),),
        selected_variants={"task": ("a",)},
    )
    indeterminate = aggregate_attack_metrics(
        [_clean(), _unfinished("error", status=GenerationStatus.ERROR, query_index=1)],
        budget=QueryBudget(1),
        expected_task_model_units=(("victim", "task"),),
        selected_variants={"task": ("a",)},
    )

    assert timeout.task_outcomes[0].status is TaskOutcomeStatus.TIMEOUT
    assert timeout.timeout_rate.value == 1.0
    assert indeterminate.task_outcomes[0].status is TaskOutcomeStatus.INDETERMINATE
    assert indeterminate.indeterminate_rate.value == 1.0
    assert timeout.fv_asr_at_q.denominator == 1
    assert indeterminate.fv_asr_at_q.denominator == 1


def test_missing_paired_clean_seed_blocks_counterfactual_not_standard_fv_asr() -> None:
    unpaired = _complete(
        "wrong-seed",
        variant="a",
        seed=2,
        query_index=1,
        functionality=FunctionalityState.FUNCTIONAL,
        security=SecurityState.VULNERABLE,
    )
    metrics = aggregate_attack_metrics(
        [_clean(), unpaired],
        budget=QueryBudget(1),
        expected_task_model_units=(("victim", "task"),),
        selected_variants={"task": ("a",)},
    )

    assert metrics.fv_asr_at_q.value == 1.0
    assert metrics.paired_counterfactual_asr_at_q.value == 0.0
    assert metrics.task_outcomes[0].status is TaskOutcomeStatus.INDETERMINATE


def test_empty_preregistered_unit_is_budget_exhausted_not_removed() -> None:
    metrics = aggregate_attack_metrics(
        [],
        budget=QueryBudget(3),
        expected_task_model_units=(("victim", "missing-task"),),
        selected_variants={"missing-task": ("a",)},
    )

    assert metrics.task_model_unit_count == 1
    assert metrics.fv_asr_at_q.value == pytest.approx(0.0)
    assert metrics.status_counts[TaskOutcomeStatus.BUDGET_EXHAUSTED.value] == 1


def test_delta_uses_only_same_budget_prefix_paired_seeds() -> None:
    records = [
        _complete(
            "clean-unpaired-vulnerable",
            variant=None,
            seed=1,
            query_index=0,
            functionality=FunctionalityState.FUNCTIONAL,
            security=SecurityState.VULNERABLE,
        ),
        _complete(
            "clean-paired-safe",
            variant=None,
            seed=2,
            query_index=0,
            functionality=FunctionalityState.FUNCTIONAL,
            security=SecurityState.SAFE,
        ),
        _complete(
            "attack-paired-vulnerable",
            variant="a",
            seed=2,
            query_index=1,
            functionality=FunctionalityState.FUNCTIONAL,
            security=SecurityState.VULNERABLE,
        ),
    ]

    metrics = aggregate_attack_metrics(
        records,
        budget=QueryBudget(1),
        expected_task_model_units=(("victim", "task"),),
        selected_variants={"task": ("a",)},
    )

    assert metrics.paired_attack_fv_asr_at_q.value == 1.0
    assert metrics.paired_clean_fv_asr_at_q.value == 0.0
    assert metrics.delta_fv_asr_at_q == 1.0
    assert metrics.diagnostic_clean_any_seed_fv_asr.value == 1.0
    assert metrics.diagnostic_unpaired_delta_fv_asr_at_q == 0.0
    assert metrics.task_outcomes[0].paired_evaluable_seed_count == 1
    document = metrics.as_dict()
    assert "clean-FV-ASR" not in document
    assert document["Delta-FV-ASR@Q"] == 1.0
    assert document["diagnostic-unpaired-any-seed-Delta-FV-ASR@Q"] == 0.0


def test_expected_units_reject_unexpected_records_instead_of_hiding_them() -> None:
    unexpected = _complete(
        "unexpected-task",
        task="other-task",
        variant=None,
        query_index=0,
        functionality=FunctionalityState.FUNCTIONAL,
        security=SecurityState.SAFE,
    )

    with pytest.raises(ValueError, match="outside expected_task_model_units"):
        aggregate_attack_metrics(
            [_clean(), unexpected],
            budget=QueryBudget(1),
            expected_task_model_units=(("victim", "task"),),
            selected_variants={"task": ("a",)},
        )


def test_budget_prefix_uses_portfolio_rank_not_original_candidate_order() -> None:
    records = [
        _clean(),
        _complete(
            "original-first-safe",
            variant="b",
            query_index=1,
            functionality=FunctionalityState.FUNCTIONAL,
            security=SecurityState.SAFE,
        ),
        _complete(
            "rank-first-vulnerable",
            variant="a",
            query_index=2,
            functionality=FunctionalityState.FUNCTIONAL,
            security=SecurityState.VULNERABLE,
        ),
    ]

    metrics = aggregate_attack_metrics(
        records,
        budget=QueryBudget(1),
        expected_task_model_units=(("victim", "task"),),
        selected_variants={"task": ("a", "b")},
    )

    assert metrics.fv_asr_at_q.value == 1.0
    assert metrics.task_outcomes[0].query_to_first_fv_success == 1
