from __future__ import annotations

import pytest

from qgapselect.attack_statistics import (
    exact_mcnemar_comparison,
    holm_adjust,
    stratified_cluster_bootstrap_difference,
    wilson_score_interval,
)


def test_wilson_interval_is_bounded_and_contains_the_estimate() -> None:
    interval = wilson_score_interval(5, 10)

    assert interval.estimate == 0.5
    assert 0.0 < interval.lower < interval.estimate
    assert interval.estimate < interval.upper < 1.0
    assert interval.confidence_level == 0.95


def test_wilson_interval_has_exact_numerical_endpoints_for_all_fail_or_pass() -> None:
    all_fail = wilson_score_interval(0, 500)
    all_pass = wilson_score_interval(500, 500)

    assert all_fail.lower == 0.0
    assert all_pass.upper == 1.0


def test_exact_mcnemar_uses_only_discordant_paired_units() -> None:
    comparison = exact_mcnemar_comparison(
        [True, True, True, True, True, False],
        [True, False, False, False, False, False],
    )

    assert comparison.sample_size == 6
    assert comparison.both_success == 1
    assert comparison.method_only_success == 4
    assert comparison.baseline_only_success == 0
    assert comparison.neither_success == 1
    assert comparison.absolute_rate_difference == pytest.approx(4 / 6)
    assert comparison.exact_mcnemar_p_value == pytest.approx(0.125)


def test_cluster_bootstrap_moves_all_rows_of_a_task_together() -> None:
    first = stratified_cluster_bootstrap_difference(
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        ["task-a", "task-a", "task-b", "task-b"],
        strata=["cwe-1", "cwe-1", "cwe-2", "cwe-2"],
        repetitions=1000,
        seed=17,
    )
    second = stratified_cluster_bootstrap_difference(
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        ["task-a", "task-a", "task-b", "task-b"],
        strata=["cwe-1", "cwe-1", "cwe-2", "cwe-2"],
        repetitions=1000,
        seed=17,
    )

    assert first == second
    assert first.cluster_count == 2
    assert first.stratum_count == 2
    assert first.point_difference == pytest.approx(0.5)
    assert first.lower <= first.point_difference <= first.upper


def test_cluster_bootstrap_rejects_a_cluster_split_across_strata() -> None:
    with pytest.raises(ValueError, match="exactly one stratum"):
        stratified_cluster_bootstrap_difference(
            [1.0, 0.0],
            [0.0, 0.0],
            ["same-task", "same-task"],
            strata=["first", "second"],
            repetitions=1000,
        )


def test_holm_adjustment_is_monotone_in_rank_order() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})

    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
