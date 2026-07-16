from __future__ import annotations

import json
from dataclasses import dataclass, replace

import pytest

from qgapselect.ccfa_evidence_gate import (
    CLAIM_SCOPE,
    CONDITIONING,
    CCFAEvidenceGateConfig,
    PreregisteredFixture,
    evaluate_ccfa_evidence_gate,
)

FAMILIES = {"f0": ("x0", "x1"), "f1": ("y0", "y1")}
CAPS = (100, 200)
REPETITIONS = 6
INFORMATION = "oracle_k_failure_budget_query_cap"


def _config(**changes: object) -> CCFAEvidenceGateConfig:
    values: dict[str, object] = {
        "candidate_method_id": "candidate",
        "strongest_baseline_method_ids": ("baseline",),
        "information_regime": INFORMATION,
        "preregistered_fixtures": tuple(
            PreregisteredFixture(family, instance)
            for family, instances in FAMILIES.items()
            for instance in instances
        ),
        "preregistered_query_caps": CAPS,
        "repetitions_per_fixture": REPETITIONS,
        "preregistration_status": "LOCKED_BEFORE_RUN",
        "preregistration_manifest_sha256": "a" * 64,
        "minimum_risk_difference": 0.20,
        "theory_statuses": {
            "new_upper_bound": "PROVED",
            "same_interface_composition_frontier": "PROVED",
            "matching_lower_bound": "PROVED",
        },
        "implementation_resource_statuses": {
            "coherent_index_execution": "VERIFIED",
            "resource_accounting": "VERIFIED",
            "strongest_baseline_fidelity": "VERIFIED",
        },
        "familywise_alpha": 0.05,
        "bootstrap_repetitions": 251,
        "bootstrap_seed": 991,
    }
    values.update(changes)
    return CCFAEvidenceGateConfig(**values)  # type: ignore[arg-type]


def _records(
    *,
    candidate_success: bool = True,
    baseline_success: bool = False,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for family, instances in FAMILIES.items():
        for instance in instances:
            for cap in CAPS:
                for repetition in range(REPETITIONS):
                    for method, success in (
                        ("candidate", candidate_success),
                        ("baseline", baseline_success),
                    ):
                        result.append(
                            {
                                "family_id": family,
                                "instance_id": instance,
                                "repetition": repetition,
                                "query_cap": cap,
                                "method_id": method,
                                "information_regime": INFORMATION,
                                "certified_exact": success,
                                "timeout": False,
                                "budget_valid": True,
                                "coherent_queries": cap // (2 if method == "candidate" else 4),
                            }
                        )
    return result


def _find_record(
    records: list[dict[str, object]],
    *,
    family: str = "f0",
    instance: str = "x0",
    cap: int = 100,
    repetition: int = 0,
    method: str = "candidate",
) -> dict[str, object]:
    return next(
        row
        for row in records
        if row["family_id"] == family
        and row["instance_id"] == instance
        and row["query_cap"] == cap
        and row["repetition"] == repetition
        and row["method_id"] == method
    )


def test_complete_strong_panel_can_pass_every_fail_closed_gate() -> None:
    report = evaluate_ccfa_evidence_gate(_records(), _config())

    assert report.advantage_claimable
    assert report.blockers == ()
    assert all(check.passed for check in report.checks)
    assert report.claim_scope == CLAIM_SCOPE
    assert "does not independently prove" in report.interpretation
    assert len(report.fixture_summaries) == 4 * 2 * 2
    assert len(report.family_cap_summaries) == 2 * 2 * 2
    assert len(report.fixed_cap_summaries) == 2 * 2
    assert len(report.pairwise_comparisons) == 2 * 2
    assert all(item.risk_difference == 1.0 for item in report.pairwise_comparisons)
    assert all(item.bootstrap.lower == 1.0 for item in report.pairwise_comparisons)
    assert all(
        item.holm_adjusted_p_value <= 0.05
        for item in report.pairwise_comparisons
    )
    assert all(item.conditioning == CONDITIONING for item in report.fixed_cap_summaries)
    assert json.loads(json.dumps(report.as_dict()))["advantage_claimable"] is True


@dataclass(frozen=True)
class AttributeRun:
    family_id: str
    instance_id: str
    repetition: int
    query_cap: int
    method_id: str
    information_regime: str
    certified_exact: bool
    timeout: bool
    budget_valid: bool
    coherent_queries: int


def test_attribute_records_and_mapping_records_have_identical_results() -> None:
    mappings = _records()
    objects = [AttributeRun(**row) for row in mappings]  # type: ignore[arg-type]

    first = evaluate_ccfa_evidence_gate(mappings, _config())
    second = evaluate_ccfa_evidence_gate(objects, _config())

    assert first.as_dict() == second.as_dict()


def test_timeout_and_budget_violation_remain_in_the_fixed_cap_denominator() -> None:
    records = _records()
    changed = _find_record(records)
    changed.update(timeout=True, budget_valid=False, coherent_queries=101)

    report = evaluate_ccfa_evidence_gate(records, _config())
    summary = next(
        row
        for row in report.fixture_summaries
        if row.family_id == "f0"
        and row.instance_id == "x0"
        and row.query_cap == 100
        and row.method_id == "candidate"
    )

    assert summary.attempt_count == REPETITIONS
    assert summary.certified_exact_count == REPETITIONS
    assert summary.valid_success_count == REPETITIONS - 1
    assert summary.timeout_count == 1
    assert summary.budget_violation_count == 1
    assert summary.success_rate == pytest.approx((REPETITIONS - 1) / REPETITIONS)
    assert "query_budget_violation" in report.blockers
    assert not report.advantage_claimable


def test_information_match_is_exact_and_fails_closed() -> None:
    records = _records()
    _find_record(records, method="baseline")["information_regime"] = (
        "oracle_k_query_cap_and_public_gap"
    )

    report = evaluate_ccfa_evidence_gate(records, _config())

    assert "information_regime_not_exactly_matched" in report.blockers
    assert not report.advantage_claimable


def test_theory_and_implementation_statuses_pass_only_at_exact_values() -> None:
    config = _config(
        theory_statuses={
            "new_upper_bound": "CONJECTURED",
            "same_interface_composition_frontier": "PROVED",
        },
        implementation_resource_statuses={"coherent_index_execution": "SIMULATED"},
    )

    report = evaluate_ccfa_evidence_gate(_records(), config)

    assert "theory_new_upper_bound_not_proved" in report.blockers
    assert "theory_matching_lower_bound_not_proved" in report.blockers
    assert "coherent_index_execution_not_verified" in report.blockers
    assert "resource_accounting_not_verified" in report.blockers
    assert "strongest_baseline_fidelity_not_verified" in report.blockers
    assert not report.advantage_claimable


@pytest.mark.parametrize(
    ("status", "digest"),
    [
        ("DRAFT", "a" * 64),
        ("LOCKED_BEFORE_RUN", None),
        ("LOCKED_BEFORE_RUN", "not-a-sha256"),
    ],
)
def test_preregistration_requires_lock_status_and_sha256(
    status: str, digest: str | None
) -> None:
    report = evaluate_ccfa_evidence_gate(
        _records(),
        _config(
            preregistration_status=status,
            preregistration_manifest_sha256=digest,
        ),
    )

    assert "preregistration_not_locked_or_manifest_missing" in report.blockers


def test_duplicate_missing_and_extra_panel_cells_are_strictly_rejected() -> None:
    records = _records()
    with pytest.raises(ValueError, match="duplicate run"):
        evaluate_ccfa_evidence_gate([*records, dict(records[0])], _config())

    with pytest.raises(ValueError, match="incomplete preregistered panel"):
        evaluate_ccfa_evidence_gate(records[:-1], _config())

    extra_cap = dict(records[0], query_cap=999)
    with pytest.raises(ValueError, match="outside the preregistered query-cap"):
        evaluate_ccfa_evidence_gate([*records, extra_cap], _config())


def test_small_fixture_or_seed_panel_is_rejected_by_configuration() -> None:
    with pytest.raises(ValueError, match="minimum_fixtures_per_family"):
        _config(
            preregistered_fixtures=(
                PreregisteredFixture("f0", "only-one-fixture"),
            )
        )

    with pytest.raises(ValueError, match="repetitions_per_fixture"):
        _config(repetitions_per_fixture=1)


def test_cluster_bootstrap_is_deterministic_and_resamples_whole_fixtures() -> None:
    first = evaluate_ccfa_evidence_gate(_records(), _config())
    second = evaluate_ccfa_evidence_gate(_records(), _config())

    assert first.pairwise_comparisons == second.pairwise_comparisons
    for comparison in first.pairwise_comparisons:
        assert comparison.bootstrap.resampling_unit == (
            "whole_fixture_with_all_seed_repetitions"
        )
        assert comparison.fixture_count == 2
        assert comparison.attempt_pairs == 2 * REPETITIONS


def test_equal_outcomes_do_not_pass_superiority_or_cross_cap_frontier_gate() -> None:
    report = evaluate_ccfa_evidence_gate(
        _records(candidate_success=True, baseline_success=True),
        _config(),
    )

    assert all(item.risk_difference == 0.0 for item in report.pairwise_comparisons)
    assert all(
        item.exact_two_sided_mcnemar_p_value == 1.0
        for item in report.pairwise_comparisons
    )
    assert "paired_statistical_superiority_not_established" in report.blockers
    assert "candidate_is_baseline_dominated_across_query_caps" in report.blockers
    assert not report.advantage_claimable


def test_minimum_risk_difference_is_preregistered_not_fit_to_results() -> None:
    report = evaluate_ccfa_evidence_gate(
        _records(),
        replace(_config(), minimum_risk_difference=1.0),
    )
    assert report.advantage_claimable

    with pytest.raises(ValueError, match="minimum_risk_difference"):
        replace(_config(), minimum_risk_difference=1.01)


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("certified_exact", 1, TypeError),
        ("coherent_queries", -1, ValueError),
        ("repetition", True, TypeError),
    ],
)
def test_run_fields_are_strictly_validated(
    field: str, value: object, exception: type[Exception]
) -> None:
    records = _records()
    records[0][field] = value

    with pytest.raises(exception):
        evaluate_ccfa_evidence_gate(records, _config())


def test_missing_mapping_field_is_rejected() -> None:
    records = _records()
    del records[0]["certified_exact"]

    with pytest.raises(ValueError, match="missing required field"):
        evaluate_ccfa_evidence_gate(records, _config())
