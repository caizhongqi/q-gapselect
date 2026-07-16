from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from qgapselect.fixed_fixture_calibration import (
    CLAIM_SCOPE,
    calibration_manifest_hash,
    one_sided_clopper_pearson_lower,
    select_hardness_quantile_anchors,
    summarize_fixed_fixture_calibration,
)


def _document(family: str, index: int, active: int, gap: float) -> dict[str, object]:
    return {
        "family_id": family,
        "instance_id": f"{family}/problem-{index:04d}",
        "difficulty_fingerprint": f"{index + (1000 if family == 'b' else 0):064x}",
        "structure_metrics": {
            "active_count": active,
            "empirical_boundary_gap": gap,
        },
        # A selector must not require or inspect outcome fields.
        "adversarial_unused_outcome": index % 2 == 0,
    }


def test_anchor_selection_is_deterministic_and_spans_mid_quantiles() -> None:
    documents = [
        _document("a", index, active=index + 1, gap=0.1)
        for index in range(10)
    ]
    first = select_hardness_quantile_anchors(documents, anchors_per_family=5)
    second = select_hardness_quantile_anchors(documents, anchors_per_family=5)

    assert first == second
    assert [record.source_rank for record in first] == [1, 3, 5, 7, 9]
    assert [record.target_quantile for record in first] == pytest.approx(
        [0.1, 0.3, 0.5, 0.7, 0.9]
    )
    assert len({record.difficulty_fingerprint for record in first}) == 5
    assert all(record.claim_scope == CLAIM_SCOPE for record in first)


def test_anchor_selection_filters_families_without_using_results() -> None:
    documents = [
        *(_document("a", index, active=2, gap=0.1 + index / 100) for index in range(6)),
        *(_document("b", index, active=3, gap=0.2 + index / 100) for index in range(6)),
    ]
    selected = select_hardness_quantile_anchors(
        documents,
        anchors_per_family=2,
        included_families=("b",),
    )

    assert {record.family_id for record in selected} == {"b"}
    assert len(selected) == 2


def test_exact_one_sided_lower_bound_has_known_all_success_form() -> None:
    lower = one_sided_clopper_pearson_lower(200, 200, alpha=0.05 / 180)

    assert lower == pytest.approx((0.05 / 180) ** (1 / 200), rel=1e-12)
    assert lower > 0.95
    assert one_sided_clopper_pearson_lower(0, 200, alpha=0.05) == 0.0
    assert one_sided_clopper_pearson_lower(100, 200, alpha=0.05) < 0.5


@dataclass(frozen=True)
class _Run:
    family_id: str
    instance_id: str
    method_id: str
    repetition: int
    certified_exact_recovery: bool
    coherent_queries: int


def _runs() -> tuple[_Run, ...]:
    records: list[_Run] = []
    for instance in ("i0", "i1"):
        for repetition in range(200):
            for method in ("method", "baseline"):
                success = method == "method" or repetition < (190 if instance == "i0" else 180)
                records.append(
                    _Run(
                        family_id="family",
                        instance_id=instance,
                        method_id=method,
                        repetition=repetition,
                        certified_exact_recovery=success,
                        coherent_queries=10 + repetition % 3,
                    )
                )
    return tuple(records)


def test_calibration_aggregates_seeds_with_fixture_as_cluster() -> None:
    report = summarize_fixed_fixture_calibration(
        _runs(),
        target_success_probability=0.95,
        familywise_alpha=0.05,
    )

    assert len(report.records) == 4
    assert report.simultaneous_alpha == pytest.approx(0.0125)
    method_records = [record for record in report.records if record.method_id == "method"]
    baseline_records = [record for record in report.records if record.method_id == "baseline"]
    assert all(record.successes == 200 for record in method_records)
    assert all(record.target_certified for record in method_records)
    assert [record.successes for record in baseline_records] == [190, 180]
    assert all(not record.target_certified for record in baseline_records)
    method_aggregate = next(
        item for item in report.aggregates if item.method_id == "method"
    )
    baseline_aggregate = next(
        item for item in report.aggregates if item.method_id == "baseline"
    )
    assert method_aggregate.mean_anchor_success_rate == 1.0
    assert method_aggregate.between_fixture_variance == 0.0
    assert baseline_aggregate.mean_anchor_success_rate == pytest.approx(0.925)
    assert baseline_aggregate.between_fixture_variance > 0.0
    assert report.claim_scope == CLAIM_SCOPE


def test_calibration_rejects_incomplete_or_duplicate_panels() -> None:
    records = list(_runs())
    records.pop()
    with pytest.raises(ValueError, match="same repetition count|same methods"):
        summarize_fixed_fixture_calibration(tuple(records))

    duplicate = (*_runs(), _runs()[0])
    with pytest.raises(ValueError, match="duplicate"):
        summarize_fixed_fixture_calibration(duplicate)


def test_calibration_manifest_is_stable_and_parameter_sensitive() -> None:
    anchors = select_hardness_quantile_anchors(
        [_document("a", index, active=index + 1, gap=0.1) for index in range(5)],
        anchors_per_family=2,
    )
    first = calibration_manifest_hash(anchors, repetitions=200, master_seed=9)
    second = calibration_manifest_hash(anchors, repetitions=200, master_seed=9)
    changed = calibration_manifest_hash(anchors, repetitions=201, master_seed=9)

    assert first == second
    assert first != changed
    assert len(first) == 64


@pytest.mark.parametrize(
    ("successes", "total", "alpha"),
    [(201, 200, 0.05), (1, 0, 0.05), (1, 2, 0.0), (1, 2, 1.0)],
)
def test_invalid_exact_interval_inputs_are_rejected(
    successes: int,
    total: int,
    alpha: float,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        one_sided_clopper_pearson_lower(successes, total, alpha=alpha)


def test_no_hidden_nan_in_variance_components() -> None:
    report = summarize_fixed_fixture_calibration(_runs())
    assert all(
        math.isfinite(item.between_fixture_variance)
        and math.isfinite(item.mean_within_fixture_seed_variance)
        for item in report.aggregates
    )
