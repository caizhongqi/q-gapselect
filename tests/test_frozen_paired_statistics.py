from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import pytest

from qgapselect.frozen_paired_statistics import (
    CLAIM_SCOPE,
    GAP_AIDED_BASELINE_METHOD_ID,
    QGAPSELECT_METHOD_ID,
    QUANTUM_REFERENCE_ADAPTER_CLAIM_SCOPE,
    PairedFixtureOutcome,
    analyze_frozen_quantum_reference_pairs,
    analyze_paired_fixtures,
    exact_two_sided_mcnemar_p_value,
    holm_fwer_adjusted_p_values,
    paired_binary_statistics,
    paired_bootstrap_risk_difference,
)


@dataclass(frozen=True)
class FakeQuantumReferenceRun:
    family_id: str
    instance_id: str
    repetition: int
    method_id: str
    certified_exact_recovery: bool
    coherent_queries: int


def _run(
    family: str,
    instance: str,
    method: str,
    success: bool,
    queries: int,
    *,
    repetition: int = 0,
) -> FakeQuantumReferenceRun:
    return FakeQuantumReferenceRun(
        family_id=family,
        instance_id=instance,
        repetition=repetition,
        method_id=method,
        certified_exact_recovery=success,
        coherent_queries=queries,
    )


def _outcomes(*values: tuple[bool, bool]) -> tuple[PairedFixtureOutcome, ...]:
    return tuple(PairedFixtureOutcome(method, baseline) for method, baseline in values)


def test_zero_discordant_pairs_have_unit_mcnemar_p_value() -> None:
    result = paired_binary_statistics(
        _outcomes((True, True), (False, False), (True, True), (False, False))
    )

    assert result.pair_count == 4
    assert result.both_success == 2
    assert result.method_only_success == 0
    assert result.baseline_only_success == 0
    assert result.neither_success == 2
    assert result.risk_difference == 0.0
    assert result.exact_two_sided_mcnemar_p_value == 1.0
    assert result.claim_scope == CLAIM_SCOPE
    assert result.as_dict()["claim_scope"] == CLAIM_SCOPE


def test_extreme_discordance_uses_exact_two_sided_binomial_tail() -> None:
    outcomes = tuple(PairedFixtureOutcome(True, False) for _ in range(10))
    result = paired_binary_statistics(outcomes)

    assert result.method_only_success == 10
    assert result.baseline_only_success == 0
    assert result.risk_difference == 1.0
    assert result.exact_two_sided_mcnemar_p_value == pytest.approx(2.0 / 2**10)
    assert exact_two_sided_mcnemar_p_value(0, 10) == pytest.approx(2.0 / 2**10)
    assert exact_two_sided_mcnemar_p_value(5, 5) == 1.0


def test_fixture_bootstrap_is_deterministic_for_explicit_seed_and_repetitions() -> None:
    outcomes = _outcomes(
        (True, False),
        (True, False),
        (False, True),
        (True, True),
        (False, False),
    )

    first = paired_bootstrap_risk_difference(
        outcomes,
        repetitions=257,
        confidence_level=0.9,
        seed=991,
    )
    second = paired_bootstrap_risk_difference(
        outcomes,
        repetitions=257,
        confidence_level=0.9,
        seed=991,
    )

    assert first == second
    assert first.as_dict() == second.as_dict()
    assert first.point_risk_difference == pytest.approx(0.2)
    assert -1.0 <= first.lower <= first.upper <= 1.0
    assert first.repetitions == 257
    assert first.seed == 991
    assert first.resampling_unit == "fixture_pair"


def test_degenerate_bootstrap_preserves_extreme_risk_difference() -> None:
    result = paired_bootstrap_risk_difference(
        tuple(PairedFixtureOutcome(False, True) for _ in range(4)),
        repetitions=19,
        seed=-7,
    )

    assert result.point_risk_difference == -1.0
    assert result.lower == -1.0
    assert result.upper == -1.0


def test_holm_adjustments_are_monotone_in_sorted_raw_p_values() -> None:
    result = holm_fwer_adjusted_p_values(
        {"first": 0.04, "second": 0.01, "third": 0.20, "fourth": 0.03},
        alpha=0.05,
    )

    assert [record.hypothesis_id for record in result.adjustments] == [
        "first",
        "second",
        "third",
        "fourth",
    ]
    by_raw = sorted(result.adjustments, key=lambda record: record.raw_p_value)
    assert [record.adjusted_p_value for record in by_raw] == pytest.approx(
        [0.04, 0.09, 0.09, 0.20]
    )
    assert all(
        first.adjusted_p_value <= second.adjusted_p_value
        for first, second in pairwise(by_raw)
    )
    assert [record.rejected_at_alpha for record in result.adjustments] == [
        False,
        True,
        False,
        False,
    ]
    assert result.as_dict()["method"] == "holm_fwer"


def test_holm_accepts_sequence_and_caps_adjusted_values_at_one() -> None:
    result = holm_fwer_adjusted_p_values([0.6, 0.9, 1.0])

    assert [item.hypothesis_id for item in result.adjustments] == ["h0", "h1", "h2"]
    assert all(item.adjusted_p_value == 1.0 for item in result.adjustments)


def test_resource_summary_is_unconditional_over_every_pair() -> None:
    outcomes = (
        PairedFixtureOutcome(True, True, 10.0, 8.0),
        # The method failure remains in the resource comparison.
        PairedFixtureOutcome(False, True, 100.0, 9.0),
        PairedFixtureOutcome(True, False, 1.0, 3.0),
    )
    result = analyze_paired_fixtures(
        outcomes,
        bootstrap_repetitions=101,
        bootstrap_seed=4,
    )

    assert result.resources is not None
    assert result.resources.pair_count == 3
    assert result.resources.method_mean_resource == 37.0
    assert result.resources.baseline_mean_resource == pytest.approx(20.0 / 3.0)
    assert result.resources.mean_paired_difference == pytest.approx(91.0 / 3.0)
    assert result.resources.min_paired_difference == -2.0
    assert result.resources.max_paired_difference == 91.0
    assert result.resources.conditioning == "all_fixture_pairs_unconditional"
    assert result.as_dict()["resources"] == result.resources.as_dict()


def test_resources_must_be_paired_and_complete_across_analysis() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        PairedFixtureOutcome(True, False, method_resource=1.0)

    complete = PairedFixtureOutcome(True, False, 1.0, 2.0)
    absent = PairedFixtureOutcome(False, True)
    with pytest.raises(ValueError, match="all fixture pairs or for none"):
        analyze_paired_fixtures(
            (complete, absent),
            bootstrap_repetitions=5,
        )


@pytest.mark.parametrize(
    ("arguments", "exception"),
    [
        ((1, False), TypeError),
        ((True, 0), TypeError),
        ((True, False, math.inf, 1.0), ValueError),
        ((True, False, -1.0, 1.0), ValueError),
    ],
)
def test_pair_validation_is_strict(
    arguments: tuple[object, ...], exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        PairedFixtureOutcome(*arguments)


def test_public_functions_reject_empty_or_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        paired_binary_statistics(())
    with pytest.raises(TypeError, match="PairedFixtureOutcome"):
        paired_binary_statistics(((True, False),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer"):
        exact_two_sided_mcnemar_p_value(True, 0)
    with pytest.raises(ValueError, match="non-empty"):
        holm_fwer_adjusted_p_values([])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_fwer_adjusted_p_values([1.1])


def test_quantum_run_adapter_pairs_panels_and_uses_all_coherent_queries() -> None:
    runs = [
        _run("family-a", "i0", QGAPSELECT_METHOD_ID, True, 100),
        _run("family-a", "i0", GAP_AIDED_BASELINE_METHOD_ID, False, 50),
        _run("family-a", "i0", "known_threshold_iae_scan", True, 25),
        _run("family-a", "i1", QGAPSELECT_METHOD_ID, False, 200),
        _run("family-a", "i1", GAP_AIDED_BASELINE_METHOD_ID, False, 80),
        _run("family-a", "i2", QGAPSELECT_METHOD_ID, True, 300),
        _run("family-a", "i2", GAP_AIDED_BASELINE_METHOD_ID, True, 90),
    ]

    result = analyze_frozen_quantum_reference_pairs(
        runs,
        master_seed=41,
        bootstrap_repetitions=101,
        confidence_level=0.9,
    )
    family = result.family_analyses[0]

    assert family.family_id == "family-a"
    assert family.analysis.binary.both_success == 1
    assert family.analysis.binary.method_only_success == 1
    assert family.analysis.binary.baseline_only_success == 0
    assert family.analysis.binary.neither_success == 1
    assert family.analysis.binary.risk_difference == pytest.approx(1.0 / 3.0)
    assert family.analysis.resources is not None
    # The failed i1 panel is retained: this is not a both-success subset.
    assert family.analysis.resources.pair_count == 3
    assert family.analysis.resources.method_mean_resource == 200.0
    assert family.analysis.resources.baseline_mean_resource == pytest.approx(220.0 / 3.0)
    assert family.analysis.resources.mean_paired_difference == pytest.approx(380.0 / 3.0)
    assert family.resource_conditioning == "all_fixture_pairs_unconditional"
    assert family.comparison_information_matched is False
    assert family.baseline_is_gap_aided is True
    assert family.baseline_information == "k_and_public_gap_floor"
    assert result.comparison_information_matched is False
    assert result.claim_scope == QUANTUM_REFERENCE_ADAPTER_CLAIM_SCOPE
    document = result.as_dict()
    assert document["comparison_information_matched"] is False
    assert document["baseline_is_gap_aided"] is True
    assert document["family_analyses"][0]["analysis"] == family.analysis.as_dict()


def test_quantum_run_adapter_holm_corrects_across_families() -> None:
    runs: list[FakeQuantumReferenceRun] = []
    for index in range(4):
        runs.extend(
            (
                _run("family-b", f"b{index}", QGAPSELECT_METHOD_ID, False, 10),
                _run(
                    "family-b",
                    f"b{index}",
                    GAP_AIDED_BASELINE_METHOD_ID,
                    True,
                    20,
                ),
            )
        )
    runs.extend(
        (
            _run("family-a", "a0", QGAPSELECT_METHOD_ID, True, 30),
            _run("family-a", "a0", GAP_AIDED_BASELINE_METHOD_ID, True, 40),
        )
    )

    result = analyze_frozen_quantum_reference_pairs(
        runs,
        bootstrap_repetitions=17,
        holm_alpha=0.1,
    )

    assert [item.family_id for item in result.family_analyses] == ["family-a", "family-b"]
    assert [item.hypothesis_id for item in result.holm_fwer.adjustments] == [
        "family-a",
        "family-b",
    ]
    raw = {item.hypothesis_id: item.raw_p_value for item in result.holm_fwer.adjustments}
    adjusted = {
        item.hypothesis_id: item.adjusted_p_value for item in result.holm_fwer.adjustments
    }
    assert raw == pytest.approx({"family-a": 1.0, "family-b": 0.125})
    assert adjusted == pytest.approx({"family-a": 1.0, "family-b": 0.25})


def test_quantum_run_adapter_is_input_order_independent_and_seed_stable() -> None:
    runs = [
        _run("z-family", "later", QGAPSELECT_METHOD_ID, True, 9, repetition=1),
        _run(
            "z-family",
            "later",
            GAP_AIDED_BASELINE_METHOD_ID,
            False,
            3,
            repetition=1,
        ),
        _run("z-family", "earlier", QGAPSELECT_METHOD_ID, False, 7),
        _run("z-family", "earlier", GAP_AIDED_BASELINE_METHOD_ID, True, 5),
    ]
    arguments = {
        "master_seed": -123,
        "bootstrap_repetitions": 211,
        "confidence_level": 0.8,
    }

    forward = analyze_frozen_quantum_reference_pairs(runs, **arguments)
    reverse = analyze_frozen_quantum_reference_pairs(list(reversed(runs)), **arguments)
    changed_master = analyze_frozen_quantum_reference_pairs(
        runs,
        master_seed=-122,
        bootstrap_repetitions=211,
        confidence_level=0.8,
    )

    assert forward == reverse
    assert forward.as_dict() == reverse.as_dict()
    assert forward.family_analyses[0].bootstrap_seed == reverse.family_analyses[0].bootstrap_seed
    assert (
        changed_master.family_analyses[0].bootstrap_seed
        != forward.family_analyses[0].bootstrap_seed
    )


def test_quantum_run_adapter_rejects_missing_or_duplicate_panels() -> None:
    missing = [_run("family", "i0", QGAPSELECT_METHOD_ID, True, 1)]
    with pytest.raises(ValueError, match="incomplete comparison panel"):
        analyze_frozen_quantum_reference_pairs(missing, bootstrap_repetitions=3)

    duplicate = [
        _run("family", "i0", QGAPSELECT_METHOD_ID, True, 1),
        _run("family", "i0", QGAPSELECT_METHOD_ID, False, 2),
        _run("family", "i0", GAP_AIDED_BASELINE_METHOD_ID, True, 3),
    ]
    with pytest.raises(ValueError, match="duplicate method record"):
        analyze_frozen_quantum_reference_pairs(duplicate, bootstrap_repetitions=3)

    control_only = [_run("family", "i0", "known_threshold_iae_scan", True, 1)]
    with pytest.raises(ValueError, match="incomplete comparison panel"):
        analyze_frozen_quantum_reference_pairs(control_only, bootstrap_repetitions=3)


def test_quantum_run_adapter_strictly_validates_protocol_fields() -> None:
    malformed = [
        _run("family", "i0", QGAPSELECT_METHOD_ID, True, 1),
        FakeQuantumReferenceRun(
            family_id="family",
            instance_id="i0",
            repetition=0,
            method_id=GAP_AIDED_BASELINE_METHOD_ID,
            certified_exact_recovery=1,  # type: ignore[arg-type]
            coherent_queries=1,
        ),
    ]
    with pytest.raises(TypeError, match="certified_exact_recovery must be bool"):
        analyze_frozen_quantum_reference_pairs(malformed, bootstrap_repetitions=3)

    with pytest.raises(TypeError, match="field 'family_id'"):
        analyze_frozen_quantum_reference_pairs(
            [object()],  # type: ignore[list-item]
            bootstrap_repetitions=3,
        )
