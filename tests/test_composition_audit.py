from __future__ import annotations

import math

import pytest

from qgapselect.composition_audit import (
    composition_audit_sweep,
    composition_loglog_slope,
    orientation_family_composition_record,
)


def test_known_outer_composition_matches_declared_layer_proxy() -> None:
    record = orientation_family_composition_record(32)

    assert record.chosen_orientation == "rejected_complement"
    assert 1.0 <= record.candidate_over_direct_layered <= math.sqrt(2.0)
    assert record.outer_composition_matches_candidate
    assert not record.explicit_family_separation_survives
    assert record.novelty_gate == "failed_explicit_family"
    assert "proof_open" in record.matching_lower_bound_status


def test_coarse_partition_plus_bai_has_same_descriptive_leading_slope() -> None:
    records = composition_audit_sweep((8, 16, 32, 64, 128, 256))

    candidate = composition_loglog_slope(records, "orientation_candidate_proxy")
    coarse_bai = composition_loglog_slope(
        records, "coarse_partition_plus_bai_proxy"
    )
    lower = composition_loglog_slope(records, "exceptional_search_lower_proxy")
    independent = composition_loglog_slope(records, "independent_all_arm_proxy")
    assert 2.3 < candidate < 2.7
    assert 2.3 < coarse_bai < 2.7
    assert lower == pytest.approx(2.5, abs=0.02)
    assert independent == pytest.approx(3.0)
    assert all(record.novelty_gate == "failed_explicit_family" for record in records)


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: orientation_family_composition_record(True), TypeError),
        (lambda: orientation_family_composition_record(1), ValueError),
        (lambda: composition_audit_sweep(()), ValueError),
        (lambda: composition_audit_sweep((8, 4)), ValueError),
        (
            lambda: composition_loglog_slope(
                composition_audit_sweep((4, 8)), "missing"
            ),
            ValueError,
        ),
    ],
)
def test_composition_audit_inputs_are_strict(
    call: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        call()  # type: ignore[operator]
