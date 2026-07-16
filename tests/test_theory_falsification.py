from __future__ import annotations

import math

import pytest

from qgapselect.theory_falsification import (
    descriptive_loglog_slope,
    orientation_separation_record,
    orientation_separation_sweep,
)


def test_orientation_family_has_declared_shape_and_rejected_orientation() -> None:
    record = orientation_separation_record(16)

    assert record.n == 48
    assert record.k == record.m == 16
    assert record.gamma == pytest.approx(16**-2)
    assert record.chosen_orientation == "rejected_complement"
    assert record.orientation_candidate_proxy == pytest.approx(
        record.rejected_candidate_proxy
    )
    assert record.rejected_candidate_proxy < record.selected_candidate_proxy
    assert record.generic_composition_audit_status == "failed_explicit_family"
    assert "no_algorithm_or_theorem" in record.claim_status


def test_declared_proxy_ratio_grows_but_remains_only_analytic_evidence() -> None:
    records = orientation_separation_sweep((8, 16, 32, 64, 128, 256))

    ratios = [record.independent_over_candidate for record in records]
    assert ratios == sorted(ratios)
    assert ratios[-1] > 6.0 * ratios[0]
    candidate_slope = descriptive_loglog_slope(
        records, "orientation_candidate_proxy"
    )
    independent_slope = descriptive_loglog_slope(
        records, "independent_all_arm_proxy"
    )
    assert 2.3 < candidate_slope < 2.7
    assert independent_slope == pytest.approx(3.0)
    assert candidate_slope < independent_slope


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: orientation_separation_record(True), TypeError),
        (lambda: orientation_separation_record(1), ValueError),
        (lambda: orientation_separation_record(2, gamma_exponent=0.0), ValueError),
        (lambda: orientation_separation_record(2, beta=math.inf), ValueError),
        (lambda: orientation_separation_sweep(()), ValueError),
        (lambda: orientation_separation_sweep((8, 4)), ValueError),
        (
            lambda: descriptive_loglog_slope(
                orientation_separation_sweep((4, 8)), "missing"
            ),
            ValueError,
        ),
    ],
)
def test_falsification_inputs_are_strict(call: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        call()  # type: ignore[operator]
