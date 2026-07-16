from __future__ import annotations

import pytest

from qgapselect.unknown_boundary_history import (
    NOVELTY_FAILED,
    NOVELTY_OPEN,
    unknown_boundary_history_loglog_slope,
    unknown_boundary_history_record,
    unknown_boundary_history_sweep,
)


def test_default_unknown_boundary_history_family_stays_open() -> None:
    record = unknown_boundary_history_record(32)

    assert record.n == 32**3
    assert record.level_count == 32
    assert record.active_base_count == 32
    assert record.total_outputs == 32
    assert record.min_encoded_valid_baseline_name == "variable_time_rebuild_rms"
    assert record.min_valid_baseline_over_candidate > 1.2
    assert not record.encoded_baseline_match_found
    assert record.novelty_gate == NOVELTY_OPEN
    assert record.known_boundary_free_history_layered_proxy < (
        record.rebuild_history_scan_layered_proxy
    )
    assert "lower_bound_open" in record.matching_lower_bound_status


def test_loose_tolerance_turns_the_same_family_into_a_failed_gate() -> None:
    record = unknown_boundary_history_record(8, baseline_match_tolerance=2.0)

    assert record.encoded_baseline_match_found
    assert record.novelty_gate == NOVELTY_FAILED


def test_unknown_boundary_history_slopes_expose_candidate_and_baselines() -> None:
    records = unknown_boundary_history_sweep((8, 16, 32, 64, 128, 256))

    candidate = unknown_boundary_history_loglog_slope(
        records, "candidate_total_proxy"
    )
    rebuild = unknown_boundary_history_loglog_slope(
        records, "variable_time_rebuild_rms_proxy"
    )
    independent = unknown_boundary_history_loglog_slope(
        records, "independent_all_arm_proxy"
    )
    lower_target = unknown_boundary_history_loglog_slope(
        records, "adversary_lower_target_proxy"
    )
    assert 3.3 < candidate < 3.6
    assert rebuild > candidate
    assert independent == pytest.approx(5.0)
    assert lower_target == pytest.approx(3.5)
    assert all(record.novelty_gate == NOVELTY_OPEN for record in records)


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: unknown_boundary_history_record(True), TypeError),
        (lambda: unknown_boundary_history_record(1), ValueError),
        (lambda: unknown_boundary_history_record(8, n_exponent=0.0), ValueError),
        (
            lambda: unknown_boundary_history_record(
                8, output_births_per_level=10_000
            ),
            ValueError,
        ),
        (lambda: unknown_boundary_history_sweep(()), ValueError),
        (lambda: unknown_boundary_history_sweep((8, 4)), ValueError),
        (
            lambda: unknown_boundary_history_loglog_slope(
                unknown_boundary_history_sweep((4, 8)), "missing"
            ),
            ValueError,
        ),
    ],
)
def test_unknown_boundary_history_inputs_are_strict(
    call: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        call()  # type: ignore[operator]
