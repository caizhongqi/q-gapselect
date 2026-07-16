from __future__ import annotations

import pytest

from qgapselect.unknown_boundary_history import unknown_boundary_history_record
from qgapselect.variable_time_charged_history import (
    CLAIM_STATUS,
    NOVELTY_FAILED,
    NOVELTY_OPEN,
    precision_bits_for_epsilon,
    variable_time_charged_history_loglog_slope,
    variable_time_charged_history_record,
    variable_time_charged_history_sweep,
)


def test_precision_bits_charge_qpe_grid_step() -> None:
    assert precision_bits_for_epsilon(0.25, precision_multiplier=2.0) == 3
    assert precision_bits_for_epsilon(0.25, precision_multiplier=2.0, min_precision_bits=5) == 5
    assert precision_bits_for_epsilon(0.001, max_precision_bits=8) == 8


def test_variable_time_charged_record_aligns_with_unknown_boundary_history() -> None:
    base = unknown_boundary_history_record(8)
    record = variable_time_charged_history_record(base, precision_multiplier=2.0)

    assert record.claim_status == CLAIM_STATUS
    assert record.m == base.m
    assert record.n == base.n
    assert record.k == base.k
    assert len(record.layers) == base.level_count
    assert all(layer.qpe_query_units > 0 for layer in record.layers)
    assert record.charged_candidate_total_proxy > base.candidate_total_proxy
    assert record.charged_serial_over_candidate > 1.0
    assert record.lower_target_over_candidate < 1.0
    assert record.novelty_gate in {NOVELTY_OPEN, NOVELTY_FAILED}


def test_variable_time_charged_sweep_exposes_open_default_family() -> None:
    records = variable_time_charged_history_sweep(
        [8, 16, 32],
        n_exponent=3.0,
        level_exponent=1.0,
        active_exponent=1.0,
        gamma_exponent=2.0,
        output_births_per_level=1,
        epsilon_growth_exponent=0.25,
        activity_decay_exponent=0.0,
        baseline_match_tolerance=1.2,
    )

    assert len(records) == 3
    assert all(record.novelty_gate == NOVELTY_OPEN for record in records)
    assert records[-1].min_valid_baseline_over_candidate > 1.2
    slope = variable_time_charged_history_loglog_slope(
        records, "charged_candidate_total_proxy"
    )
    assert slope > 0.0


def test_variable_time_charged_gate_can_fail_with_loose_tolerance() -> None:
    record = variable_time_charged_history_record(
        8,
        baseline_match_tolerance=10.0,
    )

    assert record.novelty_gate == NOVELTY_FAILED
    assert record.encoded_baseline_match_found is True


@pytest.mark.parametrize(
    ("args", "kwargs", "error"),
    [
        ((0.0,), {}, ValueError),
        ((0.1,), {"precision_multiplier": 0.0}, ValueError),
        ((0.1,), {"min_precision_bits": 0}, ValueError),
        ((0.1,), {"min_precision_bits": 4, "max_precision_bits": 3}, ValueError),
    ],
)
def test_precision_bit_inputs_are_strict(
    args: tuple[object, ...], kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        precision_bits_for_epsilon(*args, **kwargs)  # type: ignore[arg-type]
