from __future__ import annotations

import pytest

from qgapselect.composition_frontier import (
    CLAIM_STATUS,
    NOVELTY_FAILED,
    NOVELTY_OPEN,
    CompositionFrontierRecord,
    composition_frontier_loglog_slope,
    composition_frontier_record,
    composition_frontier_sweep,
)


def test_composition_frontier_keeps_known_baselines_explicit() -> None:
    record = composition_frontier_record(8)

    assert isinstance(record, CompositionFrontierRecord)
    assert record.claim_status == CLAIM_STATUS
    assert record.novelty_gate == NOVELTY_OPEN
    assert record.encoded_match_found is False
    assert record.strongest_valid_baseline_name == "loop_variable_time_rebuild"
    assert record.strongest_valid_baseline_over_candidate > 1.2
    assert any(not baseline.valid_same_interface for baseline in record.baselines)
    assert any(
        baseline.name == "free_history_qram_layered"
        and "no-free-QRAM" in baseline.limitation
        for baseline in record.baselines
    )


def test_composition_frontier_sweep_and_slopes_are_descriptive() -> None:
    records = composition_frontier_sweep([8, 16, 32, 64])

    assert [record.m for record in records] == [8, 16, 32, 64]
    assert {record.novelty_gate for record in records} == {NOVELTY_OPEN}
    assert composition_frontier_loglog_slope(records, "candidate_proxy") > 0.0
    assert (
        composition_frontier_loglog_slope(
            records,
            "strongest_valid_baseline_proxy",
        )
        > 0.0
    )


def test_composition_frontier_negative_control_fails_with_loose_gate() -> None:
    record = composition_frontier_record(8, baseline_match_tolerance=10.0)

    assert record.novelty_gate == NOVELTY_FAILED
    assert record.encoded_match_found is True


@pytest.mark.parametrize("m_values", [[], [8, 8], [16, 8]])
def test_composition_frontier_rejects_invalid_sweeps(m_values: list[int]) -> None:
    with pytest.raises(ValueError):
        composition_frontier_sweep(m_values)


def test_composition_frontier_rejects_invalid_slope_field() -> None:
    records = composition_frontier_sweep([8, 16])

    with pytest.raises(ValueError):
        composition_frontier_loglog_slope(records, "serial_rebuild_scan_proxy")
