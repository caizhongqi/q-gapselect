from __future__ import annotations

import pytest

from qgapselect.lower_bound_program import (
    CLAIM_STATUS,
    READINESS,
    LowerBoundProgramRecord,
    lower_bound_program_loglog_slope,
    lower_bound_program_record,
    lower_bound_program_sweep,
)


def test_lower_bound_program_records_open_proof_obligations() -> None:
    record = lower_bound_program_record(8)

    assert isinstance(record, LowerBoundProgramRecord)
    assert record.claim_status == CLAIM_STATUS
    assert record.lower_bound_claim_status == READINESS
    assert [block.block_id for block in record.blocks] == [
        "LB-B01",
        "LB-B02",
        "LB-B03",
        "LB-B04",
    ]
    assert record.local_facts_count == 1
    assert record.proof_obligation_count == 3
    assert record.target_over_candidate > 0.0
    assert record.strongest_block_proxy > 0.0


def test_lower_bound_program_sweep_and_slopes_are_descriptive() -> None:
    records = lower_bound_program_sweep([8, 16, 32, 64])

    assert [record.m for record in records] == [8, 16, 32, 64]
    assert {record.proof_obligation_count for record in records} == {3}
    assert lower_bound_program_loglog_slope(records, "candidate_proxy") > 0.0
    assert lower_bound_program_loglog_slope(records, "adversary_target_proxy") > 0.0
    assert lower_bound_program_loglog_slope(records, "strongest_block_proxy") > 0.0


@pytest.mark.parametrize("m_values", [[], [8, 8], [16, 8]])
def test_lower_bound_program_rejects_invalid_sweeps(m_values: list[int]) -> None:
    with pytest.raises(ValueError):
        lower_bound_program_sweep(m_values)


def test_lower_bound_program_rejects_invalid_slope_field() -> None:
    records = lower_bound_program_sweep([8, 16])

    with pytest.raises(ValueError):
        lower_bound_program_loglog_slope(records, "unknown")
