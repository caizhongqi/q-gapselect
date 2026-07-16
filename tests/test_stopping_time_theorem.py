from __future__ import annotations

import pytest

from qgapselect.stopping_time_theorem import (
    CLAIM_STATUS,
    READINESS,
    build_stopping_unitary_lemma_scaffold,
    stopping_unitary_lemma_markdown,
)
from qgapselect.stopping_time_transducer import VariableTimeStoppingTransducer


def test_stopping_unitary_lemma_scaffold_separates_checks_from_obligations() -> None:
    transducer = VariableTimeStoppingTransducer(
        [0.10, 0.49, 0.62, 0.86],
        [3, 4],
        boundary_phase=0.5,
    )

    scaffold = build_stopping_unitary_lemma_scaffold(transducer)

    assert scaffold.claim_status == CLAIM_STATUS
    assert scaffold.readiness == READINESS
    assert scaffold.failed_count == 0
    assert scaffold.passed_count == 5
    assert scaffold.proof_obligation_count == 4
    assert scaffold.compute_involution_residual_l2 <= 1e-10
    assert scaffold.phase_equivalence_residual_l2 <= 1e-10
    assert scaffold.work_garbage_probability_after_phase <= 1e-10
    assert scaffold.branch_rms_over_serial <= 1.0
    assert {check.check_id for check in scaffold.checks} == {
        "ST-U01",
        "ST-U02",
        "ST-U03",
        "ST-R01",
        "ST-R02",
        "ST-P01",
        "ST-P02",
        "ST-P03",
        "ST-P04",
    }


def test_stopping_unitary_lemma_scaffold_supports_active_phase() -> None:
    transducer = VariableTimeStoppingTransducer(
        [0.10, 0.49, 0.62, 0.86],
        [3],
        boundary_phase=0.5,
    )

    scaffold = build_stopping_unitary_lemma_scaffold(
        transducer,
        phase_on="active",
    )

    assert scaffold.phase_on == "active"
    assert scaffold.failed_count == 0
    assert scaffold.active_count == 2
    assert scaffold.output_count == 1


def test_stopping_unitary_lemma_markdown_lists_proof_obligations() -> None:
    transducer = VariableTimeStoppingTransducer(
        [0.10, 0.49, 0.62, 0.86],
        [3],
        boundary_phase=0.5,
    )
    scaffold = build_stopping_unitary_lemma_scaffold(transducer)

    markdown = stopping_unitary_lemma_markdown(scaffold)

    assert "Variable-time stopping-unitary lemma scaffold" in markdown
    assert "ST-P01" in markdown
    assert "proof_obligation" in markdown


@pytest.mark.parametrize("phase_on", ["missing", "", "selected"])
def test_stopping_unitary_lemma_rejects_unknown_phase(phase_on: str) -> None:
    transducer = VariableTimeStoppingTransducer(
        [0.10, 0.49, 0.62, 0.86],
        [3],
        boundary_phase=0.5,
    )

    with pytest.raises(ValueError):
        build_stopping_unitary_lemma_scaffold(transducer, phase_on=phase_on)
