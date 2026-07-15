from __future__ import annotations

import inspect

import pytest

import qgapselect.quantum_validation as validation_module
from qgapselect.quantum_validation import (
    run_unitary_validation,
    run_verifier_calibration,
)


def test_random_state_unitary_validation_executes_exact_query_formula() -> None:
    result = run_unitary_validation(
        (0.13, 0.52, 0.91),
        threshold=0.47,
        phase_qubits=3,
        trials=3,
        seed=71,
    )

    assert result.passed == 3
    assert len(result.trials) == 3
    assert result.max_compute_inverse_residual < 1e-10
    assert result.max_reflection_involution_residual < 1e-10
    assert result.max_norm_residual < 1e-10
    # M=8: compute/inverse plus two complete reflections costs 12M-6.
    assert {trial.expected_oracle_queries for trial in result.trials} == {90}
    assert all(trial.actual_oracle_queries == 90 for trial in result.trials)
    assert result.total_oracle_queries == 270
    assert all(trial.query_formula_exact and trial.passed for trial in result.trials)
    assert all(trial.dense_qft_matrix_dimension == 64 for trial in result.trials)


def test_verifier_calibration_separates_evaluation_and_procedure_queries() -> None:
    result = run_verifier_calibration(
        1.0,
        threshold=0.5,
        phase_qubits=2,
        shots=32,
        confidence=0.05,
        trials=5,
        seed=19,
    )

    assert result.exact_qpe_acceptance_probability == pytest.approx(1.0)
    assert result.exact_decision_side == "accepted"
    assert result.evaluation_only_oracle_queries == 7
    assert result.procedure_oracle_queries == 5 * 32 * 7
    assert result.status_counts == {"accepted": 5}
    assert result.interval_coverage_count == 5
    assert result.interval_coverage_rate == 1.0
    assert result.wrong_resolved_count == 0
    assert result.wrong_resolved_rate == 0.0
    assert all(not trial.wrong_resolved_decision for trial in result.trials)


def test_boundary_probability_is_reported_as_unresolved_not_an_error() -> None:
    # With the corrected mirror-symmetric comparator, the exact threshold grid
    # point belongs wholly to the above predicate and is not a 0.5 artifact.
    above = run_verifier_calibration(
        0.5,
        threshold=0.5,
        phase_qubits=3,
        shots=16,
        trials=2,
        seed=3,
    )
    assert above.exact_qpe_acceptance_probability == pytest.approx(1.0)
    assert above.exact_decision_side == "accepted"

    # An off-grid point can induce a QPE predicate probability close enough to
    # 1/2 that fixed shots remain unresolved; unresolved is not miscounted as a
    # wrong resolved decision.
    ambiguous = run_verifier_calibration(
        0.49,
        threshold=0.5,
        phase_qubits=2,
        shots=4,
        trials=4,
        seed=5,
    )
    assert ambiguous.status_counts.get("unresolved", 0) > 0
    assert ambiguous.wrong_resolved_count == 0


@pytest.mark.parametrize(
    "call",
    [
        lambda: run_unitary_validation((), trials=1),
        lambda: run_unitary_validation((0.5,), phase_qubits=True),
        lambda: run_unitary_validation((0.5,), trials=0),
        lambda: run_verifier_calibration(0.5, shots=True),
        lambda: run_verifier_calibration(0.5, trials=0),
        lambda: run_verifier_calibration(0.5, relation="equal"),
    ],
)
def test_validation_rejects_invalid_inputs(call: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()  # type: ignore[operator]


def test_validation_source_uses_only_public_charged_flag_operations() -> None:
    source = inspect.getsource(validation_module)

    assert "__blocks" not in source
    assert "_CanonicalRyStatevectorOracle" not in source
    assert ".acceptance_probability(" in source
    assert ".verify_index(" in source
    assert "evaluation_only" in source
