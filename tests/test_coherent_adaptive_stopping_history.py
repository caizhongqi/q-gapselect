from __future__ import annotations

import inspect

import pytest

from qgapselect.coherent import CanonicalRyStatevectorOracle
from qgapselect.coherent_adaptive_stopping_history import (
    METHOD_ID,
    OUTPUT_INCONCLUSIVE,
    OUTPUT_MASK,
    TinyCoherentAdaptiveStoppingHistory,
    TinyCoherentStoppingHistoryConfig,
    run_tiny_coherent_adaptive_stopping_history,
    run_tiny_inactive_level_subspace_audit,
)


def _run(means: tuple[float, float]):
    return run_tiny_coherent_adaptive_stopping_history(
        CanonicalRyStatevectorOracle(means),
        config=TinyCoherentStoppingHistoryConfig(
            cleanup_tolerance=1e-10,
            max_statevector_dimension=600_000,
        ),
    )


def _assert_query_ledger(result) -> None:
    ledger = result.resources.query_ledger
    expected = {
        "forward": 4,
        "inverse": 4,
        "controlled_forward": 84,
        "controlled_inverse": 84,
        "classical_sample": 0,
        "coherent_total": 176,
        "classical_total": 0,
        "total": 176,
        "qram_queries": 0,
    }
    assert dict(ledger.query_counts) == expected
    assert dict(ledger.expected_query_counts) == expected
    assert ledger.reconciled is True
    assert len(ledger.per_level_runtime_records) == 2
    level_zero, level_one = ledger.per_level_runtime_records
    assert level_zero.phase_qubits == 2
    assert level_zero.runtime_full_replay_counts["coherent_total"] == 56
    assert level_zero.runtime_derived_one_way_counts["coherent_total"] == 28
    assert level_zero.full_replay_reconciled is True
    assert level_zero.one_way_reconciled is True
    assert level_one.phase_qubits == 3
    assert level_one.runtime_full_replay_counts["forward"] == 0
    assert level_one.runtime_full_replay_counts["controlled_forward"] == 60
    assert level_one.runtime_full_replay_counts["controlled_inverse"] == 60
    assert level_one.runtime_full_replay_counts["coherent_total"] == 120
    assert level_one.runtime_derived_one_way_counts["coherent_total"] == 60
    assert level_one.full_replay_reconciled is True
    assert level_one.one_way_reconciled is True
    assert ledger.per_level_one_way_query_costs == (28, 60)
    assert ledger.worst_case_one_way_history_queries == 88
    assert ledger.worst_case_full_replay_queries == 176
    assert ledger.branch_rms_is_executed_saving is False
    assert ledger.qram_assumed is False


def test_fixed_interface_has_no_answer_gap_family_or_schedule_input() -> None:
    parameters = inspect.signature(
        run_tiny_coherent_adaptive_stopping_history
    ).parameters
    assert set(parameters) == {"oracle", "config"}
    config_fields = set(TinyCoherentStoppingHistoryConfig.__dataclass_fields__)
    assert config_fields == {"cleanup_tolerance", "max_statevector_dimension"}


def test_exact_grid_stops_at_first_level_and_full_replay_cleans() -> None:
    result = _run((1.0, 0.0))

    assert result.method_id == METHOD_ID
    assert result.output_status == OUTPUT_MASK
    assert result.membership_mask == 0b01
    assert result.membership_bits == (1, 0)
    assert result.history.stop_at_level_zero_probability == pytest.approx(1.0)
    assert result.history.stop_at_level_one_probability == pytest.approx(0.0, abs=1e-12)
    assert result.history.invalid_both_stopped_probability == pytest.approx(0.0)
    assert result.history.single_statevector_history_register is True
    assert result.history.later_level_oracles_controlled_by_active_flag is True
    assert result.durable_output.scratch_to_durable_copy_executed is True
    assert result.durable_output.full_history_replay_executed is True
    assert result.resources.cleanup.passed is True
    assert result.cleanup_error_bound < 1e-10
    _assert_query_ledger(result)
    assert result.resources.query_ledger.branch_rms_one_way_theorem_target == pytest.approx(28.0)
    assert result.resources.query_ledger.branch_rms_full_replay_theorem_target == pytest.approx(
        56.0
    )


def test_exact_grid_continues_coherently_and_stops_at_second_level() -> None:
    result = _run((0.5, 0.0))

    assert result.output_status == OUTPUT_MASK
    assert result.membership_mask == 0b01
    assert result.history.stop_at_level_zero_probability == pytest.approx(0.0, abs=1e-12)
    assert result.history.stop_at_level_one_probability == pytest.approx(1.0)
    assert result.history.unresolved_probability == pytest.approx(0.0, abs=1e-12)
    assert result.resources.cleanup.passed is True
    assert result.durable_output.dominant_probability == pytest.approx(1.0)
    _assert_query_ledger(result)
    assert result.resources.query_ledger.branch_rms_one_way_theorem_target == pytest.approx(88.0)
    assert result.resources.query_ledger.branch_rms_full_replay_theorem_target == pytest.approx(
        176.0
    )


def test_arm_one_winner_is_preserved_by_direct_mask_copy_and_replay() -> None:
    result = _run((0.0, 0.5))

    assert result.output_status == OUTPUT_MASK
    assert result.membership_mask == 0b10
    assert result.membership_bits == (0, 1)
    assert result.history.stop_at_level_one_probability == pytest.approx(1.0)
    assert result.durable_output.dominant_mask == 0b10
    assert result.resources.cleanup.passed is True
    _assert_query_ledger(result)


def test_generic_off_grid_entangles_durable_output_and_fails_closed() -> None:
    result = _run((0.82, 0.18))

    assert result.output_status == OUTPUT_INCONCLUSIVE
    assert result.membership_mask is None
    assert result.membership_bits == ()
    assert result.status == "durable_output_entanglement_cleanup_fail_closed"
    assert 0.1 < result.history.stop_at_level_zero_probability < 0.2
    assert 0.7 < result.history.stop_at_level_one_probability < 0.8
    assert 0.1 < result.history.unresolved_probability < 0.2
    cleanup = result.resources.cleanup
    assert cleanup.passed is False
    assert cleanup.executed_transient_nonzero_probability > 0.26
    assert cleanup.predicted_transient_nonzero_probability > 0.26
    assert cleanup.prediction_residual < 1e-10
    assert cleanup.purity_residual < 1e-10
    assert result.certificate.issued is False
    assert result.certificate.top_k_correctness_error_bound is None
    _assert_query_ledger(result)
    assert 28.0 < result.resources.query_ledger.branch_rms_one_way_theorem_target < 88.0


def test_true_coherent_claim_boundary_does_not_overclaim_speedup_or_theory() -> None:
    result = _run((0.5, 0.5))

    assert result.output_status == OUTPUT_INCONCLUSIVE
    assert result.status == "unresolved_history_fail_closed"
    assert result.history.unresolved_probability == pytest.approx(1.0)
    assert result.resources.cleanup.passed is True
    assert result.claim_boundary.true_coherent_stopping_history_unitary_implemented is True
    assert result.claim_boundary.durable_copy_and_full_replay_implemented is True
    assert result.claim_boundary.generic_off_grid_cleanup_proved is False
    assert result.claim_boundary.variable_time_query_speedup_proved is False
    assert result.claim_boundary.new_query_upper_bound_proved is False
    assert result.claim_boundary.matching_lower_bound_proved is False
    assert result.claim_boundary.quantum_advantage_claimable is False
    assert result.claim_boundary.ccf_a_claimable is False
    assert result.quantum_advantage_claimable is False


def test_register_inventory_and_macro_counts_are_not_transpilation_claims() -> None:
    result = _run((1.0, 0.0))
    resources = result.resources

    assert resources.declared_register_qubits == 19
    assert resources.retained_statevector_dimension == 524_288
    assert resources.register_dimensions["stopping_history"] == 4
    assert resources.register_dimensions["scratch_membership_mask"] == 4
    assert resources.register_dimensions["durable_membership_mask"] == 4
    assert resources.register_dimensions["rank_stop_work"] == 4
    assert resources.register_dimensions["active_phase_query_control"] == 2
    assert resources.executed_numpy_kernel_macro_counts[
        "scratch_to_durable_mask_copy"
    ] == 1
    assert resources.elementary_gate_ledger_available is False
    assert resources.transpiled_depth_available is False
    assert resources.compiled_ancilla_qubits_available is False


def test_inactive_level_identity_is_limited_to_clean_work_subspace() -> None:
    audit = run_tiny_inactive_level_subspace_audit(
        CanonicalRyStatevectorOracle((0.82, 0.18)),
        config=TinyCoherentStoppingHistoryConfig(),
    )

    assert audit.clean_inactive_basis_identity_residual < 1e-10
    assert audit.clean_identity_witness_passed is True
    assert audit.dirty_rank_work_negative_control_residual > 1.0
    assert audit.dirty_negative_control_activated is True
    assert audit.clean_query_counts["coherent_total"] == 60
    assert audit.clean_query_counts["controlled_forward"] == 30
    assert audit.clean_query_counts["controlled_inverse"] == 30
    assert audit.dirty_query_counts == audit.clean_query_counts
    assert audit.clean_query_ledger_reconciled is True
    assert audit.dirty_query_ledger_reconciled is True
    assert audit.theorem_status == "basis_witness_only_not_a_subspace_proof"
    assert "initialized to zero" in audit.valid_identity_subspace
    assert "nonzero rank-stop work" in audit.excluded_dirty_subspace


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"cleanup_tolerance": True}, TypeError),
        ({"cleanup_tolerance": 0.0}, ValueError),
        ({"cleanup_tolerance": float("nan")}, ValueError),
        ({"max_statevector_dimension": 0}, ValueError),
    ],
)
def test_config_rejects_invalid_caps(kwargs: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error):
        TinyCoherentStoppingHistoryConfig(**kwargs)


def test_constructor_rejects_wrong_oracle_scope_config_and_dimension_cap() -> None:
    with pytest.raises(TypeError, match="oracle"):
        TinyCoherentAdaptiveStoppingHistory(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly two arms"):
        TinyCoherentAdaptiveStoppingHistory(
            CanonicalRyStatevectorOracle((0.8, 0.1, 0.1))
        )
    with pytest.raises(TypeError, match="config"):
        TinyCoherentAdaptiveStoppingHistory(
            CanonicalRyStatevectorOracle((0.8, 0.2)),
            config=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="statevector"):
        TinyCoherentAdaptiveStoppingHistory(
            CanonicalRyStatevectorOracle((0.8, 0.2)),
            config=TinyCoherentStoppingHistoryConfig(
                max_statevector_dimension=524_287
            ),
        )
