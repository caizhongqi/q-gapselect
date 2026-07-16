from __future__ import annotations

import inspect
import math

import pytest

from qgapselect.coherent_activity_history_core import (
    CLAIM_SCOPE,
    PROOF_STATUS,
    HistoryIROp,
    HistoryOutput,
    VariableTimeCoherentActivityHistoryCore,
    VariableTimeHistoryConfig,
    run_variable_time_coherent_activity_history,
)
from qgapselect.exact_count_fixtures import generate_exact_count_fixture
from qgapselect.frozen_coherent_oracle import FrozenEmpiricalCoherentOracle


def _config(*, max_levels: int = 4) -> VariableTimeHistoryConfig:
    return VariableTimeHistoryConfig(
        confidence=0.1,
        initial_angular_precision=0.2,
        precision_decay=0.5,
        max_levels=max_levels,
        shots_per_iae_round=64,
        iae_max_rounds=6,
        iae_max_grover_power=31,
        iae_grid_points=2049,
        verification_angular_precision=0.02,
        verification_shots_per_round=96,
        verification_max_rounds=7,
        verification_max_grover_power=63,
        verification_grid_points=4097,
    )


def _separated_oracle(seed: int = 1) -> FrozenEmpiricalCoherentOracle:
    fixture = generate_exact_count_fixture(
        {
            "a0": 1.0,
            "a1": 0.85,
            "a2": 0.55,
            "a3": 0.45,
            "a4": 0.15,
            "a5": 0.0,
        },
        table_size=2048,
        seed=91,
    )
    return FrozenEmpiricalCoherentOracle(fixture, measurement_seed=seed)


def _tied_boundary_oracle(seed: int = 1) -> FrozenEmpiricalCoherentOracle:
    fixture = generate_exact_count_fixture(
        {"a0": 0.8, "a1": 0.5, "a2": 0.5, "a3": 0.2},
        table_size=1024,
        seed=92,
    )
    return FrozenEmpiricalCoherentOracle(fixture, measurement_seed=seed)


def test_core_accepts_blind_frozen_protocol_and_issues_fresh_certificate() -> None:
    oracle = _separated_oracle()

    result = run_variable_time_coherent_activity_history(
        oracle, 3, config=_config()
    )

    assert result.complete and result.certified
    assert result.extracted_selected == (0, 1, 2)
    assert result.extracted_rejected == (3, 4, 5)
    assert result.unresolved == ()
    assert result.certificate is not None
    assert result.certificate.selected == (0, 1, 2)
    assert result.certificate.verification_margin > 0.0
    assert result.verification is not None and result.verification.passed
    assert result.status == "certified_fresh_strict_separation"
    assert result.claim_scope == CLAIM_SCOPE
    assert not result.hardware_claimable
    assert not result.coherent_query_advantage_claimable


def test_selection_and_fresh_verification_ledgers_partition_actual_calls() -> None:
    oracle = _separated_oracle()
    # Calls before the core run must not leak into its resource total.
    oracle.run_grover_experiment(0, 0, 3, tag="preexisting")
    before = oracle.query_snapshot().total

    result = VariableTimeCoherentActivityHistoryCore(
        oracle, 3, config=_config()
    ).run()
    resources = result.executed_resources

    assert resources.oracle_queries == oracle.query_snapshot().total - before
    assert resources.oracle_queries == (
        resources.selection_query_counts["total"]
        + resources.verification_query_counts["total"]
    )
    assert sum(
        row["total"] for row in resources.selection_query_counts_by_arm.values()
    ) == resources.selection_query_counts["total"]
    assert sum(
        row["total"] for row in resources.verification_query_counts_by_arm.values()
    ) == resources.verification_query_counts["total"]
    assert all(
        row["total"] > 0
        for row in resources.verification_query_counts_by_arm.values()
    )
    snapshot = oracle.query_snapshot()
    assert all(
        f"vt_history_fresh_verify_arm_{arm}" in snapshot.by_tag
        for arm in range(oracle.n_arms)
    )
    assert all(
        tag.startswith("vt_history_fresh_verify_arm_")
        for tag in snapshot.by_tag
        if tag.startswith("vt_history_fresh_verify")
    )


def test_ir_has_explicit_compute_phase_uncompute_and_zero_cleanup() -> None:
    result = run_variable_time_coherent_activity_history(
        _separated_oracle(), 3, config=_config()
    )

    assert result.layers
    for layer in result.layers:
        operations = tuple(row.operation for row in layer.instructions)
        assert HistoryIROp.COMPUTE_UNKNOWN_BOUNDARY_PREDICATE in operations
        assert HistoryIROp.COMPUTE_SELECTED_PHASE_PREDICATE in operations
        assert HistoryIROp.PHASE_SELECTED_OUTPUT in operations
        assert HistoryIROp.UNCOMPUTE_SELECTED_PHASE_PREDICATE in operations
        assert HistoryIROp.UNCOMPUTE_UNKNOWN_BOUNDARY_PREDICATE in operations
        assert HistoryIROp.DIRECT_MULTI_OUTPUT_EMIT in operations
        assert layer.cleanup_passed
        assert layer.predicate_workspace_residual == 0
        assert layer.phase_workspace_residual == 0
        compute = next(
            row
            for row in layer.instructions
            if row.operation is HistoryIROp.COMPUTE_UNKNOWN_BOUNDARY_PREDICATE
        )
        uncompute = next(
            row
            for row in layer.instructions
            if row.operation is HistoryIROp.UNCOMPUTE_UNKNOWN_BOUNDARY_PREDICATE
        )
        assert uncompute.inverse_of_sequence == compute.sequence
    assert result.candidate_ir_resources.cleanup_verified
    assert all(branch.predicate_workspace_zero for branch in result.branches)
    assert all(branch.phase_workspace_zero for branch in result.branches)


def test_activity_history_and_stop_codes_record_heterogeneous_variable_time() -> None:
    result = run_variable_time_coherent_activity_history(
        _separated_oracle(), 3, config=_config()
    )
    by_arm = {branch.arm: branch for branch in result.branches}

    # With this seeded execution, far arms stop on level zero while the two
    # boundary arms continue to a finer level.
    assert by_arm[0].activity_history[:2] == (True, False)
    assert by_arm[1].activity_history[:2] == (True, False)
    assert by_arm[4].activity_history[:2] == (True, False)
    assert by_arm[5].activity_history[:2] == (True, False)
    assert by_arm[2].activity_history[:2] == (True, True)
    assert by_arm[3].activity_history[:2] == (True, True)
    assert {branch.stop_code for branch in result.branches} == {1, 2}
    assert by_arm[0].output is HistoryOutput.SELECTED
    assert by_arm[3].output is HistoryOutput.REJECTED
    assert by_arm[0].selected_phase_parity == 1
    assert by_arm[3].selected_phase_parity == 0
    assert (
        by_arm[0].selection_query_counts["total"]
        < by_arm[2].selection_query_counts["total"]
    )


def test_tied_boundary_stays_unresolved_and_never_gets_a_certificate() -> None:
    result = run_variable_time_coherent_activity_history(
        _tied_boundary_oracle(), 2, config=_config(max_levels=3)
    )

    assert not result.complete
    assert not result.certified
    assert result.certificate is None
    assert result.verification is None
    assert result.unresolved == (1, 2)
    assert result.extracted_selected == (0,)
    assert result.extracted_rejected == (3,)
    assert result.status == "max_levels_unresolved_no_certificate"
    assert result.executed_resources.verification_query_counts["total"] == 0


def test_risk_allocation_is_summable_and_certificate_uses_fresh_half() -> None:
    config = _config()
    result = run_variable_time_coherent_activity_history(
        _separated_oracle(), 3, config=config
    )
    assert result.certificate is not None

    selection_risk = result.certificate.allocated_selection_risk_upper_bound
    verification_risk = result.certificate.allocated_verification_risk
    assert selection_risk <= config.confidence / 2.0
    assert verification_risk == pytest.approx(config.confidence / 2.0)
    assert selection_risk + verification_risk <= config.confidence
    assert all(
        estimate.allocated_failure_probability
        == pytest.approx(layer.allocated_failure_probability_per_arm)
        for layer in result.layers
        for estimate in layer.estimates
    )


def test_candidate_ir_resources_are_separate_and_charge_no_free_qram() -> None:
    result = run_variable_time_coherent_activity_history(
        _separated_oracle(), 3, config=_config()
    )
    actual = result.executed_resources
    candidate = result.candidate_ir_resources

    assert actual.oracle_queries > 0
    assert candidate.serial_finite_state_gate_count > 0
    assert candidate.candidate_coherent_scheduled_depth > 0
    assert candidate.candidate_total_qubits == sum(
        (
            candidate.candidate_index_qubits,
            candidate.candidate_level_qubits,
            candidate.candidate_stop_qubits,
            candidate.candidate_activity_history_qubits,
            candidate.candidate_output_qubits,
            candidate.candidate_phase_qubits,
            candidate.candidate_precision_qubits,
            candidate.candidate_workspace_qubits,
        )
    )
    assert candidate.no_free_qram
    assert candidate.membership_compilation == (
        "explicit_multi_controlled_index_equalities_linear_in_births"
    )
    assert candidate.proof_status == PROOF_STATUS
    assert "theorem" in candidate.proof_status
    assert not hasattr(candidate, "executed_oracle_queries")


def test_constructor_has_no_supplied_activity_rows_or_mean_argument() -> None:
    parameters = inspect.signature(
        VariableTimeCoherentActivityHistoryCore.__init__
    ).parameters
    assert set(parameters) == {"self", "oracle", "k", "config"}
    assert "means" not in parameters
    assert "active_indices_by_level" not in parameters
    assert "output_indices_by_level" not in parameters

    oracle = _separated_oracle()
    assert not hasattr(oracle, "means")
    result = VariableTimeCoherentActivityHistoryCore(
        oracle, 3, config=_config()
    ).run()
    assert all(
        instruction.information_source == "executed_confidence_intervals"
        for layer in result.layers
        for instruction in layer.instructions
    )


def test_algorithm_never_reads_a_hidden_mean_attribute() -> None:
    delegate = _separated_oracle()

    class MeanTrapOracle:
        @property
        def n_arms(self) -> int:
            return delegate.n_arms

        @property
        def means(self) -> tuple[float, ...]:
            raise AssertionError("algorithm attempted to read hidden means")

        def query_snapshot(self):
            return delegate.query_snapshot()

        def run_grover_experiment(
            self,
            arm: int,
            grover_power: int,
            shots: int,
            *,
            controlled: bool = False,
            tag: str | None = None,
        ) -> int:
            return delegate.run_grover_experiment(
                arm,
                grover_power,
                shots,
                controlled=controlled,
                tag=tag,
            )

    result = run_variable_time_coherent_activity_history(
        MeanTrapOracle(), 3, config=_config()
    )
    assert result.certified


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"confidence": 1.0}, ValueError),
        ({"initial_angular_precision": math.pi / 2}, ValueError),
        ({"precision_decay": 1.0}, ValueError),
        ({"max_levels": 0}, ValueError),
        ({"iae_grid_points": 256}, ValueError),
        ({"verification_angular_precision": 0.0}, ValueError),
    ],
)
def test_invalid_config_is_rejected(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        VariableTimeHistoryConfig(**kwargs)  # type: ignore[arg-type]


def test_invalid_oracle_and_k_are_rejected() -> None:
    with pytest.raises(TypeError, match="oracle"):
        VariableTimeCoherentActivityHistoryCore(object(), 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k"):
        VariableTimeCoherentActivityHistoryCore(_separated_oracle(), 6)
    with pytest.raises(TypeError, match="config"):
        VariableTimeCoherentActivityHistoryCore(
            _separated_oracle(), 3, config=object()  # type: ignore[arg-type]
        )
