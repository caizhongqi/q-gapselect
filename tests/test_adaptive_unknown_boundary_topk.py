from __future__ import annotations

import inspect

import pytest

from qgapselect.adaptive_unknown_boundary_topk import (
    METHOD_ID,
    OUTPUT_INCONCLUSIVE,
    OUTPUT_MASK,
    AdaptiveUnknownBoundaryTopKConfig,
    run_adaptive_unknown_boundary_topk,
)
from qgapselect.coherent import CanonicalRyStatevectorOracle


def _config(*, hard_cap: int = 1_000) -> AdaptiveUnknownBoundaryTopKConfig:
    return AdaptiveUnknownBoundaryTopKConfig(
        minimum_phase_qubits=1,
        maximum_phase_qubits=4,
        target_diagnostic_error_probability=0.04,
        numerical_cleanup_tolerance=1e-10,
        max_statevector_dimension=600_000,
        max_canonical_oracle_queries=hard_cap,
    )


def test_interface_has_no_answer_gap_family_or_schedule_argument() -> None:
    parameters = inspect.signature(run_adaptive_unknown_boundary_topk).parameters
    assert set(parameters) == {"oracle", "k", "config"}
    config_fields = set(AdaptiveUnknownBoundaryTopKConfig.__dataclass_fields__)
    assert not config_fields.intersection(
        {
            "answer",
            "answer_set",
            "gap",
            "boundary",
            "family",
            "schedule",
            "activity_history",
        }
    )


def test_off_grid_case_adapts_to_phase_three_with_exact_query_history() -> None:
    oracle = CanonicalRyStatevectorOracle((0.82, 0.18))
    result = run_adaptive_unknown_boundary_topk(oracle, 1, config=_config())

    assert result.method_id == METHOD_ID
    assert result.output_status == OUTPUT_MASK
    assert result.membership_mask == 0b01
    assert result.membership_bits == (1, 0)
    assert result.status == "diagnostic_mask_selected_certificate_withheld"
    assert [level.phase_qubits for level in result.stopping_history.levels] == [1, 2, 3]
    assert result.stopping_history.emulated_history_bits == (0, 0, 1)
    assert [
        level.cumulative_coherent_queries for level in result.stopping_history.levels
    ] == [12, 40, 100]
    assert [level.query_counts["coherent_total"] for level in result.stopping_history.levels] == [
        12,
        28,
        60,
    ]
    assert all(level.query_formula_reconciled for level in result.stopping_history.levels)
    assert result.query_budget.query_counts["coherent_total"] == 100
    assert result.query_budget.expected_query_counts["coherent_total"] == 100
    assert result.query_budget.query_counts["qram_queries"] == 0
    assert result.query_budget.aggregate_reconciled is True
    assert result.query_budget.budget_valid is True
    assert result.hard_cap_respected is True
    assert result.cleanup.diagnostic_error_bound == pytest.approx(0.034769713197007945)
    assert result.cleanup.exact_numerical_cleanup_passed is False
    assert result.cleanup.diagnostic_acceptance_bound_passed is True
    assert result.durable_output.direct_k_membership_register_used is True
    assert result.durable_output.rank_relation_computed_and_uncomputed is True
    assert result.durable_output.phase_estimation_inverted is True
    assert result.durable_output.exact_transient_cleanup_passed is False
    assert result.durable_output.approximate_cleanup_bound_passed is True


def test_deeper_off_grid_case_reaches_phase_four() -> None:
    oracle = CanonicalRyStatevectorOracle((0.7, 0.3))
    result = run_adaptive_unknown_boundary_topk(oracle, 1, config=_config())

    assert result.output_status == OUTPUT_MASK
    assert result.membership_mask == 0b01
    assert result.stopping_history.first_stop_phase_qubits == 4
    assert result.stopping_history.emulated_history_bits == (0, 0, 0, 1)
    assert result.query_budget.query_counts["coherent_total"] == 224
    assert result.cleanup.diagnostic_error_bound == pytest.approx(0.005699294330581028)
    assert result.cleanup.diagnostic_error_bound < 0.04


def test_exact_tie_runs_to_public_precision_cap_and_fails_closed() -> None:
    oracle = CanonicalRyStatevectorOracle((0.5, 0.5))
    result = run_adaptive_unknown_boundary_topk(oracle, 1, config=_config())

    assert result.output_status == OUTPUT_INCONCLUSIVE
    assert result.membership_mask is None
    assert result.membership_bits == ()
    assert result.status == "inconclusive_fail_closed"
    assert result.stopping_history.emulated_history_bits == (0, 0, 0, 0)
    assert result.stopping_history.first_stop_phase_qubits is None
    assert result.query_budget.query_counts["coherent_total"] == 224
    assert result.query_budget.hard_cap_respected is True
    assert result.cleanup.nonstrict_boundary_mass == pytest.approx(1.0)
    assert result.durable_output.status == OUTPUT_INCONCLUSIVE
    assert result.durable_output.direct_k_membership_register_used is False


def test_query_hard_cap_is_checked_before_executing_next_level() -> None:
    oracle = CanonicalRyStatevectorOracle((0.82, 0.18))
    result = run_adaptive_unknown_boundary_topk(oracle, 1, config=_config(hard_cap=40))

    assert result.output_status == OUTPUT_INCONCLUSIVE
    assert [level.phase_qubits for level in result.stopping_history.levels] == [1, 2]
    assert result.query_budget.query_counts["coherent_total"] == 40
    assert result.query_budget.hard_cap_queries == 40
    assert result.query_budget.hard_cap_respected is True
    assert result.query_budget.budget_valid is True
    assert result.query_budget.blocked_before_phase_qubits == 3
    assert result.query_budget.blocked_next_level_query_cost == 60


def test_zero_query_cap_fails_closed_without_touching_oracle() -> None:
    oracle = CanonicalRyStatevectorOracle((1.0, 0.0))
    result = run_adaptive_unknown_boundary_topk(oracle, 1, config=_config(hard_cap=0))

    assert result.output_status == OUTPUT_INCONCLUSIVE
    assert result.stopping_history.levels == ()
    assert result.query_budget.query_counts["coherent_total"] == 0
    assert result.query_budget.aggregate_reconciled is True
    assert result.query_budget.budget_valid is True
    assert result.query_budget.blocked_before_phase_qubits == 1
    assert result.query_budget.blocked_next_level_query_cost == 12
    assert result.cleanup.available is False


def test_statevector_cap_fails_closed_before_oversized_level() -> None:
    oracle = CanonicalRyStatevectorOracle((0.7, 0.3))
    result = run_adaptive_unknown_boundary_topk(
        oracle,
        1,
        config=AdaptiveUnknownBoundaryTopKConfig(
            minimum_phase_qubits=1,
            maximum_phase_qubits=4,
            target_diagnostic_error_probability=0.04,
            numerical_cleanup_tolerance=1e-10,
            max_statevector_dimension=40_000,
            max_canonical_oracle_queries=1_000,
        ),
    )

    assert result.output_status == OUTPUT_INCONCLUSIVE
    assert [level.phase_qubits for level in result.stopping_history.levels] == [1, 2, 3]
    assert result.query_budget.blocked_before_phase_qubits == 4
    assert result.query_budget.blocked_next_level_query_cost is None
    assert result.query_budget.query_counts["coherent_total"] == 100


def test_controller_and_certificate_claim_boundaries_are_explicit() -> None:
    result = run_adaptive_unknown_boundary_topk(
        CanonicalRyStatevectorOracle((1.0, 0.0)),
        1,
        config=_config(),
    )

    interface = result.input_interface
    assert interface.answer_dependent_inputs_supplied is False
    assert interface.gap_supplied is False
    assert interface.boundary_supplied is False
    assert interface.family_label_supplied is False
    assert interface.precision_schedule_supplied is False
    assert interface.activity_history_supplied is False
    assert "precision_schedule" in interface.forbidden_inputs
    history = result.stopping_history
    assert history.controller_is_classical is True
    assert history.independently_coherent_level_unitaries_executed is True
    assert history.single_coherent_variable_time_unitary_implemented is False
    assert history.coherent_history_register_physically_retained is False
    assert history.coherent_history_cleanup_proved is False
    assert result.certificate.issued is False
    assert result.certificate.top_k_correctness_error_bound is None
    assert result.claim_boundary.generic_off_grid_correctness_proved is False
    assert result.claim_boundary.new_query_upper_bound_proved is False
    assert result.claim_boundary.matching_lower_bound_proved is False
    assert result.claim_boundary.quantum_advantage_claimable is False
    assert result.claim_boundary.ccf_a_claimable is False
    assert result.quantum_advantage_claimable is False


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("minimum_phase_qubits", True, TypeError),
        ("maximum_phase_qubits", 6, ValueError),
        ("target_diagnostic_error_probability", 0.0, ValueError),
        ("target_diagnostic_error_probability", 0.5, ValueError),
        ("numerical_cleanup_tolerance", float("nan"), ValueError),
        ("max_statevector_dimension", 0, ValueError),
        ("max_canonical_oracle_queries", -1, ValueError),
    ],
)
def test_config_rejects_invalid_public_limits(
    field: str, value: object, error: type[Exception]
) -> None:
    kwargs = {field: value}
    with pytest.raises(error):
        AdaptiveUnknownBoundaryTopKConfig(**kwargs)


def test_runner_rejects_invalid_oracle_k_and_config_types() -> None:
    oracle = CanonicalRyStatevectorOracle((0.8, 0.2))
    with pytest.raises(TypeError, match="oracle"):
        run_adaptive_unknown_boundary_topk(object(), 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="k"):
        run_adaptive_unknown_boundary_topk(oracle, True)
    with pytest.raises(ValueError, match="1 <= k <"):
        run_adaptive_unknown_boundary_topk(oracle, 2)
    with pytest.raises(TypeError, match="config"):
        run_adaptive_unknown_boundary_topk(oracle, 1, config=object())  # type: ignore[arg-type]
