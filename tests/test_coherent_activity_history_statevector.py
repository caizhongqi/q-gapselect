from __future__ import annotations

import inspect
from types import MappingProxyType

import numpy as np
import pytest

import qgapselect.coherent_activity_history_statevector as statevector_module
from qgapselect.coherent import CanonicalRyStatevectorOracle
from qgapselect.coherent_activity_history_statevector import (
    CLAIM_SCOPE,
    CoherentActivityHistoryStatevectorKernel,
    CoherentHistoryStatevectorConfig,
    MeasuredBoundaryBracket,
    run_coherent_activity_history_statevector,
)


def _config(*, boundary_shots: int = 48) -> CoherentHistoryStatevectorConfig:
    return CoherentHistoryStatevectorConfig(
        phase_qubits_by_level=(2, 3),
        boundary_phase_qubits=3,
        boundary_shots=boundary_shots,
        minimum_boundary_samples_per_arm=2,
        measurement_seed=7,
        cleanup_tolerance=1e-9,
        max_statevector_dimension=500_000,
    )


def _grid_oracle() -> CanonicalRyStatevectorOracle:
    # All three angles lie on both the four- and eight-bin QPE grids.  This
    # isolates circuit cleanup from the known finite-QPE leakage failure mode.
    return CanonicalRyStatevectorOracle([1.0, 0.5, 0.0], seed=2)


def _fixed_bracket() -> MeasuredBoundaryBracket:
    return MeasuredBoundaryBracket(
        lower=0.75,
        upper=0.75,
        center=0.75,
        complete=True,
        status="test_public_bracket",
        arm_histograms=(),
        query_counts=MappingProxyType({}),
    )


def test_full_layer_and_explicit_inverse_are_unitary_on_arbitrary_state() -> None:
    kernel = CoherentActivityHistoryStatevectorKernel(
        _grid_oracle(), 1, config=_config(boundary_shots=6)
    )
    rng = np.random.default_rng(91)
    state = rng.normal(size=kernel.statevector_dimension) + 1j * rng.normal(
        size=kernel.statevector_dimension
    )
    state /= np.linalg.norm(state)

    output = kernel.apply_layer_unitary(state, level=0, bracket=_fixed_bracket())
    recovered = kernel.apply_layer_unitary(output, level=0, bracket=_fixed_bracket(), inverse=True)

    assert np.linalg.norm(output) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(recovered, state, atol=1e-12)
    snapshot = kernel.oracle.query_snapshot()
    assert snapshot.counts.get("forward", 0) == 0
    assert snapshot.counts.get("inverse", 0) == 0
    assert snapshot.counts["controlled_forward"] == 14
    assert snapshot.counts["controlled_inverse"] == 14


def test_measured_boundary_then_multilevel_coherent_history_cleans_workspace() -> None:
    result = run_coherent_activity_history_statevector(_grid_oracle(), 1, config=_config())

    assert result.boundary.complete
    assert result.boundary.status == "measured_histogram_bracket_no_confidence_theorem"
    assert result.boundary.center == pytest.approx(0.75)
    assert all(row.samples >= 2 for row in result.boundary.arm_histograms)
    assert len(result.layers) == 2
    assert result.layers[0].active_probability_before == pytest.approx(1.0)
    assert result.layers[0].active_probability_after == pytest.approx(1 / 3)
    assert result.layers[1].active_probability_before == pytest.approx(1 / 3)
    assert result.layers[1].active_probability_after == pytest.approx(0.0, abs=1e-12)
    assert all(layer.cleanup_passed for layer in result.layers)
    assert all(layer.existing_stopped_branch_residual < 1e-12 for layer in result.layers)
    assert result.status == "exact_state_execution_complete_certificate_refused"
    assert result.claim_scope == CLAIM_SCOPE
    assert not result.quantum_advantage_claimable


def test_history_stop_and_output_registers_are_actual_statevector_axes() -> None:
    kernel = CoherentActivityHistoryStatevectorKernel(_grid_oracle(), 1, config=_config())
    result = kernel.run()
    view = result.state.reshape(kernel.shape)

    # Arm zero is selected at level zero.  Arm two is rejected at level zero.
    # The boundary arm remains active at level zero and rejects at level one.
    assert abs(view[0, 0, 0, 0, 0b01, 1, 0b001, 0, 0]) ** 2 == pytest.approx(1 / 3)
    assert abs(view[0, 2, 0, 0, 0b01, 1, 0, 0, 0]) ** 2 == pytest.approx(1 / 3)
    assert abs(view[0, 1, 0, 0, 0b11, 2, 0, 0, 0]) ** 2 == pytest.approx(1 / 3)
    assert result.stop_probabilities[:3] == pytest.approx((0.0, 2 / 3, 1 / 3))
    assert result.direct_output_write_executed
    assert not result.direct_multi_output_complete
    assert not result.certificate_issued
    assert (
        "branch_local_output_mask_is_not_complete_direct_multi_output_extraction" in result.blockers
    )


def test_stopped_subspace_is_unchanged_by_later_controlled_oracle_layer() -> None:
    kernel = CoherentActivityHistoryStatevectorKernel(
        _grid_oracle(), 1, config=_config(boundary_shots=6)
    )
    bracket = _fixed_bracket()
    after_zero = kernel.apply_layer_unitary(kernel.initial_state(), level=0, bracket=bracket)
    old_stopped = np.take(after_zero.reshape(kernel.shape), 0, axis=kernel._ACTIVE)
    old_stopped = np.take(old_stopped, 1, axis=kernel._STOP - 1).copy()

    before_queries = kernel.oracle.query_snapshot().coherent_total
    after_one = kernel.apply_layer_unitary(after_zero, level=1, bracket=bracket)
    new_stopped = np.take(after_one.reshape(kernel.shape), 0, axis=kernel._ACTIVE)
    new_stopped = np.take(new_stopped, 1, axis=kernel._STOP - 1)

    assert np.allclose(new_stopped, old_stopped, atol=1e-12)
    # The finer layer is genuinely executed (4*8-2 controlled calls), but its
    # oracle-control conjunction has no support on the already-stopped code.
    assert kernel.oracle.query_snapshot().coherent_total - before_queries == 30


def test_query_ledger_counts_executed_qpe_calls_not_analytic_iae_costs() -> None:
    shots = 48
    oracle = _grid_oracle()
    result = run_coherent_activity_history_statevector(
        oracle, 1, config=_config(boundary_shots=shots)
    )

    # A measured q=3 boundary copy costs 2*8-1 calls.  Complete q=2 and q=3
    # compute-copy-uncompute layers cost 4*4-2 and 4*8-2 calls respectively.
    assert result.resources.boundary_query_counts["coherent_total"] == shots * 15
    assert [row.query_counts["coherent_total"] for row in result.layers] == [14, 30]
    assert result.resources.history_query_counts["coherent_total"] == 44
    assert result.resources.oracle_queries == shots * 15 + 44
    assert result.resources.query_counts["forward"] == 0
    assert result.resources.query_counts["inverse"] == 0
    assert result.resources.retained_statevector_dimension == np.prod(
        tuple(result.resources.register_dimensions.values())
    )
    assert result.resources.qubits == sum(
        int(np.log2(value)) for value in result.resources.register_dimensions.values()
    )


def test_constructor_and_source_have_no_hidden_mean_or_supplied_boundary_input() -> None:
    parameters = inspect.signature(CoherentActivityHistoryStatevectorKernel.__init__).parameters
    assert set(parameters) == {"self", "oracle", "k", "config"}
    config_parameters = inspect.signature(CoherentHistoryStatevectorConfig).parameters
    assert not any("boundary_value" in name or "threshold" in name for name in config_parameters)
    source = inspect.getsource(statevector_module)
    assert "oracle.means" not in source
    assert "oracle.amplitudes" not in source
    assert "__blocks" not in source

    class DuckOracle:
        n_arms = 3

    with pytest.raises(TypeError, match="CanonicalRyStatevectorOracle"):
        CoherentActivityHistoryStatevectorKernel(DuckOracle(), 1)  # type: ignore[arg-type]


def test_measured_boundary_tie_refuses_layers_and_certificate() -> None:
    oracle = CanonicalRyStatevectorOracle([1.0, 0.5, 0.5, 0.0], seed=3)
    result = run_coherent_activity_history_statevector(
        oracle,
        2,
        config=CoherentHistoryStatevectorConfig(
            phase_qubits_by_level=(2, 3),
            boundary_phase_qubits=3,
            boundary_shots=96,
            minimum_boundary_samples_per_arm=3,
            measurement_seed=5,
            max_statevector_dimension=200_000,
        ),
    )

    assert not result.boundary.complete
    assert result.boundary.status == "measured_boundary_tie_fail_closed"
    assert result.layers == ()
    assert result.resources.history_query_counts["coherent_total"] == 0
    assert not result.certificate_issued
    assert "measured_boundary_tie_fail_closed" in result.blockers


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phase_qubits_by_level": ()},
        {"phase_qubits_by_level": (3, 2)},
        {"phase_qubits_by_level": (2, 2)},
        {"boundary_phase_qubits": 0},
        {"cleanup_tolerance": 0.0},
    ],
)
def test_config_rejects_nonunitary_or_ambiguous_precision_settings(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        CoherentHistoryStatevectorConfig(**kwargs)  # type: ignore[arg-type]
