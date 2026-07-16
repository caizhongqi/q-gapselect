from __future__ import annotations

import inspect

import numpy as np
import pytest

import qgapselect.coherent_rank_baseline as rank_module
from qgapselect.coherent import CanonicalRyStatevectorOracle
from qgapselect.coherent_rank_baseline import (
    CLAIM_SCOPE,
    RANK_COMPILATION_MODEL,
    CoherentMeasuredQPERankBaseline,
    CoherentRankBaselineConfig,
    run_coherent_rank_baseline,
)


def _config(*, shots: int = 4, phase_qubits: int = 3) -> CoherentRankBaselineConfig:
    return CoherentRankBaselineConfig(
        phase_qubits=phase_qubits,
        shots_per_arm=shots,
        measurement_seed=7,
        cleanup_tolerance=1e-12,
        max_statevector_dimension=500_000,
    )


def _grid_oracle() -> CanonicalRyStatevectorOracle:
    # These amplitudes occupy exact folded bins 4, 2, and 0 for q=3.
    return CanonicalRyStatevectorOracle([1.0, 0.5, 0.0], seed=2)


def test_measured_qpe_builds_complete_direct_top_k_membership() -> None:
    result = run_coherent_rank_baseline(_grid_oracle(), 2, config=_config())

    assert [row.modal_folded_bin for row in result.estimates] == [4, 2, 0]
    assert all(row.modal_unique for row in result.estimates)
    assert result.ranking == (0, 1, 2)
    assert result.boundary.strict
    assert result.boundary.inside_arm == 1
    assert result.boundary.outside_arm == 2
    assert result.membership_bits == (1, 1, 0)
    assert result.membership_mask == 0b011
    assert sum(result.membership_bits) == 2
    assert result.direct_multi_output_complete
    assert result.status == "legal_direct_membership_baseline_complete_theorem_blocked"
    assert result.certificate_issued is False
    assert result.quantum_advantage_claimable is False
    assert result.claim_scope == CLAIM_SCOPE


def test_rank_compute_copy_uncompute_cleans_every_transient_register() -> None:
    baseline = CoherentMeasuredQPERankBaseline(
        _grid_oracle(), 1, config=_config(shots=2)
    )
    result = baseline.run()
    view = result.state.reshape(baseline.shape)

    assert view[result.estimate_register_code, 0b001, 0, 0] == pytest.approx(1.0)
    assert np.count_nonzero(view[:, :, 1:, :]) == 0
    assert np.count_nonzero(view[:, :, :, 1]) == 0
    assert result.resources.cleanup.passed
    assert result.resources.cleanup.expected_output_residual_l2 == pytest.approx(0.0)
    assert result.resources.cleanup.membership_work_nonzero_probability == pytest.approx(0.0)
    assert result.resources.cleanup.comparator_nonzero_probability == pytest.approx(0.0)
    assert result.resources.cleanup.end_to_end_unitary_cleanup_proved is False
    assert "qpe_measurement_and_reset_breaks_end_to_end_coherence" in result.blockers


def test_rank_relation_is_an_involution_on_arbitrary_state() -> None:
    baseline = CoherentMeasuredQPERankBaseline(
        CanonicalRyStatevectorOracle([1.0, 0.0]),
        1,
        config=_config(shots=1, phase_qubits=2),
    )
    rng = np.random.default_rng(41)
    state = rng.normal(size=baseline.statevector_dimension) + 1j * rng.normal(
        size=baseline.statevector_dimension
    )
    state /= np.linalg.norm(state)

    restored = baseline.apply_rank_compute(baseline.apply_rank_compute(state))

    assert np.allclose(restored, state, atol=1e-12)


def test_query_and_compilation_ledgers_charge_the_expensive_baseline() -> None:
    shots = 3
    phase_qubits = 3
    baseline = CoherentMeasuredQPERankBaseline(
        _grid_oracle(),
        1,
        config=_config(shots=shots, phase_qubits=phase_qubits),
    )
    result = baseline.run()
    resources = result.resources
    phase_bins = 1 << phase_qubits
    expected_oracle_queries = 3 * shots * (2 * phase_bins - 1)

    assert resources.oracle_queries == expected_oracle_queries
    assert resources.query_counts["controlled_forward"] == 3 * shots * (phase_bins - 1)
    assert resources.query_counts["controlled_inverse"] == 3 * shots * (phase_bins - 1)
    assert resources.query_counts["forward"] == 3 * shots
    assert resources.query_counts.get("inverse", 0) == 0
    assert resources.query_counts["destructive_phase_measurements"] == 3 * shots
    assert resources.query_counts["phase_reward_resets"] == 3 * shots
    assert resources.query_counts["qram_queries"] == 0
    assert resources.qram_assumed is False
    assert resources.rank_compilation_model == RANK_COMPILATION_MODEL
    assert resources.rank_truth_table_rows == phase_bins**3
    assert resources.gate_counts["compiled_rank_truth_table_rows"] == 2 * phase_bins**3
    assert resources.gate_counts["no_ancilla_multicontrolled_rank_primitives"] > 0
    assert resources.depth > expected_oracle_queries
    assert resources.qubits == max(
        sum(int(np.log2(value)) for value in baseline.shape),
        sum(int(np.log2(value)) for value in baseline.qpe_shape),
    )


def test_measured_boundary_tie_fails_closed_without_membership_claim() -> None:
    oracle = CanonicalRyStatevectorOracle([0.5, 0.5, 0.0], seed=3)
    result = run_coherent_rank_baseline(oracle, 1, config=_config())

    assert result.ranking == (0, 1, 2)
    assert result.boundary.strict is False
    assert result.boundary.status == "ambiguous_measured_discrete_boundary_fail_closed"
    assert result.membership_bits == ()
    assert result.membership_mask is None
    assert result.direct_multi_output_complete is False
    assert result.certificate_issued is False
    assert "measured_top_k_boundary_tie_fail_closed" in result.blockers


def test_public_interface_has_no_answer_boundary_or_hidden_oracle_read() -> None:
    parameters = inspect.signature(CoherentMeasuredQPERankBaseline.__init__).parameters
    assert set(parameters) == {"self", "oracle", "k", "config"}
    config_parameters = inspect.signature(CoherentRankBaselineConfig).parameters
    forbidden = ("threshold", "truth", "answer", "boundary_value")
    assert not any(any(token in name for token in forbidden) for name in config_parameters)
    source = inspect.getsource(rank_module)
    assert "oracle.means" not in source
    assert "oracle.amplitudes" not in source
    assert "__blocks" not in source

    class DuckOracle:
        n_arms = 3

    with pytest.raises(TypeError, match="CanonicalRyStatevectorOracle"):
        CoherentMeasuredQPERankBaseline(DuckOracle(), 1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phase_qubits": 0},
        {"phase_qubits": 5},
        {"shots_per_arm": 0},
        {"cleanup_tolerance": 0.0},
        {"max_statevector_dimension": 0},
    ],
)
def test_config_rejects_invalid_precision_sampling_and_resource_limits(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        CoherentRankBaselineConfig(**kwargs)  # type: ignore[arg-type]


def test_constructor_rejects_invalid_k_and_oversized_exact_state() -> None:
    with pytest.raises(ValueError, match="k must satisfy"):
        CoherentMeasuredQPERankBaseline(_grid_oracle(), 3)
    with pytest.raises(ValueError, match="statevector"):
        CoherentMeasuredQPERankBaseline(
            _grid_oracle(),
            1,
            config=CoherentRankBaselineConfig(max_statevector_dimension=1),
        )
