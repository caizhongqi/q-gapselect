from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

import qgapselect.coherent_unknown_boundary_topk as topk_module
from qgapselect.coherent import CanonicalRyStatevectorOracle
from qgapselect.coherent_unknown_boundary_topk import (
    CLAIM_SCOPE,
    RANK_COMPILATION_MODEL,
    CoherentUnknownBoundaryTopK,
    CoherentUnknownBoundaryTopKConfig,
    run_coherent_unknown_boundary_topk,
)


def _config(*, phase_qubits: int = 3) -> CoherentUnknownBoundaryTopKConfig:
    return CoherentUnknownBoundaryTopKConfig(
        phase_qubits=phase_qubits,
        cleanup_tolerance=1e-10,
        max_statevector_dimension=600_000,
    )


def _on_grid_oracle() -> CanonicalRyStatevectorOracle:
    phase_bins = 8
    return CanonicalRyStatevectorOracle(
        [
            math.sin(3 * math.pi / phase_bins) ** 2,
            math.sin(math.pi / phase_bins) ** 2,
        ]
    )


def test_on_grid_strict_instance_returns_complete_durable_membership() -> None:
    result = run_coherent_unknown_boundary_topk(
        _on_grid_oracle(),
        1,
        config=_config(),
    )

    assert result.membership_bits == (1, 0)
    assert result.membership_mask == 0b01
    assert result.direct_multi_output_complete
    assert result.rounding_promise_witnessed
    assert result.strict_boundary_witnessed
    assert result.certificate_issued is False
    assert result.quantum_advantage_claimable is False
    assert result.status == "rounding_promise_top_k_unitary_complete_theorem_blocked"
    assert result.claim_scope == CLAIM_SCOPE
    assert result.boundary.dominant_mask == 0b01
    assert result.boundary.dominant_probability == pytest.approx(1.0, abs=1e-10)
    assert result.boundary.strict_probability == pytest.approx(1.0, abs=1e-10)
    assert sum(result.membership_bits) == 1


def test_three_arm_execution_copies_every_membership_bit_and_cleans() -> None:
    result = run_coherent_unknown_boundary_topk(
        CanonicalRyStatevectorOracle([1.0, 1.0, 0.0]),
        2,
        config=_config(phase_qubits=1),
    )

    assert result.membership_bits == (1, 1, 0)
    assert result.membership_mask == 0b011
    assert len(result.membership_bits) == 3
    assert result.resources.register_dimensions["durable_membership_output"] == 8
    assert result.resources.cleanup.passed
    assert result.resources.cleanup.executed_transient_nonzero_probability < 1e-10
    assert result.resources.retained_statevector_dimension == 524_288


def test_rank_copy_cleanup_identity_holds_exactly_on_the_qpe_grid() -> None:
    result = run_coherent_unknown_boundary_topk(
        _on_grid_oracle(),
        1,
        config=_config(),
    )
    cleanup = result.resources.cleanup

    assert cleanup.predicted_transient_nonzero_probability < 1e-10
    assert cleanup.executed_transient_nonzero_probability < 1e-10
    assert cleanup.prediction_residual < 1e-10
    assert cleanup.output_collision_probability == pytest.approx(1.0, abs=1e-10)
    assert cleanup.output_reduced_purity == pytest.approx(1.0, abs=1e-10)
    assert cleanup.purity_residual < 1e-10
    assert cleanup.phase_nonzero_probability < 1e-10
    assert cleanup.compiled_index_nonzero_probability < 1e-10
    assert cleanup.reward_nonzero_probability < 1e-10
    assert cleanup.rank_work_nonzero_probability < 1e-10


def test_off_grid_output_entanglement_matches_prediction_and_fails_closed() -> None:
    result = run_coherent_unknown_boundary_topk(
        CanonicalRyStatevectorOracle([0.82, 0.18]),
        1,
        config=_config(),
    )
    cleanup = result.resources.cleanup
    nonzero_masks = [
        mask
        for mask, probability in result.boundary.output_mask_probabilities.items()
        if probability > 1e-8
    ]

    assert len(nonzero_masks) >= 2
    assert cleanup.predicted_transient_nonzero_probability > 1e-3
    assert cleanup.executed_transient_nonzero_probability > 1e-3
    assert cleanup.executed_transient_nonzero_probability == pytest.approx(
        cleanup.predicted_transient_nonzero_probability,
        abs=1e-10,
    )
    assert cleanup.prediction_residual < 1e-10
    assert cleanup.output_reduced_purity == pytest.approx(
        cleanup.output_collision_probability,
        abs=1e-10,
    )
    assert not cleanup.passed
    assert result.membership_bits == ()
    assert result.membership_mask is None
    assert not result.direct_multi_output_complete
    assert not result.rounding_promise_witnessed
    assert result.status == "finite_qpe_output_entanglement_fail_closed"
    assert "finite_qpe_output_entanglement_cleanup_failed" in result.blockers


def test_exact_grid_boundary_tie_cleans_but_does_not_emit_membership() -> None:
    result = run_coherent_unknown_boundary_topk(
        CanonicalRyStatevectorOracle([0.5, 0.5]),
        1,
        config=_config(phase_qubits=2),
    )

    assert result.resources.cleanup.passed
    assert result.rounding_promise_witnessed
    assert not result.strict_boundary_witnessed
    assert result.boundary.strict_probability == pytest.approx(0.0, abs=1e-10)
    assert result.boundary.dominant_mask == 0
    assert result.boundary.dominant_probability == pytest.approx(1.0, abs=1e-10)
    assert result.membership_mask is None
    assert result.membership_bits == ()
    assert result.status == "coherent_discrete_boundary_not_strict_fail_closed"
    assert "coherent_discrete_boundary_not_strict" in result.blockers


def test_rank_boundary_and_output_copy_relations_are_involutions() -> None:
    algorithm = CoherentUnknownBoundaryTopK(
        CanonicalRyStatevectorOracle([1.0, 0.0]),
        1,
        config=_config(phase_qubits=1),
    )
    rng = np.random.default_rng(91)
    state = rng.normal(size=algorithm.statevector_dimension) + 1j * rng.normal(
        size=algorithm.statevector_dimension
    )
    state /= np.linalg.norm(state)

    rank_roundtrip = algorithm.apply_rank_boundary_relation(
        algorithm.apply_rank_boundary_relation(state)
    )
    copy_roundtrip = algorithm.apply_durable_output_copy(
        algorithm.apply_durable_output_copy(state)
    )

    assert np.allclose(rank_roundtrip, state, atol=1e-12)
    assert np.allclose(copy_roundtrip, state, atol=1e-12)


def test_resource_ledger_charges_all_oracle_directions_and_no_qram() -> None:
    oracle = _on_grid_oracle()
    algorithm = CoherentUnknownBoundaryTopK(oracle, 1, config=_config())
    result = algorithm.run()
    resources = result.resources
    phase_bins = 8
    expected_controlled = 2 * oracle.n_arms * (phase_bins - 1)
    expected_total = 2 * oracle.n_arms * (2 * phase_bins - 1)

    assert resources.query_counts["forward"] == oracle.n_arms
    assert resources.query_counts["inverse"] == oracle.n_arms
    assert resources.query_counts["controlled_forward"] == expected_controlled
    assert resources.query_counts["controlled_inverse"] == expected_controlled
    assert resources.query_counts["coherent_total"] == expected_total
    assert resources.query_counts["qram_queries"] == 0
    assert resources.oracle_queries == expected_total
    assert resources.qram_assumed is False
    assert resources.qpe_calls == 2 * oracle.n_arms
    assert resources.controlled_grover_iterations == expected_controlled
    assert resources.rank_truth_table_rows == phase_bins**oracle.n_arms
    assert resources.rank_compilation_model == RANK_COMPILATION_MODEL
    executed = resources.executed_numpy_kernel_operation_counts
    assert executed["exhaustive_rank_boundary_permutation"] == 2
    assert executed["durable_output_permutation"] == 1
    assert executed["dense_qft_matrix_multiply"] == oracle.n_arms
    assert executed["dense_inverse_qft_matrix_multiply"] == oracle.n_arms
    macros = resources.logical_circuit_macro_counts
    assert macros["rank_boundary_relation"] == 2
    assert macros["durable_output_copy"] == 1
    assert macros["controlled_grover_forward"] == oracle.n_arms * (phase_bins - 1)
    assert macros["controlled_grover_inverse"] == oracle.n_arms * (phase_bins - 1)
    proxies = resources.rank_compilation_proxies
    assert proxies["phase_control_width"] == oracle.n_arms * 3
    assert proxies["rank_relation_rows_per_call"] == phase_bins**oracle.n_arms
    assert proxies["rank_relation_calls"] == 2
    assert proxies["rank_relation_output_bit_incidences_across_calls"] > 0
    assert resources.elementary_gate_ledger_available is False
    assert resources.transpiled_depth_available is False
    assert resources.transpiled_depth is None
    assert resources.compiled_ancilla_qubits_available is False
    assert not hasattr(resources, "gate_counts")
    assert not hasattr(resources, "depth")
    assert "not_elementary" in resources.logical_macro_count_semantics
    assert "no_gate_or_depth_synthesis" in resources.compilation_proxy_semantics
    assert resources.declared_register_qubits == 15
    assert resources.retained_statevector_dimension == 32_768
    assert resources.estimated_peak_complex_amplitudes == 65_600

    for arm in range(oracle.n_arms):
        tag = f"coherent_unknown_boundary_topk_arm_{arm}"
        assert oracle.query_snapshot().by_tag[tag] == {
            "forward": 1,
            "controlled_inverse": 2 * (phase_bins - 1),
            "controlled_forward": 2 * (phase_bins - 1),
            "inverse": 1,
        }


def test_public_interface_has_no_answer_dependent_or_hidden_oracle_input() -> None:
    parameters = inspect.signature(CoherentUnknownBoundaryTopK.__init__).parameters
    assert set(parameters) == {"self", "oracle", "k", "config"}
    config_parameters = inspect.signature(CoherentUnknownBoundaryTopKConfig).parameters
    forbidden_parameters = ("answer", "boundary", "schedule", "truth", "mean")
    assert not any(
        token in name
        for name in config_parameters
        for token in forbidden_parameters
    )
    source = inspect.getsource(topk_module)
    assert "oracle.means" not in source
    assert "oracle.amplitudes" not in source
    assert "oracle.__blocks" not in source
    assert "oracle._CanonicalRyStatevectorOracle__blocks" not in source

    class DuckOracle:
        n_arms = 2

    with pytest.raises(TypeError, match="CanonicalRyStatevectorOracle"):
        CoherentUnknownBoundaryTopK(DuckOracle(), 1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phase_qubits": 0},
        {"phase_qubits": 6},
        {"cleanup_tolerance": 0.0},
        {"max_statevector_dimension": 0},
    ],
)
def test_config_rejects_invalid_precision_and_resource_limits(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        CoherentUnknownBoundaryTopKConfig(**kwargs)  # type: ignore[arg-type]


def test_constructor_rejects_invalid_k_arm_count_and_oversized_state() -> None:
    with pytest.raises(ValueError, match="k must satisfy"):
        CoherentUnknownBoundaryTopK(_on_grid_oracle(), 2)
    with pytest.raises(ValueError, match="2 <= n <= 3"):
        CoherentUnknownBoundaryTopK(CanonicalRyStatevectorOracle([0.5]), 1)
    with pytest.raises(ValueError, match="statevector"):
        CoherentUnknownBoundaryTopK(
            _on_grid_oracle(),
            1,
            config=CoherentUnknownBoundaryTopKConfig(
                phase_qubits=3,
                max_statevector_dimension=1,
            ),
        )
