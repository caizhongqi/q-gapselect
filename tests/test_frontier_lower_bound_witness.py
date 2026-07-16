from __future__ import annotations

import math
from dataclasses import replace

import pytest

import qgapselect.frontier_lower_bound_witness as frontier_module
from qgapselect.coherent import CanonicalRyStatevectorOracle
from qgapselect.coherent_adaptive_stopping_history import (
    TinyCoherentStoppingHistoryConfig,
    run_tiny_coherent_adaptive_stopping_history,
)
from qgapselect.coherent_unknown_boundary_topk import (
    CoherentUnknownBoundaryTopKConfig,
    run_coherent_unknown_boundary_topk,
)
from qgapselect.frontier_lower_bound_witness import (
    COMPOSITION_WITNESS_TYPE,
    JOHNSON_WITNESS_TYPE,
    PAIR_WITNESS_TYPE,
    composition_falsification_certificate,
    johnson_adversary_certificate,
    paired_rotation_hybrid_certificate,
)


def _comparison_results(means: tuple[float, float] = (1.0, 0.0)):
    candidate = run_tiny_coherent_adaptive_stopping_history(
        CanonicalRyStatevectorOracle(means),
        config=TinyCoherentStoppingHistoryConfig(
            cleanup_tolerance=1e-10,
            max_statevector_dimension=600_000,
        ),
    )
    baseline = run_coherent_unknown_boundary_topk(
        CanonicalRyStatevectorOracle(means),
        1,
        config=CoherentUnknownBoundaryTopKConfig(
            phase_qubits=2,
            cleanup_tolerance=1e-10,
            max_statevector_dimension=600_000,
        ),
    )
    return candidate, baseline


def _composition_certificate(means: tuple[float, float] = (1.0, 0.0)):
    return composition_falsification_certificate(
        witness_id="finite_diagnostic",
        means=means,
        k=1,
        phase_qubits=2,
        cleanup_tolerance=1e-10,
        max_statevector_dimension=600_000,
    )


def test_canonical_rotation_hybrid_uses_actual_repository_parameterization() -> None:
    certificate = paired_rotation_hybrid_certificate(
        witness_id="pair_parameterization",
        n=2,
        k=1,
        low_angle=0.0,
        high_angle=math.pi / 2.0,
        error_probability=0.1,
    )

    expected = math.sqrt(2.0)
    assert certificate.witness_type == PAIR_WITNESS_TYPE
    assert certificate.forward_difference_norm_numeric == pytest.approx(expected)
    assert certificate.forward_difference_norm_analytic == pytest.approx(expected)
    assert certificate.inverse_difference_norm_numeric == pytest.approx(expected)
    assert certificate.controlled_difference_norm_numeric == pytest.approx(expected)
    assert certificate.norm_formula_residual < 1e-12
    assert certificate.forward_difference_norm_analytic == pytest.approx(
        2.0 * math.sin((math.pi / 2.0) / 2.0)
    )
    # 2*sin(theta-phi) would be the wrong convention for B_theta=R_y(2 theta).
    assert certificate.forward_difference_norm_analytic != pytest.approx(
        2.0 * math.sin(math.pi / 2.0)
    )
    assert certificate.computed_quantity == pytest.approx(0.8 / math.sqrt(2.0))
    assert certificate.integer_query_lower_bound == 1
    assert certificate.verification_passed
    assert certificate.instance_x_top_k != certificate.instance_y_top_k
    assert not certificate.composition_match
    assert not certificate.composition_kill_flag
    assert not certificate.matching_lower_bound_claimable


def test_pair_hybrid_small_gap_scales_only_as_local_inverse_gap() -> None:
    coarse = paired_rotation_hybrid_certificate(
        witness_id="coarse",
        n=6,
        k=2,
        low_angle=0.4,
        high_angle=0.6,
        error_probability=0.05,
    )
    fine = paired_rotation_hybrid_certificate(
        witness_id="fine",
        n=6,
        k=2,
        low_angle=0.4,
        high_angle=0.5,
        error_probability=0.05,
    )

    assert fine.computed_quantity > coarse.computed_quantity
    assert fine.blockers == coarse.blockers
    assert "two-input" in fine.explicit_non_theorem_boundary
    assert "direct-sum" in fine.explicit_non_theorem_boundary


@pytest.mark.parametrize(
    ("n", "k", "expected"),
    ((2, 1, 1.0), (4, 2, 2.0), (6, 2, math.sqrt(8.0))),
)
def test_johnson_adversary_matrix_has_expected_small_n_objective(
    n: int,
    k: int,
    expected: float,
) -> None:
    certificate = johnson_adversary_certificate(
        witness_id=f"johnson_{n}_{k}",
        n=n,
        k=k,
    )

    assert certificate.witness_type == JOHNSON_WITNESS_TYPE
    assert certificate.numerator_spectral_norm == pytest.approx(k * (n - k))
    assert certificate.maximum_filtered_spectral_norm == pytest.approx(expected)
    assert certificate.computed_quantity == pytest.approx(expected)
    assert certificate.objective_expected == pytest.approx(expected)
    assert all(value == pytest.approx(expected) for value in certificate.filtered_spectral_norms)
    assert certificate.maximum_spectral_residual < 1e-10
    assert certificate.symmetric
    assert certificate.zero_diagonal
    assert certificate.support_respects_distinct_outputs
    assert certificate.verification_passed
    assert "discrete" in certificate.explicit_non_theorem_boundary
    assert not certificate.composition_kill_flag


def test_johnson_witness_refuses_unbounded_matrix_materialization() -> None:
    with pytest.raises(ValueError, match="exceeds maximum_input_count"):
        johnson_adversary_certificate(
            witness_id="too_large",
            n=12,
            k=6,
            maximum_input_count=100,
        )


def test_exact_grid_comparison_is_only_a_same_oracle_model_diagnostic() -> None:
    certificate = _composition_certificate()

    assert certificate.witness_type == COMPOSITION_WITNESS_TYPE
    assert certificate.baseline_query_count == 28
    assert certificate.candidate_query_count == 176
    assert certificate.computed_quantity == pytest.approx(28 / 176)
    assert certificate.same_oracle_model_verified
    assert certificate.same_fixture_harness_verified
    assert certificate.distinct_oracle_instances_verified
    assert certificate.distinct_implementation_classes_verified
    assert not certificate.same_public_algorithm_interface_verified
    assert not certificate.same_certified_output_contract_verified
    assert certificate.finite_fixture_query_dominance_verified
    assert certificate.candidate_complete_output_verified
    assert certificate.candidate_cleanup_verified
    assert certificate.candidate_truth_match_verified
    assert not certificate.candidate_certificate_issued
    assert certificate.candidate_query_ledger_reconciled
    assert certificate.baseline_complete_output_verified
    assert certificate.baseline_cleanup_verified
    assert certificate.baseline_truth_match_verified
    assert not certificate.baseline_certificate_issued
    assert certificate.baseline_query_ledger_reconciled
    assert not certificate.composition_match
    assert not certificate.composition_kill_flag
    assert certificate.verification_passed
    assert not certificate.registered_published_baseline_fidelity_verified
    assert not certificate.global_composition_frontier_closed
    assert not certificate.quantum_advantage_claimable
    assert "finite diagnostic" in certificate.explicit_non_theorem_boundary
    assert "registered_public_algorithm_interface_not_shared" in certificate.blockers
    assert "delta_sound_output_contract_not_implemented" in certificate.blockers


@pytest.mark.parametrize("means", ((0.5,), (0.0, 0.0), (-0.1, 0.5), (0.2, 1.1)))
def test_finite_diagnostic_rejects_invalid_or_nonstrict_fixtures(
    means: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        composition_falsification_certificate(
            witness_id="invalid_fixture",
            means=means,
            k=1,
            phase_qubits=2,
            cleanup_tolerance=1e-10,
            max_statevector_dimension=600_000,
        )


def test_mutating_either_ledger_fails_the_finite_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, baseline = _comparison_results()
    forged_candidate_ledger = replace(
        candidate.resources.query_ledger,
        qram_assumed=True,
    )
    forged_candidate = replace(
        candidate,
        resources=replace(
            candidate.resources,
            query_ledger=forged_candidate_ledger,
        ),
    )
    monkeypatch.setattr(
        frontier_module,
        "run_tiny_coherent_adaptive_stopping_history",
        lambda oracle, config=None: forged_candidate,
    )
    certificate = _composition_certificate()

    assert not certificate.candidate_query_ledger_reconciled
    assert not certificate.finite_fixture_query_dominance_verified
    assert not certificate.composition_kill_flag

    monkeypatch.setattr(
        frontier_module,
        "run_tiny_coherent_adaptive_stopping_history",
        lambda oracle, config=None: candidate,
    )
    forged_baseline = replace(
        baseline,
        resources=replace(baseline.resources, qram_assumed=True),
    )
    monkeypatch.setattr(
        frontier_module,
        "run_coherent_unknown_boundary_topk",
        lambda oracle, k, config=None: forged_baseline,
    )
    certificate = _composition_certificate()

    assert not certificate.baseline_query_ledger_reconciled
    assert not certificate.finite_fixture_query_dominance_verified
    assert not certificate.composition_kill_flag


@pytest.mark.parametrize(
    "kwargs",
    (
        {"n": 1, "k": 1, "low_angle": 0.1, "high_angle": 0.2},
        {"n": 2, "k": 2, "low_angle": 0.1, "high_angle": 0.2},
        {"n": 2, "k": 1, "low_angle": 0.2, "high_angle": 0.1},
    ),
)
def test_pair_certificate_rejects_invalid_instances(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        paired_rotation_hybrid_certificate(
            witness_id="invalid",
            error_probability=0.1,
            **kwargs,  # type: ignore[arg-type]
        )
