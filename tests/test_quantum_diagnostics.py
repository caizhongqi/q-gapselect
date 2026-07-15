from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from qgapselect.coherent import CanonicalRyStatevectorOracle
from qgapselect.direct_phase import DirectAmplitudeThresholdFlag
from qgapselect.quantum_diagnostics import (
    INVALID_INDEX_METHOD,
    VALID_FULL_METHOD,
    QPEAcceptancePoint,
    joint_acceptance_probability,
    make_threshold_angular_gap_instance,
    run_diffusion_ablation,
    run_phase_grid_sweep,
    run_qpe_acceptance_sweep,
)


def _random_state(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    return state / np.linalg.norm(state)


def _mean_classification_error(points: tuple[QPEAcceptancePoint, ...]) -> float:
    losses = [
        1.0 - point.joint_acceptance_probability
        if point.truth_accept
        else point.joint_acceptance_probability
        for point in points
    ]
    return float(np.mean(losses))


def test_angular_gap_instance_is_reproducible_and_truth_is_evaluation_only() -> None:
    first = make_threshold_angular_gap_instance(
        n_below=3,
        n_above=2,
        threshold=0.5,
        angular_gap=0.12,
        seed=71,
    )
    second = make_threshold_angular_gap_instance(
        n_below=3,
        n_above=2,
        threshold=0.5,
        angular_gap=0.12,
        seed=71,
    )

    assert first == second
    assert first.truth_usage == "experiment_evaluation_only_never_passed_to_search"
    assert len(first.truth_below) == 3
    assert len(first.truth_above) == 2
    assert set(first.truth_below).isdisjoint(first.truth_above)
    for index in first.truth_below:
        assert first.threshold_angle - first.angles[index] == pytest.approx(0.12)
    for index in first.truth_above:
        assert first.angles[index] - first.threshold_angle == pytest.approx(0.12)

    # The diagnostic consumes means, not the generator's truth partitions.
    parameters = inspect.signature(run_qpe_acceptance_sweep).parameters
    assert "truth_above" not in parameters
    assert "truth_below" not in parameters


def test_full_threshold_reflection_matches_public_compute_kickback_uncompute() -> None:
    means = (0.17, 0.74)
    direct_oracle = CanonicalRyStatevectorOracle(means)
    direct_flag = DirectAmplitudeThresholdFlag(
        direct_oracle,
        0.43,
        phase_qubits=3,
    )
    state = _random_state(direct_flag.statevector_dimension, 19)
    direct = direct_flag.reflect(state)

    manual_oracle = CanonicalRyStatevectorOracle(means)
    manual_flag = DirectAmplitudeThresholdFlag(
        manual_oracle,
        0.43,
        phase_qubits=3,
    )
    computed = manual_flag.compute(state)
    marked = computed.reshape(
        manual_flag.phase_bins,
        manual_flag.index_dimension,
        2,
    ).copy()
    marked[manual_flag.acceptance_mask] *= -1.0
    recovered = manual_flag.inverse_compute(marked.reshape(-1))

    assert np.allclose(direct.state, recovered, atol=1e-11)
    assert np.linalg.norm(direct.state) == pytest.approx(1.0)
    # Reflection C^dagger P C: 2 * (2L - 1) for L=8.
    assert direct.resources.oracle_queries == 30
    assert direct.resources.comparator_residual < 1e-12


def test_public_joint_acceptance_matches_exact_grid_and_rejects_bad_state() -> None:
    phase_bins = 8
    angle = 2.0 * math.pi / phase_bins
    oracle = CanonicalRyStatevectorOracle([math.sin(angle) ** 2])
    flag = DirectAmplitudeThresholdFlag(oracle, 0.4, phase_qubits=3)
    computed = flag.compute(flag.initial_state((0,)))

    assert joint_acceptance_probability(flag, computed) == pytest.approx(1.0)
    assert joint_acceptance_probability(flag, -computed) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="length"):
        joint_acceptance_probability(flag, np.asarray([1.0 + 0.0j]))
    with pytest.raises(ValueError, match="normalized"):
        joint_acceptance_probability(flag, np.zeros(flag.statevector_dimension))


def test_phase_grid_sweep_has_mirrored_peaks_and_exact_resource_ledger() -> None:
    sweep = run_phase_grid_sweep(
        phase_qubits=3,
        threshold=0.4,
        relation="above",
        seed=12,
    )

    assert len(sweep.points) == 5
    assert [point.mirrored_peak_bins for point in sweep.points] == [
        (0,),
        (1, 7),
        (2, 6),
        (3, 5),
        (4,),
    ]
    for point in sweep.points:
        expected = 1.0 if point.truth_accept else 0.0
        assert point.joint_acceptance_probability == pytest.approx(expected, abs=1e-11)
        assert point.resources.oracle_queries == 15
        assert point.resources.qpe_calls == 1
        assert point.resources.controlled_grover_iterations == 7
        # Compute-only QPE does not allocate a comparator flag, but this NumPy
        # backend materializes the dense M x M QFT matrix beside input/output.
        assert point.resources.comparator_expanded_statevector_dimension == 0
        assert point.resources.dense_qft_matrix_dimension == 64
        assert point.resources.peak_statevector_dimension == 96
        assert point.resources.estimated_peak_complex_amplitudes == 96
    assert sweep.resources.oracle_queries == 5 * 15


def test_finite_phase_diffusion_ablation_marks_index_only_branch_invalid() -> None:
    kwargs = {
        "means": (0.62, 0.18, 0.81),
        "threshold": 0.5,
        "phase_qubits": 3,
        "grover_iterations": 1,
        "seed": 4,
    }
    first = run_diffusion_ablation(**kwargs)
    second = run_diffusion_ablation(**kwargs)

    assert first == second
    assert first.finite_phase_leakage_detected
    assert first.state_distance_up_to_global_phase > 0.1
    assert first.full_workspace.method == VALID_FULL_METHOD
    assert first.full_workspace.algorithmically_valid
    assert first.full_workspace.output_eligible
    assert first.full_workspace.warning is None
    assert first.invalid_index_only.method == INVALID_INDEX_METHOD
    assert not first.invalid_index_only.algorithmically_valid
    assert not first.invalid_index_only.output_eligible
    assert first.invalid_index_only.warning is not None
    assert "INVALID" in first.invalid_index_only.warning
    assert first.full_workspace.joint_acceptance_probability != pytest.approx(
        first.invalid_index_only.joint_acceptance_probability,
        abs=1e-4,
    )


@pytest.mark.parametrize("iterations", [0, 1, 2])
def test_diffusion_ablation_charges_exact_executed_queries(iterations: int) -> None:
    phase_qubits = 3
    phase_bins = 1 << phase_qubits
    result = run_diffusion_ablation(
        (0.62, 0.18, 0.81),
        threshold=0.5,
        phase_qubits=phase_qubits,
        grover_iterations=iterations,
        seed=9,
    )
    expected_queries = iterations * (4 * phase_bins - 2) + (2 * phase_bins - 1)

    assert result.full_workspace.resources.oracle_queries == expected_queries
    assert result.invalid_index_only.resources.oracle_queries == expected_queries
    assert result.full_workspace.resources.statevector_dimension == phase_bins * 4 * 2
    assert result.invalid_index_only.resources.statevector_dimension == phase_bins * 4 * 2
    if iterations == 0:
        assert result.state_distance_up_to_global_phase == pytest.approx(0.0)
        assert not result.finite_phase_leakage_detected
        assert (
            result.full_workspace.resources.comparator_expanded_statevector_dimension
            == 0
        )
    else:
        assert (
            result.full_workspace.resources.comparator_expanded_statevector_dimension
            == 2 * result.full_workspace.resources.statevector_dimension
        )
    # M=8, padded index dimension=4, S=64.  Both QFT and comparator paths
    # have a conservative peak of 192 live complex entries here.
    assert result.full_workspace.resources.peak_statevector_dimension == 192


def test_selected_precision_and_gap_endpoints_show_expected_resolution_trend() -> None:
    near = make_threshold_angular_gap_instance(
        n_below=2,
        n_above=2,
        angular_gap=0.03,
        seed=2,
    )
    far = make_threshold_angular_gap_instance(
        n_below=2,
        n_above=2,
        angular_gap=0.2,
        seed=2,
    )
    near_sweep = run_qpe_acceptance_sweep(
        near.means,
        threshold=near.threshold,
        phase_qubits=(2, 5),
        seed=2,
    )
    far_sweep = run_qpe_acceptance_sweep(
        far.means,
        threshold=far.threshold,
        phase_qubits=(2, 5),
        seed=2,
    )

    near_high_precision = tuple(
        point for point in near_sweep.points if point.phase_qubits == 5
    )
    far_low_precision = tuple(
        point for point in far_sweep.points if point.phase_qubits == 2
    )
    far_high_precision = tuple(
        point for point in far_sweep.points if point.phase_qubits == 5
    )

    # These are calibrated endpoints, not a claim of pointwise monotonicity:
    # finite-QPE sidelobes can make intermediate precisions non-monotone.
    assert _mean_classification_error(far_high_precision) < (
        _mean_classification_error(near_high_precision)
    )
    assert _mean_classification_error(far_high_precision) < (
        _mean_classification_error(far_low_precision)
    )
    assert near_sweep.resources.oracle_queries == 4 * (7 + 63)
    assert far_sweep.resources.oracle_queries == 4 * (7 + 63)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_below": 0, "n_above": 0},
        {"n_below": 1, "n_above": 1, "angular_gap": 0.9},
    ],
)
def test_angular_gap_generator_rejects_invalid_instances(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        make_threshold_angular_gap_instance(**kwargs)


def test_diffusion_ablation_rejects_negative_iteration_count() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        run_diffusion_ablation(
            (0.2, 0.8),
            threshold=0.5,
            phase_qubits=3,
            grover_iterations=-1,
        )
