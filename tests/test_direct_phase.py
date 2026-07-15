from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from qgapselect.coherent import CanonicalRyStatevectorOracle
from qgapselect.direct_phase import DirectAmplitudeThresholdFlag


def _random_state(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    return state / np.linalg.norm(state)


def test_embedded_oracle_is_unitary_with_spectator_and_control_registers() -> None:
    oracle = CanonicalRyStatevectorOracle([0.2, 0.8])
    shape = (3, 2, oracle.index_dimension, 2)
    state = _random_state(np.prod(shape), 101)
    output = oracle.apply_embedded(
        state,
        register_shape=shape,
        index_axis=2,
        reward_axis=3,
        control_axis=1,
        tag="embedded",
    )
    recovered = oracle.apply_embedded(
        output,
        register_shape=shape,
        index_axis=2,
        reward_axis=3,
        control_axis=1,
        inverse=True,
        tag="embedded",
    )

    assert np.isclose(np.linalg.norm(output), 1.0)
    assert np.allclose(recovered, state, atol=1e-12)
    snapshot = oracle.query_snapshot()
    assert snapshot.by_tag["embedded"] == {
        "controlled_forward": 1,
        "controlled_inverse": 1,
    }


def test_embedded_control_zero_basis_branch_is_fixed() -> None:
    oracle = CanonicalRyStatevectorOracle([0.36])
    shape = (2, 5, oracle.index_dimension, 2)
    state = np.zeros(shape, dtype=np.complex128)
    state[0, 3, 0, 1] = 1.0
    output = oracle.apply_embedded(
        state.reshape(-1),
        register_shape=shape,
        index_axis=2,
        reward_axis=3,
        control_axis=0,
    )
    assert np.array_equal(output, state.reshape(-1))
    assert oracle.query_snapshot().flat()["controlled_forward"] == 1


def test_embedded_oracle_matches_an_independent_dense_construction() -> None:
    means = (0.2, 0.8, 0.35)
    oracle = CanonicalRyStatevectorOracle(means)
    shape = (2, 3, oracle.index_dimension, 2)
    state = _random_state(int(np.prod(shape)), 73)

    actual = oracle.apply_embedded(
        state,
        register_shape=shape,
        index_axis=2,
        reward_axis=3,
        control_axis=0,
    ).reshape(shape)

    expected = state.copy().reshape(shape)
    for spectator in range(shape[1]):
        for arm, mean in enumerate(means):
            cosine = math.sqrt(1.0 - mean)
            sine = math.sqrt(mean)
            block = np.asarray(
                ((cosine, -sine), (sine, cosine)),
                dtype=np.complex128,
            )
            expected[1, spectator, arm, :] = (
                block @ expected[1, spectator, arm, :]
            )

    assert np.allclose(actual, expected, atol=1e-12)
    assert oracle.query_snapshot().coherent_total == 1


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"register_shape": (2, True, 2), "index_axis": 1, "reward_axis": 2}, TypeError),
        ({"register_shape": (2, 2), "index_axis": True, "reward_axis": 1}, TypeError),
        ({"register_shape": (2, 2), "index_axis": 0, "reward_axis": 0}, ValueError),
        (
            {
                "register_shape": (2, 2, 2),
                "index_axis": 1,
                "reward_axis": 2,
                "control_axis": 1,
            },
            ValueError,
        ),
    ],
)
def test_embedded_oracle_rejects_invalid_register_descriptions(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    oracle = CanonicalRyStatevectorOracle([0.5, 0.5])
    dimension = int(np.prod(kwargs["register_shape"]))
    state = np.zeros(dimension, dtype=np.complex128)
    state[0] = 1.0
    with pytest.raises(error):
        oracle.apply_embedded(state, **kwargs)


def test_direct_flag_properties_and_zero_workspace_preparation() -> None:
    oracle = CanonicalRyStatevectorOracle([0.9, 0.2, 0.1])
    flag = DirectAmplitudeThresholdFlag(oracle, 0.5, phase_qubits=3)
    state = flag.initial_state((0, 2))
    view = state.reshape(flag.phase_bins, flag.index_dimension, 2)

    assert flag.valid_indices == (0, 1, 2)
    assert flag.workspace_dimension == 16
    assert flag.statevector_dimension == 8 * 4 * 2
    assert np.allclose(view[0, (0, 2), 0], 1 / np.sqrt(2))
    assert np.count_nonzero(view) == 2


def test_above_and_below_masks_are_complementary_except_exclusions() -> None:
    oracle = CanonicalRyStatevectorOracle([0.8, 0.4, 0.1])
    above = DirectAmplitudeThresholdFlag(
        oracle,
        0.5,
        phase_qubits=3,
        relation="above",
        excluded_indices=(1,),
    )
    below = DirectAmplitudeThresholdFlag(
        oracle,
        0.5,
        phase_qubits=3,
        relation="below",
        excluded_indices=(1,),
    )

    above_mask = above.acceptance_mask
    below_mask = below.acceptance_mask
    for index in (0, 2):
        assert np.all(np.logical_xor(above_mask[:, index, :], below_mask[:, index, :]))
    assert not np.any(above_mask[:, 1, :])
    assert not np.any(below_mask[:, 1, :])
    assert not np.any(above_mask[:, 3, :])  # padded index
    assert not above_mask.flags.writeable


def test_qpe_threshold_probability_separates_high_and_low_unknown_arms() -> None:
    oracle = CanonicalRyStatevectorOracle([0.95, 0.05])
    flag = DirectAmplitudeThresholdFlag(oracle, 0.5, phase_qubits=4)

    high = flag.acceptance_probability(0, tag="diagnostic")
    low = flag.acceptance_probability(1, tag="diagnostic")
    assert high > 0.98
    assert low < 0.02
    # Each diagnostic is C_tau: 1 + 2*(L-1) = 31 charged queries.
    snapshot = oracle.query_snapshot()
    assert snapshot.by_tag["diagnostic"] == {
        "forward": 2,
        "controlled_inverse": 30,
        "controlled_forward": 30,
    }
    assert snapshot.coherent_total == 62


def test_exact_phase_grid_has_the_expected_mirrored_qpe_peaks() -> None:
    phase_bins = 8
    expected_support = ((0,), (1, 7), (2, 6), (3, 5), (4,))
    for grid_index, support in enumerate(expected_support):
        angle = math.pi * grid_index / phase_bins
        oracle = CanonicalRyStatevectorOracle([math.sin(angle) ** 2])
        flag = DirectAmplitudeThresholdFlag(oracle, 0.5, phase_qubits=3)
        computed = flag.compute(flag.initial_state((0,)))
        marginal = np.sum(
            np.abs(computed.reshape(phase_bins, 1, 2)) ** 2,
            axis=(1, 2),
        )

        assert tuple(np.flatnonzero(marginal > 1e-10)) == support
        assert float(np.sum(marginal[list(support)])) == pytest.approx(1.0)


def test_phase_predicate_is_mirror_symmetric_and_api_has_no_marked_set() -> None:
    oracle = CanonicalRyStatevectorOracle([0.9, 0.1])
    flag = DirectAmplitudeThresholdFlag(oracle, 0.4, phase_qubits=4)

    assert all(
        ((-phase_bin) % flag.phase_bins in flag.marked_phase_bins)
        for phase_bin in flag.marked_phase_bins
    )
    assert "marked_indices" not in inspect.signature(
        DirectAmplitudeThresholdFlag
    ).parameters


def test_threshold_endpoints_and_padded_indices_have_explicit_semantics() -> None:
    oracle = CanonicalRyStatevectorOracle([0.4, 0.6, 0.8])
    above_zero = DirectAmplitudeThresholdFlag(oracle, 0.0, phase_qubits=2)
    below_zero = DirectAmplitudeThresholdFlag(
        oracle,
        0.0,
        phase_qubits=2,
        relation="below",
    )
    above_one = DirectAmplitudeThresholdFlag(oracle, 1.0, phase_qubits=2)

    assert above_zero.marked_phase_bins == (0, 1, 2, 3)
    assert below_zero.marked_phase_bins == ()
    assert above_one.marked_phase_bins == (2,)
    assert not np.any(above_zero.acceptance_mask[:, 3, :])


def test_superposition_acceptance_is_the_basis_probability_mixture() -> None:
    oracle = CanonicalRyStatevectorOracle([0.95, 0.05])
    flag = DirectAmplitudeThresholdFlag(oracle, 0.5, phase_qubits=4)
    high = flag.acceptance_probability(0)
    low = flag.acceptance_probability(1)
    result = flag.reflect(flag.initial_state((0, 1)))
    assert result.acceptance_probability == pytest.approx((high + low) / 2, abs=1e-12)


def test_full_workspace_reflection_is_unitary_and_involutory() -> None:
    oracle = CanonicalRyStatevectorOracle([0.82, 0.17])
    flag = DirectAmplitudeThresholdFlag(oracle, 0.5, phase_qubits=2)
    state = _random_state(flag.statevector_dimension, 44)
    once = flag.apply_reflection(state, tag="reflection")
    twice = flag.apply_reflection(once, tag="reflection")

    assert np.isclose(np.linalg.norm(once), 1.0)
    assert not np.allclose(once, state, atol=1e-8)
    assert np.allclose(twice, state, atol=1e-11)
    # One reflection costs 4L-2 = 14 queries for L=4.
    snapshot = oracle.query_snapshot()
    assert snapshot.by_tag["reflection"] == {
        "forward": 2,
        "controlled_inverse": 12,
        "controlled_forward": 12,
        "inverse": 2,
    }
    assert snapshot.coherent_total == 28


def test_compute_inverse_compute_roundtrip_and_exact_query_budget() -> None:
    oracle = CanonicalRyStatevectorOracle([0.9, 0.1])
    flag = DirectAmplitudeThresholdFlag(oracle, 0.5, phase_qubits=3)
    state = _random_state(flag.statevector_dimension, 8)
    computed = flag.compute(state, tag="decode")
    recovered = flag.inverse_compute(computed, tag="decode")

    assert np.allclose(recovered, state, atol=1e-11)
    # C_tau and C_tau^dagger each cost 2L-1 = 15 queries.
    snapshot = oracle.query_snapshot()
    assert snapshot.by_tag["decode"] == {
        "forward": 1,
        "controlled_inverse": 14,
        "controlled_forward": 14,
        "inverse": 1,
    }
    assert snapshot.coherent_total == 30


def test_reflection_reports_finite_qpe_leakage_without_discarding_it() -> None:
    oracle = CanonicalRyStatevectorOracle([0.37])
    flag = DirectAmplitudeThresholdFlag(oracle, 0.5, phase_qubits=3)
    result = flag.reflect(flag.initial_state((0,)))

    assert result.resources.comparator_residual < 1e-14
    assert result.resources.phase_ancilla_residual > 0.0
    assert result.resources.zero_workspace_residual >= (
        result.resources.phase_ancilla_residual
    )
    assert result.resources.query_counts["coherent_total"] == 30
    assert result.state.shape == (flag.statevector_dimension,)


def test_verify_index_reruns_and_charges_every_qpe_shot() -> None:
    oracle = CanonicalRyStatevectorOracle([1.0, 0.0])
    flag = DirectAmplitudeThresholdFlag(oracle, 0.5, phase_qubits=2)
    accepted = flag.verify_index(0, shots=32, confidence=0.05, seed=4)
    rejected = flag.verify_index(1, shots=32, confidence=0.05, seed=5)

    assert accepted.accepted and accepted.successes == 32
    assert rejected.rejected and rejected.successes == 0
    assert accepted.resources.oracle_queries == 32 * 7
    assert rejected.resources.oracle_queries == 32 * 7
    assert oracle.query_snapshot().coherent_total == 64 * 7


def test_joint_accept_index_sampling_respects_exclusion() -> None:
    oracle = CanonicalRyStatevectorOracle([1.0, 1.0])
    flag = DirectAmplitudeThresholdFlag(
        oracle,
        0.5,
        phase_qubits=2,
        excluded_indices=(0,),
    )
    computed = flag.compute(flag.initial_state())
    assert {flag.sample_accept_index(computed, seed=seed) for seed in range(10)} <= {
        None,
        1,
    }
    assert any(flag.sample_accept_index(computed, seed=seed) == 1 for seed in range(10))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"threshold": True}, TypeError),
        ({"threshold": -0.1}, ValueError),
        ({"threshold": 0.5, "phase_qubits": 2.5}, TypeError),
        ({"threshold": 0.5, "phase_qubits": 0}, ValueError),
        ({"threshold": 0.5, "relation": 1}, TypeError),
        ({"threshold": 0.5, "relation": "equal"}, ValueError),
        ({"threshold": 0.5, "excluded_indices": (True,)}, TypeError),
        ({"threshold": 0.5, "excluded_indices": (0, 0)}, ValueError),
        ({"threshold": 0.5, "excluded_indices": (2,)}, IndexError),
    ],
)
def test_invalid_direct_flag_construction_is_rejected(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    oracle = CanonicalRyStatevectorOracle([0.9, 0.1])
    with pytest.raises(error):
        DirectAmplitudeThresholdFlag(oracle, **kwargs)


def test_invalid_states_indices_and_shot_types_are_rejected() -> None:
    oracle = CanonicalRyStatevectorOracle([0.9, 0.1])
    flag = DirectAmplitudeThresholdFlag(oracle, 0.5, phase_qubits=2)
    with pytest.raises(ValueError, match="length"):
        flag.apply_reflection(np.asarray([1.0, 0.0]))
    with pytest.raises(ValueError, match="normalized"):
        flag.prepare_zero_workspace(np.zeros(flag.index_dimension))
    with pytest.raises(TypeError, match="integer"):
        flag.verify_index(0, shots=3.5)
    with pytest.raises(TypeError, match="integer"):
        flag.acceptance_probability(True)
