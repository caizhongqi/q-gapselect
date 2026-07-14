from __future__ import annotations

import math

import numpy as np
import pytest

from qgapselect.contracts import OracleModel
from qgapselect.natural_oracles import (
    NaturalArmDistribution,
    NaturalPurificationStatevectorOracle,
)
from qgapselect.oracles import QueryKind


def _oracle() -> NaturalPurificationStatevectorOracle:
    return NaturalPurificationStatevectorOracle(
        (
            NaturalArmDistribution.from_sequences((0.75, 0.25), (0, 1)),
            NaturalArmDistribution.from_sequences((0.2, 0.3, 0.5), (0, 1, 1)),
            NaturalArmDistribution.from_sequences((1.0,), (0,)),
        ),
        seed=7,
    )


def test_natural_sampler_prepares_work_garbage_and_reward_amplitudes() -> None:
    oracle = _oracle()
    prepared = oracle.apply(oracle.index_superposition())
    shaped = prepared.reshape(
        oracle.index_dimension,
        oracle.workspace_dimension,
        2,
    )

    assert oracle.contract.model is OracleModel.NATURAL_PURIFICATION
    assert np.sum(np.abs(shaped[0, :, 1]) ** 2) == pytest.approx(0.25 / 3.0)
    assert np.sum(np.abs(shaped[1, :, 1]) ** 2) == pytest.approx(0.8 / 3.0)
    assert np.sum(np.abs(shaped[2, :, 1]) ** 2) == pytest.approx(0.0)
    assert np.linalg.norm(shaped[3]) == pytest.approx(0.0)
    assert np.linalg.norm(prepared) == pytest.approx(1.0)


def test_forward_inverse_roundtrip_on_arbitrary_superposition() -> None:
    oracle = _oracle()
    rng = np.random.default_rng(20260714)
    state = rng.normal(size=oracle.statevector_dimension) + 1j * rng.normal(
        size=oracle.statevector_dimension
    )
    state = np.asarray(state / np.linalg.norm(state), dtype=np.complex128)

    restored = oracle.apply(oracle.apply(state), inverse=True)

    assert np.linalg.norm(restored - state) < 1e-12
    snapshot = oracle.query_snapshot()
    assert snapshot.counts[QueryKind.FORWARD.value] == 1
    assert snapshot.counts[QueryKind.INVERSE.value] == 1


def test_controlled_sampler_has_real_control_semantics() -> None:
    oracle = _oracle()
    inactive = oracle.index_superposition(controlled=True, active_control=False)
    active = oracle.index_superposition(controlled=True, active_control=True)

    inactive_after = oracle.apply(inactive, controlled=True)
    active_after = oracle.apply(active, controlled=True)

    assert np.array_equal(inactive_after, inactive)
    assert not np.allclose(active_after, active)
    restored = oracle.apply(active_after, inverse=True, controlled=True)
    assert np.linalg.norm(restored - active) < 1e-12
    assert oracle.query_snapshot().counts[QueryKind.CONTROLLED_FORWARD.value] == 2
    assert oracle.query_snapshot().counts[QueryKind.CONTROLLED_INVERSE.value] == 1


def test_good_reflection_is_involutory_and_not_an_oracle_query() -> None:
    oracle = _oracle()
    state = oracle.apply(oracle.index_superposition())
    reflected = oracle.reflect_good(state)
    restored = oracle.reflect_good(reflected)

    assert np.linalg.norm(restored - state) < 1e-12
    assert oracle.resource_snapshot().good_reflections == 2
    assert oracle.query_snapshot().coherent_total == 1


def test_reward_measurement_obeys_prepared_probability() -> None:
    oracle = NaturalPurificationStatevectorOracle(
        (NaturalArmDistribution.from_sequences((1.0,), (1,)),),
        seed=1,
    )
    prepared = oracle.apply(oracle.index_superposition())

    assert oracle.measure_reward(prepared) == 1


def test_distribution_and_state_validation() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        NaturalArmDistribution.from_sequences((0.2, 0.2), (0, 1))
    with pytest.raises(ValueError, match="equal length"):
        NaturalArmDistribution.from_sequences((1.0,), (0, 1))
    oracle = _oracle()
    with pytest.raises(ValueError, match="statevector"):
        oracle.apply(np.zeros(oracle.statevector_dimension, dtype=np.complex128))
    with pytest.raises(ValueError, match="length"):
        oracle.apply(np.array([1.0 + 0.0j]))
    with pytest.raises(IndexError):
        oracle.index_superposition((oracle.n_arms,))


def test_householder_completion_is_a_full_unitary_not_only_a_first_column() -> None:
    oracle = _oracle()
    dimension = oracle.statevector_dimension
    columns = []
    for basis in range(dimension):
        state = np.zeros(dimension, dtype=np.complex128)
        state[basis] = 1.0
        columns.append(oracle.apply(state))
    unitary = np.column_stack(columns)

    assert np.linalg.norm(unitary.conj().T @ unitary - np.eye(dimension)) < 1e-11
    assert oracle.query_snapshot().coherent_total == dimension
    assert oracle.contract.workspace_qubits == int(
        math.log2(oracle.workspace_dimension)
    )
    assert not hasattr(oracle, "means")
    assert not hasattr(oracle, "probabilities")


def test_natural_oracle_requires_boolean_control_flags() -> None:
    oracle = NaturalPurificationStatevectorOracle(
        (NaturalArmDistribution.from_sequences((1.0,), (0,)),)
    )
    with pytest.raises(TypeError, match="controlled must be bool"):
        oracle.zero_state(controlled=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="controlled and active_control must be bool"):
        oracle.index_superposition(active_control=1)  # type: ignore[arg-type]
