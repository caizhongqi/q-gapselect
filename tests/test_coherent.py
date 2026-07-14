from __future__ import annotations

import math

import numpy as np
import pytest

from qgapselect.coherent import CanonicalRyStatevectorOracle


def _random_state(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    return state / np.linalg.norm(state)


def test_full_ry_block_acts_on_reward_zero_and_one() -> None:
    mean = 0.36
    oracle = CanonicalRyStatevectorOracle([mean])
    cosine = math.sqrt(1.0 - mean)
    sine = math.sqrt(mean)

    zero = np.asarray([1.0, 0.0], dtype=np.complex128)
    one = np.asarray([0.0, 1.0], dtype=np.complex128)
    assert np.allclose(oracle.apply(zero), [cosine, sine])
    assert np.allclose(oracle.apply(one), [-sine, cosine])
    assert oracle.query_snapshot().flat()["forward"] == 2


@pytest.mark.parametrize("n_arms", [1, 2, 3, 5])
def test_forward_inverse_is_unitary_on_arbitrary_superpositions(n_arms: int) -> None:
    means = np.linspace(0.05, 0.95, n_arms)
    oracle = CanonicalRyStatevectorOracle(means)
    state = _random_state(oracle.statevector_dimension, 100 + n_arms)
    output = oracle.apply(state, tag="unitarity")
    recovered = oracle.apply(output, inverse=True, tag="unitarity")

    assert np.isclose(np.linalg.norm(output), 1.0)
    assert np.allclose(recovered, state, atol=1e-12)
    snapshot = oracle.query_snapshot()
    assert snapshot.counts["forward"] == 1
    assert snapshot.counts["inverse"] == 1
    assert snapshot.by_tag["unitarity"] == {"forward": 1, "inverse": 1}


def test_invalid_padded_index_is_fixed_by_identity() -> None:
    oracle = CanonicalRyStatevectorOracle([0.1, 0.4, 0.9])
    state = np.zeros(oracle.statevector_dimension, dtype=np.complex128)
    # The fourth index is padding because index_dimension == 4.
    state[2 * 3 + 1] = 1.0
    assert np.array_equal(oracle.apply(state), state)


def test_controlled_call_acts_only_on_active_control_branch() -> None:
    oracle = CanonicalRyStatevectorOracle([0.25, 0.75])
    state = _random_state(2 * oracle.statevector_dimension, 77)
    view = state.reshape(2, oracle.index_dimension, 2)
    inactive_before = view[0].copy()

    output = oracle.apply(state, controlled=True, tag="controlled")
    output_view = output.reshape(2, oracle.index_dimension, 2)
    assert np.allclose(output_view[0], inactive_before)
    recovered = oracle.apply(
        output,
        controlled=True,
        inverse=True,
        tag="controlled",
    )
    assert np.allclose(recovered, state, atol=1e-12)

    snapshot = oracle.query_snapshot()
    assert snapshot.counts["controlled_forward"] == 1
    assert snapshot.counts["controlled_inverse"] == 1
    assert snapshot.counts.get("forward", 0) == 0


def test_endpoint_reward_experiments_are_charged_not_read_from_means() -> None:
    oracle = CanonicalRyStatevectorOracle([0.0, 1.0], seed=4)
    assert oracle.reward_experiment(0, 7, tag="endpoint") == 0
    assert oracle.reward_experiment(1, 9, tag="endpoint") == 9
    snapshot = oracle.query_snapshot()
    assert snapshot.counts["forward"] == 16
    assert snapshot.by_tag["endpoint"]["forward"] == 16


def test_good_reflection_is_involutory_and_not_an_oracle_query() -> None:
    oracle = CanonicalRyStatevectorOracle([0.2, 0.8])
    state = _random_state(oracle.statevector_dimension, 3)
    reflected = oracle.reflect_good(state)
    recovered = oracle.reflect_good(reflected)
    assert np.allclose(recovered, state)
    assert oracle.query_snapshot().total == 0
    assert oracle.resource_snapshot().good_reflections == 2


@pytest.mark.parametrize(
    "means",
    [[], [-0.1], [1.1], [float("nan")], [float("inf")]],
)
def test_invalid_oracle_construction_is_rejected(means: list[float]) -> None:
    with pytest.raises(ValueError):
        CanonicalRyStatevectorOracle(means)


def test_state_shape_and_norm_are_validated() -> None:
    oracle = CanonicalRyStatevectorOracle([0.5, 0.5])
    with pytest.raises(ValueError, match="length"):
        oracle.apply(np.asarray([1.0, 0.0]))
    with pytest.raises(ValueError, match="normalized"):
        oracle.apply(np.zeros(oracle.statevector_dimension))
