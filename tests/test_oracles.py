from __future__ import annotations

import math

import pytest

from qgapselect.diagnostics import (
    amplified_success_probability,
    controlled_reward_rotation_matrix,
    reward_rotation_matrix,
)
from qgapselect.oracles import (
    CanonicalBernoulliOracleSimulator,
    QueryKind,
    QueryLedger,
)


def _transpose(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(matrix[row][column] for row in range(len(matrix)))
        for column in range(len(matrix))
    )


@pytest.mark.parametrize("mean", (0.0, 0.125, 0.5, 0.9, 1.0))
def test_reward_rotation_is_unitary_and_encodes_the_bernoulli_mean(mean: float) -> None:
    oracle = CanonicalBernoulliOracleSimulator((mean,))
    forward = reward_rotation_matrix(mean)
    inverse = reward_rotation_matrix(mean, inverse=True)

    assert inverse == _transpose(forward)
    assert sum(entry**2 for entry in (forward[0][0], forward[1][0])) == pytest.approx(1.0)
    assert forward[1][0] ** 2 == pytest.approx(mean)
    assert oracle.query_snapshot().total == 0


def test_controlled_rotation_is_identity_or_reward_rotation_by_control() -> None:
    oracle = CanonicalBernoulliOracleSimulator((0.36,))
    controlled = controlled_reward_rotation_matrix(0.36)
    reward = reward_rotation_matrix(0.36)

    assert controlled[0] == (1.0, 0.0, 0.0, 0.0)
    assert controlled[1] == (0.0, 1.0, 0.0, 0.0)
    assert tuple(row[2:] for row in controlled[2:]) == reward
    assert oracle.query_snapshot().total == 0


def test_grover_experiment_expands_powers_into_canonical_oracle_calls() -> None:
    oracle = CanonicalBernoulliOracleSimulator((1.0,), seed=3)
    successes = oracle.run_grover_experiment(0, grover_power=3, shots=5, tag="audit")
    snapshot = oracle.query_snapshot()

    assert successes == 5
    assert snapshot.counts[QueryKind.FORWARD.value] == 20
    assert snapshot.counts[QueryKind.INVERSE.value] == 15
    assert snapshot.coherent_total == 35
    assert snapshot.classical_total == 0
    assert snapshot.by_arm[0][QueryKind.FORWARD.value] == 20
    assert snapshot.by_tag["audit"][QueryKind.INVERSE.value] == 15


def test_controlled_and_classical_resources_remain_separate() -> None:
    oracle = CanonicalBernoulliOracleSimulator((0.0,), seed=8)
    assert oracle.run_grover_experiment(0, 2, 7, controlled=True) == 0
    assert oracle.sample(0, shots=11) == 0
    snapshot = oracle.query_snapshot()

    assert snapshot.counts[QueryKind.CONTROLLED_FORWARD.value] == 21
    assert snapshot.counts[QueryKind.CONTROLLED_INVERSE.value] == 14
    assert snapshot.coherent_total == 35
    assert snapshot.classical_total == 11
    assert snapshot.total == 46


def test_analytic_probability_matches_amplitude_amplification_formula() -> None:
    mean = 0.2
    oracle = CanonicalBernoulliOracleSimulator((mean,))
    theta = math.asin(math.sqrt(mean))

    for power in range(6):
        assert amplified_success_probability(mean, power) == pytest.approx(
            math.sin((2 * power + 1) * theta) ** 2
        )
    assert oracle.query_snapshot().total == 0


def test_ledger_difference_and_input_validation() -> None:
    ledger = QueryLedger()
    before = ledger.snapshot()
    ledger.record(QueryKind.FORWARD, 4, arm=2)
    after = ledger.snapshot()

    assert QueryLedger.difference(after, before)["coherent_total"] == 4
    with pytest.raises(ValueError, match="predates"):
        QueryLedger.difference(before, after)
    with pytest.raises(ValueError, match="negative"):
        ledger.record(QueryKind.FORWARD, -1)
    with pytest.raises(TypeError, match="integer"):
        ledger.record(QueryKind.FORWARD, 1.5)
    with pytest.raises(IndexError):
        CanonicalBernoulliOracleSimulator((0.5,)).run_grover_experiment(1, 0, 1)
    oracle = CanonicalBernoulliOracleSimulator((0.5,))
    with pytest.raises(TypeError, match="integer"):
        oracle.run_grover_experiment(0.5, 0, 1)
    with pytest.raises(TypeError, match="integer"):
        oracle.run_grover_experiment(0, 0.5, 1)
    with pytest.raises(TypeError, match="controlled"):
        oracle.run_grover_experiment(0, 0, 1, controlled=1)  # type: ignore[arg-type]


def test_query_snapshots_are_deeply_immutable_copies() -> None:
    ledger = QueryLedger()
    ledger.record(QueryKind.FORWARD, 2, arm=0, tag="immutable")
    snapshot = ledger.snapshot()

    with pytest.raises(TypeError):
        snapshot.counts[QueryKind.FORWARD.value] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        snapshot.by_arm[0][QueryKind.FORWARD.value] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        snapshot.by_tag["immutable"][QueryKind.FORWARD.value] = 99  # type: ignore[index]
    assert ledger.snapshot().counts[QueryKind.FORWARD.value] == 2
