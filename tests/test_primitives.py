from __future__ import annotations

import numpy as np
import pytest

from qgapselect.coherent import CanonicalRyStatevectorOracle
from qgapselect.natural_oracles import (
    NaturalArmDistribution,
    NaturalPurificationStatevectorOracle,
)
from qgapselect.primitives import (
    DovetailTopKController,
    QBoundaryEstimator,
    QGapFlag,
    qbatch_extract,
    qboundary,
    qgap_flag,
)


def _random_state(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    return state / np.linalg.norm(state)


def test_qgapflag_marks_phases_and_exactly_uncomputes_workspace() -> None:
    state = np.asarray([0.5, 0.5j, -0.5, -0.5j], dtype=np.complex128)
    result = qgap_flag(state, [1, 3])
    assert np.allclose(result.state, [0.5, -0.5j, -0.5, 0.5j])
    assert result.resources.phase_oracle_queries == 1
    assert result.resources.workspace_qubits == 1
    assert result.resources.uncompute_residual == 0.0
    assert result.resources.gate_counts["qgapflag_compute"] == 1
    assert result.resources.gate_counts["qgapflag_uncompute"] == 1


@pytest.mark.parametrize("seed", range(5))
def test_qgapflag_full_workspace_operator_is_an_involution(seed: int) -> None:
    flag = QGapFlag(8, [0, 3, 6])
    state = _random_state(16, seed)
    once = flag.apply_workspace(state)
    twice = flag.apply_workspace(once)
    assert np.isclose(np.linalg.norm(once), 1.0)
    assert np.allclose(twice, state, atol=1e-12)


def test_qboundary_uses_only_charged_oracle_experiments() -> None:
    oracle = CanonicalRyStatevectorOracle([1.0, 1.0, 0.0, 0.0])
    assert not hasattr(oracle, "means")
    result = qboundary(
        oracle,
        2,
        confidence=0.1,
        shots_per_round=32,
        max_rounds=2,
        seed=9,
    )
    assert result.complete
    assert result.candidate_selected == (0, 1)
    assert result.candidate_rejected == (2, 3)
    assert result.certificate is not None
    assert result.certificate.mean_margin > 0.0
    assert result.certificate.angular_margin > 0.0
    assert result.order_statistic_mean_interval[0] <= 1.0
    assert result.resources.query_counts["forward"] == 4 * 32
    assert result.resources.query_counts["inverse"] == 0
    assert result.resources.gate_counts["coherent_reward_oracle"] == 4 * 32


def test_qboundary_runs_against_natural_purification_protocol() -> None:
    oracle = NaturalPurificationStatevectorOracle(
        [
            NaturalArmDistribution.from_sequences([1.0], [1]),
            NaturalArmDistribution.from_sequences([1.0], [0]),
        ],
        seed=3,
    )
    result = qboundary(
        oracle,
        1,
        confidence=0.1,
        shots_per_round=32,
        max_rounds=2,
        seed=4,
    )
    assert result.complete
    assert result.candidate_selected == (0,)
    assert result.resources.query_counts["forward"] == 64
    assert result.resources.workspace_qubits == oracle.contract.workspace_qubits


@pytest.mark.parametrize("strategy", ["known", "bbht"])
def test_qbatch_extract_enumerates_unique_verified_outputs(strategy: str) -> None:
    marked = (1, 4, 6)
    flag = QGapFlag(8, marked)
    result = qbatch_extract(
        flag,
        len(marked),
        strategy=strategy,
        seed=25,
        max_attempts_per_output=80,
    )
    assert result.complete
    assert result.verified
    assert result.outputs == marked
    assert result.attempts >= len(marked)
    # Every measured candidate and final output is explicitly verified; Grover
    # phase calls, when present, add further predicate charges.
    assert result.resources.phase_oracle_queries >= result.attempts + len(marked)
    assert result.resources.gate_counts["qgapflag_verification_compute"] >= (
        result.attempts + len(marked)
    )
    assert result.resources.uncompute_residual == 0.0


@pytest.mark.parametrize("dimension", [2, 4, 8, 16])
def test_known_count_grover_recovers_random_marked_subsets(dimension: int) -> None:
    rng = np.random.default_rng(1000 + dimension)
    count = max(1, dimension // 4)
    marked = tuple(sorted(int(value) for value in rng.choice(dimension, count, False)))
    result = qbatch_extract(
        QGapFlag(dimension, marked),
        count,
        strategy="known",
        seed=dimension,
        max_attempts_per_output=100,
    )
    assert result.complete
    assert result.outputs == marked


def test_dovetail_controller_pauses_resumes_and_gates_output_on_certificate() -> None:
    oracle = CanonicalRyStatevectorOracle([0.98, 0.9, 0.1, 0.02])
    controller = DovetailTopKController(
        oracle,
        2,
        confidence=0.1,
        shots_per_round=32,
        max_boundary_rounds=3,
        seed=12,
    )
    paused = controller.run(max_steps=1)
    assert not paused.complete
    assert paused.selected == ()
    assert paused.rejected == ()
    assert paused.status == "paused_resumable"

    completed = controller.resume(max_steps=8)
    assert completed.complete
    assert completed.status == "complete_certificate"
    assert completed.selected == (0, 1)
    assert completed.rejected == (2, 3)
    winner = next(
        item
        for item in completed.certificates
        if item.orientation == completed.winning_orientation
    )
    assert winner.complete and winner.verified
    assert winner.boundary is not None and winner.boundary.complete
    assert completed.resources.query_counts["forward"] > 0
    assert completed.resources.phase_oracle_queries > 0
    assert completed.resources.uncompute_residual == 0.0


def test_controller_never_certifies_overlapping_intervals() -> None:
    oracle = CanonicalRyStatevectorOracle([0.51, 0.5, 0.49, 0.48])
    result = DovetailTopKController(
        oracle,
        2,
        confidence=0.01,
        shots_per_round=1,
        max_boundary_rounds=1,
        seed=1,
    ).run()
    assert not result.complete
    assert result.selected == ()
    assert result.winning_orientation is None
    assert all(not certificate.complete for certificate in result.certificates)


@pytest.mark.parametrize(
    "factory",
    [
        lambda oracle: QBoundaryEstimator(oracle, 1, shots_per_round=0.5),
        lambda oracle: QBoundaryEstimator(oracle, 1, max_rounds=1.5),
        lambda oracle: DovetailTopKController(oracle, 1).run(max_steps=2.5),
    ],
)
def test_integer_budgets_are_not_silently_truncated(factory) -> None:
    oracle = CanonicalRyStatevectorOracle([0.9, 0.1])
    with pytest.raises(TypeError, match="integer"):
        factory(oracle)


def test_batch_integer_arguments_are_not_silently_truncated() -> None:
    flag = QGapFlag(4, [1])
    with pytest.raises(TypeError, match="integer"):
        qbatch_extract(flag, 0.5)
    with pytest.raises(TypeError, match="integer"):
        qbatch_extract(flag, 1, max_attempts_per_output=2.5)
