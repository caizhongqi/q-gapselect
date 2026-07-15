from __future__ import annotations

import numpy as np
import pytest

from qgapselect.coherent import CanonicalRyStatevectorOracle
from qgapselect.direct_search import (
    FullWorkspaceBBHT,
    full_workspace_rank_one_diffusion,
)


def _random_state(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    return state / np.linalg.norm(state)


def test_rank_one_diffusion_reflects_the_complete_workspace_state() -> None:
    initial = np.zeros(16, dtype=np.complex128)
    initial[[0, 4]] = 1.0 / np.sqrt(2.0)
    state = _random_state(16, 9)

    reflected = full_workspace_rank_one_diffusion(state, initial)
    expected = 2.0 * initial * np.vdot(initial, state) - state

    assert np.allclose(reflected, expected, atol=1e-12)
    assert np.isclose(np.linalg.norm(reflected), 1.0)
    assert np.allclose(
        full_workspace_rank_one_diffusion(reflected, initial),
        state,
        atol=1e-12,
    )
    # Amplitudes outside the zero-workspace support are negated, proving this
    # is not an index diffusion tensored with identity on spectator registers.
    assert reflected[3] == pytest.approx(-state[3])


@pytest.mark.parametrize(
    ("mean", "relation"),
    [(1.0, "above"), (0.0, "below")],
)
def test_single_arm_exact_grid_query_formula_and_fresh_verification(
    mean: float,
    relation: str,
) -> None:
    oracle = CanonicalRyStatevectorOracle([mean])
    search = FullWorkspaceBBHT(
        oracle,
        0.5,
        1,
        phase_qubits=2,
        relation=relation,
        verification_shots=32,
        max_attempts_per_output=2,
        seed=3,
    )

    result = search.run()

    assert result.complete and result.verified
    assert result.outputs == (0,)
    assert result.found_indices == result.outputs
    assert result.selected_indices == result.outputs
    assert result.status == "complete_fixed_confidence_qpe_predicate"
    assert not result.absence_certified
    assert result.attempts == 1
    attempt = result.trace[0]
    assert attempt.grover_iterations == 0
    assert attempt.verification is not None
    assert attempt.verification.accepted
    assert attempt.resources.verification_shots == 32
    # L=4: final C_tau costs 2L-1=7 and 32 fresh C_tau shots cost 224.
    assert attempt.resources.oracle_queries == 7 + 32 * 7
    assert result.resources.oracle_queries == 231
    assert result.resources.qpe_calls == 33
    assert result.resources.rank_one_diffusions == 0
    assert oracle.query_snapshot().coherent_total == 231
    assert result.resources.query_counts == oracle.query_snapshot().flat()
    assert result.verification_failure_budget == 0.05
    assert result.per_verification_failure_budget == 0.025


def test_above_search_enumerates_unique_outputs_with_dynamic_exclusion() -> None:
    oracle = CanonicalRyStatevectorOracle([1.0, 1.0, 0.0, 0.0])
    search = FullWorkspaceBBHT(
        oracle,
        0.5,
        2,
        phase_qubits=2,
        verification_shots=32,
        max_attempts_per_output=10,
        seed=2,
    )

    result = search.run()

    assert result.complete
    assert set(result.outputs) == {0, 1}
    assert len(set(result.outputs)) == 2
    accepted = [attempt for attempt in result.trace if attempt.accepted_output]
    assert len(accepted) == 2
    assert accepted[1].eligible_count == 3
    assert all(attempt.verification is not None for attempt in accepted)


def test_nonzero_bbht_iteration_composes_reflection_decode_and_verifier_costs() -> None:
    oracle = CanonicalRyStatevectorOracle([1.0, 1.0, 0.0, 0.0])
    result = FullWorkspaceBBHT(
        oracle,
        0.5,
        2,
        phase_qubits=2,
        verification_shots=32,
        max_attempts_per_output=10,
        seed=1,
    ).run()

    attempt = next(item for item in result.trace if item.grover_iterations > 0)
    assert attempt.grover_iterations == 1
    assert attempt.candidate is not None
    assert attempt.verification is not None
    # L=4: one reflection is 4L-2, decode is 2L-1, and each fresh
    # verification shot reruns the 2L-1 compute circuit.
    expected = 1 * (4 * 4 - 2) + (2 * 4 - 1) + 32 * (2 * 4 - 1)
    assert attempt.resources.oracle_queries == expected == 245
    assert attempt.resources.rank_one_diffusions == 1
    assert attempt.resources.amplitude_amplification_iterations == 1


def test_static_exclusions_are_never_prepared_or_returned() -> None:
    oracle = CanonicalRyStatevectorOracle([1.0, 1.0, 1.0])
    search = FullWorkspaceBBHT(
        oracle,
        0.5,
        2,
        phase_qubits=2,
        excluded_indices=(1,),
        verification_shots=32,
        max_attempts_per_output=10,
        seed=6,
    )

    result = search.run()

    assert result.complete
    assert set(result.outputs) == {0, 2}
    assert all(attempt.candidate != 1 for attempt in result.trace)


def test_zero_step_pause_then_resume_is_deterministic() -> None:
    oracle = CanonicalRyStatevectorOracle([1.0])
    search = FullWorkspaceBBHT(
        oracle,
        0.5,
        1,
        phase_qubits=2,
        verification_shots=32,
        seed=17,
    )

    paused = search.run(max_steps=0)
    assert paused.status == "paused_resumable"
    assert paused.attempts == 0
    assert paused.resources.oracle_queries == 0

    completed = search.resume(max_steps=1)
    assert completed.complete
    assert completed.outputs == (0,)
    assert completed.attempts == 1


def test_randomized_budget_exhaustion_never_claims_absence() -> None:
    oracle = CanonicalRyStatevectorOracle([0.0])
    search = FullWorkspaceBBHT(
        oracle,
        0.5,
        1,
        phase_qubits=2,
        relation="above",
        verification_shots=8,
        max_attempts_per_output=2,
        seed=8,
    )

    result = search.run()

    assert not result.complete
    assert result.outputs == ()
    assert result.status == "search_attempt_budget_exhausted"
    assert result.failure_reason == (
        "randomized_budget_exhaustion_does_not_certify_absence"
    )
    assert not result.absence_certified
    assert result.attempts == 2
    assert all(attempt.candidate is None for attempt in result.trace)
    assert result.resources.oracle_queries == 2 * 7
    # A terminal fixed budget cannot silently restart on resume.
    assert search.resume(max_steps=10) == result


def test_statevector_limit_blocks_before_any_oracle_query() -> None:
    oracle = CanonicalRyStatevectorOracle([1.0, 0.0])
    search = FullWorkspaceBBHT(
        oracle,
        0.5,
        1,
        phase_qubits=4,
        # The retained 64 amplitudes fit, but the transient comparator doubles
        # the peak allocation to 128 amplitudes.
        max_statevector_dimension=64,
        seed=1,
    )

    result = search.run()

    assert search.statevector_dimension == 64
    assert search.peak_statevector_dimension == 128
    assert result.status == "statevector_budget_exceeded"
    assert not result.complete
    assert not result.absence_certified
    assert result.attempts == 0
    assert result.resources.oracle_queries == 0
    assert result.resources.statevector_dimension == 64
    assert result.resources.peak_statevector_dimension == 128
    assert oracle.query_snapshot().total == 0

    allowed = FullWorkspaceBBHT(
        oracle,
        0.5,
        0,
        phase_qubits=4,
        max_statevector_dimension=128,
        seed=1,
    ).run()
    assert allowed.complete


def test_external_validator_runs_only_after_fresh_qpe_acceptance() -> None:
    calls: list[int] = []

    def reject(index: int) -> bool:
        calls.append(index)
        return False

    oracle = CanonicalRyStatevectorOracle([1.0])
    search = FullWorkspaceBBHT(
        oracle,
        0.5,
        1,
        phase_qubits=2,
        verification_shots=8,
        max_attempts_per_output=1,
        candidate_validator=reject,
        seed=10,
    )

    result = search.run()

    assert calls == [0]
    assert result.outputs == ()
    assert result.status == "search_attempt_budget_exhausted"
    attempt = result.trace[0]
    assert attempt.verification is not None and attempt.verification.accepted
    assert attempt.validator_accepted is False
    assert attempt.outcome == "classifier_false_positive_rejected"
    assert not attempt.accepted_output


def test_seed_reproduces_bbht_and_measurement_trace() -> None:
    def execute() -> tuple[tuple[object, ...], ...]:
        oracle = CanonicalRyStatevectorOracle([1.0, 1.0, 0.0, 0.0])
        search = FullWorkspaceBBHT(
            oracle,
            0.5,
            2,
            phase_qubits=2,
            verification_shots=32,
            max_attempts_per_output=10,
            seed=1,
        )
        result = search.run()
        assert result.complete
        return tuple(
            (
                attempt.grover_iterations,
                attempt.measurement_seed,
                attempt.verification_seed,
                attempt.candidate,
                attempt.outcome,
            )
            for attempt in result.trace
        )

    assert execute() == execute()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"threshold": True, "target_count": 1}, TypeError),
        ({"threshold": 0.5, "target_count": 1.0}, TypeError),
        ({"threshold": 0.5, "target_count": True}, TypeError),
        ({"threshold": 0.5, "target_count": 1, "phase_qubits": 2.0}, TypeError),
        ({"threshold": 0.5, "target_count": 1, "verification_shots": True}, TypeError),
        (
            {"threshold": 0.5, "target_count": 1, "max_attempts_per_output": 2.0},
            TypeError,
        ),
        (
            {"threshold": 0.5, "target_count": 1, "max_statevector_dimension": True},
            TypeError,
        ),
        ({"threshold": 0.5, "target_count": 1, "seed": 3.0}, TypeError),
        ({"threshold": 0.5, "target_count": 1, "relation": "equal"}, ValueError),
        ({"threshold": 0.5, "target_count": 3}, ValueError),
        ({"threshold": 0.5, "target_count": 1, "excluded_indices": (True,)}, TypeError),
        (
            {"threshold": 0.5, "target_count": 1, "candidate_validator": 4},
            TypeError,
        ),
    ],
)
def test_constructor_rejects_invalid_or_lossy_arguments(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    oracle = CanonicalRyStatevectorOracle([1.0, 0.0])
    with pytest.raises(error):
        FullWorkspaceBBHT(oracle, **kwargs)


def test_run_rejects_lossy_attempt_budgets_and_validator_non_bool() -> None:
    oracle = CanonicalRyStatevectorOracle([1.0])
    search = FullWorkspaceBBHT(oracle, 0.5, 1, seed=1)
    with pytest.raises(TypeError, match="integer"):
        search.run(max_steps=1.0)
    with pytest.raises(TypeError, match="integer"):
        search.run(max_steps=True)
    with pytest.raises(ValueError, match="negative"):
        search.run(max_steps=-1)

    invalid = FullWorkspaceBBHT(
        oracle,
        0.5,
        1,
        phase_qubits=2,
        verification_shots=8,
        candidate_validator=lambda _index: np.bool_(True),
        max_attempts_per_output=1,
        seed=2,
    )
    with pytest.raises(TypeError, match="return bool"):
        invalid.step()


def test_rank_one_diffusion_rejects_mismatched_or_unnormalized_states() -> None:
    with pytest.raises(ValueError, match="normalized"):
        full_workspace_rank_one_diffusion(np.zeros(4), np.asarray([1.0, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="equal length"):
        full_workspace_rank_one_diffusion(
            np.asarray([1.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
        )
