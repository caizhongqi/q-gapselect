from __future__ import annotations

import inspect

import pytest

import qgapselect.direct_baselines as baseline_module
from qgapselect.coherent import CanonicalRyStatevectorOracle
from qgapselect.direct_baselines import (
    ClassicalThresholdScan,
    IndependentQPEThresholdScan,
)


def test_independent_qpe_scan_executes_fresh_verifier_and_exact_query_formula() -> None:
    oracle = CanonicalRyStatevectorOracle([1.0, 0.0])
    result = IndependentQPEThresholdScan(
        oracle,
        0.5,
        1,
        phase_qubits=2,
        verification_shots=32,
        confidence=0.05,
    ).run()

    assert result.complete and result.verified
    assert result.outputs == (0,)
    assert result.found_indices == result.selected_indices == (0,)
    assert result.status == "complete_fixed_confidence_qpe_predicate"
    assert not result.absence_certified
    assert result.per_arm_failure_budget == pytest.approx(0.025)
    assert len(result.trace) == 1
    assert result.trace[0].accepted
    assert result.trace[0].interval_domain == (
        "qpe_predicate_acceptance_probability"
    )
    # L=4, hence each of 32 freshly executed verifier shots costs 2L-1=7.
    assert result.resources.oracle_queries == 32 * 7
    assert result.resources.coherent_oracle_queries == 32 * 7
    assert result.resources.measurement_shots == 32
    assert result.resources.qpe_calls == 32
    assert result.resources.access_mode == (
        "coherent_controlled_qpe_from_canonical_rotation"
    )
    assert "no_complexity_claim" in result.resources.claim_status
    assert result.trace[0].query_counts["coherent_total"] == 32 * 7
    assert oracle.query_snapshot().coherent_total == 32 * 7


def test_independent_qpe_below_scan_rejects_then_accepts_with_union_bound() -> None:
    oracle = CanonicalRyStatevectorOracle([1.0, 0.0])
    result = IndependentQPEThresholdScan(
        oracle,
        0.5,
        1,
        phase_qubits=2,
        relation="below",
        verification_shots=32,
    ).run()

    assert result.complete
    assert result.outputs == (1,)
    assert [record.status for record in result.trace] == ["rejected", "accepted"]
    assert result.resources.verifier_calls == 2
    assert result.resources.oracle_queries == 2 * 32 * 7
    assert all(record.failure_budget == pytest.approx(0.025) for record in result.trace)


def test_independent_qpe_query_budget_exhaustion_makes_no_absence_claim() -> None:
    oracle = CanonicalRyStatevectorOracle([1.0])
    result = IndependentQPEThresholdScan(
        oracle,
        0.5,
        1,
        phase_qubits=2,
        verification_shots=8,
        max_oracle_queries=55,
    ).run()

    assert not result.complete
    assert result.outputs == ()
    assert result.status == "query_budget_exhausted"
    assert result.failure_reason == "budget_exhaustion_does_not_certify_absence"
    assert not result.absence_certified
    assert result.resources.oracle_queries == 0
    assert result.trace == ()
    assert oracle.query_snapshot().total == 0


@pytest.mark.parametrize(
    ("means", "relation", "excluded", "expected_output"),
    [
        ((1.0, 0.0), "above", (), 0),
        ((1.0, 0.0), "below", (0,), 1),
    ],
)
def test_classical_scan_executes_charged_forward_measurements_for_both_relations(
    means: tuple[float, ...],
    relation: str,
    excluded: tuple[int, ...],
    expected_output: int,
) -> None:
    oracle = CanonicalRyStatevectorOracle(means, seed=7)
    result = ClassicalThresholdScan(
        oracle,
        0.5,
        1,
        relation=relation,
        excluded_indices=excluded,
        shots_per_arm=128,
        confidence=0.05,
    ).run()

    assert result.complete and result.verified
    assert result.outputs == (expected_output,)
    assert result.status == "complete_simultaneous_hoeffding"
    assert result.trace[0].accepted
    assert result.trace[0].interval_domain == "reward_mean"
    assert result.resources.oracle_queries == 128
    assert result.resources.query_counts["forward"] == 128
    # reward_experiment prepares a basis state and calls the canonical forward
    # oracle once per shot.  It is deliberately not mislabeled as a free
    # classical-sample ledger event.
    assert result.resources.query_counts["classical_sample"] == 0
    assert result.resources.qpe_calls == 0
    assert result.resources.access_mode == (
        "basis_state_forward_oracle_and_reward_measurement"
    )
    assert result.resources.gate_counts == {
        "reward_oracle_forward": 128,
        "reward_measurement": 128,
    }
    assert oracle.query_snapshot().counts["forward"] == 128


def test_classical_scan_uses_simultaneous_bounds_and_does_not_accept_ambiguity() -> None:
    oracle = CanonicalRyStatevectorOracle([0.5], seed=11)
    result = ClassicalThresholdScan(
        oracle,
        0.5,
        1,
        shots_per_arm=8,
        confidence=0.05,
    ).run()

    assert not result.complete
    assert result.outputs == ()
    assert result.trace[0].status == "unresolved"
    assert result.trace[0].interval[0] <= 0.5 <= result.trace[0].interval[1]
    assert result.status == "scan_exhausted_without_target"
    assert result.failure_reason == "finite_scan_does_not_certify_absence"
    assert not result.absence_certified
    assert result.resources.oracle_queries == 8


def test_classical_budget_is_checked_before_running_a_partial_arm() -> None:
    oracle = CanonicalRyStatevectorOracle([1.0], seed=2)
    result = ClassicalThresholdScan(
        oracle,
        0.5,
        1,
        shots_per_arm=32,
        max_oracle_queries=31,
    ).run()

    assert result.status == "query_budget_exhausted"
    assert result.trace == ()
    assert result.resources.oracle_queries == 0
    assert not result.absence_certified
    assert oracle.query_snapshot().total == 0


def test_zero_expected_count_is_complete_without_oracle_access() -> None:
    qpe_oracle = CanonicalRyStatevectorOracle([0.5])
    qpe = IndependentQPEThresholdScan(qpe_oracle, 0.5, 0).run()
    classical_oracle = CanonicalRyStatevectorOracle([0.5])
    classical = ClassicalThresholdScan(classical_oracle, 0.5, 0).run()

    assert qpe.complete and qpe.verified and qpe.outputs == ()
    assert classical.complete and classical.verified and classical.outputs == ()
    assert qpe.resources.oracle_queries == classical.resources.oracle_queries == 0


@pytest.mark.parametrize(
    ("baseline", "kwargs"),
    [
        (IndependentQPEThresholdScan, {"expected_count": True}),
        (IndependentQPEThresholdScan, {"phase_qubits": True}),
        (IndependentQPEThresholdScan, {"verification_shots": 1.5}),
        (ClassicalThresholdScan, {"expected_count": True}),
        (ClassicalThresholdScan, {"shots_per_arm": True}),
        (ClassicalThresholdScan, {"max_arms": 1.5}),
    ],
)
def test_baselines_reject_non_integral_and_boolean_integer_arguments(
    baseline: type[IndependentQPEThresholdScan] | type[ClassicalThresholdScan],
    kwargs: dict[str, object],
) -> None:
    oracle = CanonicalRyStatevectorOracle([1.0])
    parameters: dict[str, object] = {"expected_count": 1}
    parameters.update(kwargs)
    with pytest.raises(TypeError):
        baseline(oracle, 0.5, **parameters)


def test_baseline_implementation_never_reaches_private_or_hidden_oracle_data() -> None:
    source = inspect.getsource(baseline_module)

    assert "__blocks" not in source
    assert "_CanonicalRyStatevectorOracle" not in source
    assert ".acceptance_probability(" not in source
    assert ".reward_experiment(" in source
    assert ".verify_index(" in source
