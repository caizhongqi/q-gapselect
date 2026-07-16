from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qgapselect.attack_oracles import (
    FrozenCandidateGraph,
    freeze_source_streams,
)
from qgapselect.frozen_coherent_oracle import (
    BACKEND,
    CLAIM_SCOPE,
    FrozenCoherentOracleProtocol,
    FrozenEmpiricalCoherentOracle,
    build_frozen_empirical_coherent_oracle,
)
from qgapselect.gapselect import QGapSelect
from qgapselect.iterative_ae_baseline import IterativeAEThresholdScan
from qgapselect.models import GapSelectConfig, IAEConfig
from qgapselect.oracles import CanonicalBernoulliOracleSimulator, QueryKind


def _endpoint_fixture():
    graph = FrozenCandidateGraph.from_ids(("zero-a", "one-a", "one-b", "zero-b"))
    return freeze_source_streams(
        graph,
        reward_streams={
            "zero-a": [0, 0, 0, 0],
            "one-a": [1, 1, 1, 1],
            "one-b": [1, 1, 1, 1],
            "zero-b": [0, 0, 0, 0],
        },
        cost_streams={candidate_id: [1.0] * 4 for candidate_id in graph.candidate_ids},
        # Deliberately contradict the empirical tensor.  Coherent emulation
        # must be derived from frozen observations, not evaluator-only truth.
        configured_means={
            "zero-a": 1.0,
            "one-a": 0.0,
            "one-b": 0.0,
            "zero-b": 1.0,
        },
    )


def _iae_config() -> IAEConfig:
    return IAEConfig(
        target_angular_precision=0.1,
        confidence=0.05,
        shots_per_round=32,
        max_rounds=3,
        max_grover_power=3,
        grid_points=1025,
    )


def test_capability_is_blind_and_retains_only_public_commitments() -> None:
    fixture = _endpoint_fixture()
    oracle = build_frozen_empirical_coherent_oracle(fixture, measurement_seed=7)

    assert isinstance(oracle, FrozenCoherentOracleProtocol)
    assert isinstance(oracle, CanonicalBernoulliOracleSimulator)
    assert oracle.claim_scope == CLAIM_SCOPE
    assert oracle.claim_scope == (
        "empirical_tensor_coherent_oracle_emulation_no_hardware_claim"
    )
    assert oracle.backend == BACKEND
    assert not oracle.hardware_claimable
    assert oracle.manifest_hash == fixture.manifest_hash
    assert oracle.candidate_ids == ("zero-a", "one-a", "one-b", "zero-b")
    assert not hasattr(oracle, "means")
    assert not hasattr(oracle, "configured_means")
    assert not hasattr(oracle, "frozen_means")
    assert not hasattr(oracle, "reward_streams")
    assert not hasattr(oracle, "cost_streams")
    assert not hasattr(oracle, "fixture")
    assert not hasattr(oracle, "tensor")
    assert not hasattr(oracle, "evaluator")
    assert not hasattr(oracle, "__dict__")


def test_descriptor_has_no_reward_or_mean_field_and_is_immutable() -> None:
    oracle = FrozenEmpiricalCoherentOracle(_endpoint_fixture(), measurement_seed=1)
    descriptor = oracle.descriptor

    assert descriptor.as_dict() == {
        "manifest_hash": oracle.manifest_hash,
        "candidate_ids": ["zero-a", "one-a", "one-b", "zero-b"],
        "stream_lengths": [4, 4, 4, 4],
        "backend": BACKEND,
        "claim_scope": CLAIM_SCOPE,
        "hardware_claimable": False,
    }
    assert descriptor.stream_length_by_candidate == {
        "zero-a": 4,
        "one-a": 4,
        "one-b": 4,
        "zero-b": 4,
    }
    assert "mean" not in repr(descriptor).lower()
    assert "reward" not in repr(descriptor).lower()
    with pytest.raises(FrozenInstanceError):
        descriptor.hardware_claimable = True


def test_grover_measurements_use_frozen_empirical_frequency_not_evaluator_truth() -> None:
    oracle = FrozenEmpiricalCoherentOracle(_endpoint_fixture(), measurement_seed=3)

    # sin^2((2m+1) theta) remains exactly 0/1 at endpoint amplitudes.
    assert oracle.run_grover_experiment(0, 7, 11) == 0
    assert oracle.run_grover_experiment(1, 7, 11) == 11
    assert oracle.run_grover_experiment(2, 2, 5) == 5
    assert oracle.run_grover_experiment(3, 2, 5) == 0


def test_forward_inverse_and_controlled_query_ledger_is_exact() -> None:
    oracle = FrozenEmpiricalCoherentOracle(_endpoint_fixture(), measurement_seed=5)

    oracle.run_grover_experiment(0, grover_power=2, shots=3, tag="plain")
    oracle.run_grover_experiment(
        1,
        grover_power=1,
        shots=2,
        controlled=True,
        tag="controlled",
    )
    snapshot = oracle.query_snapshot()

    assert snapshot.counts[QueryKind.FORWARD.value] == 9
    assert snapshot.counts[QueryKind.INVERSE.value] == 6
    assert snapshot.counts[QueryKind.CONTROLLED_FORWARD.value] == 4
    assert snapshot.counts[QueryKind.CONTROLLED_INVERSE.value] == 2
    assert snapshot.coherent_total == 21
    assert snapshot.classical_total == 0
    assert snapshot.total == 21
    assert snapshot.by_arm[0] == {"forward": 9, "inverse": 6}
    assert snapshot.by_arm[1] == {
        "controlled_forward": 4,
        "controlled_inverse": 2,
    }
    assert snapshot.by_tag["plain"] == {"forward": 9, "inverse": 6}
    assert snapshot.by_tag["controlled"] == {
        "controlled_forward": 4,
        "controlled_inverse": 2,
    }


def test_existing_qgapselect_runs_against_blind_frozen_capability() -> None:
    oracle = FrozenEmpiricalCoherentOracle(_endpoint_fixture(), measurement_seed=17)
    config = GapSelectConfig(
        confidence=0.1,
        initial_angular_epsilon=0.25,
        max_rounds=2,
        shots_per_iae_round=16,
        iae_max_rounds=2,
        iae_max_grover_power=1,
        iae_grid_points=1025,
    )

    result = QGapSelect(config).run(oracle, 2)

    assert result.selected == (1, 2)
    assert result.executed_query_counts["coherent_total"] > 0
    assert result.executed_query_counts == oracle.query_snapshot().flat()
    assert result.candidate_theory_accounting.proof_status == (
        "conjectural_not_a_query_bound"
    )


def test_existing_iterative_ae_scan_accepts_the_blind_subtype() -> None:
    oracle = FrozenEmpiricalCoherentOracle(_endpoint_fixture(), measurement_seed=19)

    result = IterativeAEThresholdScan(
        oracle,
        threshold=0.5,
        expected_count=2,
        relation="above",
        target_angular_precision=0.1,
        config=_iae_config(),
        seed=4,
    ).run()

    assert result.complete and result.verified
    assert set(result.outputs) == {1, 2}
    assert result.resources.oracle_queries > 0
    assert result.resources.query_counts == oracle.query_snapshot().flat()
    assert "no_hardware" in result.claim_status
    assert oracle.claim_scope == CLAIM_SCOPE


def test_candidate_index_mapping_is_public_but_contains_no_score() -> None:
    oracle = FrozenEmpiricalCoherentOracle(_endpoint_fixture())

    assert oracle.candidate_id(2) == "one-b"
    assert oracle.arm_index("zero-b") == 3
    with pytest.raises(IndexError):
        oracle.candidate_id(4)
    with pytest.raises(KeyError):
        oracle.arm_index("missing")
    with pytest.raises(TypeError):
        oracle.candidate_id(True)


@pytest.mark.parametrize("seed", [True, 1.5, "7"])
def test_invalid_measurement_seed_is_rejected(seed: object) -> None:
    with pytest.raises(TypeError):
        FrozenEmpiricalCoherentOracle(_endpoint_fixture(), measurement_seed=seed)  # type: ignore[arg-type]


def test_factory_requires_a_frozen_source_fixture() -> None:
    with pytest.raises(TypeError, match="FrozenSourceFixture"):
        FrozenEmpiricalCoherentOracle(object())  # type: ignore[arg-type]
