from __future__ import annotations

import gc
import json
import math

import pytest

import qgapselect
import qgapselect.frozen_quantum_reference_benchmarking as benchmark_module
from qgapselect.attack_oracles import FrozenCandidateGraph, freeze_source_streams
from qgapselect.frozen_quantum_reference_benchmarking import (
    CLAIM_SCOPE,
    DEFAULT_METHOD_IDS,
    INDEPENDENT_IAE_INFORMATION_REGIME,
    KNOWN_THRESHOLD_IAE_INFORMATION_REGIME,
    QGAPSELECT_INFORMATION_REGIME,
    FrozenQuantumMethodConfigs,
    FrozenQuantumReferenceInstance,
    run_frozen_quantum_reference_benchmark,
)
from qgapselect.models import GapSelectConfig, IAEConfig


def _endpoint_fixture():
    graph = FrozenCandidateGraph.from_ids(("zero-a", "one-a", "one-b", "zero-b"))
    return freeze_source_streams(
        graph,
        reward_streams={
            "zero-a": [0] * 4,
            "one-a": [1] * 4,
            "one-b": [1] * 4,
            "zero-b": [0] * 4,
        },
        cost_streams={candidate_id: [1.0] * 4 for candidate_id in graph.candidate_ids},
        # The benchmark truth must be the frozen empirical tensor, not these
        # deliberately contradictory evaluator-configured expectations.
        configured_means={
            "zero-a": 1.0,
            "one-a": 0.0,
            "one-b": 0.0,
            "zero-b": 1.0,
        },
    )


def _endpoint_instance() -> FrozenQuantumReferenceInstance:
    return FrozenQuantumReferenceInstance(
        family_id="endpoint",
        instance_id="four-arms",
        fixture=_endpoint_fixture(),
        public_threshold=0.5,
        public_gap_floor=math.pi / 4.0,
        k=2,
    )


def _endpoint_instances(count: int):
    for index in range(count):
        yield FrozenQuantumReferenceInstance(
            family_id="endpoint-stream",
            instance_id=f"endpoint-{index}",
            fixture=_endpoint_fixture(),
            public_threshold=0.5,
            public_gap_floor=math.pi / 4.0,
            k=2,
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


def _method_configs(
    method_ids: tuple[str, ...] = DEFAULT_METHOD_IDS,
) -> FrozenQuantumMethodConfigs:
    iae = _iae_config()
    return FrozenQuantumMethodConfigs(
        qgapselect=GapSelectConfig(
            confidence=0.05,
            initial_angular_epsilon=0.25,
            max_rounds=2,
            shots_per_iae_round=16,
            iae_max_rounds=2,
            iae_max_grover_power=1,
            iae_grid_points=1025,
        ),
        independent_iae=iae,
        known_threshold_iae=iae,
        failure_probability=0.05,
        precision_fraction=0.25,
        method_ids=method_ids,
    )


def test_information_regime_constants_are_exported_from_package_root() -> None:
    assert qgapselect.QGAPSELECT_INFORMATION_REGIME == "k_only"
    assert qgapselect.INDEPENDENT_IAE_INFORMATION_REGIME == ("k_and_public_gap_floor")
    assert qgapselect.KNOWN_THRESHOLD_IAE_INFORMATION_REGIME == (
        "k_public_gap_floor_and_public_threshold"
    )


def test_endpoint_matrix_has_fresh_ledgers_and_strict_certified_recovery() -> None:
    report = run_frozen_quantum_reference_benchmark(
        (_endpoint_instance(),),
        _method_configs(),
        repetitions=2,
        master_seed=7,
    )

    assert len(report.runs) == 2 * len(DEFAULT_METHOD_IDS)
    assert len(report.aggregates) == len(DEFAULT_METHOD_IDS)
    assert len(report.instances) == 1
    assert report.method_ids == DEFAULT_METHOD_IDS
    assert report.claim_scope == CLAIM_SCOPE
    assert all(run.certified_exact_recovery for run in report.runs)
    assert all(not run.timeout and not run.heuristic_only for run in report.runs)
    assert all(run.classical_queries == 0 for run in report.runs)
    assert all(run.coherent_queries == run.total_queries > 0 for run in report.runs)
    assert all(
        run.coherent_query_ledger["coherent_total"] == run.coherent_queries for run in report.runs
    )

    # Endpoint measurement outcomes are deterministic.  Equal per-repetition
    # ledgers show that each method started from zero rather than sharing one.
    for method_id in DEFAULT_METHOD_IDS:
        method_runs = [run for run in report.runs if run.method_id == method_id]
        assert len({run.coherent_queries for run in method_runs}) == 1
        assert len({run.measurement_seed for run in method_runs}) == 2

    public_instance = report.as_dict()["instances"][0]
    assert (
        public_instance["difficulty_fingerprint"] == report.instances[0]["difficulty_fingerprint"]
    )
    assert len(public_instance["difficulty_fingerprint"]) == 64
    assert public_instance["structure_metrics"]["n_arms"] == 4
    assert "difficulty_fingerprint" not in report.runs[0].as_dict()
    assert "structure_metrics" not in report.runs[0].as_dict()


def test_streamed_and_eager_small_matrices_are_exactly_equivalent() -> None:
    eager = run_frozen_quantum_reference_benchmark(
        tuple(_endpoint_instances(3)),
        _method_configs(),
        repetitions=2,
        master_seed=29,
        instance_chunk_size=3,
    )
    streamed = run_frozen_quantum_reference_benchmark(
        _endpoint_instances(3),
        _method_configs(),
        repetitions=2,
        master_seed=29,
        instance_chunk_size=1,
    )

    assert streamed == eager
    assert streamed.as_dict() == eager.as_dict()
    assert [run.measurement_seed for run in streamed.runs] == [
        run.measurement_seed for run in eager.runs
    ]
    assert [run.coherent_query_ledger for run in streamed.runs] == [
        run.coherent_query_ledger for run in eager.runs
    ]


def test_streaming_runner_never_retains_more_records_than_chunk_size() -> None:
    state = {"live": 0, "peak": 0}

    class TrackedInstance(FrozenQuantumReferenceInstance):
        __slots__ = ()

        def __del__(self) -> None:
            state["live"] -= 1

    def records():
        for index in range(7):
            record = TrackedInstance(
                family_id="tracked",
                instance_id=f"tracked-{index}",
                fixture=_endpoint_fixture(),
                public_threshold=0.5,
                public_gap_floor=math.pi / 4.0,
                k=2,
            )
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])
            yield record
            del record

    report = benchmark_module.run_frozen_quantum_reference_benchmark(
        records(),
        _method_configs(("qgapselect",)),
        master_seed=31,
        instance_chunk_size=2,
    )
    gc.collect()

    assert len(report.runs) == 7
    assert len(report.instances) == 7
    assert state == {"live": 0, "peak": 2}


def test_frozen_empirical_truth_overrides_contradictory_configured_means() -> None:
    report = run_frozen_quantum_reference_benchmark(
        (_endpoint_instance(),),
        _method_configs(("independent_iae_topk",)),
        master_seed=3,
    )
    run = report.runs[0]

    assert run.reference_top_k == (1, 2)
    assert run.reference_top_k_candidate_ids == ("one-a", "one-b")
    assert set(run.selected) == {1, 2}
    assert run.certified_exact_recovery


def test_three_information_regimes_and_qgapselect_matching_are_explicit() -> None:
    report = run_frozen_quantum_reference_benchmark(
        (_endpoint_instance(),),
        _method_configs(),
        master_seed=11,
    )
    runs = {run.method_id: run for run in report.runs}

    assert runs["qgapselect"].information_matched_to_qgapselect
    assert runs["qgapselect"].information_regime == QGAPSELECT_INFORMATION_REGIME
    assert runs["qgapselect"].algorithm_inputs == ("k",)

    independent = runs["independent_iae_topk"]
    assert not independent.information_matched_to_qgapselect
    assert independent.information_regime == INDEPENDENT_IAE_INFORMATION_REGIME
    assert independent.algorithm_inputs == ("k", "public_gap_floor")
    assert "public_threshold" not in independent.algorithm_inputs
    assert independent.target_angular_precision == pytest.approx(math.pi / 16.0)

    stronger = runs["known_threshold_iae_scan"]
    assert not stronger.information_matched_to_qgapselect
    assert stronger.information_regime == KNOWN_THRESHOLD_IAE_INFORMATION_REGIME
    assert stronger.algorithm_inputs == (
        "k",
        "public_gap_floor",
        "public_threshold",
    )
    assert all("same_information" not in run.as_dict() for run in runs.values())
    aggregates = {aggregate.method_id: aggregate for aggregate in report.aggregates}
    assert aggregates["qgapselect"].information_matched_to_qgapselect
    assert not aggregates["independent_iae_topk"].information_matched_to_qgapselect
    assert not aggregates["known_threshold_iae_scan"].information_matched_to_qgapselect


def test_qgapselect_exact_heuristic_completion_is_not_a_certificate() -> None:
    graph = FrozenCandidateGraph.from_ids(("high", "low"))
    fixture = freeze_source_streams(
        graph,
        reward_streams={"high": [1, 1, 1, 0], "low": [0, 0, 0, 1]},
        cost_streams={"high": [1.0] * 4, "low": [1.0] * 4},
    )
    instance = FrozenQuantumReferenceInstance(
        family_id="finite-budget",
        instance_id="two-arms",
        fixture=fixture,
        public_threshold=0.5,
        public_gap_floor=0.2,
        k=1,
    )
    configs = FrozenQuantumMethodConfigs(
        qgapselect=GapSelectConfig(
            confidence=0.05,
            initial_angular_epsilon=0.25,
            max_rounds=1,
            shots_per_iae_round=1,
            iae_max_rounds=1,
            iae_max_grover_power=0,
            iae_grid_points=257,
        ),
        method_ids=("qgapselect",),
    )
    run = run_frozen_quantum_reference_benchmark(
        (instance,),
        configs,
        master_seed=1,
    ).runs[0]

    assert run.exact_recovery  # post-run scoring happens to like the completion
    assert run.timeout
    assert run.heuristic_only
    assert not run.certified
    assert not run.certified_exact_recovery
    assert run.status == "max_rounds"
    assert "not a certificate" in (run.failure_reason or "")


def test_aggregates_claim_boundaries_and_report_are_serializable() -> None:
    first = run_frozen_quantum_reference_benchmark(
        (_endpoint_instance(),),
        _method_configs(),
        repetitions=2,
        master_seed=19,
    )
    second = run_frozen_quantum_reference_benchmark(
        (_endpoint_instance(),),
        _method_configs(),
        repetitions=2,
        master_seed=19,
    )

    assert first == second
    assert first.as_dict() == second.as_dict()
    assert json.loads(json.dumps(first.as_dict(), sort_keys=True)) == first.as_dict()
    assert len(first.manifest_hash) == 64
    assert all(aggregate.run_count == 2 for aggregate in first.aggregates)
    assert all(aggregate.certified_exact_recovery_rate == 1.0 for aggregate in first.aggregates)
    assert all(
        aggregate.certified_exact_wilson_lower
        <= aggregate.certified_exact_recovery_rate
        <= aggregate.certified_exact_wilson_upper
        for aggregate in first.aggregates
    )
    assert any("hardware" in item for item in first.claim_boundaries["does_not_support"])
    assert any("stronger-information" in item for item in first.claim_boundaries["fairness"])


def test_invalid_threshold_and_gap_promises_are_rejected() -> None:
    fixture = _endpoint_fixture()
    with pytest.raises(ValueError, match="public_threshold"):
        FrozenQuantumReferenceInstance(
            "bad",
            "threshold",
            fixture,
            public_threshold=1.0,
            public_gap_floor=0.1,
            k=2,
        )
    with pytest.raises(ValueError, match="public_gap_floor"):
        FrozenQuantumReferenceInstance(
            "bad",
            "gap",
            fixture,
            public_threshold=0.5,
            public_gap_floor=math.pi / 4.0 + 0.01,
            k=2,
        )


def test_runner_rejects_duplicate_records_and_mismatched_failure_budget() -> None:
    instance = _endpoint_instance()
    with pytest.raises(ValueError, match="pairs must be unique"):
        run_frozen_quantum_reference_benchmark(
            (instance, instance),
            _method_configs(("qgapselect",)),
        )
    with pytest.raises(ValueError, match="must equal failure_probability"):
        FrozenQuantumMethodConfigs(
            qgapselect=GapSelectConfig(confidence=0.1),
            failure_probability=0.05,
        )
    with pytest.raises(ValueError, match="instance_chunk_size"):
        run_frozen_quantum_reference_benchmark(
            (_endpoint_instance(),),
            _method_configs(("qgapselect",)),
            instance_chunk_size=0,
        )
