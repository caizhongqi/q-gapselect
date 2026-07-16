from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

import qgapselect.quantum_benchmarking as benchmark_module
from qgapselect.quantum_benchmarking import (
    FAMILIES,
    BenchmarkRecord,
    QuantumBenchmarkConfig,
    QuantumBenchmarkRunner,
    aggregate_benchmark_records,
    aggregate_paired_query_ratios,
    make_benchmark_instance,
    make_benchmark_suite,
    paired_query_ratios,
    wilson_success_interval,
)


def _small_config(*, phase_qubits: int = 2) -> QuantumBenchmarkConfig:
    return QuantumBenchmarkConfig(
        phase_qubits=phase_qubits,
        verification_shots=32,
        confidence=0.1,
        max_attempts_per_output=10,
        max_statevector_dimension=4096,
        classical_shots_per_arm=128,
        boundary_shots_per_round=32,
        max_boundary_rounds=3,
    )


def _endpoint_instance() -> object:
    return make_benchmark_instance("endpoint_angular", n_arms=2, k=1, seed=2)


@pytest.mark.parametrize("family", FAMILIES)
def test_all_instance_families_are_reproducible_and_truth_is_evaluation_only(
    family: str,
) -> None:
    first = make_benchmark_instance(family, n_arms=4, k=2, seed=17)
    second = make_benchmark_instance(family, n_arms=4, k=2, seed=17)

    assert first == second
    assert first.truth_usage == "evaluation_only_never_passed_as_membership_to_algorithm"
    assert len(first.truth_above) == first.k
    assert len(first.truth_below) == first.n_arms - first.k
    assert set(first.truth_above).isdisjoint(first.truth_below)
    assert first.topk_truth == first.truth_above
    assert all(first.means[index] > first.threshold for index in first.truth_above)
    assert all(first.means[index] < first.threshold for index in first.truth_below)
    assert all(
        mean == pytest.approx(math.sin(angle) ** 2)
        for mean, angle in zip(first.means, first.angles, strict=True)
    )


def test_family_specific_angular_constructions_cover_grid_random_and_endpoints() -> None:
    equal = make_benchmark_instance("equal_grid", n_arms=4, k=2, seed=4)
    heterogeneous = make_benchmark_instance(
        "heterogeneous_dyadic", n_arms=4, k=2, seed=4
    )
    random_first = make_benchmark_instance("off_grid_random", n_arms=4, k=2, seed=4)
    random_other = make_benchmark_instance("off_grid_random", n_arms=4, k=2, seed=5)
    endpoints = make_benchmark_instance("endpoint_angular", n_arms=4, k=2, seed=4)

    equal_sorted = sorted(equal.angles)
    equal_steps = [
        equal_sorted[index + 1] - equal_sorted[index]
        for index in range(len(equal_sorted) - 1)
    ]
    assert max(equal_steps) == pytest.approx(min(equal_steps))
    dyadic_gaps = sorted(
        abs(angle - heterogeneous.threshold_angle) for angle in heterogeneous.angles
    )
    assert dyadic_gaps[-1] == pytest.approx(2.0 * dyadic_gaps[0])
    assert random_first.angles != random_other.angles
    assert min(endpoints.angles) == pytest.approx(0.0)
    assert max(endpoints.angles) == pytest.approx(math.pi / 2.0)
    assert min(endpoints.means) == pytest.approx(0.0)
    assert max(endpoints.means) == pytest.approx(1.0)


def test_suite_order_is_deterministic_family_major_then_seed_minor() -> None:
    suite = make_benchmark_suite(
        families=("equal_grid", "endpoint_angular"),
        n_arms=4,
        k=2,
        seeds=(7, 8),
    )

    assert [(item.family, item.instance_seed) for item in suite] == [
        ("equal_grid", 7),
        ("equal_grid", 8),
        ("endpoint_angular", 7),
        ("endpoint_angular", 8),
    ]


def test_direct_adapter_splits_executed_query_categories_and_has_no_trace_field() -> None:
    runner = QuantumBenchmarkRunner(_small_config())
    record = runner.run("direct_bbht", _endpoint_instance(), trial_seed=0)

    assert record.complete and record.exact and record.certified
    assert record.total_queries == 308
    assert record.reflection_queries == 42
    assert record.decode_queries == 42
    assert record.fresh_verification_queries == 224
    assert record.basis_sampling_queries == 0
    assert record.other_queries == 0
    assert record.total_queries == (
        record.reflection_queries
        + record.decode_queries
        + record.fresh_verification_queries
    )
    assert record.gates > 0 and record.depth > 0 and record.qubits > 0
    assert len(record.instance_manifest_sha256) == 64
    assert record.minimum_angular_boundary_gap > 0.0
    assert record.minimum_mean_boundary_gap > 0.0
    assert record.peak_statevector_dimension >= record.retained_statevector_dimension
    assert "trace" not in record.as_flat_dict()
    assert all(
        not isinstance(value, (dict, list, tuple))
        for value in record.as_flat_dict().values()
    )


def test_direct_adapter_resumes_until_terminal_instead_of_treating_pause_as_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _endpoint_instance()
    calls = {"run": 0, "resume": 0}

    def result(status: str, complete: bool) -> object:
        resources = SimpleNamespace(
            query_counts={"total": 0},
            gate_counts={},
            depth=0,
            qubits=1,
            workspace_qubits=0,
            retained_statevector_dimension=8,
            peak_statevector_dimension=16,
        )
        return SimpleNamespace(
            outputs=instance.truth_above if complete else (),
            complete=complete,
            verified=complete,
            status=status,
            failure_reason=None,
            attempts=calls["run"] + calls["resume"],
            trace=(),
            resources=resources,
        )

    class FakeResumableSearch:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run(self) -> object:
            calls["run"] += 1
            return result("paused_resumable", False)

        def resume(self) -> object:
            calls["resume"] += 1
            return result("complete_fixed_confidence_qpe_predicate", True)

    monkeypatch.setattr(benchmark_module, "FullWorkspaceBBHT", FakeResumableSearch)
    record = QuantumBenchmarkRunner(_small_config()).run(
        "direct_bbht", instance, trial_seed=3
    )

    assert calls == {"run": 1, "resume": 1}
    assert record.complete and record.exact
    # Compatibility accepts the newer retained/peak resource field names.
    assert record.retained_statevector_dimension == 8
    assert record.peak_statevector_dimension == 16


def test_independent_and_classical_adapters_use_disjoint_query_categories() -> None:
    runner = QuantumBenchmarkRunner(_small_config())
    instance = _endpoint_instance()
    independent = runner.run("independent_qpe_scan", instance, trial_seed=5)
    classical = runner.run("classical_threshold_scan", instance, trial_seed=5)

    assert independent.complete and independent.exact
    assert independent.fresh_verification_queries == independent.total_queries
    assert independent.basis_sampling_queries == 0
    retained = (1 << independent.phase_qubits) * 2 * 2
    assert independent.retained_statevector_dimension == retained
    assert independent.peak_statevector_dimension == (
        (1 << independent.phase_qubits) ** 2 + 2 * retained
    )
    assert independent.comparator_expanded_statevector_dimension == 0
    assert independent.dense_qft_matrix_dimension == 16
    assert independent.estimated_peak_bytes == 16 * (
        independent.peak_statevector_dimension
    )
    assert classical.complete and classical.exact
    assert classical.basis_sampling_queries == classical.total_queries
    assert classical.fresh_verification_queries == 0
    assert classical.phase_qubits is None
    assert classical.quantum_discovery_claim_allowed is False


def test_boundary_negative_control_forbids_quantum_discovery_claim() -> None:
    record = QuantumBenchmarkRunner(_small_config(phase_qubits=4)).run(
        "boundary_only_negative_control",
        _endpoint_instance(),
        trial_seed=6,
    )

    assert record.complete and record.exact and record.certified
    assert record.boundary_certificate_available
    assert not record.quantum_discovery_claim_allowed
    assert "NEGATIVE_CONTROL" in record.control_role
    assert "FORBIDDEN" in record.interpretation
    assert "membership_certificate" in record.status
    assert record.basis_sampling_queries == record.total_queries
    assert record.reflection_queries == record.decode_queries == 0


def test_calibrated_topk_record_separates_boundary_and_direct_search_queries() -> None:
    record = QuantumBenchmarkRunner(_small_config(phase_qubits=4)).run(
        "calibrated_direct_topk",
        _endpoint_instance(),
        trial_seed=5,
    )

    assert record.complete and record.exact and record.certified
    assert record.boundary_certificate_available
    assert record.control_role == "calibrated_direct_rediscovery_information_firewall"
    assert record.basis_sampling_queries == 64
    assert record.decode_queries > 0
    assert record.fresh_verification_queries > 0
    assert record.total_queries == (
        record.reflection_queries
        + record.decode_queries
        + record.fresh_verification_queries
        + record.basis_sampling_queries
        + record.other_queries
    )


def test_adaptive_topk_uses_smallest_feasible_precision_and_fixed_max_control() -> None:
    config = replace(
        _small_config(phase_qubits=2),
        max_phase_qubits=5,
        max_statevector_dimension=4096,
    )
    runner = QuantumBenchmarkRunner(config)
    instance = _endpoint_instance()

    adaptive = runner.run(
        "adaptive_calibrated_direct_topk", instance, trial_seed=5
    )
    fixed_max = runner.run("fixed_max_precision_topk", instance, trial_seed=5)
    refined_boundary = runner.run(
        "refined_boundary_only_negative_control", instance, trial_seed=5
    )

    assert adaptive.complete and adaptive.exact and adaptive.certified
    assert fixed_max.complete and fixed_max.exact and fixed_max.certified
    assert refined_boundary.complete and refined_boundary.exact
    assert not refined_boundary.quantum_discovery_claim_allowed
    assert not adaptive.quantum_discovery_claim_allowed
    assert not fixed_max.quantum_discovery_claim_allowed
    assert adaptive.initial_phase_qubits == 2
    assert adaptive.max_phase_qubits == 5
    assert adaptive.phase_qubits == 4
    assert adaptive.phase_candidate_levels == "2,3,4,5"
    assert fixed_max.initial_phase_qubits == fixed_max.max_phase_qubits == 5
    assert fixed_max.phase_qubits == 5
    assert fixed_max.phase_candidate_levels == "5"
    assert adaptive.boundary_rounds == fixed_max.boundary_rounds
    assert adaptive.boundary_rounds == refined_boundary.boundary_rounds
    assert (
        adaptive.basis_sampling_queries
        == fixed_max.basis_sampling_queries
        == refined_boundary.basis_sampling_queries
    )
    assert adaptive.dense_qft_matrix_dimension < fixed_max.dense_qft_matrix_dimension
    assert adaptive.control_role == "resource_aware_measured_margin_phase_schedule"
    assert fixed_max.control_role == (
        "fixed_max_precision_matched_boundary_refinement_control"
    )
    assert "NEGATIVE_CONTROL_refined_boundary" in refined_boundary.control_role


def test_budget_failure_status_and_noncertificate_semantics_are_preserved() -> None:
    blocked_config = replace(_small_config(phase_qubits=4), max_statevector_dimension=1)
    record = QuantumBenchmarkRunner(blocked_config).run(
        "direct_bbht",
        _endpoint_instance(),
        trial_seed=4,
    )

    assert not record.complete
    assert not record.exact
    assert not record.certified
    assert record.status == "statevector_budget_exceeded"
    assert record.failure_reason == "statevector_budget_exceeded"
    assert record.total_queries == 0


def test_wilson_interval_and_group_aggregates_include_status_and_quantiles() -> None:
    lower, upper = wilson_success_interval(5, 10)
    assert lower == pytest.approx(0.236593, rel=1e-5)
    assert upper == pytest.approx(0.763407, rel=1e-5)

    base = QuantumBenchmarkRunner(_small_config()).run(
        "classical_threshold_scan", _endpoint_instance(), trial_seed=1
    )
    failed = replace(
        base,
        trial_seed=2,
        complete=False,
        exact=False,
        certified=False,
        status="query_budget_exhausted",
        total_queries=0,
        basis_sampling_queries=0,
        gates=0,
        depth=0,
    )
    aggregate = aggregate_benchmark_records((base, failed))[0]

    assert aggregate.trials == 2
    assert aggregate.successes == 1
    assert aggregate.success_rate == pytest.approx(0.5)
    assert aggregate.wilson_lower < 0.5 < aggregate.wilson_upper
    assert aggregate.status_counts == {
        "complete_simultaneous_hoeffding": 1,
        "query_budget_exhausted": 1,
    }
    assert aggregate.metrics["total_queries"].mean == pytest.approx(
        base.total_queries / 2.0
    )
    assert set(aggregate.metrics["total_queries"].quantiles) == {
        "q25",
        "q50",
        "q75",
    }


def test_query_ratios_are_paired_by_instance_and_trial_before_aggregation() -> None:
    instance = _endpoint_instance()
    runner = QuantumBenchmarkRunner(_small_config())
    rows: list[BenchmarkRecord] = []
    for trial_seed in (1, 2):
        rows.append(runner.run("direct_bbht", instance, trial_seed=trial_seed))
        rows.append(
            runner.run("classical_threshold_scan", instance, trial_seed=trial_seed)
        )

    pairs = paired_query_ratios(
        rows,
        "direct_bbht",
        "classical_threshold_scan",
    )
    assert len(pairs) == 2
    assert {pair.trial_seed for pair in pairs} == {1, 2}
    assert all(
        pair.status == "both_certified_success_paired" and pair.ratio is not None
        for pair in pairs
    )
    for pair in pairs:
        assert pair.ratio == pytest.approx(
            pair.numerator_queries / pair.denominator_queries
        )

    aggregate = aggregate_paired_query_ratios(pairs)
    assert aggregate.pairs == aggregate.finite_pairs == 2
    assert aggregate.status_counts == {"both_certified_success_paired": 2}
    assert aggregate.ratios is not None
    assert aggregate.ratios.mean == pytest.approx(
        sum(float(pair.ratio) for pair in pairs) / 2.0
    )


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: make_benchmark_instance("equal_grid", n_arms=True), TypeError),
        (lambda: make_benchmark_instance("equal_grid", k=1.5), TypeError),
        (lambda: make_benchmark_instance("missing"), ValueError),
        (lambda: QuantumBenchmarkConfig(phase_qubits=True), TypeError),
        (
            lambda: QuantumBenchmarkConfig(
                phase_qubits=5, max_phase_qubits=4
            ),
            ValueError,
        ),
        (lambda: QuantumBenchmarkConfig(confidence=True), TypeError),
        (lambda: wilson_success_interval(True, 2), TypeError),
        (lambda: wilson_success_interval(3, 2), ValueError),
    ],
)
def test_public_benchmark_inputs_are_strict(call: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        call()
