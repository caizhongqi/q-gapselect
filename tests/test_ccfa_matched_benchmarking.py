from __future__ import annotations

import json
import math

import pytest

from qgapselect.ccfa_evidence_gate import (
    CCFAEvidenceGateConfig,
    PreregisteredFixture,
    evaluate_ccfa_evidence_gate,
)
from qgapselect.ccfa_matched_benchmarking import (
    CLAIM_SCOPE,
    COHERENT_METHOD_ID,
    K_ONLY_INFORMATION_REGIME,
    KNOWN_TIME_METHOD_ID,
    PRIMARY_COMPARISON_GROUP,
    PRIMARY_METHOD_IDS,
    STRONGER_INFORMATION_GROUP,
    AtomicQueryCapOracle,
    CCFAMatchedBenchmarkConfig,
    KnownTimeControlSpec,
    aggregate_ccfa_matched_trials,
    iter_ccfa_matched_trials,
    matched_campaign_manifest_hash,
    run_capped_coherent_activity_history,
    validate_complete_matched_panel,
)
from qgapselect.coherent_activity_history_core import VariableTimeHistoryConfig
from qgapselect.exact_count_fixtures import generate_exact_count_fixture
from qgapselect.frozen_coherent_oracle import build_frozen_empirical_coherent_oracle
from qgapselect.frozen_quantum_reference_benchmarking import (
    FrozenQuantumReferenceInstance,
)
from qgapselect.matched_quantum_baselines import (
    MatchedBaselineConfig,
    QueryCapExceeded,
)
from qgapselect.models import IAEConfig


def _instance(instance_id: str = "fixture-0") -> FrozenQuantumReferenceInstance:
    fixture = generate_exact_count_fixture(
        {
            "a0": 1.0,
            "a1": 0.95,
            "a2": 0.90,
            "a3": 0.10,
            "a4": 0.05,
            "a5": 0.0,
        },
        table_size=100,
        seed=91,
    )
    return FrozenQuantumReferenceInstance(
        family_id="separated",
        instance_id=instance_id,
        fixture=fixture,
        public_threshold=0.5,
        public_gap_floor=0.45,
        k=3,
    )


def _coherent_config() -> VariableTimeHistoryConfig:
    return VariableTimeHistoryConfig(
        confidence=0.1,
        initial_angular_precision=0.2,
        precision_decay=0.5,
        max_levels=4,
        shots_per_iae_round=64,
        iae_max_rounds=6,
        iae_max_grover_power=31,
        iae_grid_points=2049,
        verification_angular_precision=0.02,
        verification_shots_per_round=96,
        verification_max_rounds=7,
        verification_max_grover_power=63,
        verification_grid_points=4097,
    )


def _baseline_config() -> MatchedBaselineConfig:
    return MatchedBaselineConfig(
        initial_angular_precision=math.pi / 8.0,
        precision_decay=0.5,
        max_levels=4,
        iae=IAEConfig(
            target_angular_precision=0.1,
            confidence=0.05,
            shots_per_round=64,
            max_rounds=5,
            max_grover_power=15,
            grid_points=1025,
        ),
    )


def _campaign(*, repetitions: int, cap: int, known: bool) -> CCFAMatchedBenchmarkConfig:
    controls = (
        (KnownTimeControlSpec("separated", "fixture-0", (4, 4, 4, 4, 4, 4)),) if known else ()
    )
    return CCFAMatchedBenchmarkConfig(
        master_seed=17,
        repetitions=repetitions,
        query_caps=(cap,),
        failure_budget=0.1,
        coherent=_coherent_config(),
        baselines=_baseline_config(),
        coarse_partition_block_size=2,
        coarse_partition_seed=8,
        known_time_controls=controls,
    )


def test_atomic_cap_rejects_the_whole_next_experiment_before_charge() -> None:
    oracle = build_frozen_empirical_coherent_oracle(
        _instance().fixture,
        measurement_seed=3,
    )
    capped = AtomicQueryCapOracle(oracle, 100)

    capped.run_grover_experiment(0, 0, 64, tag="affordable")
    before_rejection = oracle.query_snapshot().flat()
    with pytest.raises(QueryCapExceeded) as raised:
        capped.run_grover_experiment(0, 1, 16, tag="rejected")

    # The rejected m=1 experiment costs 48 calls; no prefix is charged.
    assert raised.value.spent == 64
    assert raised.value.requested == 48
    assert oracle.query_snapshot().flat() == before_rejection
    assert capped.spent == 64
    assert capped.remaining == 36
    assert capped.rejected_experiments == 1
    assert "rejected" not in oracle.query_snapshot().by_tag


def test_capped_coherent_timeout_stays_in_denominator_with_zero_overshoot() -> None:
    execution = run_capped_coherent_activity_history(
        build_frozen_empirical_coherent_oracle(
            _instance().fixture,
            measurement_seed=5,
        ),
        3,
        query_cap=100,
        failure_budget=0.1,
        config=_coherent_config(),
    )

    assert execution.timeout
    assert execution.status == "query_cap_exhausted"
    assert execution.result is None
    assert execution.actual_queries == 64
    assert execution.selection_queries == 64
    assert execution.verification_queries == 0
    assert execution.budget_valid
    assert not execution.cleanup_passed
    assert execution.rejected_experiments == 1


def test_balanced_panel_has_same_information_primary_and_separate_known_control() -> None:
    rows = tuple(
        iter_ccfa_matched_trials(
            (_instance(),),
            _campaign(
                repetitions=1,
                cap=1_000_000,
                known=True,
            ),
        )
    )

    assert tuple(row.method_id for row in rows) == (*PRIMARY_METHOD_IDS, KNOWN_TIME_METHOD_ID)
    primary = [row for row in rows if row.comparison_group == PRIMARY_COMPARISON_GROUP]
    known = [row for row in rows if row.comparison_group == STRONGER_INFORMATION_GROUP]
    assert {row.method_id for row in primary} == set(PRIMARY_METHOD_IDS)
    assert {row.information_regime for row in primary} == {K_ONLY_INFORMATION_REGIME}
    assert len(known) == 1 and known[0].method_id == KNOWN_TIME_METHOD_ID
    assert "stop_levels" in known[0].information_regime
    assert {row.block_seed for row in rows} == {rows[0].block_seed}
    assert all(row.measurement_seed == rows[0].block_seed for row in rows)
    assert all(row.oracle_ledger_start_queries == 0 for row in rows)
    assert all(row.actual_queries <= row.query_cap for row in rows)
    assert all(row.coherent_queries == row.actual_queries for row in rows)
    assert all(not row.hardware_claimable for row in rows)
    assert all(not row.theorem_claimable for row in rows)
    assert all(not row.quantum_advantage_claimable for row in rows)
    assert all(row.claim_scope == CLAIM_SCOPE for row in rows)


def test_coherent_success_records_cleanup_direct_output_and_fresh_verification() -> None:
    rows = tuple(
        iter_ccfa_matched_trials(
            (_instance(),),
            _campaign(repetitions=1, cap=1_000_000, known=False),
        )
    )
    coherent = next(row for row in rows if row.method_id == COHERENT_METHOD_ID)

    assert coherent.complete
    assert coherent.certified
    assert coherent.exact
    assert coherent.certified_exact
    assert coherent.cleanup_passed is True
    assert coherent.direct_multi_output
    assert coherent.as_dict()["finite_state_direct_output_tape"]
    assert not coherent.as_dict()["coherent_direct_multi_output_verified"]
    assert "not_coherent_cross_index_union" in str(
        coherent.as_dict()["direct_multi_output_semantics"]
    )
    assert coherent.selection_queries > 0
    assert coherent.verification_queries > 0
    assert coherent.selection_queries + coherent.verification_queries == (coherent.actual_queries)
    assert coherent.expected_top_k == (0, 1, 2)
    assert coherent.selected == coherent.expected_top_k
    assert coherent.empirical_truth_source.startswith("trusted_harness")
    assert all(row.cleanup_passed is None for row in rows[1:])
    assert all(not row.direct_multi_output for row in rows[1:])


def test_every_timeout_is_retained_and_each_method_gets_a_fresh_ledger() -> None:
    rows = tuple(
        iter_ccfa_matched_trials(
            (_instance(),),
            _campaign(repetitions=2, cap=0, known=True),
        )
    )
    aggregates = aggregate_ccfa_matched_trials(rows)

    assert len(rows) == 2 * 6
    assert len(aggregates) == 6
    assert all(row.actual_queries == 0 for row in rows)
    assert all(row.oracle_ledger_start_queries == 0 for row in rows)
    assert all(row.timeout for row in rows)
    assert all(not row.certified_exact for row in rows)
    assert all(aggregate.attempts == 2 for aggregate in aggregates)
    assert all(aggregate.timeout_count == 2 for aggregate in aggregates)
    assert all(aggregate.certified_exact_count == 0 for aggregate in aggregates)
    assert all(aggregate.certified_exact_rate == 0.0 for aggregate in aggregates)
    assert all(aggregate.budget_violation_count == 0 for aggregate in aggregates)


def test_aggregate_refuses_a_missing_method_instead_of_changing_denominator() -> None:
    rows = tuple(
        iter_ccfa_matched_trials(
            (_instance(),),
            _campaign(repetitions=1, cap=0, known=False),
        )
    )
    validate_complete_matched_panel(rows)

    with pytest.raises(ValueError, match="missing a preregistered method"):
        aggregate_ccfa_matched_trials(rows[:-1])


def test_aggregate_refuses_an_entire_missing_repetition_block() -> None:
    rows = tuple(
        iter_ccfa_matched_trials(
            (_instance(),),
            _campaign(repetitions=2, cap=0, known=False),
        )
    )
    first_repetition_only = tuple(row for row in rows if row.repetition == 0)

    with pytest.raises(ValueError, match="missing an entire"):
        aggregate_ccfa_matched_trials(first_repetition_only)


def test_fixture_shards_merge_to_the_full_deterministic_campaign() -> None:
    instances = (_instance("fixture-0"), _instance("fixture-1"))
    config = _campaign(repetitions=2, cap=0, known=False)
    full = tuple(iter_ccfa_matched_trials(instances, config))
    shards = tuple(
        row
        for instance in instances
        for row in iter_ccfa_matched_trials(
            instances,
            config,
            execution_fixture_keys=((instance.family_id, instance.instance_id),),
        )
    )

    def stable_key(row: object) -> tuple[object, ...]:
        return (
            row.family_id,
            row.instance_id,
            row.repetition,
            row.query_cap,
            row.method_id,
        )

    full_sorted = tuple(sorted(full, key=stable_key))
    shards_sorted = tuple(sorted(shards, key=stable_key))
    assert [row.as_dict() for row in shards_sorted] == [row.as_dict() for row in full_sorted]
    expected_manifest = matched_campaign_manifest_hash(instances, config)
    assert {row.campaign_manifest_hash for row in shards} == {expected_manifest}
    assert all(
        row.preregistered_fixture_keys == (("separated", "fixture-0"), ("separated", "fixture-1"))
        for row in shards
    )
    validate_complete_matched_panel(shards_sorted)

    first_shard = tuple(
        iter_ccfa_matched_trials(
            instances,
            config,
            execution_fixture_keys=(("separated", "fixture-0"),),
        )
    )
    with pytest.raises(ValueError, match="missing an entire"):
        validate_complete_matched_panel(first_shard)


@pytest.mark.parametrize(
    ("execution_keys", "error", "message"),
    [
        ((), ValueError, "cannot be empty"),
        (
            (("separated", "fixture-0"), ("separated", "fixture-0")),
            ValueError,
            "must be unique",
        ),
        (
            (("separated", "not-preregistered"),),
            ValueError,
            "unregistered fixture",
        ),
        (("separated", "fixture-0"), TypeError, "each execution"),
        ((["separated", "fixture-0"],), TypeError, "must be a"),
    ],
)
def test_fixture_shards_reject_empty_duplicate_unknown_or_malformed_keys(
    execution_keys: object,
    error: type[Exception],
    message: str,
) -> None:
    instances = (_instance("fixture-0"), _instance("fixture-1"))
    with pytest.raises(error, match=message):
        tuple(
            iter_ccfa_matched_trials(
                instances,
                _campaign(repetitions=1, cap=0, known=False),
                execution_fixture_keys=execution_keys,  # type: ignore[arg-type]
            )
        )


def test_campaign_inputs_and_outputs_are_canonical_json_serializable() -> None:
    instance = _instance()
    config = _campaign(repetitions=1, cap=0, known=False)
    rows = tuple(iter_ccfa_matched_trials((instance,), config))
    aggregates = aggregate_ccfa_matched_trials(rows)
    first_hash = matched_campaign_manifest_hash((instance,), config)
    second_hash = matched_campaign_manifest_hash((instance,), config)

    document = {
        "config": config.as_dict(),
        "records": [row.as_dict() for row in rows],
        "aggregates": [row.as_dict() for row in aggregates],
        "manifest_hash": first_hash,
    }
    encoded = json.dumps(document, sort_keys=True, allow_nan=False)
    assert json.loads(encoded)["manifest_hash"] == first_hash == second_hash
    assert len(first_hash) == 64


def test_primary_records_feed_fail_closed_statistical_gate_without_translation() -> None:
    instances = (_instance("fixture-0"), _instance("fixture-1"))
    config = _campaign(repetitions=2, cap=1, known=False)
    rows = tuple(iter_ccfa_matched_trials(instances, config))
    manifest_hash = matched_campaign_manifest_hash(instances, config)
    gate = CCFAEvidenceGateConfig(
        candidate_method_id=COHERENT_METHOD_ID,
        strongest_baseline_method_ids=PRIMARY_METHOD_IDS[1:],
        information_regime=K_ONLY_INFORMATION_REGIME,
        preregistered_fixtures=tuple(
            PreregisteredFixture(row.family_id, row.instance_id) for row in instances
        ),
        preregistered_query_caps=(1,),
        repetitions_per_fixture=2,
        preregistration_status="LOCKED_BEFORE_RUN",
        preregistration_manifest_sha256=manifest_hash,
        minimum_risk_difference=0.0,
        bootstrap_repetitions=20,
        minimum_fixtures_per_family=2,
    )

    report = evaluate_ccfa_evidence_gate(rows, gate)

    assert not report.advantage_claimable
    assert report.blockers
    assert all(summary.attempt_count == 4 for summary in report.fixed_cap_summaries)


def test_known_time_schedules_are_preregistered_for_every_fixture() -> None:
    config = CCFAMatchedBenchmarkConfig(
        repetitions=1,
        query_caps=(0,),
        coherent=_coherent_config(),
        baselines=_baseline_config(),
        known_time_controls=(KnownTimeControlSpec("different", "fixture", (4, 4, 4, 4, 4, 4)),),
    )

    with pytest.raises(ValueError, match="exactly one schedule per fixture"):
        tuple(iter_ccfa_matched_trials((_instance(),), config))
