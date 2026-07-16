from __future__ import annotations

import inspect
import json

import pytest

from qgapselect.oracles import CanonicalBernoulliOracleSimulator
from qgapselect.strong_composition_s3 import (
    BACKEND,
    FIDELITY_STATUS,
    INFORMATION_REGIME,
    OUTPUT_EXACT,
    OUTPUT_INCONCLUSIVE,
    FixedPrecisionGlobalTopKBAI,
    PublicCapUnknownTimeSearchComposition,
    RepeatedFixedPrecisionPhaseBAI,
    S3ExecutionConfig,
    aggregate_s3_attempts,
    score_s3_attempt,
)


def _easy_config() -> S3ExecutionConfig:
    return S3ExecutionConfig(
        phase_powers=(0, 1, 3, 7, 15),
        fixed_shots_per_power=512,
        unknown_time_shots_per_level=512,
        grid_points=2049,
    )


@pytest.mark.parametrize(
    "method",
    [
        FixedPrecisionGlobalTopKBAI(_easy_config()),
        RepeatedFixedPrecisionPhaseBAI(_easy_config()),
        PublicCapUnknownTimeSearchComposition(_easy_config()),
    ],
)
def test_same_interface_controls_certify_an_easy_instance(method) -> None:
    oracle = CanonicalBernoulliOracleSimulator((0.99, 0.90, 0.10, 0.01), seed=19)
    result = method.run(
        n=4,
        k=2,
        delta=0.05,
        oracle=oracle,
        atomic_query_cap=10_000_000,
    )

    assert result.certified
    assert result.output_relation == OUTPUT_EXACT
    assert result.output_indices == (0, 1)
    assert result.output_mask == 0b0011
    assert result.status == "CERTIFIED_EXACT_TOP_K"
    assert result.exact_canonical_query_count == result.query_counts["total"]
    assert result.exact_canonical_query_count == sum(result.per_arm_query_counts.values())
    assert result.exact_canonical_query_count == sum(
        stage.query_counts["total"] for stage in result.stages
    )
    fixed_arm_charge = 512 * sum(2 * power + 1 for power in (0, 1, 3, 7, 15))
    expected_queries = {
        "fixed_precision_global_topk_bai": 4 * fixed_arm_charge,
        "repeated_fixed_precision_phase_bai": (4 + 3) * fixed_arm_charge,
        "public_cap_unknown_time_search_composition": 4 * 512,
    }
    assert result.exact_canonical_query_count == expected_queries[result.method_id]
    assert result.budget_valid
    assert result.information_regime == INFORMATION_REGIME
    assert result.fidelity_status == FIDELITY_STATUS
    assert result.backend == BACKEND
    assert not result.qpe_circuit_executed
    assert not result.coherent_variable_time_search_executed
    assert not result.official_literature_reproduction
    assert not result.registry_coverage_activated
    assert not result.hardware_claimable
    assert not result.quantum_advantage_claimable
    assert result.attempt_must_remain_in_denominator
    assert json.loads(json.dumps(result.as_dict()))["budget_valid"] is True
    if result.method_id == "public_cap_unknown_time_search_composition":
        assert result.query_counts["forward"] > 0
        assert result.query_counts["controlled_forward"] == 0
    else:
        assert result.query_counts["controlled_forward"] > 0
        assert result.query_counts["forward"] == 0


def test_runtime_signature_has_only_the_canonical_five_fields() -> None:
    expected = {"self", "n", "k", "delta", "oracle", "atomic_query_cap"}
    for method_type in (
        FixedPrecisionGlobalTopKBAI,
        RepeatedFixedPrecisionPhaseBAI,
        PublicCapUnknownTimeSearchComposition,
    ):
        signature = inspect.signature(method_type.run)
        assert set(signature.parameters) == expected
        assert all(
            forbidden not in signature.parameters
            for forbidden in (
                "gap",
                "family",
                "truth",
                "schedule",
                "threshold",
                "stop_levels",
                "partition",
            )
        )


@pytest.mark.parametrize(
    ("method", "cap", "expected_queries"),
    [
        (
            RepeatedFixedPrecisionPhaseBAI(
                S3ExecutionConfig(
                    phase_powers=(0, 1),
                    fixed_shots_per_power=8,
                    unknown_time_shots_per_level=8,
                    grid_points=257,
                )
            ),
            10,
            8,
        ),
        (
            PublicCapUnknownTimeSearchComposition(
                S3ExecutionConfig(
                    phase_powers=(0, 1),
                    fixed_shots_per_power=8,
                    unknown_time_shots_per_level=8,
                    grid_points=257,
                )
            ),
            20,
            16,
        ),
    ],
)
def test_hard_cap_rejects_the_first_overshooting_atomic_experiment(
    method, cap: int, expected_queries: int
) -> None:
    result = method.run(
        n=3,
        k=1,
        delta=0.05,
        oracle=CanonicalBernoulliOracleSimulator((0.9, 0.5, 0.1), seed=4),
        atomic_query_cap=cap,
    )

    assert not result.certified
    assert result.inconclusive
    assert result.output_relation == OUTPUT_INCONCLUSIVE
    assert result.output_indices is None
    assert result.output_mask is None
    assert result.status == "INCONCLUSIVE_QUERY_CAP"
    assert result.exact_canonical_query_count == expected_queries
    assert result.exact_canonical_query_count <= cap
    assert result.budget_valid


@pytest.mark.parametrize(
    "method",
    [
        RepeatedFixedPrecisionPhaseBAI(
            S3ExecutionConfig(
                phase_powers=(0, 1, 3, 7),
                fixed_shots_per_power=128,
                unknown_time_shots_per_level=128,
                grid_points=1025,
            )
        ),
        PublicCapUnknownTimeSearchComposition(
            S3ExecutionConfig(
                phase_powers=(0, 1, 3, 7),
                fixed_shots_per_power=128,
                unknown_time_shots_per_level=128,
                grid_points=1025,
            )
        ),
    ],
)
def test_exact_tie_fails_closed_without_a_partial_mask(method) -> None:
    result = method.run(
        n=3,
        k=1,
        delta=0.01,
        oracle=CanonicalBernoulliOracleSimulator((0.8, 0.8, 0.2), seed=3),
        atomic_query_cap=1_000_000,
    )
    score = score_s3_attempt(result, None)

    assert result.inconclusive
    assert result.output_indices is None
    assert result.output_mask is None
    assert score.included_in_all_attempt_denominator
    assert score.all_attempt_success
    assert score.fail_closed_success
    assert not score.certified_exact_success
    assert not score.incorrect_certificate


def test_scoring_and_aggregation_never_filter_inconclusive_attempts() -> None:
    method = PublicCapUnknownTimeSearchComposition(
        S3ExecutionConfig(
            phase_powers=(0, 1),
            fixed_shots_per_power=8,
            unknown_time_shots_per_level=8,
            grid_points=257,
        )
    )
    strict_result = method.run(
        n=3,
        k=1,
        delta=0.05,
        oracle=CanonicalBernoulliOracleSimulator((0.9, 0.5, 0.1), seed=1),
        atomic_query_cap=20,
    )
    tie_result = method.run(
        n=3,
        k=1,
        delta=0.05,
        oracle=CanonicalBernoulliOracleSimulator((0.8, 0.8, 0.2), seed=2),
        atomic_query_cap=20,
    )
    scores = (
        score_s3_attempt(strict_result, (0,)),
        score_s3_attempt(tie_result, None),
    )
    aggregate = aggregate_s3_attempts(scores)

    assert aggregate.all_attempt_count == 2
    assert aggregate.all_attempt_success_count == 1
    assert aggregate.all_attempt_success_rate == 0.5
    assert aggregate.certified_exact_count == 0
    assert aggregate.certified_exact_rate_all_attempts == 0.0
    assert aggregate.fail_closed_success_count == 1
    assert aggregate.inconclusive_count == 2
    assert aggregate.inconclusive_rate_all_attempts == 1.0
    assert aggregate.incorrect_certificate_count == 0
    assert aggregate.budget_violation_count == 0


def test_trusted_scorer_exposes_an_incorrect_certificate_without_filtering_it() -> None:
    result = FixedPrecisionGlobalTopKBAI(_easy_config()).run(
        n=4,
        k=2,
        delta=0.05,
        oracle=CanonicalBernoulliOracleSimulator((0.99, 0.90, 0.10, 0.01), seed=19),
        atomic_query_cap=10_000_000,
    )
    score = score_s3_attempt(result, (0, 2))

    assert result.certified
    assert score.included_in_all_attempt_denominator
    assert not score.all_attempt_success
    assert not score.certified_exact_success
    assert score.incorrect_certificate


def test_nonfresh_oracle_and_mismatched_n_are_rejected() -> None:
    method = RepeatedFixedPrecisionPhaseBAI(_easy_config())
    oracle = CanonicalBernoulliOracleSimulator((0.9, 0.1), seed=2)
    with pytest.raises(ValueError, match="exactly equal"):
        method.run(
            n=3,
            k=1,
            delta=0.05,
            oracle=oracle,
            atomic_query_cap=1000,
        )

    oracle.run_grover_experiment(0, 0, 1)
    with pytest.raises(ValueError, match="fresh zero-ledger"):
        method.run(
            n=2,
            k=1,
            delta=0.05,
            oracle=oracle,
            atomic_query_cap=1000,
        )


@pytest.mark.parametrize(
    "invalid",
    [
        {"phase_powers": ()},
        {"phase_powers": (0, 2)},
        {"phase_powers": (1, 0)},
        {"fixed_shots_per_power": 0},
        {"unknown_time_shots_per_level": 0},
        {"grid_points": 256},
    ],
)
def test_compile_schedule_rejects_invalid_values(invalid: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        S3ExecutionConfig(**invalid)
