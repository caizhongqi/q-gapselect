"""Falsifiable small-instance witnesses for the S3 proof frontier.

The module deliberately separates three statements that are easy to conflate:

* a two-input hybrid lower bound for the repository's canonical indexed
  :math:`R_y(2\theta)` oracle;
* a finite Johnson-graph adversary-matrix objective for the *standard discrete*
  fixed-weight bit-query oracle; and
* a same-oracle-model finite-fixture diagnostic against the existing all-arm
  coherent-QPE implementation.

All three records expose the same claim-boundary fields.  Numerical spectral
checks certify the stated finite witnesses only.  They do not prove an
asymptotic lower bound for continuous-angle Top-k, a matching lower bound for
Q-GapSelect, or a composition-frontier theorem.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .coherent import CanonicalRyStatevectorOracle
from .coherent_adaptive_stopping_history import (
    TinyCoherentStoppingHistoryConfig,
    TinyCoherentStoppingHistoryResult,
    run_tiny_coherent_adaptive_stopping_history,
)
from .coherent_unknown_boundary_topk import (
    CoherentUnknownBoundaryTopKConfig,
    CoherentUnknownBoundaryTopKResult,
    run_coherent_unknown_boundary_topk,
)

FloatMatrix = NDArray[np.float64]

CANONICAL_INTERFACE_ID = "canonical_blind_exact_topk_v1"
CANONICAL_ROTATION_ORACLE = (
    "indexed_B_theta_with_forward_inverse_and_controlled_calls_B_theta_block_equals_Ry_2theta"
)
STANDARD_FIXED_WEIGHT_BIT_ORACLE = "standard_indexed_bit_query_oracle_on_weight_k_strings"
COMPLETE_UNIQUE_TOPK_RELATION = "complete_unique_exact_top_k_membership"
PAIR_WITNESS_TYPE = "paired_canonical_rotation_hybrid"
JOHNSON_WITNESS_TYPE = "johnson_positive_adversary_matrix"
COMPOSITION_WITNESS_TYPE = "same_oracle_model_exact_grid_finite_diagnostic"
PAIR_NON_THEOREM_BOUNDARY = (
    "This is a local two-input all-algorithms hybrid bound. It does not imply "
    "an n-arm direct-sum factor, an activity-history lower bound, or a matching "
    "lower bound for Q-GapSelect."
)
JOHNSON_NON_THEOREM_BOUNDARY = (
    "This finite matrix is for a standard discrete symbol oracle. Transferring "
    "its Johnson factor to the continuous canonical rotation oracle requires a "
    "separate general-oracle adversary/composition argument; no such transfer "
    "is certified here."
)
COMPOSITION_NON_THEOREM_BOUNDARY = (
    "The harness independently executes two implementations on one exact-grid "
    "fixture in the same oracle model. Neither implementation accepts the full "
    "registered (n,k,delta,atomic_query_cap) interface or issues a delta-sound "
    "certificate. The query ratio is therefore a finite diagnostic, not a "
    "same-interface composition kill, published-baseline result, or theorem."
)


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _spectral_norm(matrix: FloatMatrix) -> float:
    return float(np.linalg.svd(matrix, compute_uv=False)[0])


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _rotation_block(theta: float) -> FloatMatrix:
    return np.asarray(
        (
            (math.cos(theta), -math.sin(theta)),
            (math.sin(theta), math.cos(theta)),
        ),
        dtype=np.float64,
    )


def _indexed_rotation_matrix(angles: Sequence[float]) -> FloatMatrix:
    dimension = _next_power_of_two(len(angles))
    result = np.zeros((2 * dimension, 2 * dimension), dtype=np.float64)
    for index in range(dimension):
        block = _rotation_block(angles[index]) if index < len(angles) else np.eye(2)
        start = 2 * index
        result[start : start + 2, start : start + 2] = block
    return result


def _controlled(matrix: FloatMatrix) -> FloatMatrix:
    dimension = matrix.shape[0]
    result = np.zeros((2 * dimension, 2 * dimension), dtype=np.float64)
    result[:dimension, :dimension] = np.eye(dimension)
    result[dimension:, dimension:] = matrix
    return result


def _top_k_indices(angles: Sequence[float], k: int) -> tuple[int, ...]:
    ranking = sorted(range(len(angles)), key=lambda index: (-angles[index], index))
    if not angles[ranking[k - 1]] > angles[ranking[k]]:
        raise ValueError("the explicit hard instance must have a strict Top-k boundary")
    return tuple(sorted(ranking[:k]))


@dataclass(frozen=True, slots=True)
class PairedRotationHybridCertificate:
    """One explicit two-input hybrid certificate for canonical rotations."""

    witness_id: str
    witness_type: str
    oracle_interface: str
    output_relation: str
    computed_quantity_name: str
    computed_quantity: float
    verified_local_statement: str
    explicit_non_theorem_boundary: str
    composition_match: bool
    composition_kill_flag: bool
    n: int
    k: int
    error_probability: float
    low_angle: float
    high_angle: float
    angle_gap: float
    swapped_indices: tuple[int, int]
    instance_x_top_k: tuple[int, ...]
    instance_y_top_k: tuple[int, ...]
    required_output_total_variation: float
    forward_difference_norm_numeric: float
    forward_difference_norm_analytic: float
    inverse_difference_norm_numeric: float
    controlled_difference_norm_numeric: float
    norm_formula_residual: float
    integer_query_lower_bound: int
    verification_passed: bool
    blockers: tuple[str, ...]
    quantum_advantage_claimable: bool = False
    matching_lower_bound_claimable: bool = False


def paired_rotation_hybrid_certificate(
    *,
    witness_id: str,
    n: int,
    k: int,
    low_angle: float,
    high_angle: float,
    error_probability: float,
    tolerance: float = 1e-12,
) -> PairedRotationHybridCertificate:
    r"""Build and check a local hybrid lower-bound certificate.

    The canonical block is ``R_y(2 theta)`` in quantum-gate notation but its
    real matrix is a planar rotation by ``theta``.  Consequently

    ``||B_theta - B_phi|| = 2 sin(|theta-phi| / 2)``.

    If two unique Top-k outputs differ, success at least ``1-epsilon`` on both
    makes their output distributions have total variation at least
    ``1-2 epsilon``.  A T-query hybrid has distance at most ``T`` times the
    maximum forward/inverse/controlled oracle difference, giving the recorded
    local bound.
    """

    if not isinstance(witness_id, str) or not witness_id.strip():
        raise TypeError("witness_id must be a non-empty string")
    n = _integer(n, "n", minimum=2)
    k = _integer(k, "k", minimum=1)
    if k >= n:
        raise ValueError("k must be smaller than n")
    low = _finite_real(low_angle, "low_angle")
    high = _finite_real(high_angle, "high_angle")
    if not 0.0 <= low < high <= math.pi / 2.0:
        raise ValueError("angles must satisfy 0 <= low_angle < high_angle <= pi/2")
    error = _finite_real(error_probability, "error_probability")
    if not 0.0 <= error < 0.5:
        raise ValueError("error_probability must lie in [0, 1/2)")
    atol = _finite_real(tolerance, "tolerance")
    if atol <= 0.0:
        raise ValueError("tolerance must be positive")

    angles_x = [high] * k + [low] * (n - k)
    angles_y = list(angles_x)
    left, right = k - 1, k
    angles_y[left], angles_y[right] = angles_y[right], angles_y[left]
    top_x = _top_k_indices(angles_x, k)
    top_y = _top_k_indices(angles_y, k)

    oracle_x = _indexed_rotation_matrix(angles_x)
    oracle_y = _indexed_rotation_matrix(angles_y)
    forward_norm = _spectral_norm(oracle_x - oracle_y)
    inverse_norm = _spectral_norm(oracle_x.T - oracle_y.T)
    controlled_norm = _spectral_norm(_controlled(oracle_x) - _controlled(oracle_y))
    gap = high - low
    analytic_norm = 2.0 * math.sin(gap / 2.0)
    residual = max(
        abs(forward_norm - analytic_norm),
        abs(inverse_norm - analytic_norm),
        abs(controlled_norm - analytic_norm),
    )
    required_tv = 1.0 - 2.0 * error
    lower_bound = required_tv / analytic_norm
    integer_lower_bound = max(1, math.ceil(lower_bound - atol))
    verified = top_x != top_y and residual <= atol
    blockers = (
        "only_two_explicit_inputs",
        "no_n_arm_direct_sum_or_johnson_factor_for_rotation_oracle",
        "no_activity_history_or_multi_output_lower_bound",
        "no_matching_upper_lower_closure",
    )
    statement = (
        f"For the explicit pair, every algorithm in the declared canonical "
        f"forward/inverse/controlled query model with error <= {error:.17g} "
        f"obeys T >= (1-2*error)/||B_x-B_y|| = {lower_bound:.17g}."
    )
    return PairedRotationHybridCertificate(
        witness_id=witness_id,
        witness_type=PAIR_WITNESS_TYPE,
        oracle_interface=CANONICAL_ROTATION_ORACLE,
        output_relation=COMPLETE_UNIQUE_TOPK_RELATION,
        computed_quantity_name="hybrid_query_lower_bound_real",
        computed_quantity=lower_bound,
        verified_local_statement=statement,
        explicit_non_theorem_boundary=PAIR_NON_THEOREM_BOUNDARY,
        composition_match=False,
        composition_kill_flag=False,
        n=n,
        k=k,
        error_probability=error,
        low_angle=low,
        high_angle=high,
        angle_gap=gap,
        swapped_indices=(left, right),
        instance_x_top_k=top_x,
        instance_y_top_k=top_y,
        required_output_total_variation=required_tv,
        forward_difference_norm_numeric=forward_norm,
        forward_difference_norm_analytic=analytic_norm,
        inverse_difference_norm_numeric=inverse_norm,
        controlled_difference_norm_numeric=controlled_norm,
        norm_formula_residual=residual,
        integer_query_lower_bound=integer_lower_bound,
        verification_passed=verified,
        blockers=blockers,
    )


@dataclass(frozen=True, slots=True)
class JohnsonAdversaryCertificate:
    """A finite positive-adversary witness on weight-k bit strings."""

    witness_id: str
    witness_type: str
    oracle_interface: str
    output_relation: str
    computed_quantity_name: str
    computed_quantity: float
    verified_local_statement: str
    explicit_non_theorem_boundary: str
    composition_match: bool
    composition_kill_flag: bool
    n: int
    k: int
    input_count: int
    johnson_degree: int
    numerator_spectral_norm: float
    numerator_expected: float
    filtered_spectral_norms: tuple[float, ...]
    maximum_filtered_spectral_norm: float
    filtered_norm_expected: float
    objective_expected: float
    maximum_spectral_residual: float
    symmetric: bool
    zero_diagonal: bool
    support_respects_distinct_outputs: bool
    verification_passed: bool
    primary_source: str
    source_locator: str
    blockers: tuple[str, ...]
    quantum_advantage_claimable: bool = False
    matching_lower_bound_claimable: bool = False


def johnson_adversary_certificate(
    *,
    witness_id: str,
    n: int,
    k: int,
    tolerance: float = 1e-10,
    maximum_input_count: int = 512,
) -> JohnsonAdversaryCertificate:
    """Compute a Johnson-adjacency positive-adversary objective for small n."""

    if not isinstance(witness_id, str) or not witness_id.strip():
        raise TypeError("witness_id must be a non-empty string")
    n = _integer(n, "n", minimum=2)
    k = _integer(k, "k", minimum=1)
    if k >= n:
        raise ValueError("k must be smaller than n")
    cap = _integer(maximum_input_count, "maximum_input_count", minimum=2)
    atol = _finite_real(tolerance, "tolerance")
    if atol <= 0.0:
        raise ValueError("tolerance must be positive")

    subsets = tuple(itertools.combinations(range(n), k))
    if len(subsets) > cap:
        raise ValueError(f"C(n,k)={len(subsets)} exceeds maximum_input_count={cap}")
    encoded = tuple(frozenset(subset) for subset in subsets)
    dimension = len(encoded)
    gamma = np.zeros((dimension, dimension), dtype=np.float64)
    for row, left in enumerate(encoded):
        for column in range(row + 1, dimension):
            right = encoded[column]
            if len(left.symmetric_difference(right)) == 2:
                gamma[row, column] = gamma[column, row] = 1.0

    numerator = _spectral_norm(gamma)
    filtered: list[float] = []
    for arm in range(n):
        delta = np.fromiter(
            (float((arm in left) != (arm in right)) for left in encoded for right in encoded),
            dtype=np.float64,
            count=dimension * dimension,
        ).reshape(dimension, dimension)
        filtered.append(_spectral_norm(gamma * delta))
    denominator = max(filtered)
    objective = numerator / denominator

    expected_numerator = float(k * (n - k))
    expected_filtered = math.sqrt(k * (n - k))
    expected_objective = expected_filtered
    residual = max(
        abs(numerator - expected_numerator),
        *(abs(value - expected_filtered) for value in filtered),
        abs(objective - expected_objective),
    )
    symmetric = bool(np.allclose(gamma, gamma.T, atol=atol, rtol=0.0))
    zero_diagonal = bool(np.allclose(np.diag(gamma), 0.0, atol=atol, rtol=0.0))
    support_ok = all(
        row == column or gamma[row, column] == 0.0 or encoded[row] != encoded[column]
        for row in range(dimension)
        for column in range(dimension)
    )
    verified = symmetric and zero_diagonal and support_ok and residual <= atol
    statement = (
        f"The explicit {dimension}x{dimension} Johnson adjacency matrix is a "
        f"feasible positive-adversary witness for the standard weight-{k} "
        f"symbol oracle, with computed objective {objective:.17g}."
    )
    return JohnsonAdversaryCertificate(
        witness_id=witness_id,
        witness_type=JOHNSON_WITNESS_TYPE,
        oracle_interface=STANDARD_FIXED_WEIGHT_BIT_ORACLE,
        output_relation=COMPLETE_UNIQUE_TOPK_RELATION,
        computed_quantity_name="positive_adversary_objective",
        computed_quantity=objective,
        verified_local_statement=statement,
        explicit_non_theorem_boundary=JOHNSON_NON_THEOREM_BOUNDARY,
        composition_match=False,
        composition_kill_flag=False,
        n=n,
        k=k,
        input_count=dimension,
        johnson_degree=k * (n - k),
        numerator_spectral_norm=numerator,
        numerator_expected=expected_numerator,
        filtered_spectral_norms=tuple(filtered),
        maximum_filtered_spectral_norm=denominator,
        filtered_norm_expected=expected_filtered,
        objective_expected=expected_objective,
        maximum_spectral_residual=residual,
        symmetric=symmetric,
        zero_diagonal=zero_diagonal,
        support_respects_distinct_outputs=support_ok,
        verification_passed=verified,
        primary_source="https://arxiv.org/abs/quant-ph/0611054v2",
        source_locator="positive/general adversary spectral-ratio formulation",
        blockers=(
            "standard_discrete_symbol_oracle_not_canonical_rotation_oracle",
            "continuous_angle_scaling_not_composed",
            "finite_n_numeric_certificate_not_general_n_symbolic_proof",
        ),
    )


@dataclass(frozen=True, slots=True)
class CompositionFalsificationCertificate:
    """A fail-closed same-oracle-model finite-fixture diagnostic."""

    witness_id: str
    witness_type: str
    oracle_interface: str
    output_relation: str
    computed_quantity_name: str
    computed_quantity: float
    verified_local_statement: str
    explicit_non_theorem_boundary: str
    composition_match: bool
    composition_kill_flag: bool
    candidate_name: str
    baseline_name: str
    candidate_query_count: int
    baseline_query_count: int
    match_tolerance: float
    fixture_sha256: str
    same_oracle_model_verified: bool
    same_fixture_harness_verified: bool
    distinct_oracle_instances_verified: bool
    distinct_implementation_classes_verified: bool
    same_public_algorithm_interface_verified: bool
    same_certified_output_contract_verified: bool
    finite_fixture_query_dominance_verified: bool
    candidate_complete_output_verified: bool
    candidate_cleanup_verified: bool
    candidate_truth_match_verified: bool
    candidate_certificate_issued: bool
    candidate_query_ledger_reconciled: bool
    baseline_complete_output_verified: bool
    baseline_cleanup_verified: bool
    baseline_truth_match_verified: bool
    baseline_certificate_issued: bool
    baseline_query_ledger_reconciled: bool
    registered_published_baseline_fidelity_verified: bool
    global_composition_frontier_closed: bool
    verification_passed: bool
    blockers: tuple[str, ...]
    quantum_advantage_claimable: bool = False
    matching_lower_bound_claimable: bool = False


def composition_falsification_certificate(
    *,
    witness_id: str,
    means: Sequence[float],
    k: int,
    phase_qubits: int,
    cleanup_tolerance: float,
    max_statevector_dimension: int,
    match_tolerance: float = 1.0,
) -> CompositionFalsificationCertificate:
    """Execute an S3 candidate and all-arm QPE on one harness-owned fixture.

    The function itself constructs two distinct oracle objects from the same
    private fixture and executes two distinct implementation classes.  This
    closes the earlier result-pair binding hole.  It intentionally refuses to
    call the result a same-interface composition match because neither
    implementation exposes ``delta`` and ``atomic_query_cap`` or issues the
    registered delta-sound output certificate.
    """

    if not isinstance(witness_id, str) or not witness_id.strip():
        raise TypeError("witness_id must be a non-empty string")
    fixture = tuple(_finite_real(mean, "means item") for mean in means)
    if len(fixture) != 2:
        raise ValueError("the tiny S3 diagnostic requires exactly two means")
    if any(not 0.0 <= mean <= 1.0 for mean in fixture):
        raise ValueError("means must lie in [0, 1]")
    k = _integer(k, "k", minimum=1)
    if k != 1:
        raise ValueError("the tiny S3 diagnostic requires k=1")
    phase_qubits = _integer(phase_qubits, "phase_qubits", minimum=1)
    max_dimension = _integer(
        max_statevector_dimension,
        "max_statevector_dimension",
        minimum=1,
    )
    cleanup = _finite_real(cleanup_tolerance, "cleanup_tolerance")
    if cleanup <= 0.0:
        raise ValueError("cleanup_tolerance must be positive")
    tolerance = _finite_real(match_tolerance, "match_tolerance")
    if tolerance < 1.0:
        raise ValueError("match_tolerance must be at least one")

    ranking = sorted(range(len(fixture)), key=lambda index: (-fixture[index], index))
    if not fixture[ranking[k - 1]] > fixture[ranking[k]]:
        raise ValueError("the finite diagnostic requires a strict Top-k fixture")
    truth_mask = sum(1 << arm for arm in ranking[:k])
    fixture_payload = json.dumps(
        {"k": k, "means": fixture},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    fixture_sha256 = hashlib.sha256(fixture_payload).hexdigest()

    candidate_oracle = CanonicalRyStatevectorOracle(fixture)
    baseline_oracle = CanonicalRyStatevectorOracle(fixture)
    distinct_oracles = candidate_oracle is not baseline_oracle
    candidate_result = run_tiny_coherent_adaptive_stopping_history(
        candidate_oracle,
        config=TinyCoherentStoppingHistoryConfig(
            cleanup_tolerance=cleanup,
            max_statevector_dimension=max_dimension,
        ),
    )
    baseline_result = run_coherent_unknown_boundary_topk(
        baseline_oracle,
        k,
        config=CoherentUnknownBoundaryTopKConfig(
            phase_qubits=phase_qubits,
            cleanup_tolerance=cleanup,
            max_statevector_dimension=max_dimension,
        ),
    )
    if not isinstance(candidate_result, TinyCoherentStoppingHistoryResult):
        raise TypeError("candidate runner returned the wrong result type")
    if not isinstance(baseline_result, CoherentUnknownBoundaryTopKResult):
        raise TypeError("baseline runner returned the wrong result type")

    candidate_query_counts = candidate_result.resources.query_ledger.query_counts
    executed_candidate_queries = int(candidate_query_counts.get("coherent_total", -1))
    candidate_charged_sum = sum(
        int(candidate_query_counts.get(name, 0))
        for name in (
            "forward",
            "inverse",
            "controlled_forward",
            "controlled_inverse",
        )
    )
    candidate_expected = candidate_result.resources.query_ledger.expected_query_counts
    per_level_ok = all(
        record.full_replay_reconciled and record.one_way_reconciled
        for record in candidate_result.resources.query_ledger.per_level_runtime_records
    )
    candidate_ledger_ok = (
        executed_candidate_queries > 0
        and candidate_charged_sum == executed_candidate_queries
        and int(candidate_query_counts.get("classical_sample", -1)) == 0
        and int(candidate_query_counts.get("classical_total", -1)) == 0
        and int(candidate_query_counts.get("total", -1)) == executed_candidate_queries
        and int(candidate_query_counts.get("qram_queries", -1)) == 0
        and not candidate_result.resources.query_ledger.qram_assumed
        and dict(candidate_query_counts) == dict(candidate_expected)
        and candidate_result.resources.query_ledger.reconciled
        and per_level_ok
        and candidate_result.fixed_expected_query_ledger_respected
        and candidate_result.budget_valid
    )
    query_counts = baseline_result.resources.query_counts
    charged_sum = sum(
        int(query_counts.get(name, 0))
        for name in (
            "forward",
            "inverse",
            "controlled_forward",
            "controlled_inverse",
        )
    )
    baseline_reported_queries = int(query_counts.get("coherent_total", -1))
    ledger_ok = (
        baseline_reported_queries > 0
        and charged_sum == baseline_reported_queries
        and baseline_reported_queries == baseline_result.resources.oracle_queries
        and int(query_counts.get("classical_sample", -1)) == 0
        and int(query_counts.get("classical_total", -1)) == 0
        and int(query_counts.get("total", -1)) == baseline_reported_queries
        and int(query_counts.get("qram_queries", -1)) == 0
        and not baseline_result.resources.qram_assumed
    )
    baseline_queries = charged_sum
    candidate_complete = candidate_result.durable_output.status == "MASK"
    candidate_cleanup = candidate_result.resources.cleanup.passed
    candidate_truth_match = candidate_result.membership_mask == truth_mask
    baseline_complete = baseline_result.direct_multi_output_complete
    baseline_cleanup = baseline_result.resources.cleanup.passed
    baseline_truth_match = baseline_result.membership_mask == truth_mask
    ratio = (
        baseline_queries / executed_candidate_queries
        if executed_candidate_queries > 0
        else -1.0
    )
    same_oracle_model = (
        candidate_oracle.contract.model == baseline_oracle.contract.model
        and candidate_oracle.contract == baseline_oracle.contract
    )
    same_fixture = distinct_oracles and candidate_oracle.n_arms == baseline_oracle.n_arms
    distinct_implementations = type(candidate_result) is not type(baseline_result)
    finite_dominance = (
        same_oracle_model
        and same_fixture
        and distinct_oracles
        and distinct_implementations
        and candidate_complete
        and candidate_cleanup
        and candidate_truth_match
        and candidate_ledger_ok
        and baseline_complete
        and baseline_cleanup
        and baseline_truth_match
        and ledger_ok
        and ratio >= 0.0
        and ratio <= tolerance
    )
    blockers: list[str] = [
        "registered_public_algorithm_interface_not_shared",
        "delta_sound_output_contract_not_implemented",
        "published_baseline_runtime_fidelity_not_verified",
        "finite_fixture_only_no_asymptotic_composition_statement",
        "exact_grid_rounding_promise_only",
    ]
    if not same_oracle_model:
        blockers.append("oracle_model_mismatch")
    if not same_fixture:
        blockers.append("same_fixture_harness_not_verified")
    if not distinct_oracles:
        blockers.append("distinct_oracle_instances_not_verified")
    if not distinct_implementations:
        blockers.append("distinct_implementation_classes_not_verified")
    if not candidate_complete:
        blockers.append("candidate_complete_output_not_verified")
    if not candidate_cleanup:
        blockers.append("candidate_cleanup_not_verified")
    if not candidate_truth_match:
        blockers.append("candidate_truth_match_not_verified")
    if not candidate_ledger_ok:
        blockers.append("candidate_query_ledger_not_reconciled")
    if not baseline_complete:
        blockers.append("baseline_complete_output_not_verified")
    if not baseline_cleanup:
        blockers.append("baseline_cleanup_not_verified")
    if not baseline_truth_match:
        blockers.append("baseline_truth_match_not_verified")
    if not ledger_ok:
        blockers.append("baseline_query_ledger_not_reconciled")
    if ratio < 0.0 or ratio > tolerance:
        blockers.append("baseline_does_not_match_candidate_budget")
    statement = (
        "Two distinct implementations on one harness-owned exact-grid fixture "
        "have reconciled query ledgers, and the all-arm comparator uses no more "
        "queries than the S3 candidate. Because their public confidence/budget "
        "interfaces and certified output contracts are incomplete, this is a "
        "finite same-oracle-model diagnostic and no composition kill is issued."
        if finite_dominance
        else "The finite same-oracle-model diagnostic did not pass every fixture, "
        "implementation, output, cleanup, truth, ledger, and budget check."
    )
    return CompositionFalsificationCertificate(
        witness_id=witness_id,
        witness_type=COMPOSITION_WITNESS_TYPE,
        oracle_interface=CANONICAL_ROTATION_ORACLE,
        output_relation=COMPLETE_UNIQUE_TOPK_RELATION,
        computed_quantity_name="baseline_queries_over_candidate_queries",
        computed_quantity=ratio,
        verified_local_statement=statement,
        explicit_non_theorem_boundary=COMPOSITION_NON_THEOREM_BOUNDARY,
        composition_match=False,
        composition_kill_flag=False,
        candidate_name=candidate_result.method_id,
        baseline_name="all_arm_coherent_qpe_rank_copy_cleanup",
        candidate_query_count=executed_candidate_queries,
        baseline_query_count=baseline_queries,
        match_tolerance=tolerance,
        fixture_sha256=fixture_sha256,
        same_oracle_model_verified=same_oracle_model,
        same_fixture_harness_verified=same_fixture,
        distinct_oracle_instances_verified=distinct_oracles,
        distinct_implementation_classes_verified=distinct_implementations,
        same_public_algorithm_interface_verified=False,
        same_certified_output_contract_verified=False,
        finite_fixture_query_dominance_verified=finite_dominance,
        candidate_complete_output_verified=candidate_complete,
        candidate_cleanup_verified=candidate_cleanup,
        candidate_truth_match_verified=candidate_truth_match,
        candidate_certificate_issued=candidate_result.certificate.issued,
        candidate_query_ledger_reconciled=candidate_ledger_ok,
        baseline_complete_output_verified=baseline_complete,
        baseline_cleanup_verified=baseline_cleanup,
        baseline_truth_match_verified=baseline_truth_match,
        baseline_certificate_issued=baseline_result.certificate_issued,
        baseline_query_ledger_reconciled=ledger_ok,
        registered_published_baseline_fidelity_verified=False,
        global_composition_frontier_closed=False,
        verification_passed=finite_dominance,
        blockers=tuple(blockers),
    )


__all__ = [
    "CANONICAL_INTERFACE_ID",
    "CANONICAL_ROTATION_ORACLE",
    "COMPLETE_UNIQUE_TOPK_RELATION",
    "COMPOSITION_NON_THEOREM_BOUNDARY",
    "COMPOSITION_WITNESS_TYPE",
    "JOHNSON_NON_THEOREM_BOUNDARY",
    "JOHNSON_WITNESS_TYPE",
    "PAIR_NON_THEOREM_BOUNDARY",
    "PAIR_WITNESS_TYPE",
    "STANDARD_FIXED_WEIGHT_BIT_ORACLE",
    "CompositionFalsificationCertificate",
    "JohnsonAdversaryCertificate",
    "PairedRotationHybridCertificate",
    "composition_falsification_certificate",
    "johnson_adversary_certificate",
    "paired_rotation_hybrid_certificate",
]
