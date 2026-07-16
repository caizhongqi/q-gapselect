"""Fail-closed adaptive-precision audit for unknown-boundary Top-k.

This module composes the tiny coherent level unitary implemented in
``coherent_unknown_boundary_topk`` at consecutive public QPE precisions.  The
algorithm-facing constructor receives only an opaque canonical oracle, ``k``,
and public error/resource caps.  It receives no answer, gap, boundary, family
label, precision schedule, or activity history.

The implementation deliberately separates two facts that are easy to conflate:

* every *level* is the executed coherent compute-rank-copy-uncompute circuit;
* the controller that inspects the exact statevector diagnostics and decides
  whether to run the next level is classical simulation code.

Consequently the returned stopping history is an auditable emulation of the
history a variable-time construction would have to retain.  It is not a single
coherent variable-time stopping unitary, and its acceptance predicate is not a
free observable in the canonical oracle model.  The prototype never issues a
correctness certificate or claims a query-complexity advantage.
"""

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .coherent import CanonicalRyStatevectorOracle
from .coherent_unknown_boundary_topk import (
    CoherentUnknownBoundaryTopKConfig,
    run_coherent_unknown_boundary_topk,
)
from .oracles import QueryLedger

METHOD_ID = "adaptive_unknown_boundary_topk_exact_state_diagnostic_v1"
BACKEND = "numpy_sequential_coherent_levels_classical_adaptive_controller"
# The sequential controller has no confidence certificate and may retain
# numerical cleanup error.  Its selected set is therefore deliberately not
# serialized under the claim-bearing ``MASK`` label.
OUTPUT_MASK = "DIAGNOSTIC_MASK"
OUTPUT_INCONCLUSIVE = "INCONCLUSIVE"
QUERY_FORMULA = "Q_m = 2*n*(2^(m+1)-1) per executed precision m"
STOPPING_HISTORY_SEMANTICS = (
    "classical_exact_state_diagnostic_history_over_independently_executed_"
    "coherent_level_unitaries_not_a_single_variable_time_unitary"
)
ERROR_BOUND_SCOPE = (
    "exact_simulator_mass_relative_to_the_dominant_durable_mask_not_a_"
    "top_k_correctness_or_confidence_bound"
)
CLAIM_SCOPE = (
    "adaptive_off_grid_exact_state_diagnostic_with_exact_queries_"
    "no_coherent_variable_time_controller_no_correctness_certificate_"
    "no_new_upper_bound_no_lower_bound_no_quantum_advantage"
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


def _probability(value: object, name: str, *, strictly_positive: bool) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real probability, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real probability") from error
    lower_ok = result > 0.0 if strictly_positive else result >= 0.0
    if not math.isfinite(result) or not lower_ok or result >= 0.5:
        interval = "(0, 0.5)" if strictly_positive else "[0, 0.5)"
        raise ValueError(f"{name} must be finite and in {interval}")
    return result


def _frozen_counts(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType({str(key): int(value) for key, value in values.items()})


def _query_counts_with_qram(values: Mapping[str, int]) -> Mapping[str, int]:
    result = {str(key): int(value) for key, value in values.items()}
    result["qram_queries"] = 0
    return _frozen_counts(result)


def _expected_level_query_counts(n_arms: int, phase_qubits: int) -> dict[str, int]:
    bins = 1 << phase_qubits
    controlled_each_kind = 2 * n_arms * (bins - 1)
    coherent_total = 2 * n_arms * (2 * bins - 1)
    return {
        "forward": n_arms,
        "inverse": n_arms,
        "controlled_forward": controlled_each_kind,
        "controlled_inverse": controlled_each_kind,
        "classical_sample": 0,
        "coherent_total": coherent_total,
        "classical_total": 0,
        "total": coherent_total,
        "qram_queries": 0,
    }


def _statevector_dimension(oracle: CanonicalRyStatevectorOracle, phase_qubits: int) -> int:
    n = oracle.n_arms
    phase_dimension = (1 << phase_qubits) ** n
    index_pack_dimension = oracle.index_dimension**n
    reward_pack_dimension = 1 << n
    output_dimension = 1 << n
    rank_work_dimension = 1 << (n + 1)
    return (
        phase_dimension
        * index_pack_dimension
        * reward_pack_dimension
        * output_dimension
        * rank_work_dimension
    )


@dataclass(frozen=True, slots=True)
class AdaptiveUnknownBoundaryTopKConfig:
    """Public precision/error limits; no instance-dependent schedule is accepted."""

    minimum_phase_qubits: int = 1
    maximum_phase_qubits: int = 4
    target_diagnostic_error_probability: float = 0.04
    numerical_cleanup_tolerance: float = 1e-10
    max_statevector_dimension: int = 600_000
    max_canonical_oracle_queries: int = 10_000

    def __post_init__(self) -> None:
        minimum = _integer(
            self.minimum_phase_qubits, "minimum_phase_qubits", minimum=1
        )
        maximum = _integer(
            self.maximum_phase_qubits, "maximum_phase_qubits", minimum=minimum
        )
        if maximum > 5:
            raise ValueError("maximum_phase_qubits exceeds the tiny exact-state limit of 5")
        object.__setattr__(self, "minimum_phase_qubits", minimum)
        object.__setattr__(self, "maximum_phase_qubits", maximum)
        object.__setattr__(
            self,
            "target_diagnostic_error_probability",
            _probability(
                self.target_diagnostic_error_probability,
                "target_diagnostic_error_probability",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "numerical_cleanup_tolerance",
            _probability(
                self.numerical_cleanup_tolerance,
                "numerical_cleanup_tolerance",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "max_statevector_dimension",
            _integer(
                self.max_statevector_dimension,
                "max_statevector_dimension",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "max_canonical_oracle_queries",
            _integer(
                self.max_canonical_oracle_queries,
                "max_canonical_oracle_queries",
                minimum=0,
            ),
        )


@dataclass(frozen=True, slots=True)
class AdaptiveUnknownBoundaryInputInterface:
    """Machine-checkable statement of allowed and forbidden inputs."""

    oracle: str
    k: int
    public_precision_bounds: tuple[int, int]
    public_target_diagnostic_error_probability: float
    public_numerical_cleanup_tolerance: float
    public_statevector_dimension_cap: int
    public_canonical_query_hard_cap: int
    internally_constructed_precision_policy: str
    forbidden_inputs: tuple[str, ...]
    answer_dependent_inputs_supplied: bool = False
    gap_supplied: bool = False
    boundary_supplied: bool = False
    family_label_supplied: bool = False
    precision_schedule_supplied: bool = False
    activity_history_supplied: bool = False


@dataclass(frozen=True, slots=True)
class AdaptiveStoppingLevel:
    """One executed precision and its exact-state diagnostic stop bit."""

    level_index: int
    phase_qubits: int
    phase_bins: int
    retained_statevector_dimension: int
    dominant_mask: int
    dominant_mask_bit_count: int
    dominant_probability: float
    statevector_output_disagreement_mass: float
    strict_boundary_probability: float
    nonstrict_boundary_mass: float
    executed_transient_nonzero_probability: float
    predicted_transient_nonzero_probability: float
    cleanup_prediction_residual: float
    output_reduced_purity: float
    output_collision_probability: float
    purity_residual: float
    exact_numerical_cleanup_passed: bool
    diagnostic_error_bound: float
    diagnostic_acceptance_bound_passed: bool
    query_counts: Mapping[str, int]
    expected_query_counts: Mapping[str, int]
    query_formula_reconciled: bool
    cumulative_coherent_queries: int
    emulated_stop_bit: int
    stop_latched_after_level: bool
    level_status: str
    stop_decision_source: str = (
        "exact_statevector_introspection_not_an_observable_free_oracle_output"
    )
    error_bound_scope: str = ERROR_BOUND_SCOPE


@dataclass(frozen=True, slots=True)
class AdaptiveStoppingHistory:
    """Durable classical audit trail for the intended stopping-register semantics."""

    levels: tuple[AdaptiveStoppingLevel, ...]
    emulated_history_bits: tuple[int, ...]
    first_stop_level_index: int | None
    first_stop_phase_qubits: int | None
    controller_is_classical: bool
    independently_coherent_level_unitaries_executed: bool
    single_coherent_variable_time_unitary_implemented: bool
    coherent_history_register_physically_retained: bool
    coherent_history_cleanup_proved: bool
    semantics: str = STOPPING_HISTORY_SEMANTICS


@dataclass(frozen=True, slots=True)
class AdaptiveDurableOutput:
    """Direct membership-register diagnostic, or an explicit inconclusive result."""

    status: str
    membership_mask: int | None
    membership_bits: tuple[int, ...]
    selected_phase_qubits: int | None
    direct_k_membership_register_used: bool
    rank_relation_computed_and_uncomputed: bool
    phase_estimation_inverted: bool
    exact_transient_cleanup_passed: bool
    approximate_cleanup_bound_passed: bool
    output_disagreement_mass: float | None
    transient_nonzero_probability: float | None
    dominant_mask_selected_by_statevector_introspection: bool
    physical_measurement_performed: bool = False


@dataclass(frozen=True, slots=True)
class AdaptiveCertificate:
    """Certificate is intentionally withheld until an observable stop test exists."""

    issued: bool
    certificate_type: str | None
    top_k_correctness_error_bound: float | None
    reason: str
    exact_statevector_diagnostic_error_bound: float | None
    diagnostic_error_bound_scope: str = ERROR_BOUND_SCOPE


@dataclass(frozen=True, slots=True)
class AdaptiveCleanupSummary:
    """Cleanup evidence at the selected or last executed precision."""

    available: bool
    exact_numerical_cleanup_passed: bool
    diagnostic_acceptance_bound_passed: bool
    executed_transient_nonzero_probability: float | None
    predicted_transient_nonzero_probability: float | None
    cleanup_prediction_residual: float | None
    statevector_output_disagreement_mass: float | None
    nonstrict_boundary_mass: float | None
    diagnostic_error_bound: float | None
    target_diagnostic_error_probability: float
    top_k_correctness_error_bound: float | None
    error_bound_scope: str = ERROR_BOUND_SCOPE


@dataclass(frozen=True, slots=True)
class AdaptiveQueryBudget:
    """Executed and formula-derived canonical query accounting."""

    query_counts: Mapping[str, int]
    expected_query_counts: Mapping[str, int]
    query_formula: str
    all_executed_levels_reconciled: bool
    aggregate_reconciled: bool
    hard_cap_queries: int
    hard_cap_respected: bool
    budget_valid: bool
    blocked_before_phase_qubits: int | None
    blocked_next_level_query_cost: int | None
    qram_assumed: bool = False


@dataclass(frozen=True, slots=True)
class AdaptiveClaimBoundary:
    """Positive evidence and claims that remain forbidden."""

    supports: tuple[str, ...]
    does_not_support: tuple[str, ...]
    single_coherent_variable_time_unitary_implemented: bool = False
    generic_off_grid_correctness_proved: bool = False
    new_query_upper_bound_proved: bool = False
    matching_lower_bound_proved: bool = False
    quantum_advantage_claimable: bool = False
    ccf_a_claimable: bool = False
    claim_scope: str = CLAIM_SCOPE


@dataclass(frozen=True, slots=True)
class AdaptiveUnknownBoundaryTopKResult:
    """Stable fail-closed result consumed by the S3 experiment campaign."""

    method_id: str
    input_interface: AdaptiveUnknownBoundaryInputInterface
    output_status: str
    membership_mask: int | None
    membership_bits: tuple[int, ...]
    durable_output: AdaptiveDurableOutput
    certificate: AdaptiveCertificate
    query_budget: AdaptiveQueryBudget
    cleanup: AdaptiveCleanupSummary
    stopping_history: AdaptiveStoppingHistory
    hard_cap_respected: bool
    budget_valid: bool
    status: str
    blockers: tuple[str, ...]
    claim_boundary: AdaptiveClaimBoundary
    backend: str = BACKEND
    claim_scope: str = CLAIM_SCOPE
    quantum_advantage_claimable: bool = False


def _input_interface(
    oracle: CanonicalRyStatevectorOracle,
    k: int,
    config: AdaptiveUnknownBoundaryTopKConfig,
) -> AdaptiveUnknownBoundaryInputInterface:
    return AdaptiveUnknownBoundaryInputInterface(
        oracle="opaque_canonical_ry_statevector_oracle_handle",
        k=k,
        public_precision_bounds=(
            config.minimum_phase_qubits,
            config.maximum_phase_qubits,
        ),
        public_target_diagnostic_error_probability=(
            config.target_diagnostic_error_probability
        ),
        public_numerical_cleanup_tolerance=config.numerical_cleanup_tolerance,
        public_statevector_dimension_cap=config.max_statevector_dimension,
        public_canonical_query_hard_cap=config.max_canonical_oracle_queries,
        internally_constructed_precision_policy=(
            "all_consecutive_integer_precisions_within_public_bounds"
        ),
        forbidden_inputs=(
            "answer_set",
            "top_k_membership_mask",
            "gap",
            "boundary_value",
            "family_label",
            "precision_schedule",
            "activity_schedule",
            "activity_history",
            "free_qram_list",
        ),
    )


def _claim_boundary() -> AdaptiveClaimBoundary:
    return AdaptiveClaimBoundary(
        supports=(
            "sequential execution of tiny coherent unknown-boundary level unitaries",
            "consecutive precision construction without an instance schedule input",
            "exact per-level and cumulative canonical-oracle query reconciliation",
            "hard-cap checks before every executed precision",
            "durable direct-k membership-register and cleanup diagnostics",
            "fail-closed output when the diagnostic predicate or budget fails",
        ),
        does_not_support=(
            "a single coherent variable-time stopping unitary",
            "a physically retained coherent stopping-history register",
            "free observation of exact output probabilities or garbage mass",
            "a Top-k correctness or confidence certificate",
            "generic off-grid correctness under finite QPE",
            "a cleanup/resource theorem for coherent stopping-history replay",
            "a new query-complexity upper bound or composition separation",
            "a matching oracle lower bound",
            "hardware evidence, quantum advantage, or a CCF-A publication claim",
        ),
    )


def run_adaptive_unknown_boundary_topk(
    oracle: CanonicalRyStatevectorOracle,
    k: int,
    *,
    config: AdaptiveUnknownBoundaryTopKConfig | None = None,
) -> AdaptiveUnknownBoundaryTopKResult:
    """Run consecutive coherent levels under a classical fail-closed controller."""

    if not isinstance(oracle, CanonicalRyStatevectorOracle):
        raise TypeError("oracle must be a CanonicalRyStatevectorOracle")
    if not 2 <= oracle.n_arms <= 3:
        raise ValueError("the tiny exact-state implementation requires 2 <= n <= 3")
    k = _integer(k, "k", minimum=1)
    if k >= oracle.n_arms:
        raise ValueError("k must satisfy 1 <= k < oracle.n_arms")
    if config is not None and not isinstance(config, AdaptiveUnknownBoundaryTopKConfig):
        raise TypeError("config must be AdaptiveUnknownBoundaryTopKConfig")
    resolved = config or AdaptiveUnknownBoundaryTopKConfig()

    before = oracle.query_snapshot()
    levels: list[AdaptiveStoppingLevel] = []
    cumulative_expected: Counter[str] = Counter()
    cumulative_queries = 0
    selected: AdaptiveStoppingLevel | None = None
    blocked_before: int | None = None
    blocked_cost: int | None = None
    stop_reason: str | None = None

    for level_index, phase_qubits in enumerate(
        range(resolved.minimum_phase_qubits, resolved.maximum_phase_qubits + 1)
    ):
        expected = _expected_level_query_counts(oracle.n_arms, phase_qubits)
        next_cost = expected["coherent_total"]
        if cumulative_queries + next_cost > resolved.max_canonical_oracle_queries:
            blocked_before = phase_qubits
            blocked_cost = next_cost
            stop_reason = "canonical_query_hard_cap_before_next_level"
            break
        retained_dimension = _statevector_dimension(oracle, phase_qubits)
        if retained_dimension > resolved.max_statevector_dimension:
            blocked_before = phase_qubits
            stop_reason = "exact_statevector_dimension_cap_before_next_level"
            break

        level_result = run_coherent_unknown_boundary_topk(
            oracle,
            k,
            config=CoherentUnknownBoundaryTopKConfig(
                phase_qubits=phase_qubits,
                cleanup_tolerance=resolved.numerical_cleanup_tolerance,
                max_statevector_dimension=resolved.max_statevector_dimension,
            ),
        )
        observed = dict(level_result.resources.query_counts)
        observed["qram_queries"] = 0
        reconciled = all(observed.get(name) == value for name, value in expected.items())
        cumulative_queries += int(observed["coherent_total"])
        cumulative_expected.update(expected)

        boundary = level_result.boundary
        cleanup = level_result.resources.cleanup
        disagreement = max(0.0, 1.0 - boundary.dominant_probability)
        nonstrict = max(0.0, 1.0 - boundary.strict_probability)
        diagnostic_error = max(
            disagreement,
            nonstrict,
            cleanup.executed_transient_nonzero_probability,
        )
        identity_passed = (
            cleanup.prediction_residual
            <= 10.0 * resolved.numerical_cleanup_tolerance
            and cleanup.purity_residual
            <= 10.0 * resolved.numerical_cleanup_tolerance
        )
        acceptance = (
            reconciled
            and identity_passed
            and boundary.dominant_mask.bit_count() == k
            and diagnostic_error <= resolved.target_diagnostic_error_probability
        )
        level = AdaptiveStoppingLevel(
            level_index=level_index,
            phase_qubits=phase_qubits,
            phase_bins=1 << phase_qubits,
            retained_statevector_dimension=retained_dimension,
            dominant_mask=boundary.dominant_mask,
            dominant_mask_bit_count=boundary.dominant_mask.bit_count(),
            dominant_probability=boundary.dominant_probability,
            statevector_output_disagreement_mass=disagreement,
            strict_boundary_probability=boundary.strict_probability,
            nonstrict_boundary_mass=nonstrict,
            executed_transient_nonzero_probability=(
                cleanup.executed_transient_nonzero_probability
            ),
            predicted_transient_nonzero_probability=(
                cleanup.predicted_transient_nonzero_probability
            ),
            cleanup_prediction_residual=cleanup.prediction_residual,
            output_reduced_purity=cleanup.output_reduced_purity,
            output_collision_probability=cleanup.output_collision_probability,
            purity_residual=cleanup.purity_residual,
            exact_numerical_cleanup_passed=cleanup.passed,
            diagnostic_error_bound=diagnostic_error,
            diagnostic_acceptance_bound_passed=acceptance,
            query_counts=_query_counts_with_qram(observed),
            expected_query_counts=_frozen_counts(expected),
            query_formula_reconciled=reconciled,
            cumulative_coherent_queries=cumulative_queries,
            emulated_stop_bit=int(acceptance),
            stop_latched_after_level=acceptance,
            level_status=(
                "diagnostic_stop_latched_certificate_withheld"
                if acceptance
                else "continue_to_finer_precision"
            ),
        )
        levels.append(level)
        if acceptance:
            selected = level
            stop_reason = "diagnostic_acceptance_predicate"
            break

    if stop_reason is None:
        stop_reason = "maximum_phase_precision_reached_without_acceptance"

    after = oracle.query_snapshot()
    aggregate_observed = QueryLedger.difference(after, before)
    aggregate_observed["qram_queries"] = 0
    expected_aggregate = {
        name: int(cumulative_expected.get(name, 0))
        for name in _expected_level_query_counts(oracle.n_arms, 1)
    }
    aggregate_reconciled = all(
        aggregate_observed.get(name) == value
        for name, value in expected_aggregate.items()
    )
    per_level_reconciled = all(level.query_formula_reconciled for level in levels)
    hard_cap_respected = (
        int(aggregate_observed["coherent_total"])
        <= resolved.max_canonical_oracle_queries
    )
    budget_valid = per_level_reconciled and aggregate_reconciled and hard_cap_respected

    if selected is None:
        output_status = OUTPUT_INCONCLUSIVE
        membership_mask = None
        membership_bits: tuple[int, ...] = ()
        status = "inconclusive_fail_closed"
    else:
        output_status = OUTPUT_MASK
        membership_mask = selected.dominant_mask
        membership_bits = tuple(
            (membership_mask >> arm) & 1 for arm in range(oracle.n_arms)
        )
        status = "diagnostic_mask_selected_certificate_withheld"

    audit_level = selected if selected is not None else (levels[-1] if levels else None)
    durable_output = AdaptiveDurableOutput(
        status=output_status,
        membership_mask=membership_mask,
        membership_bits=membership_bits,
        selected_phase_qubits=(None if selected is None else selected.phase_qubits),
        direct_k_membership_register_used=selected is not None,
        rank_relation_computed_and_uncomputed=selected is not None,
        phase_estimation_inverted=selected is not None,
        exact_transient_cleanup_passed=(
            False if selected is None else selected.exact_numerical_cleanup_passed
        ),
        approximate_cleanup_bound_passed=(
            False if selected is None else selected.diagnostic_acceptance_bound_passed
        ),
        output_disagreement_mass=(
            None if selected is None else selected.statevector_output_disagreement_mass
        ),
        transient_nonzero_probability=(
            None
            if selected is None
            else selected.executed_transient_nonzero_probability
        ),
        dominant_mask_selected_by_statevector_introspection=selected is not None,
    )
    certificate = AdaptiveCertificate(
        issued=False,
        certificate_type=None,
        top_k_correctness_error_bound=None,
        reason=(
            "the stop predicate reads exact simulator probabilities and cleanup mass; "
            "an observable charged estimator and correctness theorem are not implemented"
        ),
        exact_statevector_diagnostic_error_bound=(
            None if selected is None else selected.diagnostic_error_bound
        ),
    )
    cleanup_summary = AdaptiveCleanupSummary(
        available=audit_level is not None,
        exact_numerical_cleanup_passed=(
            False if audit_level is None else audit_level.exact_numerical_cleanup_passed
        ),
        diagnostic_acceptance_bound_passed=(
            False
            if audit_level is None
            else audit_level.diagnostic_acceptance_bound_passed
        ),
        executed_transient_nonzero_probability=(
            None
            if audit_level is None
            else audit_level.executed_transient_nonzero_probability
        ),
        predicted_transient_nonzero_probability=(
            None
            if audit_level is None
            else audit_level.predicted_transient_nonzero_probability
        ),
        cleanup_prediction_residual=(
            None if audit_level is None else audit_level.cleanup_prediction_residual
        ),
        statevector_output_disagreement_mass=(
            None
            if audit_level is None
            else audit_level.statevector_output_disagreement_mass
        ),
        nonstrict_boundary_mass=(
            None if audit_level is None else audit_level.nonstrict_boundary_mass
        ),
        diagnostic_error_bound=(
            None if audit_level is None else audit_level.diagnostic_error_bound
        ),
        target_diagnostic_error_probability=(
            resolved.target_diagnostic_error_probability
        ),
        top_k_correctness_error_bound=None,
    )
    stopping_history = AdaptiveStoppingHistory(
        levels=tuple(levels),
        emulated_history_bits=tuple(level.emulated_stop_bit for level in levels),
        first_stop_level_index=(None if selected is None else selected.level_index),
        first_stop_phase_qubits=(None if selected is None else selected.phase_qubits),
        controller_is_classical=True,
        independently_coherent_level_unitaries_executed=bool(levels),
        single_coherent_variable_time_unitary_implemented=False,
        coherent_history_register_physically_retained=False,
        coherent_history_cleanup_proved=False,
    )
    query_budget = AdaptiveQueryBudget(
        query_counts=_query_counts_with_qram(aggregate_observed),
        expected_query_counts=_frozen_counts(expected_aggregate),
        query_formula=QUERY_FORMULA,
        all_executed_levels_reconciled=per_level_reconciled,
        aggregate_reconciled=aggregate_reconciled,
        hard_cap_queries=resolved.max_canonical_oracle_queries,
        hard_cap_respected=hard_cap_respected,
        budget_valid=budget_valid,
        blocked_before_phase_qubits=blocked_before,
        blocked_next_level_query_cost=blocked_cost,
    )
    blockers = (
        "adaptive_controller_reads_exact_statevector_diagnostics",
        "no_single_coherent_variable_time_stopping_unitary",
        "no_physically_retained_coherent_history_register",
        "no_observable_charged_acceptance_estimator",
        "no_generic_off_grid_top_k_correctness_certificate",
        "no_coherent_history_cleanup_resource_bound",
        "no_new_query_complexity_upper_bound",
        "no_same_interface_composition_separation",
        "no_matching_oracle_lower_bound",
        "tiny_exact_state_simulation_not_hardware_evidence",
    )
    return AdaptiveUnknownBoundaryTopKResult(
        method_id=METHOD_ID,
        input_interface=_input_interface(oracle, k, resolved),
        output_status=output_status,
        membership_mask=membership_mask,
        membership_bits=membership_bits,
        durable_output=durable_output,
        certificate=certificate,
        query_budget=query_budget,
        cleanup=cleanup_summary,
        stopping_history=stopping_history,
        hard_cap_respected=hard_cap_respected,
        budget_valid=budget_valid,
        status=status,
        blockers=blockers,
        claim_boundary=_claim_boundary(),
    )


__all__ = [
    "BACKEND",
    "CLAIM_SCOPE",
    "ERROR_BOUND_SCOPE",
    "METHOD_ID",
    "OUTPUT_INCONCLUSIVE",
    "OUTPUT_MASK",
    "QUERY_FORMULA",
    "STOPPING_HISTORY_SEMANTICS",
    "AdaptiveCertificate",
    "AdaptiveClaimBoundary",
    "AdaptiveCleanupSummary",
    "AdaptiveDurableOutput",
    "AdaptiveQueryBudget",
    "AdaptiveStoppingHistory",
    "AdaptiveStoppingLevel",
    "AdaptiveUnknownBoundaryInputInterface",
    "AdaptiveUnknownBoundaryTopKConfig",
    "AdaptiveUnknownBoundaryTopKResult",
    "run_adaptive_unknown_boundary_topk",
]
