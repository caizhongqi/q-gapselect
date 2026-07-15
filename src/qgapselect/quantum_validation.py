"""Executed validation experiments for the direct QPE quantum core.

This module contains two diagnostics that are easy to omit from algorithm
benchmarks: random-state unitary/involution checks and repeated calibration of
the fresh measured QPE verifier.  Both reach the canonical oracle only through
its public charged interface.  Exact probabilities are evaluation references,
not free predicates supplied to an algorithm.

All results describe a small NumPy exact-state simulation.  They establish
finite implementation invariants only; they are neither a complexity proof nor
hardware evidence.
"""

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from types import MappingProxyType

import numpy as np

from .coherent import CanonicalRyStatevectorOracle
from .direct_phase import DirectAmplitudeThresholdFlag

BACKEND = "numpy_exact_statevector_small_scale"
CLAIM_STATUS = "executed_quantum_validation_no_complexity_or_hardware_claim"
TRUTH_USAGE = "evaluation_only_never_passed_to_verifier"


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _probability(value: object, name: str, *, closed: bool) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    valid = 0.0 <= result <= 1.0 if closed else 0.0 < result < 1.0
    if not math.isfinite(result) or not valid:
        interval = "[0, 1]" if closed else "(0, 1)"
        raise ValueError(f"{name} must be finite and lie in {interval}")
    return result


def _relation(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("relation must be a string")
    if value not in {"above", "below"}:
        raise ValueError("relation must be 'above' or 'below'")
    return value


def _seed(value: object) -> int:
    result = _integer(value, "seed")
    if result < 0:
        raise ValueError("seed cannot be negative")
    return result


def _immutable(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType({str(key): int(value) for key, value in values.items()})


def _random_state(dimension: int, rng: np.random.Generator) -> np.ndarray:
    state = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    return np.asarray(state / np.linalg.norm(state), dtype=np.complex128)


def _wilson(successes: int, trials: int, confidence: float) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


@dataclass(frozen=True, slots=True)
class UnitaryValidationTrial:
    """One random-state compute/uncompute and reflection-involution check."""

    trial: int
    seed: int
    n_arms: int
    index_dimension: int
    phase_qubits: int
    phase_bins: int
    statevector_dimension: int
    compute_inverse_residual: float
    reflection_involution_residual: float
    compute_norm_residual: float
    reflection_norm_residual: float
    actual_oracle_queries: int
    expected_oracle_queries: int
    query_formula_exact: bool
    passed: bool
    retained_statevector_dimension: int
    comparator_expanded_statevector_dimension: int
    dense_qft_matrix_dimension: int
    estimated_peak_complex_amplitudes: int
    backend: str = BACKEND
    claim_status: str = CLAIM_STATUS


@dataclass(frozen=True, slots=True)
class UnitaryValidationResult:
    """Aggregate of independently seeded random-state invariant checks."""

    trials: tuple[UnitaryValidationTrial, ...]
    passed: int
    max_compute_inverse_residual: float
    max_reflection_involution_residual: float
    max_norm_residual: float
    total_oracle_queries: int
    tolerance: float
    backend: str = BACKEND
    claim_status: str = CLAIM_STATUS


def run_unitary_validation(
    means: Sequence[float],
    *,
    threshold: float = 0.5,
    phase_qubits: int = 4,
    relation: str = "above",
    trials: int = 8,
    seed: int = 0,
    tolerance: float = 1e-10,
) -> UnitaryValidationResult:
    """Execute random-state QPE round trips and threshold-reflection squares."""

    values = tuple(_probability(mean, "mean", closed=True) for mean in means)
    if not values:
        raise ValueError("means cannot be empty")
    threshold = _probability(threshold, "threshold", closed=True)
    phase_qubits = _integer(phase_qubits, "phase_qubits")
    trials = _integer(trials, "trials")
    seed = _seed(seed)
    relation = _relation(relation)
    if not 1 <= phase_qubits <= 12:
        raise ValueError("phase_qubits must lie in [1, 12]")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if isinstance(tolerance, bool):
        raise TypeError("tolerance must be a real number, not bool")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")

    rng = np.random.default_rng(seed)
    records: list[UnitaryValidationTrial] = []
    phase_bins = 1 << phase_qubits
    # compute + inverse_compute costs 2(2M-1).  Two reflections cost
    # 2(4M-2), so the complete diagnostic costs 12M-6 oracle calls.
    expected_queries = 12 * phase_bins - 6
    for trial in range(trials):
        trial_seed = int(rng.integers(0, 2**63, dtype=np.int64))
        oracle = CanonicalRyStatevectorOracle(values, seed=trial_seed)
        flag = DirectAmplitudeThresholdFlag(
            oracle,
            threshold,
            phase_qubits=phase_qubits,
            relation=relation,
        )
        state = _random_state(
            flag.statevector_dimension,
            np.random.default_rng(trial_seed),
        )
        computed = flag.compute(state, tag="unitary_validation_compute")
        recovered = flag.inverse_compute(
            computed,
            tag="unitary_validation_inverse",
        )
        reflected = flag.apply_reflection(
            state,
            tag="unitary_validation_reflection",
        )
        reflected_twice = flag.apply_reflection(
            reflected,
            tag="unitary_validation_reflection",
        )
        compute_residual = float(np.linalg.norm(recovered - state))
        reflection_residual = float(np.linalg.norm(reflected_twice - state))
        compute_norm_residual = abs(float(np.linalg.norm(computed)) - 1.0)
        reflection_norm_residual = abs(float(np.linalg.norm(reflected)) - 1.0)
        resources = flag.resources()
        actual_queries = oracle.query_snapshot().coherent_total
        formula_exact = actual_queries == expected_queries
        passed = (
            compute_residual <= tolerance
            and reflection_residual <= tolerance
            and compute_norm_residual <= tolerance
            and reflection_norm_residual <= tolerance
            and formula_exact
        )
        records.append(
            UnitaryValidationTrial(
                trial=trial,
                seed=trial_seed,
                n_arms=len(values),
                index_dimension=oracle.index_dimension,
                phase_qubits=phase_qubits,
                phase_bins=phase_bins,
                statevector_dimension=flag.statevector_dimension,
                compute_inverse_residual=compute_residual,
                reflection_involution_residual=reflection_residual,
                compute_norm_residual=compute_norm_residual,
                reflection_norm_residual=reflection_norm_residual,
                actual_oracle_queries=actual_queries,
                expected_oracle_queries=expected_queries,
                query_formula_exact=formula_exact,
                passed=passed,
                retained_statevector_dimension=int(
                    resources.retained_statevector_dimension
                ),
                comparator_expanded_statevector_dimension=int(
                    resources.comparator_expanded_statevector_dimension
                ),
                dense_qft_matrix_dimension=int(resources.dense_qft_matrix_dimension),
                estimated_peak_complex_amplitudes=int(
                    resources.estimated_peak_complex_amplitudes
                ),
            )
        )
    return UnitaryValidationResult(
        trials=tuple(records),
        passed=sum(record.passed for record in records),
        max_compute_inverse_residual=max(
            record.compute_inverse_residual for record in records
        ),
        max_reflection_involution_residual=max(
            record.reflection_involution_residual for record in records
        ),
        max_norm_residual=max(
            max(record.compute_norm_residual, record.reflection_norm_residual)
            for record in records
        ),
        total_oracle_queries=sum(record.actual_oracle_queries for record in records),
        tolerance=tolerance,
    )


@dataclass(frozen=True, slots=True)
class VerifierCalibrationTrial:
    """One independently rerun fixed-shot QPE verifier decision."""

    trial: int
    seed: int
    status: str
    successes: int
    shots: int
    estimate: float
    interval: tuple[float, float]
    interval_covers_exact_qpe_probability: bool
    wrong_resolved_decision: bool
    oracle_queries: int


@dataclass(frozen=True, slots=True)
class VerifierCalibrationResult:
    """Repeated calibration against the exact public QPE-predicate probability."""

    mean: float
    threshold: float
    relation: str
    phase_qubits: int
    phase_bins: int
    shots: int
    confidence: float
    trials: tuple[VerifierCalibrationTrial, ...]
    exact_qpe_acceptance_probability: float
    exact_decision_side: str
    evaluation_only_oracle_queries: int
    procedure_oracle_queries: int
    status_counts: Mapping[str, int]
    interval_coverage_count: int
    interval_coverage_rate: float
    coverage_wilson_interval: tuple[float, float]
    wrong_resolved_count: int
    wrong_resolved_rate: float
    wrong_resolved_wilson_interval: tuple[float, float]
    truth_usage: str = TRUTH_USAGE
    backend: str = BACKEND
    claim_status: str = CLAIM_STATUS


def run_verifier_calibration(
    mean: float,
    *,
    threshold: float = 0.5,
    phase_qubits: int = 4,
    relation: str = "above",
    shots: int = 64,
    confidence: float = 0.05,
    trials: int = 100,
    seed: int = 0,
    interval_report_confidence: float = 0.95,
) -> VerifierCalibrationResult:
    """Calibrate measured decisions against an independently charged QPE value."""

    mean = _probability(mean, "mean", closed=True)
    threshold = _probability(threshold, "threshold", closed=True)
    confidence = _probability(confidence, "confidence", closed=False)
    interval_report_confidence = _probability(
        interval_report_confidence,
        "interval_report_confidence",
        closed=False,
    )
    phase_qubits = _integer(phase_qubits, "phase_qubits")
    shots = _integer(shots, "shots")
    trials = _integer(trials, "trials")
    seed = _seed(seed)
    relation = _relation(relation)
    if not 1 <= phase_qubits <= 12:
        raise ValueError("phase_qubits must lie in [1, 12]")
    if shots <= 0 or trials <= 0:
        raise ValueError("shots and trials must be positive")

    evaluation_oracle = CanonicalRyStatevectorOracle([mean], seed=seed)
    evaluation_flag = DirectAmplitudeThresholdFlag(
        evaluation_oracle,
        threshold,
        phase_qubits=phase_qubits,
        relation=relation,
    )
    exact_probability = evaluation_flag.acceptance_probability(
        0,
        tag="verifier_calibration_evaluation_only",
    )
    evaluation_queries = evaluation_oracle.query_snapshot().coherent_total
    if math.isclose(exact_probability, 0.5, rel_tol=0.0, abs_tol=1e-15):
        exact_side = "boundary"
    elif exact_probability > 0.5:
        exact_side = "accepted"
    else:
        exact_side = "rejected"

    rng = np.random.default_rng(seed)
    records: list[VerifierCalibrationTrial] = []
    for trial in range(trials):
        trial_seed = int(rng.integers(0, 2**63, dtype=np.int64))
        oracle = CanonicalRyStatevectorOracle([mean], seed=trial_seed)
        flag = DirectAmplitudeThresholdFlag(
            oracle,
            threshold,
            phase_qubits=phase_qubits,
            relation=relation,
        )
        result = flag.verify_index(
            0,
            shots=shots,
            confidence=confidence,
            seed=trial_seed,
            tag="verifier_calibration_procedure",
        )
        wrong = (
            (result.status == "accepted" and exact_side == "rejected")
            or (result.status == "rejected" and exact_side == "accepted")
        )
        records.append(
            VerifierCalibrationTrial(
                trial=trial,
                seed=trial_seed,
                status=result.status,
                successes=result.successes,
                shots=result.shots,
                estimate=result.estimate,
                interval=result.interval,
                interval_covers_exact_qpe_probability=(
                    result.interval[0]
                    <= exact_probability
                    <= result.interval[1]
                ),
                wrong_resolved_decision=wrong,
                oracle_queries=oracle.query_snapshot().coherent_total,
            )
        )

    coverage = sum(
        record.interval_covers_exact_qpe_probability for record in records
    )
    wrong_count = sum(record.wrong_resolved_decision for record in records)
    status_counts = Counter(record.status for record in records)
    return VerifierCalibrationResult(
        mean=mean,
        threshold=threshold,
        relation=relation,
        phase_qubits=phase_qubits,
        phase_bins=1 << phase_qubits,
        shots=shots,
        confidence=confidence,
        trials=tuple(records),
        exact_qpe_acceptance_probability=exact_probability,
        exact_decision_side=exact_side,
        evaluation_only_oracle_queries=evaluation_queries,
        procedure_oracle_queries=sum(record.oracle_queries for record in records),
        status_counts=_immutable(status_counts),
        interval_coverage_count=coverage,
        interval_coverage_rate=coverage / trials,
        coverage_wilson_interval=_wilson(
            coverage,
            trials,
            interval_report_confidence,
        ),
        wrong_resolved_count=wrong_count,
        wrong_resolved_rate=wrong_count / trials,
        wrong_resolved_wilson_interval=_wilson(
            wrong_count,
            trials,
            interval_report_confidence,
        ),
    )
