"""Auditable small-state diagnostics for the direct quantum core.

The routines in this module exercise the charged, public QPE/reflection API.
They are deliberately separated from the selection controllers: diagnostic
ground truth is used to construct and label instances, but it is never handed
to a search routine.

``run_diffusion_ablation`` also executes an index-only diffusion as a negative
control.  That branch is explicitly labelled ``INVALID`` because finite-QPE
workspace leakage makes index-only diffusion the wrong operator.  Its output
is diagnostic evidence only and is never an algorithm result.
"""

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from .coherent import CanonicalRyStatevectorOracle
from .direct_phase import (
    PHASE_BOUNDARY_ATOL,
    RESOURCE_SEMANTICS,
    DirectAmplitudeThresholdFlag,
    DirectPhaseFlagResources,
)
from .direct_search import full_workspace_rank_one_diffusion
from .oracles import QueryKind

ComplexState = NDArray[np.complex128]
BACKEND = "numpy_exact_statevector_small_scale"
DIAGNOSTIC_CLAIM = "quantum_core_diagnostic_no_complexity_or_advantage_claim"
VALID_FULL_METHOD = "full_workspace_rank_one_diffusion"
INVALID_INDEX_METHOD = "INVALID_index_only_diffusion_diagnostic_only"


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _real(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(value: object, name: str) -> float:
    result = _real(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _seed(value: object | None) -> int | None:
    if value is None:
        return None
    return _integer(value, "seed")


def _immutable(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType({str(key): int(value) for key, value in values.items()})


def _query_counts(resources: DirectPhaseFlagResources) -> Mapping[str, int]:
    return _immutable(
        {
            kind.value: int(resources.query_counts.get(kind.value, 0))
            for kind in QueryKind
        }
        | {
            "coherent_total": resources.oracle_queries,
            "classical_total": int(
                resources.query_counts.get(QueryKind.CLASSICAL_SAMPLE.value, 0)
            ),
            "total": int(resources.query_counts.get("total", resources.oracle_queries)),
        }
    )


@dataclass(frozen=True, slots=True)
class DiagnosticResources:
    """Executed resources for exactly one diagnostic method or sweep."""

    method: str
    query_counts: Mapping[str, int]
    gate_counts: Mapping[str, int]
    qpe_calls: int
    controlled_grover_iterations: int
    depth: int
    qubits: int
    workspace_qubits: int
    statevector_dimension: int
    peak_statevector_dimension: int
    retained_statevector_dimension: int
    comparator_expanded_statevector_dimension: int
    dense_qft_matrix_dimension: int
    estimated_peak_complex_amplitudes: int
    phase_ancilla_residual: float
    zero_workspace_residual: float
    comparator_residual: float
    backend: str = BACKEND
    resource_semantics: str = RESOURCE_SEMANTICS

    @property
    def oracle_queries(self) -> int:
        return int(self.query_counts.get("coherent_total", 0))

    @property
    def gates(self) -> int:
        return sum(int(count) for count in self.gate_counts.values())


@dataclass(frozen=True, slots=True)
class ThresholdAngularGapInstance:
    """Synthetic threshold instance with labels reserved for evaluation only."""

    means: tuple[float, ...]
    threshold: float
    threshold_angle: float
    angular_gap: float
    angles: tuple[float, ...]
    truth_above: tuple[int, ...]
    truth_below: tuple[int, ...]
    seed: int | None
    truth_usage: str = "experiment_evaluation_only_never_passed_to_search"


@dataclass(frozen=True, slots=True)
class QPEAcceptancePoint:
    """One charged QPE acceptance measurement computed from public state data."""

    arm: int
    mean: float
    angle: float
    phase_qubits: int
    phase_bins: int
    threshold: float
    relation: str
    truth_accept: bool
    joint_acceptance_probability: float
    phase_support_size: int
    resources: DiagnosticResources


@dataclass(frozen=True, slots=True)
class QPEAcceptanceSweep:
    """Acceptance probabilities across arms and finite phase precisions."""

    points: tuple[QPEAcceptancePoint, ...]
    phase_qubits: tuple[int, ...]
    threshold: float
    relation: str
    seed: int | None
    resources: DiagnosticResources
    claim_status: str = DIAGNOSTIC_CLAIM


@dataclass(frozen=True, slots=True)
class PhaseGridPoint:
    """Exact representable eigenphase and its mirrored QPE support."""

    grid_index: int
    angle: float
    mean: float
    mirrored_peak_bins: tuple[int, ...]
    truth_accept: bool
    joint_acceptance_probability: float
    resources: DiagnosticResources


@dataclass(frozen=True, slots=True)
class PhaseGridSweep:
    """Charged exact-grid QPE diagnostic for angles in ``[0, pi/2]``."""

    phase_qubits: int
    phase_bins: int
    threshold: float
    relation: str
    points: tuple[PhaseGridPoint, ...]
    resources: DiagnosticResources
    seed: int | None
    claim_status: str = DIAGNOSTIC_CLAIM


@dataclass(frozen=True, slots=True)
class DiffusionMethodDiagnostic:
    """One branch of the full-workspace versus invalid-control ablation."""

    method: str
    algorithmically_valid: bool
    output_eligible: bool
    joint_acceptance_probability: float
    sampled_accept_index: int | None
    resources: DiagnosticResources
    warning: str | None


@dataclass(frozen=True, slots=True)
class DiffusionAblation:
    """Paired diagnostic; the invalid branch must not feed algorithm output."""

    full_workspace: DiffusionMethodDiagnostic
    invalid_index_only: DiffusionMethodDiagnostic
    state_distance_up_to_global_phase: float
    finite_phase_leakage_detected: bool
    phase_qubits: int
    threshold: float
    relation: str
    grover_iterations: int
    seed: int | None
    claim_status: str = DIAGNOSTIC_CLAIM


def make_threshold_angular_gap_instance(
    *,
    n_below: int,
    n_above: int,
    threshold: float = 0.5,
    angular_gap: float = 0.1,
    seed: int | None = 0,
) -> ThresholdAngularGapInstance:
    """Create two angular bands at a controlled distance from the threshold.

    The returned truth labels are for plots, assertions, and evaluation.  The
    quantum diagnostic functions below consume only ``means`` and
    ``threshold``; no label is supplied to a flag or a search controller.
    """

    n_below = _integer(n_below, "n_below")
    n_above = _integer(n_above, "n_above")
    if n_below < 0 or n_above < 0 or n_below + n_above == 0:
        raise ValueError("arm counts must be nonnegative and sum to a positive value")
    threshold = _probability(threshold, "threshold")
    angular_gap = _real(angular_gap, "angular_gap")
    if angular_gap <= 0.0:
        raise ValueError("angular_gap must be positive")
    threshold_angle = math.asin(math.sqrt(threshold))
    if angular_gap > min(threshold_angle, math.pi / 2.0 - threshold_angle):
        raise ValueError("angular_gap would place an angle outside [0, pi/2]")
    seed = _seed(seed)

    labelled = [
        (threshold_angle - angular_gap, False) for _ in range(n_below)
    ] + [(threshold_angle + angular_gap, True) for _ in range(n_above)]
    order = np.random.default_rng(seed).permutation(len(labelled))
    shuffled = tuple(labelled[int(index)] for index in order)
    angles = tuple(angle for angle, _ in shuffled)
    means = tuple(math.sin(angle) ** 2 for angle in angles)
    truth_above = tuple(index for index, (_, above) in enumerate(shuffled) if above)
    truth_below = tuple(index for index, (_, above) in enumerate(shuffled) if not above)
    return ThresholdAngularGapInstance(
        means=means,
        threshold=threshold,
        threshold_angle=threshold_angle,
        angular_gap=angular_gap,
        angles=angles,
        truth_above=truth_above,
        truth_below=truth_below,
        seed=seed,
    )


def joint_acceptance_probability(
    flag: DirectAmplitudeThresholdFlag,
    computed_state: ComplexState,
) -> float:
    """Compute joint phase-predicate/index acceptance from public data only."""

    if not isinstance(flag, DirectAmplitudeThresholdFlag):
        raise TypeError("flag must be a DirectAmplitudeThresholdFlag")
    state = np.asarray(computed_state, dtype=np.complex128)
    if state.ndim != 1 or state.size != flag.statevector_dimension:
        raise ValueError(
            f"computed_state must have length {flag.statevector_dimension}"
        )
    if not np.isclose(np.linalg.norm(state), 1.0, atol=1e-10):
        raise ValueError("computed_state must be normalized")
    probabilities = np.abs(
        state.reshape(flag.phase_bins, flag.index_dimension, 2)
    ) ** 2
    result = float(np.sum(probabilities[flag.acceptance_mask]))
    return min(max(result, 0.0), 1.0)


def _angle_at_or_above(angle: float, threshold_angle: float) -> bool:
    return angle >= threshold_angle or math.isclose(
        angle,
        threshold_angle,
        rel_tol=0.0,
        abs_tol=PHASE_BOUNDARY_ATOL,
    )


def _diagnostic_resources(
    method: str,
    resources: DirectPhaseFlagResources,
    *,
    statevector_dimension: int,
    extra_gates: Mapping[str, int] | None = None,
    extra_depth: int = 0,
) -> DiagnosticResources:
    gates = Counter(resources.gate_counts)
    if extra_gates is not None:
        gates.update(extra_gates)
    comparator_executed = gates.get("phase_bin_comparator_compute", 0) > 0
    retained = int(
        getattr(resources, "retained_statevector_dimension", statevector_dimension)
    )
    dense_qft = int(
        getattr(
            resources,
            "dense_qft_matrix_dimension",
            resources.phase_bins * resources.phase_bins,
        )
    )
    comparator = 2 * retained if comparator_executed else 0
    estimated_peak = max(
        dense_qft + 2 * retained,
        3 * retained if comparator_executed else 0,
    )
    return DiagnosticResources(
        method=method,
        query_counts=_query_counts(resources),
        gate_counts=_immutable(gates),
        qpe_calls=resources.qpe_calls,
        controlled_grover_iterations=resources.controlled_grover_iterations,
        depth=resources.depth + extra_depth,
        # DirectPhaseFlagResources declares capacity for the transient
        # comparator.  Remove that qubit for compute-only diagnostics where
        # no comparator gate was actually executed.
        qubits=resources.qubits if comparator_executed else resources.qubits - 1,
        workspace_qubits=resources.workspace_qubits,
        statevector_dimension=retained,
        peak_statevector_dimension=estimated_peak,
        retained_statevector_dimension=retained,
        comparator_expanded_statevector_dimension=comparator,
        dense_qft_matrix_dimension=dense_qft,
        estimated_peak_complex_amplitudes=estimated_peak,
        phase_ancilla_residual=resources.phase_ancilla_residual,
        zero_workspace_residual=resources.zero_workspace_residual,
        comparator_residual=resources.comparator_residual,
    )


def _aggregate_resources(
    method: str,
    resources: Sequence[DiagnosticResources],
) -> DiagnosticResources:
    if not resources:
        raise ValueError("resources cannot be empty")
    queries: Counter[str] = Counter()
    gates: Counter[str] = Counter()
    for item in resources:
        queries.update(item.query_counts)
        gates.update(item.gate_counts)
    return DiagnosticResources(
        method=method,
        query_counts=_immutable(queries),
        gate_counts=_immutable(gates),
        qpe_calls=sum(item.qpe_calls for item in resources),
        controlled_grover_iterations=sum(
            item.controlled_grover_iterations for item in resources
        ),
        depth=sum(item.depth for item in resources),
        qubits=max(item.qubits for item in resources),
        workspace_qubits=max(item.workspace_qubits for item in resources),
        statevector_dimension=max(item.statevector_dimension for item in resources),
        peak_statevector_dimension=max(
            item.peak_statevector_dimension for item in resources
        ),
        retained_statevector_dimension=max(
            item.retained_statevector_dimension for item in resources
        ),
        comparator_expanded_statevector_dimension=max(
            item.comparator_expanded_statevector_dimension for item in resources
        ),
        dense_qft_matrix_dimension=max(
            item.dense_qft_matrix_dimension for item in resources
        ),
        estimated_peak_complex_amplitudes=max(
            item.estimated_peak_complex_amplitudes for item in resources
        ),
        phase_ancilla_residual=max(
            item.phase_ancilla_residual for item in resources
        ),
        zero_workspace_residual=max(
            item.zero_workspace_residual for item in resources
        ),
        comparator_residual=max(item.comparator_residual for item in resources),
    )


def _validated_phase_qubits(values: Sequence[int]) -> tuple[int, ...]:
    try:
        checked = tuple(_integer(value, "phase_qubit") for value in values)
    except TypeError as error:
        raise TypeError("phase_qubits must be a sequence of integers") from error
    if not checked:
        raise ValueError("phase_qubits cannot be empty")
    if len(set(checked)) != len(checked):
        raise ValueError("phase_qubits must be unique")
    if any(value <= 0 or value > 12 for value in checked):
        raise ValueError("each phase-qubit count must lie in [1, 12]")
    return checked


def _relation(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("relation must be a string")
    if value not in {"above", "below"}:
        raise ValueError("relation must be 'above' or 'below'")
    return value


def run_qpe_acceptance_sweep(
    means: Sequence[float],
    *,
    threshold: float,
    phase_qubits: Sequence[int],
    relation: str = "above",
    seed: int | None = 0,
) -> QPEAcceptanceSweep:
    """Run charged QPE for every arm/precision and report marked joint mass."""

    values = tuple(_probability(mean, "mean") for mean in means)
    if not values:
        raise ValueError("means cannot be empty")
    threshold = _probability(threshold, "threshold")
    bits = _validated_phase_qubits(phase_qubits)
    relation = _relation(relation)
    seed = _seed(seed)
    points: list[QPEAcceptancePoint] = []
    point_resources: list[DiagnosticResources] = []
    threshold_angle = math.asin(math.sqrt(threshold))

    for precision in bits:
        for arm, mean in enumerate(values):
            # A fresh object keeps the per-point ledger exact and local.
            oracle = CanonicalRyStatevectorOracle(values, seed=seed)
            flag = DirectAmplitudeThresholdFlag(
                oracle,
                threshold,
                phase_qubits=precision,
                relation=relation,
            )
            computed = flag.compute(
                flag.initial_state((arm,)),
                tag=f"qpe_sweep_m{precision}_arm{arm}",
            )
            probability = joint_acceptance_probability(flag, computed)
            raw = flag.last_resources
            if raw is None:
                raise RuntimeError("QPE compute did not publish resources")
            item_resources = _diagnostic_resources(
                "qpe_acceptance_compute",
                raw,
                statevector_dimension=flag.statevector_dimension,
            )
            marginal = np.sum(
                np.abs(
                    computed.reshape(flag.phase_bins, flag.index_dimension, 2)
                )
                ** 2,
                axis=(1, 2),
            )
            angle = math.asin(math.sqrt(mean))
            truth_above = _angle_at_or_above(angle, threshold_angle)
            points.append(
                QPEAcceptancePoint(
                    arm=arm,
                    mean=mean,
                    angle=angle,
                    phase_qubits=precision,
                    phase_bins=1 << precision,
                    threshold=threshold,
                    relation=relation,
                    truth_accept=(truth_above == (relation == "above")),
                    joint_acceptance_probability=probability,
                    phase_support_size=int(np.count_nonzero(marginal > 1e-12)),
                    resources=item_resources,
                )
            )
            point_resources.append(item_resources)

    return QPEAcceptanceSweep(
        points=tuple(points),
        phase_qubits=bits,
        threshold=threshold,
        relation=relation,
        seed=seed,
        resources=_aggregate_resources("qpe_acceptance_sweep", point_resources),
    )


def run_phase_grid_sweep(
    *,
    phase_qubits: int,
    threshold: float = 0.5,
    relation: str = "above",
    seed: int | None = 0,
) -> PhaseGridSweep:
    """Run exact-QPE grid points and expose their mirrored peak bins."""

    phase_qubits = _integer(phase_qubits, "phase_qubits")
    if not 1 <= phase_qubits <= 12:
        raise ValueError("phase_qubits must lie in [1, 12]")
    threshold = _probability(threshold, "threshold")
    relation = _relation(relation)
    seed = _seed(seed)
    phase_bins = 1 << phase_qubits
    points: list[PhaseGridPoint] = []
    point_resources: list[DiagnosticResources] = []

    for grid_index in range(phase_bins // 2 + 1):
        angle = math.pi * grid_index / phase_bins
        mean = math.sin(angle) ** 2
        oracle = CanonicalRyStatevectorOracle([mean], seed=seed)
        flag = DirectAmplitudeThresholdFlag(
            oracle,
            threshold,
            phase_qubits=phase_qubits,
            relation=relation,
        )
        computed = flag.compute(flag.initial_state((0,)), tag="phase_grid_sweep")
        probability = joint_acceptance_probability(flag, computed)
        raw = flag.last_resources
        if raw is None:
            raise RuntimeError("QPE compute did not publish resources")
        item_resources = _diagnostic_resources(
            "phase_grid_qpe_compute",
            raw,
            statevector_dimension=flag.statevector_dimension,
        )
        peaks = (grid_index,) if grid_index in {0, phase_bins // 2} else (
            grid_index,
            phase_bins - grid_index,
        )
        threshold_angle = math.asin(math.sqrt(threshold))
        truth_above = _angle_at_or_above(angle, threshold_angle)
        points.append(
            PhaseGridPoint(
                grid_index=grid_index,
                angle=angle,
                mean=mean,
                mirrored_peak_bins=peaks,
                truth_accept=(truth_above == (relation == "above")),
                joint_acceptance_probability=probability,
                resources=item_resources,
            )
        )
        point_resources.append(item_resources)

    return PhaseGridSweep(
        phase_qubits=phase_qubits,
        phase_bins=phase_bins,
        threshold=threshold,
        relation=relation,
        points=tuple(points),
        resources=_aggregate_resources("phase_grid_sweep", point_resources),
        seed=seed,
    )


def _invalid_index_only_diffusion(
    full_state: ComplexState,
    initial_index_state: ComplexState,
    *,
    phase_bins: int,
) -> ComplexState:
    """INVALID negative control: reflect index independently per workspace.

    This is intentionally private.  It is not the full rank-one reflection
    required by amplitude amplification and must never be called by search.
    """

    state = np.asarray(full_state, dtype=np.complex128)
    index_state = np.asarray(initial_index_state, dtype=np.complex128)
    if state.ndim != 1 or index_state.ndim != 1:
        raise ValueError("states must be flat")
    expected = phase_bins * index_state.size * 2
    if state.size != expected:
        raise ValueError(f"full_state must have length {expected}")
    if not np.isclose(np.linalg.norm(state), 1.0, atol=1e-10):
        raise ValueError("full_state must be normalized")
    if not np.isclose(np.linalg.norm(index_state), 1.0, atol=1e-10):
        raise ValueError("initial_index_state must be normalized")
    view = state.reshape(phase_bins, index_state.size, 2)
    overlaps = np.einsum("i,pir->pr", index_state.conj(), view)
    reflected = 2.0 * np.einsum("i,pr->pir", index_state, overlaps) - view
    result = reflected.reshape(-1)
    if not np.isclose(np.linalg.norm(result), 1.0, atol=1e-10):
        raise RuntimeError("invalid control diffusion unexpectedly changed norm")
    return result


def _state_distance_up_to_global_phase(left: ComplexState, right: ComplexState) -> float:
    overlap = np.vdot(left, right)
    if abs(overlap) > 1e-15:
        right = right * np.exp(-1j * np.angle(overlap))
    return float(np.linalg.norm(left - right))


def _run_diffusion_method(
    means: tuple[float, ...],
    *,
    threshold: float,
    phase_qubits: int,
    relation: str,
    grover_iterations: int,
    seed: int | None,
    valid: bool,
) -> tuple[DiffusionMethodDiagnostic, ComplexState]:
    oracle = CanonicalRyStatevectorOracle(means, seed=seed)
    flag = DirectAmplitudeThresholdFlag(
        oracle,
        threshold,
        phase_qubits=phase_qubits,
        relation=relation,
    )
    initial = flag.initial_state()
    index_state = initial.reshape(flag.phase_bins, flag.index_dimension, 2)[0, :, 0]
    state = initial.copy()
    for _ in range(grover_iterations):
        state = flag.apply_reflection(state, tag="diffusion_ablation_reflection")
        if valid:
            state = full_workspace_rank_one_diffusion(state, initial)
        else:
            state = _invalid_index_only_diffusion(
                state,
                index_state,
                phase_bins=flag.phase_bins,
            )
    computed = flag.compute(state, tag="diffusion_ablation_decode")
    acceptance = joint_acceptance_probability(flag, computed)
    candidate = flag.sample_accept_index(computed, seed=seed)
    raw = flag.resources()
    method = VALID_FULL_METHOD if valid else INVALID_INDEX_METHOD
    resources = _diagnostic_resources(
        method,
        raw,
        statevector_dimension=flag.statevector_dimension,
        extra_gates={method: grover_iterations, "joint_accept_index_measurement": 1},
        extra_depth=grover_iterations + 1,
    )
    result = DiffusionMethodDiagnostic(
        method=method,
        algorithmically_valid=valid,
        output_eligible=valid,
        joint_acceptance_probability=acceptance,
        sampled_accept_index=candidate,
        resources=resources,
        warning=(
            None
            if valid
            else "INVALID negative control; never use this branch as algorithm output"
        ),
    )
    return result, computed


def run_diffusion_ablation(
    means: Sequence[float],
    *,
    threshold: float,
    phase_qubits: int,
    relation: str = "above",
    grover_iterations: int = 1,
    seed: int | None = 0,
) -> DiffusionAblation:
    """Compare correct full diffusion with an explicitly invalid negative control."""

    values = tuple(_probability(mean, "mean") for mean in means)
    if not values:
        raise ValueError("means cannot be empty")
    threshold = _probability(threshold, "threshold")
    phase_qubits = _integer(phase_qubits, "phase_qubits")
    if not 1 <= phase_qubits <= 12:
        raise ValueError("phase_qubits must lie in [1, 12]")
    relation = _relation(relation)
    grover_iterations = _integer(grover_iterations, "grover_iterations")
    if grover_iterations < 0:
        raise ValueError("grover_iterations must be nonnegative")
    seed = _seed(seed)

    full, full_state = _run_diffusion_method(
        values,
        threshold=threshold,
        phase_qubits=phase_qubits,
        relation=relation,
        grover_iterations=grover_iterations,
        seed=seed,
        valid=True,
    )
    invalid, invalid_state = _run_diffusion_method(
        values,
        threshold=threshold,
        phase_qubits=phase_qubits,
        relation=relation,
        grover_iterations=grover_iterations,
        seed=seed,
        valid=False,
    )
    distance = _state_distance_up_to_global_phase(full_state, invalid_state)
    return DiffusionAblation(
        full_workspace=full,
        invalid_index_only=invalid,
        state_distance_up_to_global_phase=distance,
        finite_phase_leakage_detected=distance > 1e-10,
        phase_qubits=phase_qubits,
        threshold=threshold,
        relation=relation,
        grover_iterations=grover_iterations,
        seed=seed,
    )
