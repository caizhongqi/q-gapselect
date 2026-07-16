"""Reproducible, auditable benchmarks for the direct quantum selection core.

Ground-truth membership is constructed with each synthetic instance but is
used only after an algorithm has returned.  Every execution adapter creates a
fresh charged oracle and passes the algorithm only that oracle plus public
problem parameters such as the threshold and requested output count.

The boundary-only adapter is an explicit negative control: QBoundaryEstimator
already returns a complete membership certificate.  Its record therefore
forbids describing the result as quantum discovery.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from types import MappingProxyType

import numpy as np

from .coherent import CanonicalRyStatevectorOracle
from .direct_baselines import ClassicalThresholdScan, IndependentQPEThresholdScan
from .direct_search import FullWorkspaceBBHT
from .direct_topk import CalibratedDirectTopKController
from .primitives import QBoundaryEstimator

FAMILIES = (
    "equal_grid",
    "heterogeneous_dyadic",
    "off_grid_random",
    "endpoint_angular",
)
METHODS = (
    "direct_bbht",
    "independent_qpe_scan",
    "classical_threshold_scan",
    "boundary_only_negative_control",
    "refined_boundary_only_negative_control",
    "calibrated_direct_topk",
    "adaptive_calibrated_direct_topk",
    "fixed_max_precision_topk",
)
TRUTH_USAGE = "evaluation_only_never_passed_as_membership_to_algorithm"
CLAIM_STATUS = "executed_small_state_benchmark_no_quantum_advantage_theorem"


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


def _open_probability(value: object, name: str) -> float:
    result = _real(value, name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return result


def _relation(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("relation must be a string")
    if value not in {"above", "below"}:
        raise ValueError("relation must be 'above' or 'below'")
    return value


def _method(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("method must be a string")
    if value not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    return value


def _csv_indices(values: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in sorted(values))


def _instance_manifest_hash(instance: QuantumBenchmarkInstance) -> str:
    manifest = {
        "family": instance.family,
        "instance_seed": instance.instance_seed,
        "n_arms": instance.n_arms,
        "k": instance.k,
        "means": instance.means,
        "angles": instance.angles,
        "threshold": instance.threshold,
        "threshold_angle": instance.threshold_angle,
    }
    encoded = json.dumps(
        manifest,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _query_total(resources: object) -> int:
    query_counts = getattr(resources, "query_counts", {})
    if not isinstance(query_counts, Mapping):
        return 0
    if "total" in query_counts:
        return int(query_counts["total"])
    return int(query_counts.get("coherent_total", 0)) + int(
        query_counts.get("classical_total", 0)
    )


def _resource_integer(resources: object, *names: str, default: int = 0) -> int:
    for name in names:
        value = getattr(resources, name, None)
        if value is not None:
            return int(value)
    return default


def _gate_total(resources: object) -> int:
    gate_counts = getattr(resources, "gate_counts", {})
    if not isinstance(gate_counts, Mapping):
        return 0
    return sum(int(value) for value in gate_counts.values())


def _outputs(result: object) -> tuple[int, ...]:
    for name in ("outputs", "found_indices", "selected"):
        values = getattr(result, name, None)
        if values is not None:
            return tuple(int(value) for value in values)
    return ()


def _run_resumable_to_terminal(controller: object, *, step_bound: int) -> object:
    """Keep advancing a resumable implementation until its status is terminal."""

    if step_bound <= 0:
        raise ValueError("step_bound must be positive")
    run = controller.run
    result = run()
    advances = 1
    while str(getattr(result, "status", "")) == "paused_resumable":
        if advances >= step_bound:
            raise RuntimeError("resumable benchmark exceeded its deterministic step bound")
        resume = getattr(controller, "resume", run)
        result = resume()
        advances += 1
    return result


@dataclass(frozen=True, slots=True)
class QuantumBenchmarkInstance:
    """Synthetic threshold/Top-k instance with evaluation-only labels."""

    family: str
    instance_seed: int
    n_arms: int
    k: int
    means: tuple[float, ...]
    angles: tuple[float, ...]
    threshold: float
    threshold_angle: float
    truth_above: tuple[int, ...]
    truth_below: tuple[int, ...]
    topk_truth: tuple[int, ...]
    construction: str
    truth_usage: str = TRUTH_USAGE


def make_benchmark_instance(
    family: str,
    *,
    n_arms: int = 4,
    k: int = 2,
    seed: int = 0,
) -> QuantumBenchmarkInstance:
    """Construct one seeded angular instance from a named benchmark family."""

    if not isinstance(family, str):
        raise TypeError("family must be a string")
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    n_arms = _integer(n_arms, "n_arms")
    k = _integer(k, "k")
    seed = _integer(seed, "seed")
    if n_arms < 2:
        raise ValueError("n_arms must be at least two")
    if not 1 <= k < n_arms:
        raise ValueError("k must satisfy 1 <= k < n_arms")
    rng = np.random.default_rng(seed)

    if family == "equal_grid":
        grid_bits = max(3, int(math.ceil(math.log2(2 * (n_arms + 1)))))
        phase_bins = 1 << grid_bits
        descending = np.asarray(
            [math.pi * grid_index / phase_bins for grid_index in range(n_arms, 0, -1)],
            dtype=np.float64,
        )
        construction = f"exact_equal_angular_grid_phase_bins_{phase_bins}"
    elif family == "heterogeneous_dyadic":
        center = math.pi / 4.0
        base = math.pi / 8.0
        above = sorted(
            (center + base / (2**level) for level in range(k)),
            reverse=True,
        )
        below = sorted(
            (center - base / (2**level) for level in range(n_arms - k)),
            reverse=True,
        )
        descending = np.asarray(above + below, dtype=np.float64)
        construction = "dyadic_heterogeneous_angular_gaps_about_pi_over_four"
    elif family == "off_grid_random":
        lower = 0.07
        upper = math.pi / 2.0 - 0.07
        descending = np.sort(rng.uniform(lower, upper, size=n_arms))[::-1]
        construction = "seeded_continuous_off_grid_angular_draws"
    else:
        descending = np.linspace(math.pi / 2.0, 0.0, n_arms, dtype=np.float64)
        construction = "angular_endpoints_including_zero_and_pi_over_two"

    if not np.all(np.diff(descending) < 0.0):
        raise RuntimeError("benchmark construction failed to produce distinct angles")
    threshold_angle = float(0.5 * (descending[k - 1] + descending[k]))

    # Shuffle arm identities independently of ranking.  Membership labels are
    # computed only for the returned evaluation object, never handed to a
    # selection implementation by the execution adapter below.
    permutation = rng.permutation(n_arms)
    angles_array = np.empty(n_arms, dtype=np.float64)
    angles_array[permutation] = descending
    angles = tuple(float(value) for value in angles_array)
    means = tuple(float(math.sin(angle) ** 2) for angle in angles)
    threshold = float(math.sin(threshold_angle) ** 2)
    truth_above = tuple(
        index for index, angle in enumerate(angles) if angle > threshold_angle
    )
    truth_below = tuple(
        index for index, angle in enumerate(angles) if angle < threshold_angle
    )
    if len(truth_above) != k or len(truth_below) != n_arms - k:
        raise RuntimeError("benchmark truth partition is inconsistent with k")
    return QuantumBenchmarkInstance(
        family=family,
        instance_seed=seed,
        n_arms=n_arms,
        k=k,
        means=means,
        angles=angles,
        threshold=threshold,
        threshold_angle=threshold_angle,
        truth_above=tuple(sorted(truth_above)),
        truth_below=tuple(sorted(truth_below)),
        topk_truth=tuple(sorted(truth_above)),
        construction=construction,
    )


def make_benchmark_suite(
    *,
    families: Sequence[str] = FAMILIES,
    n_arms: int = 4,
    k: int = 2,
    seeds: Sequence[int] = (0,),
) -> tuple[QuantumBenchmarkInstance, ...]:
    """Generate a deterministic family-major, seed-minor instance suite."""

    try:
        family_values = tuple(families)
    except TypeError as error:
        raise TypeError("families must be a sequence of strings") from error
    try:
        seed_values = tuple(_integer(seed, "seed") for seed in seeds)
    except TypeError as error:
        raise TypeError("seeds must be a sequence of integers") from error
    if not family_values:
        raise ValueError("families cannot be empty")
    if not seed_values:
        raise ValueError("seeds cannot be empty")
    return tuple(
        make_benchmark_instance(family, n_arms=n_arms, k=k, seed=seed)
        for family in family_values
        for seed in seed_values
    )


@dataclass(frozen=True, slots=True)
class QuantumBenchmarkConfig:
    """Shared strict execution budgets for all benchmark adapters."""

    phase_qubits: int = 4
    max_phase_qubits: int = 7
    verification_shots: int = 32
    confidence: float = 0.05
    max_attempts_per_output: int = 12
    max_statevector_dimension: int = 1_048_576
    classical_shots_per_arm: int = 128
    boundary_shots_per_round: int = 32
    max_boundary_rounds: int = 4

    def __post_init__(self) -> None:
        for name in (
            "phase_qubits",
            "max_phase_qubits",
            "verification_shots",
            "max_attempts_per_output",
            "max_statevector_dimension",
            "classical_shots_per_arm",
            "boundary_shots_per_round",
            "max_boundary_rounds",
        ):
            value = _integer(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.phase_qubits > 12:
            raise ValueError("phase_qubits exceeds the small-state limit of 12")
        if self.max_phase_qubits < self.phase_qubits:
            raise ValueError("max_phase_qubits must be at least phase_qubits")
        if self.max_phase_qubits > 12:
            raise ValueError("max_phase_qubits exceeds the small-state limit of 12")
        object.__setattr__(
            self,
            "confidence",
            _open_probability(self.confidence, "confidence"),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """Flat, JSON-ready execution summary; no algorithm trace is embedded."""

    family: str
    instance_manifest_sha256: str
    instance_seed: int
    trial_seed: int
    method: str
    n_arms: int
    k: int
    relation: str
    threshold: float
    threshold_angle: float
    minimum_angular_boundary_gap: float
    minimum_mean_boundary_gap: float
    phase_qubits: int | None
    expected_count: int
    outputs: str
    truth: str
    complete: bool
    exact: bool
    certified: bool
    status: str
    failure_reason: str | None
    total_queries: int
    reflection_queries: int
    decode_queries: int
    fresh_verification_queries: int
    basis_sampling_queries: int
    other_queries: int
    gates: int
    depth: int
    qubits: int
    workspace_qubits: int
    retained_statevector_dimension: int
    peak_statevector_dimension: int
    comparator_expanded_statevector_dimension: int
    dense_qft_matrix_dimension: int
    estimated_peak_bytes: int
    attempts: int
    boundary_certificate_available: bool
    quantum_discovery_claim_allowed: bool
    control_role: str
    interpretation: str
    truth_usage: str = TRUTH_USAGE
    claim_status: str = CLAIM_STATUS
    initial_phase_qubits: int | None = None
    max_phase_qubits: int | None = None
    phase_candidate_levels: str = ""
    boundary_rounds: int = 0
    memory_proxy_semantics: str = (
        "analytic_array_size_proxy_not_measured_numpy_peak"
    )

    @property
    def success(self) -> bool:
        return self.complete and self.exact and self.certified

    def as_flat_dict(self) -> dict[str, object]:
        """Return only scalar/string fields suitable for a CSV or JSON row."""

        return asdict(self)


def _direct_query_split(result: object, phase_qubits: int) -> tuple[int, int, int]:
    """Split executed direct-search calls without retaining its full trace."""

    phase_bins = 1 << phase_qubits
    reflection = 0
    decode = 0
    verification = 0
    attempts = getattr(result, "trace", None)
    if attempts is None:
        attempts = getattr(result, "attempts", ())
    for attempt in tuple(attempts):
        resources = getattr(attempt, "resources", None)
        iterations = int(getattr(attempt, "grover_iterations", 0))
        reflection += _resource_integer(
            resources,
            "reflection_queries",
            "reflection_oracle_queries",
            default=iterations * (4 * phase_bins - 2),
        )
        decode += _resource_integer(
            resources,
            "decode_queries",
            "decode_oracle_queries",
            default=2 * phase_bins - 1,
        )
        verification_shots = _resource_integer(resources, "verification_shots")
        verification += _resource_integer(
            resources,
            "fresh_verification_queries",
            "fresh_verification_oracle_queries",
            default=verification_shots * (2 * phase_bins - 1),
        )
    return reflection, decode, verification


def _record(
    *,
    instance: QuantumBenchmarkInstance,
    trial_seed: int,
    method: str,
    relation: str,
    phase_qubits: int | None,
    expected_count: int,
    outputs: tuple[int, ...],
    truth: tuple[int, ...],
    complete: bool,
    certified: bool,
    status: str,
    failure_reason: str | None,
    total_queries: int,
    reflection_queries: int,
    decode_queries: int,
    fresh_verification_queries: int,
    basis_sampling_queries: int,
    gates: int,
    depth: int,
    qubits: int,
    workspace_qubits: int,
    retained_statevector_dimension: int,
    peak_statevector_dimension: int,
    comparator_expanded_statevector_dimension: int,
    dense_qft_matrix_dimension: int,
    attempts: int,
    boundary_certificate_available: bool,
    quantum_discovery_claim_allowed: bool,
    control_role: str,
    interpretation: str,
    initial_phase_qubits: int | None = None,
    max_phase_qubits: int | None = None,
    phase_candidate_levels: str = "",
    boundary_rounds: int = 0,
) -> BenchmarkRecord:
    split = (
        reflection_queries
        + decode_queries
        + fresh_verification_queries
        + basis_sampling_queries
    )
    if min(total_queries, reflection_queries, decode_queries, fresh_verification_queries) < 0:
        raise RuntimeError("query counters cannot be negative")
    if basis_sampling_queries < 0 or split > total_queries:
        raise RuntimeError("query split exceeds the measured total")
    return BenchmarkRecord(
        family=instance.family,
        instance_manifest_sha256=_instance_manifest_hash(instance),
        instance_seed=instance.instance_seed,
        trial_seed=trial_seed,
        method=method,
        n_arms=instance.n_arms,
        k=instance.k,
        relation=relation,
        threshold=instance.threshold,
        threshold_angle=instance.threshold_angle,
        minimum_angular_boundary_gap=min(
            abs(angle - instance.threshold_angle) for angle in instance.angles
        ),
        minimum_mean_boundary_gap=min(
            abs(mean - instance.threshold) for mean in instance.means
        ),
        phase_qubits=phase_qubits,
        expected_count=expected_count,
        outputs=_csv_indices(outputs),
        truth=_csv_indices(truth),
        complete=complete,
        exact=complete and set(outputs) == set(truth),
        certified=certified,
        status=status,
        failure_reason=failure_reason,
        total_queries=total_queries,
        reflection_queries=reflection_queries,
        decode_queries=decode_queries,
        fresh_verification_queries=fresh_verification_queries,
        basis_sampling_queries=basis_sampling_queries,
        other_queries=total_queries - split,
        gates=gates,
        depth=depth,
        qubits=qubits,
        workspace_qubits=workspace_qubits,
        retained_statevector_dimension=retained_statevector_dimension,
        peak_statevector_dimension=peak_statevector_dimension,
        comparator_expanded_statevector_dimension=(
            comparator_expanded_statevector_dimension
        ),
        dense_qft_matrix_dimension=dense_qft_matrix_dimension,
        estimated_peak_bytes=16 * peak_statevector_dimension,
        attempts=attempts,
        boundary_certificate_available=boundary_certificate_available,
        quantum_discovery_claim_allowed=quantum_discovery_claim_allowed,
        control_role=control_role,
        interpretation=interpretation,
        initial_phase_qubits=initial_phase_qubits,
        max_phase_qubits=max_phase_qubits,
        phase_candidate_levels=phase_candidate_levels,
        boundary_rounds=boundary_rounds,
    )


class QuantumBenchmarkRunner:
    """Execute all benchmark methods through one resource-normalizing adapter."""

    def __init__(self, config: QuantumBenchmarkConfig | None = None) -> None:
        if config is None:
            config = QuantumBenchmarkConfig()
        if not isinstance(config, QuantumBenchmarkConfig):
            raise TypeError("config must be a QuantumBenchmarkConfig")
        self.config = config

    def run(
        self,
        method: str,
        instance: QuantumBenchmarkInstance,
        *,
        trial_seed: int,
        relation: str = "above",
    ) -> BenchmarkRecord:
        method = _method(method)
        if not isinstance(instance, QuantumBenchmarkInstance):
            raise TypeError("instance must be a QuantumBenchmarkInstance")
        trial_seed = _integer(trial_seed, "trial_seed")
        relation = _relation(relation)
        oracle = CanonicalRyStatevectorOracle(instance.means, seed=trial_seed)
        config = self.config

        if method == "direct_bbht":
            truth = instance.truth_above if relation == "above" else instance.truth_below
            expected_count = instance.k if relation == "above" else instance.n_arms - instance.k
            search = FullWorkspaceBBHT(
                oracle,
                instance.threshold,
                expected_count,
                phase_qubits=config.phase_qubits,
                relation=relation,
                verification_shots=config.verification_shots,
                verification_confidence=config.confidence,
                max_attempts_per_output=config.max_attempts_per_output,
                max_statevector_dimension=config.max_statevector_dimension,
                seed=trial_seed,
            )
            result = _run_resumable_to_terminal(
                search,
                step_bound=max(2, expected_count * config.max_attempts_per_output + 2),
            )
            resources = result.resources
            reflection, decode, verification = _direct_query_split(
                result, config.phase_qubits
            )
            retained = _resource_integer(
                resources,
                "retained_statevector_dimension",
                "statevector_dimension",
            )
            peak = _resource_integer(
                resources,
                "peak_statevector_dimension",
                default=retained,
            )
            comparator = _resource_integer(
                resources,
                "comparator_expanded_statevector_dimension",
            )
            dense_qft = _resource_integer(
                resources,
                "dense_qft_matrix_dimension",
            )
            return _record(
                instance=instance,
                trial_seed=trial_seed,
                method=method,
                relation=relation,
                phase_qubits=config.phase_qubits,
                expected_count=expected_count,
                outputs=_outputs(result),
                truth=truth,
                complete=bool(result.complete),
                certified=bool(result.complete and result.verified),
                status=str(result.status),
                failure_reason=result.failure_reason,
                total_queries=_query_total(resources),
                reflection_queries=reflection,
                decode_queries=decode,
                fresh_verification_queries=verification,
                basis_sampling_queries=0,
                gates=_gate_total(resources),
                depth=_resource_integer(resources, "depth"),
                qubits=_resource_integer(resources, "qubits"),
                workspace_qubits=_resource_integer(resources, "workspace_qubits"),
                retained_statevector_dimension=retained,
                peak_statevector_dimension=peak,
                comparator_expanded_statevector_dimension=comparator,
                dense_qft_matrix_dimension=dense_qft,
                attempts=int(result.attempts),
                boundary_certificate_available=False,
                quantum_discovery_claim_allowed=True,
                control_role="primary_unknown_oracle_threshold_discovery",
                interpretation="direct_full_workspace_qpe_bbht_no_advantage_theorem",
            )

        if method == "independent_qpe_scan":
            truth = instance.truth_above if relation == "above" else instance.truth_below
            expected_count = instance.k if relation == "above" else instance.n_arms - instance.k
            result = IndependentQPEThresholdScan(
                oracle,
                instance.threshold,
                expected_count,
                phase_qubits=config.phase_qubits,
                relation=relation,
                verification_shots=config.verification_shots,
                confidence=config.confidence,
                seed=trial_seed,
            ).run()
            resources = result.resources
            retained = (1 << config.phase_qubits) * oracle.index_dimension * 2
            dense_qft = (1 << config.phase_qubits) ** 2
            peak = dense_qft + 2 * retained
            return _record(
                instance=instance,
                trial_seed=trial_seed,
                method=method,
                relation=relation,
                phase_qubits=config.phase_qubits,
                expected_count=expected_count,
                outputs=_outputs(result),
                truth=truth,
                complete=result.complete,
                certified=result.complete and result.verified,
                status=result.status,
                failure_reason=result.failure_reason,
                total_queries=_query_total(resources),
                reflection_queries=0,
                decode_queries=0,
                fresh_verification_queries=_query_total(resources),
                basis_sampling_queries=0,
                gates=_gate_total(resources),
                depth=resources.depth,
                qubits=config.phase_qubits + oracle.index_qubits + 2,
                workspace_qubits=config.phase_qubits + 1,
                retained_statevector_dimension=retained,
                peak_statevector_dimension=peak,
                comparator_expanded_statevector_dimension=0,
                dense_qft_matrix_dimension=dense_qft,
                attempts=resources.verifier_calls,
                boundary_certificate_available=False,
                quantum_discovery_claim_allowed=True,
                control_role="independent_qpe_no_amplitude_amplification_baseline",
                interpretation="per_arm_fresh_qpe_scan_union_bound",
            )

        if method == "classical_threshold_scan":
            truth = instance.truth_above if relation == "above" else instance.truth_below
            expected_count = instance.k if relation == "above" else instance.n_arms - instance.k
            result = ClassicalThresholdScan(
                oracle,
                instance.threshold,
                expected_count,
                relation=relation,
                shots_per_arm=config.classical_shots_per_arm,
                confidence=config.confidence,
                seed=trial_seed,
            ).run()
            resources = result.resources
            total = _query_total(resources)
            return _record(
                instance=instance,
                trial_seed=trial_seed,
                method=method,
                relation=relation,
                phase_qubits=None,
                expected_count=expected_count,
                outputs=_outputs(result),
                truth=truth,
                complete=result.complete,
                certified=result.complete and result.verified,
                status=result.status,
                failure_reason=result.failure_reason,
                total_queries=total,
                reflection_queries=0,
                decode_queries=0,
                fresh_verification_queries=0,
                basis_sampling_queries=total,
                gates=_gate_total(resources),
                depth=resources.depth,
                qubits=oracle.index_qubits + 1,
                workspace_qubits=0,
                retained_statevector_dimension=oracle.statevector_dimension,
                peak_statevector_dimension=oracle.statevector_dimension,
                comparator_expanded_statevector_dimension=0,
                dense_qft_matrix_dimension=0,
                attempts=resources.arms_examined,
                boundary_certificate_available=False,
                quantum_discovery_claim_allowed=False,
                control_role="classical_basis_sampling_baseline",
                interpretation="simultaneous_hoeffding_threshold_scan",
            )

        if method in {
            "boundary_only_negative_control",
            "refined_boundary_only_negative_control",
        }:
            refined = method == "refined_boundary_only_negative_control"
            boundary_seed = (
                int(np.random.default_rng(trial_seed).integers(0, 2**32))
                if refined
                else trial_seed
            )
            minimum_margin = (
                2.0 * math.pi / (1 << config.max_phase_qubits)
                if refined
                else 0.0
            )
            result = QBoundaryEstimator(
                oracle,
                instance.k,
                confidence=config.confidence,
                shots_per_round=config.boundary_shots_per_round,
                max_rounds=config.max_boundary_rounds,
                minimum_angular_margin=minimum_margin,
                seed=boundary_seed,
                tag="benchmark_boundary_negative_control",
            ).run()
            resources = result.resources
            certificate = result.certificate
            outputs = () if certificate is None else tuple(certificate.selected)
            total = _query_total(resources)
            status = (
                "complete_refined_membership_certificate_forbidden_quantum_discovery_claim"
                if refined and result.complete
                else "complete_membership_certificate_forbidden_quantum_discovery_claim"
                if result.complete
                else "boundary_not_separated"
            )
            return _record(
                instance=instance,
                trial_seed=trial_seed,
                method=method,
                relation="topk",
                phase_qubits=None,
                expected_count=instance.k,
                outputs=outputs,
                truth=instance.topk_truth,
                complete=result.complete,
                certified=result.complete and certificate is not None,
                status=status,
                failure_reason=None if result.complete else "boundary_budget_exhausted",
                total_queries=total,
                reflection_queries=0,
                decode_queries=0,
                fresh_verification_queries=0,
                basis_sampling_queries=total,
                gates=_gate_total(resources),
                depth=resources.depth,
                qubits=resources.qubits,
                workspace_qubits=resources.workspace_qubits,
                retained_statevector_dimension=oracle.statevector_dimension,
                peak_statevector_dimension=oracle.statevector_dimension,
                comparator_expanded_statevector_dimension=0,
                dense_qft_matrix_dimension=0,
                attempts=len(result.rounds),
                boundary_certificate_available=certificate is not None,
                quantum_discovery_claim_allowed=False,
                control_role=(
                    "NEGATIVE_CONTROL_refined_boundary_already_knows_membership"
                    if refined
                    else "NEGATIVE_CONTROL_boundary_already_knows_membership"
                ),
                interpretation=(
                    "FORBIDDEN_as_quantum_discovery_refined_boundary_contains_full_certificate"
                    if refined
                    else "FORBIDDEN_as_quantum_discovery_boundary_contains_full_certificate"
                ),
                initial_phase_qubits=(config.phase_qubits if refined else None),
                max_phase_qubits=(config.max_phase_qubits if refined else None),
                phase_candidate_levels=(
                    ",".join(
                        str(value)
                        for value in range(
                            config.phase_qubits, config.max_phase_qubits + 1
                        )
                    )
                    if refined
                    else ""
                ),
                boundary_rounds=len(result.rounds),
            )

        adaptive_phase = method in {
            "adaptive_calibrated_direct_topk",
            "fixed_max_precision_topk",
        }
        initial_phase_qubits = (
            config.max_phase_qubits
            if method == "fixed_max_precision_topk"
            else config.phase_qubits
        )
        maximum_phase_qubits = (
            config.max_phase_qubits
            if adaptive_phase
            else initial_phase_qubits
        )
        controller = CalibratedDirectTopKController(
            oracle,
            instance.k,
            phase_qubits=initial_phase_qubits,
            adaptive_phase_qubits=adaptive_phase,
            max_phase_qubits=maximum_phase_qubits,
            confidence=config.confidence,
            boundary_shots_per_round=config.boundary_shots_per_round,
            max_boundary_rounds=config.max_boundary_rounds,
            max_attempts_per_output=config.max_attempts_per_output,
            verification_shots=config.verification_shots,
            verification_confidence=config.confidence,
            max_statevector_dimension=config.max_statevector_dimension,
            seed=trial_seed,
        )
        result = _run_resumable_to_terminal(
            controller,
            step_bound=max(2, 2 * instance.n_arms * config.max_attempts_per_output + 2),
        )
        total = _query_total(result.resources)
        # DirectTopK freezes the boundary query ledger at calibration time.
        # The interval-shot reconstruction is still recorded as a redundant
        # audit check because the boundary certificate already determines
        # membership and therefore forbids a discovery-advantage claim.
        basis = sum(interval.shots for interval in result.boundary.intervals)
        reflection = 0
        decode = 0
        verification = 0
        gates = _gate_total(result.boundary.resources)
        depth = result.boundary.resources.depth
        qubits = result.boundary.resources.qubits
        workspace = result.boundary.resources.workspace_qubits
        retained = oracle.statevector_dimension
        peak = oracle.statevector_dimension
        comparator = 0
        dense_qft = 0
        attempts = 0
        for branch in result.branches:
            branch_resources = branch.resources
            if branch_resources is not None:
                gates += _gate_total(branch_resources)
                depth += _resource_integer(branch_resources, "depth")
                qubits = max(qubits, _resource_integer(branch_resources, "qubits"))
                workspace = max(
                    workspace,
                    _resource_integer(branch_resources, "workspace_qubits"),
                )
                retained = max(
                    retained,
                    _resource_integer(
                        branch_resources,
                        "retained_statevector_dimension",
                        "statevector_dimension",
                    ),
                )
                peak = max(
                    peak,
                    _resource_integer(
                        branch_resources,
                        "peak_statevector_dimension",
                        default=retained,
                    ),
                )
                comparator = max(
                    comparator,
                    _resource_integer(
                        branch_resources,
                        "comparator_expanded_statevector_dimension",
                    ),
                )
                dense_qft = max(
                    dense_qft,
                    _resource_integer(
                        branch_resources,
                        "dense_qft_matrix_dimension",
                    ),
                )
            attempts += len(branch.attempts)
            branch_reflection, branch_decode, branch_verification = _direct_query_split(
                branch, result.resources.phase_qubits
            )
            reflection += branch_reflection
            decode += branch_decode
            verification += branch_verification
        certificate_available = result.boundary.certificate is not None
        roles = {
            "calibrated_direct_topk": (
                "calibrated_direct_rediscovery_information_firewall",
                "fixed_precision_coherent_rediscovery_no_advantage_claim",
            ),
            "adaptive_calibrated_direct_topk": (
                "resource_aware_measured_margin_phase_schedule",
                "adaptive_precision_selected_before_search_no_advantage_claim",
            ),
            "fixed_max_precision_topk": (
                "fixed_max_precision_matched_boundary_refinement_control",
                "same_refined_boundary_fixed_max_precision_control",
            ),
        }
        control_role, interpretation = roles[method]
        schedule = result.phase_schedule
        candidate_levels = (
            ""
            if schedule is None
            else ",".join(
                str(candidate.phase_qubits) for candidate in schedule.candidates
            )
        )
        return _record(
            instance=instance,
            trial_seed=trial_seed,
            method=method,
            relation="topk",
            phase_qubits=(
                schedule.selected_phase_qubits
                if schedule is not None
                else result.resources.phase_qubits
                if result.branches
                else None
            ),
            expected_count=instance.k,
            outputs=tuple(result.selected),
            truth=instance.topk_truth,
            complete=result.complete,
            certified=result.complete,
            status=result.status,
            failure_reason=None if result.complete else result.status,
            total_queries=total,
            reflection_queries=reflection,
            decode_queries=decode,
            fresh_verification_queries=verification,
            basis_sampling_queries=basis,
            gates=gates,
            depth=depth,
            qubits=qubits,
            workspace_qubits=workspace,
            retained_statevector_dimension=retained,
            peak_statevector_dimension=peak,
            comparator_expanded_statevector_dimension=comparator,
            dense_qft_matrix_dimension=dense_qft,
            attempts=attempts,
            boundary_certificate_available=certificate_available,
            quantum_discovery_claim_allowed=False,
            control_role=control_role,
            interpretation=interpretation,
            initial_phase_qubits=result.initial_phase_qubits,
            max_phase_qubits=result.max_phase_qubits,
            phase_candidate_levels=candidate_levels,
            boundary_rounds=len(result.boundary.rounds),
        )


def wilson_success_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a binomial benchmark success rate."""

    successes = _integer(successes, "successes")
    trials = _integer(trials, "trials")
    confidence = _open_probability(confidence, "confidence")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in {0, ..., trials}")
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
class NumericAggregate:
    mean: float
    minimum: float
    maximum: float
    quantiles: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class BenchmarkAggregate:
    family: str
    n_arms: int
    k: int
    method: str
    relation: str
    trials: int
    successes: int
    success_rate: float
    wilson_lower: float
    wilson_upper: float
    complete_count: int
    exact_count: int
    certified_count: int
    status_counts: Mapping[str, int]
    metrics: Mapping[str, NumericAggregate]


def _quantile_labels(values: Sequence[float]) -> tuple[tuple[str, float], ...]:
    try:
        quantiles = tuple(_real(value, "quantile") for value in values)
    except TypeError as error:
        raise TypeError("quantiles must be a sequence of real numbers") from error
    if not quantiles:
        raise ValueError("quantiles cannot be empty")
    if any(not 0.0 <= value <= 1.0 for value in quantiles):
        raise ValueError("quantiles must lie in [0, 1]")
    if len(set(quantiles)) != len(quantiles):
        raise ValueError("quantiles must be unique")
    return tuple((f"q{round(value * 100):02d}", value) for value in sorted(quantiles))


def _numeric_aggregate(
    values: Sequence[float],
    quantiles: tuple[tuple[str, float], ...],
) -> NumericAggregate:
    array = np.asarray(values, dtype=np.float64)
    return NumericAggregate(
        mean=float(np.mean(array)),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        quantiles=MappingProxyType(
            {
                label: float(np.quantile(array, quantile))
                for label, quantile in quantiles
            }
        ),
    )


def aggregate_benchmark_records(
    records: Sequence[BenchmarkRecord],
    *,
    quantiles: Sequence[float] = (0.25, 0.5, 0.75),
    confidence: float = 0.95,
) -> tuple[BenchmarkAggregate, ...]:
    """Group records and aggregate resources, status counts, and Wilson CIs."""

    try:
        rows = tuple(records)
    except TypeError as error:
        raise TypeError("records must be a sequence of BenchmarkRecord") from error
    if not rows:
        raise ValueError("records cannot be empty")
    if any(not isinstance(record, BenchmarkRecord) for record in rows):
        raise TypeError("every record must be a BenchmarkRecord")
    confidence = _open_probability(confidence, "confidence")
    labels = _quantile_labels(quantiles)
    groups: defaultdict[
        tuple[str, int, int, str, str], list[BenchmarkRecord]
    ] = defaultdict(list)
    for record in rows:
        groups[
            (
                record.family,
                record.n_arms,
                record.k,
                record.method,
                record.relation,
            )
        ].append(record)
    metric_names = (
        "total_queries",
        "reflection_queries",
        "decode_queries",
        "fresh_verification_queries",
        "basis_sampling_queries",
        "gates",
        "depth",
        "qubits",
        "retained_statevector_dimension",
        "peak_statevector_dimension",
        "comparator_expanded_statevector_dimension",
        "dense_qft_matrix_dimension",
        "estimated_peak_bytes",
        "attempts",
        "boundary_rounds",
    )
    results: list[BenchmarkAggregate] = []
    for (family, n_arms, k, method, relation), group in sorted(groups.items()):
        successes = sum(record.success for record in group)
        wilson_lower, wilson_upper = wilson_success_interval(
            successes,
            len(group),
            confidence=confidence,
        )
        status_counts = Counter(record.status for record in group)
        metrics = {
            name: _numeric_aggregate(
                [float(getattr(record, name)) for record in group],
                labels,
            )
            for name in metric_names
        }
        for name in (
            "phase_qubits",
            "initial_phase_qubits",
            "max_phase_qubits",
        ):
            values = [getattr(record, name) for record in group]
            if all(value is not None for value in values):
                metrics[name] = _numeric_aggregate(
                    [float(value) for value in values if value is not None],
                    labels,
                )
        results.append(
            BenchmarkAggregate(
                family=family,
                n_arms=n_arms,
                k=k,
                method=method,
                relation=relation,
                trials=len(group),
                successes=successes,
                success_rate=successes / len(group),
                wilson_lower=wilson_lower,
                wilson_upper=wilson_upper,
                complete_count=sum(record.complete for record in group),
                exact_count=sum(record.exact for record in group),
                certified_count=sum(record.certified for record in group),
                status_counts=MappingProxyType(dict(sorted(status_counts.items()))),
                metrics=MappingProxyType(metrics),
            )
        )
    return tuple(results)


@dataclass(frozen=True, slots=True)
class PairedQueryRatio:
    family: str
    instance_seed: int
    trial_seed: int
    relation: str
    numerator_method: str
    denominator_method: str
    query_field: str
    numerator_queries: float
    denominator_queries: float
    ratio: float | None
    status: str


@dataclass(frozen=True, slots=True)
class PairedRatioAggregate:
    numerator_method: str
    denominator_method: str
    query_field: str
    pairs: int
    finite_pairs: int
    status_counts: Mapping[str, int]
    ratios: NumericAggregate | None


def paired_query_ratios(
    records: Sequence[BenchmarkRecord],
    numerator_method: str,
    denominator_method: str,
    *,
    query_field: str = "total_queries",
) -> tuple[PairedQueryRatio, ...]:
    """Pair methods within the same instance/trial before computing ratios."""

    numerator_method = _method(numerator_method)
    denominator_method = _method(denominator_method)
    allowed_fields = {
        "total_queries",
        "reflection_queries",
        "decode_queries",
        "fresh_verification_queries",
        "basis_sampling_queries",
    }
    if not isinstance(query_field, str):
        raise TypeError("query_field must be a string")
    if query_field not in allowed_fields:
        raise ValueError(f"query_field must be one of {sorted(allowed_fields)}")
    rows = tuple(records)
    if any(not isinstance(record, BenchmarkRecord) for record in rows):
        raise TypeError("every record must be a BenchmarkRecord")
    keyed: dict[tuple[str, int, int, int, int, str, str], BenchmarkRecord] = {}
    for record in rows:
        key = (
            record.family,
            record.instance_seed,
            record.trial_seed,
            record.n_arms,
            record.k,
            record.relation,
            record.method,
        )
        if key in keyed:
            raise ValueError("duplicate method record for one paired trial")
        keyed[key] = record

    pairs: list[PairedQueryRatio] = []
    for key, numerator in sorted(keyed.items()):
        if numerator.method != numerator_method:
            continue
        denominator_key = key[:-1] + (denominator_method,)
        denominator = keyed.get(denominator_key)
        if denominator is None:
            continue
        numerator_value = float(getattr(numerator, query_field))
        denominator_value = float(getattr(denominator, query_field))
        if not numerator.success or not denominator.success:
            ratio = None
            status = "not_both_certified_success_excluded"
        elif denominator_value == 0.0:
            ratio = None
            status = "zero_denominator_excluded"
        else:
            ratio = numerator_value / denominator_value
            status = "both_certified_success_paired"
        pairs.append(
            PairedQueryRatio(
                family=numerator.family,
                instance_seed=numerator.instance_seed,
                trial_seed=numerator.trial_seed,
                relation=numerator.relation,
                numerator_method=numerator_method,
                denominator_method=denominator_method,
                query_field=query_field,
                numerator_queries=numerator_value,
                denominator_queries=denominator_value,
                ratio=ratio,
                status=status,
            )
        )
    return tuple(pairs)


def aggregate_paired_query_ratios(
    pairs: Sequence[PairedQueryRatio],
    *,
    quantiles: Sequence[float] = (0.25, 0.5, 0.75),
) -> PairedRatioAggregate:
    """Aggregate already trial-paired query ratios without ratio-of-means bias."""

    rows = tuple(pairs)
    if not rows:
        raise ValueError("pairs cannot be empty")
    if any(not isinstance(pair, PairedQueryRatio) for pair in rows):
        raise TypeError("every pair must be a PairedQueryRatio")
    identity = {
        (pair.numerator_method, pair.denominator_method, pair.query_field)
        for pair in rows
    }
    if len(identity) != 1:
        raise ValueError("all pairs must summarize the same methods and query field")
    numerator_method, denominator_method, query_field = identity.pop()
    finite = [float(pair.ratio) for pair in rows if pair.ratio is not None]
    labels = _quantile_labels(quantiles)
    status_counts = Counter(pair.status for pair in rows)
    return PairedRatioAggregate(
        numerator_method=numerator_method,
        denominator_method=denominator_method,
        query_field=query_field,
        pairs=len(rows),
        finite_pairs=len(finite),
        status_counts=MappingProxyType(dict(sorted(status_counts.items()))),
        ratios=None if not finite else _numeric_aggregate(finite, labels),
    )
