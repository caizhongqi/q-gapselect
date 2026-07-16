"""Frozen empirical Layer-C quantum-reference benchmarks.

The runner in this module compares executable *analytic measurement-law*
references on exactly the same frozen empirical reward tensor.  A fresh blind
coherent-oracle emulator is constructed for every method and repetition.  The
fixture's empirical means are consulted by the trusted runner only for promise
validation and post-run scoring; algorithms receive only their documented
public inputs.

This is deliberately a Layer-C logical-query benchmark.  It executes no
quantum circuit or hardware, assumes no state-preparation implementation, and
does not establish an asymptotic quantum advantage.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import statistics
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from itertools import islice
from types import MappingProxyType

from .attack_oracles import FrozenSourceFixture
from .attack_statistics import wilson_score_interval
from .frozen_coherent_oracle import build_frozen_empirical_coherent_oracle
from .gapselect import QGapSelect
from .independent_iae_topk import run_independent_iae_topk
from .iterative_ae_baseline import IterativeAEThresholdScan
from .models import GapSelectConfig, IAEConfig, TerminationStatus

CLAIM_SCOPE = "frozen_empirical_layer_c_quantum_reference_benchmark_no_hardware_or_advantage_claim"
DEFAULT_METHOD_IDS = (
    "qgapselect",
    "independent_iae_topk",
    "known_threshold_iae_scan",
)
QGAPSELECT_INFORMATION_REGIME = "k_only"
INDEPENDENT_IAE_INFORMATION_REGIME = "k_and_public_gap_floor"
KNOWN_THRESHOLD_IAE_INFORMATION_REGIME = "k_public_gap_floor_and_public_threshold"

CLAIM_BOUNDARIES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "supports": (
            "post-run certified exact-recovery measurement on frozen empirical tensors",
            "fresh per-method logical coherent-query ledgers under one measurement law",
            "finite-instance comparison of executable analytic reference procedures",
        ),
        "does_not_support": (
            "quantum hardware, circuit-depth, gate-count, or wall-clock claims",
            "state-preparation, QRAM, QROM, or reversible reward-oracle construction costs",
            "an asymptotic quantum speedup or a proved QGapSelect complexity theorem",
            "LLM execution, attack effectiveness, transferability, or security claims",
        ),
        "fairness": (
            "qgapselect receives an oracle and k only",
            "independent_iae_topk is a gap-aided conservative reference, not an "
            "information-matched QGapSelect baseline",
            "known_threshold_iae_scan is a still-stronger-information control",
            "heuristic rankings and QGapSelect empirical completions are never certificates",
            "timeout denotes configured schedule exhaustion, not elapsed wall-clock time",
        ),
    }
)


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


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


def _probability(value: object, name: str, *, open_interval: bool) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    valid = 0.0 < result < 1.0 if open_interval else 0.0 <= result <= 1.0
    interval = "(0, 1)" if open_interval else "[0, 1]"
    if not math.isfinite(result) or not valid:
        raise ValueError(f"{name} must be finite and lie in {interval}")
    return result


def _positive_angle(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(result) or not 0.0 < result < math.pi / 2.0:
        raise ValueError(f"{name} must lie in (0, pi/2)")
    return result


def _immutable_counts(values: Mapping[str, int]) -> Mapping[str, int]:
    counts = MappingProxyType({str(key): int(value) for key, value in values.items()})
    if any(value < 0 for value in counts.values()):
        raise ValueError("query counts cannot be negative")
    required = {"coherent_total", "classical_total", "total"}
    if not required.issubset(counts):
        raise ValueError("query ledger is missing aggregate totals")
    return counts


def _canonical_hash(document: Mapping[str, object]) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_hex(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 hexadecimal digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a 64-character SHA-256 hexadecimal digest") from error
    return value.lower()


def _coefficient_of_variation(values: tuple[float, ...]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def _derived_instance_metadata(
    fixture: FrozenSourceFixture,
    *,
    public_threshold: float,
    public_gap_floor: float,
    k: int,
) -> tuple[str, dict[str, object]]:
    """Derive permutation-invariant fallback metadata from a frozen tensor.

    Campaign instances normally supply the richer generator fingerprint.  This
    fallback keeps the public instance type backwards compatible for hand-made
    fixtures while still preventing candidate labels or arm order from
    changing the reported difficulty identity.
    """

    threshold_angle = math.asin(math.sqrt(public_threshold))
    means = fixture.evaluator.frozen_means
    gaps = tuple(abs(math.asin(math.sqrt(mean)) - threshold_angle) for mean in means)
    selected_records: list[tuple[int, int]] = []
    rejected_records: list[tuple[int, int]] = []
    for mean, stream in zip(means, fixture.tensor.reward_streams, strict=True):
        record = (sum(stream), len(stream))
        if mean > public_threshold:
            selected_records.append(record)
        else:
            rejected_records.append(record)
    fingerprint = _canonical_hash(
        {
            "schema": "qgapselect.frozen-difficulty-fingerprint.fallback.v1",
            "n_arms": len(means),
            "k": k,
            "public_threshold": public_threshold,
            "public_gap_floor": public_gap_floor,
            "selected_success_tables": sorted(selected_records),
            "rejected_success_tables": sorted(rejected_records),
        }
    )
    metrics: dict[str, object] = {
        "n_arms": len(means),
        "empirical_boundary_gap": min(gaps),
        "empirical_maximum_gap": max(gaps),
        "empirical_mean_gap": statistics.fmean(gaps),
        "empirical_gap_cv": _coefficient_of_variation(gaps),
        "distinct_empirical_gap_count": len({round(gap, 15) for gap in gaps}),
    }
    return fingerprint, metrics


def _immutable_structure_metrics(
    values: Mapping[str, object],
) -> Mapping[str, object]:
    normalized: dict[str, object] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise TypeError("structure_metrics keys must be non-empty strings")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("structure_metrics floating values must be finite")
        if not isinstance(value, (bool, int, float, str)):
            raise TypeError("structure_metrics values must be JSON scalar values")
        normalized[key] = value
    return MappingProxyType(dict(sorted(normalized.items())))


def _derived_seed(master_seed: int, *parts: object) -> int:
    material = "\0".join(
        ("qgapselect-frozen-quantum-reference-v1", str(master_seed), *map(str, parts))
    )
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _iae_config_document(config: IAEConfig) -> dict[str, object]:
    return {
        "target_angular_precision": config.target_angular_precision,
        "confidence": config.confidence,
        "shots_per_round": config.shots_per_round,
        "max_rounds": config.max_rounds,
        "max_grover_power": config.max_grover_power,
        "grid_points": config.grid_points,
    }


def _gapselect_config_document(config: GapSelectConfig) -> dict[str, object]:
    return {
        "confidence": config.confidence,
        "initial_angular_epsilon": config.initial_angular_epsilon,
        "epsilon_decay": config.epsilon_decay,
        "max_rounds": config.max_rounds,
        "shots_per_iae_round": config.shots_per_iae_round,
        "iae_max_rounds": config.iae_max_rounds,
        "iae_max_grover_power": config.iae_max_grover_power,
        "iae_grid_points": config.iae_grid_points,
    }


@dataclass(frozen=True, slots=True)
class FrozenQuantumReferenceInstance:
    """One frozen empirical tensor and its preregistered public promises."""

    family_id: str
    instance_id: str
    fixture: FrozenSourceFixture
    public_threshold: float
    public_gap_floor: float
    k: int
    difficulty_fingerprint: str | None = None
    structure_metrics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        family_id = _nonempty_string(self.family_id, "family_id")
        instance_id = _nonempty_string(self.instance_id, "instance_id")
        if not isinstance(self.fixture, FrozenSourceFixture):
            raise TypeError("fixture must be a FrozenSourceFixture")
        threshold = _probability(
            self.public_threshold,
            "public_threshold",
            open_interval=False,
        )
        gap_floor = _positive_angle(self.public_gap_floor, "public_gap_floor")
        k = _integer(self.k, "k", minimum=1)
        if k > len(self.fixture.tensor.graph.candidate_ids):
            raise ValueError("k cannot exceed the frozen candidate count")
        if not isinstance(self.structure_metrics, Mapping):
            raise TypeError("structure_metrics must be a mapping")

        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "instance_id", instance_id)
        object.__setattr__(self, "public_threshold", threshold)
        object.__setattr__(self, "public_gap_floor", gap_floor)
        object.__setattr__(self, "k", k)
        self._validate_frozen_promises()
        fallback_fingerprint, fallback_metrics = _derived_instance_metadata(
            self.fixture,
            public_threshold=threshold,
            public_gap_floor=gap_floor,
            k=k,
        )
        fingerprint = (
            fallback_fingerprint
            if self.difficulty_fingerprint is None
            else _sha256_hex(self.difficulty_fingerprint, "difficulty_fingerprint")
        )
        merged_metrics = {**fallback_metrics, **dict(self.structure_metrics)}
        object.__setattr__(self, "difficulty_fingerprint", fingerprint)
        object.__setattr__(
            self,
            "structure_metrics",
            _immutable_structure_metrics(merged_metrics),
        )

    def _validate_frozen_promises(self) -> None:
        """Validate promises in the trusted harness, never in an algorithm."""

        candidate_ids = self.fixture.tensor.graph.candidate_ids
        evaluator = self.fixture.evaluator
        if evaluator.candidate_ids != candidate_ids:
            raise ValueError("fixture evaluator and tensor candidate orders differ")
        frozen_means = evaluator.frozen_means
        ranking = sorted(
            range(len(candidate_ids)),
            key=lambda arm: (-frozen_means[arm], arm),
        )
        top_k = set(ranking[: self.k])
        threshold_side = {
            arm for arm, mean in enumerate(frozen_means) if mean > self.public_threshold
        }
        if threshold_side != top_k or len(threshold_side) != self.k:
            raise ValueError(
                "public_threshold must strictly separate exactly the frozen empirical Top-k"
            )

        threshold_angle = math.asin(math.sqrt(self.public_threshold))
        actual_floor = min(
            abs(math.asin(math.sqrt(mean)) - threshold_angle) for mean in frozen_means
        )
        if actual_floor + 1e-12 < self.public_gap_floor:
            raise ValueError(
                "public_gap_floor exceeds an arm's frozen empirical angular distance "
                "from public_threshold"
            )

    def public_document(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "instance_id": self.instance_id,
            "fixture_manifest_hash": self.fixture.manifest_hash,
            "public_threshold": self.public_threshold,
            "public_gap_floor": self.public_gap_floor,
            "k": self.k,
            "difficulty_fingerprint": self.difficulty_fingerprint,
            "structure_metrics": dict(self.structure_metrics),
        }


@dataclass(frozen=True, slots=True)
class FrozenQuantumMethodConfigs:
    """Preregistered method settings shared across all benchmark records."""

    qgapselect: GapSelectConfig = field(default_factory=GapSelectConfig)
    independent_iae: IAEConfig = field(default_factory=IAEConfig)
    known_threshold_iae: IAEConfig = field(default_factory=IAEConfig)
    failure_probability: float = 0.05
    precision_fraction: float = 0.25
    method_ids: tuple[str, ...] = DEFAULT_METHOD_IDS

    def __post_init__(self) -> None:
        if not isinstance(self.qgapselect, GapSelectConfig):
            raise TypeError("qgapselect must be a GapSelectConfig")
        if not isinstance(self.independent_iae, IAEConfig):
            raise TypeError("independent_iae must be an IAEConfig")
        if not isinstance(self.known_threshold_iae, IAEConfig):
            raise TypeError("known_threshold_iae must be an IAEConfig")
        failure_probability = _probability(
            self.failure_probability,
            "failure_probability",
            open_interval=True,
        )
        precision_fraction = _probability(
            self.precision_fraction,
            "precision_fraction",
            open_interval=True,
        )
        method_ids = tuple(self.method_ids)
        if not method_ids:
            raise ValueError("method_ids cannot be empty")
        if any(not isinstance(method_id, str) for method_id in method_ids):
            raise TypeError("method_ids must contain strings")
        if len(set(method_ids)) != len(method_ids):
            raise ValueError("method_ids must be unique")
        unknown = set(method_ids) - set(DEFAULT_METHOD_IDS)
        if unknown:
            raise ValueError(f"unknown method IDs: {sorted(unknown)}")
        if not math.isclose(
            self.qgapselect.confidence,
            failure_probability,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "qgapselect.confidence must equal failure_probability for a fair comparison"
            )
        object.__setattr__(self, "failure_probability", failure_probability)
        object.__setattr__(self, "precision_fraction", precision_fraction)
        object.__setattr__(self, "method_ids", method_ids)

    def as_dict(self) -> dict[str, object]:
        return {
            "qgapselect": _gapselect_config_document(self.qgapselect),
            "independent_iae": _iae_config_document(self.independent_iae),
            "known_threshold_iae": _iae_config_document(self.known_threshold_iae),
            "failure_probability": self.failure_probability,
            "precision_fraction": self.precision_fraction,
            "method_ids": list(self.method_ids),
        }


@dataclass(frozen=True, slots=True)
class FrozenQuantumReferenceRun:
    """One method execution, post-run truth score, and exact query ledger."""

    family_id: str
    instance_id: str
    repetition: int
    method_id: str
    measurement_seed: int
    fixture_manifest_hash: str
    k: int
    public_threshold: float
    public_gap_floor: float
    target_angular_precision: float | None
    information_matched_to_qgapselect: bool
    information_regime: str
    algorithm_inputs: tuple[str, ...]
    selected: tuple[int, ...]
    selected_candidate_ids: tuple[str, ...]
    reference_top_k: tuple[int, ...]
    reference_top_k_candidate_ids: tuple[str, ...]
    certified: bool
    exact_recovery: bool
    certified_exact_recovery: bool
    heuristic_only: bool
    timeout: bool
    status: str
    failure_reason: str | None
    coherent_query_ledger: Mapping[str, int]
    coherent_queries: int
    classical_queries: int
    total_queries: int
    backend: str
    method_claim_scope: str
    hardware_claimable: bool = False
    claim_scope: str = CLAIM_SCOPE

    def as_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "instance_id": self.instance_id,
            "repetition": self.repetition,
            "method_id": self.method_id,
            "measurement_seed": self.measurement_seed,
            "fixture_manifest_hash": self.fixture_manifest_hash,
            "k": self.k,
            "public_threshold": self.public_threshold,
            "public_gap_floor": self.public_gap_floor,
            "target_angular_precision": self.target_angular_precision,
            "information_matched_to_qgapselect": self.information_matched_to_qgapselect,
            "information_regime": self.information_regime,
            "algorithm_inputs": list(self.algorithm_inputs),
            "selected": list(self.selected),
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "reference_top_k": list(self.reference_top_k),
            "reference_top_k_candidate_ids": list(self.reference_top_k_candidate_ids),
            "certified": self.certified,
            "exact_recovery": self.exact_recovery,
            "certified_exact_recovery": self.certified_exact_recovery,
            "heuristic_only": self.heuristic_only,
            "timeout": self.timeout,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "coherent_query_ledger": dict(self.coherent_query_ledger),
            "coherent_queries": self.coherent_queries,
            "classical_queries": self.classical_queries,
            "total_queries": self.total_queries,
            "backend": self.backend,
            "method_claim_scope": self.method_claim_scope,
            "hardware_claimable": self.hardware_claimable,
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class FrozenQuantumReferenceAggregate:
    """Repetition/instance aggregate for one family and method."""

    family_id: str
    method_id: str
    information_matched_to_qgapselect: bool
    information_regime: str
    run_count: int
    certified_count: int
    exact_recovery_count: int
    certified_exact_recovery_count: int
    timeout_count: int
    certified_rate: float
    exact_recovery_rate: float
    certified_exact_recovery_rate: float
    certified_exact_wilson_lower: float
    certified_exact_wilson_upper: float
    timeout_rate: float
    mean_coherent_queries: float
    median_coherent_queries: float
    std_coherent_queries: float
    mean_total_queries: float
    claim_scope: str = CLAIM_SCOPE

    def as_dict(self) -> dict[str, object]:
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class FrozenQuantumReferenceBenchmarkReport:
    """Serializable benchmark matrix with explicit evidentiary boundaries."""

    manifest_hash: str
    master_seed: int
    repetitions: int
    method_ids: tuple[str, ...]
    instances: tuple[Mapping[str, object], ...]
    runs: tuple[FrozenQuantumReferenceRun, ...]
    aggregates: tuple[FrozenQuantumReferenceAggregate, ...]
    claim_boundaries: Mapping[str, tuple[str, ...]]
    claim_scope: str = CLAIM_SCOPE

    def as_dict(self) -> dict[str, object]:
        return {
            "manifest_hash": self.manifest_hash,
            "master_seed": self.master_seed,
            "repetitions": self.repetitions,
            "method_ids": list(self.method_ids),
            "instances": [
                {
                    **dict(instance),
                    "structure_metrics": dict(instance["structure_metrics"]),
                }
                for instance in self.instances
            ],
            "runs": [run.as_dict() for run in self.runs],
            "aggregates": [aggregate.as_dict() for aggregate in self.aggregates],
            "claim_boundaries": {
                key: list(values) for key, values in self.claim_boundaries.items()
            },
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class _MethodOutcome:
    selected: tuple[int, ...]
    certified: bool
    timeout: bool
    status: str
    failure_reason: str | None
    query_counts: Mapping[str, int]
    backend: str
    method_claim_scope: str


def _reference_top_k(instance: FrozenQuantumReferenceInstance) -> tuple[int, ...]:
    """Trusted-runner frozen truth; configured means are intentionally ignored."""

    frozen_means = instance.fixture.evaluator.frozen_means
    ranking = sorted(
        range(len(frozen_means)),
        key=lambda arm: (-frozen_means[arm], arm),
    )
    return tuple(sorted(ranking[: instance.k]))


def _run_method(
    *,
    instance: FrozenQuantumReferenceInstance,
    method_id: str,
    method_configs: FrozenQuantumMethodConfigs,
    measurement_seed: int,
) -> tuple[_MethodOutcome, float | None, bool, str, tuple[str, ...]]:
    oracle = build_frozen_empirical_coherent_oracle(
        instance.fixture,
        measurement_seed=measurement_seed,
    )
    if oracle.query_snapshot().total != 0:
        raise RuntimeError("a benchmark method did not receive a fresh oracle ledger")

    precision = instance.public_gap_floor * method_configs.precision_fraction
    if method_id == "qgapselect":
        result = QGapSelect(method_configs.qgapselect).run(oracle, instance.k)
        certified = (
            result.interval_resolved
            and not result.unresolved_at_stop
            and len(result.accepted_by_intervals) == instance.k
            and set(result.selected) == set(result.accepted_by_intervals)
        )
        outcome = _MethodOutcome(
            selected=tuple(result.selected),
            certified=certified,
            timeout=result.status is TerminationStatus.MAX_ROUNDS,
            status=result.status.value,
            failure_reason=(
                None
                if certified
                else "empirical completion or unresolved intervals are not a certificate"
            ),
            query_counts=result.executed_query_counts,
            backend=result.backend,
            method_claim_scope=result.paper_claim_status,
        )
        target_precision: float | None = None
        information_matched_to_qgapselect = True
        information_regime = QGAPSELECT_INFORMATION_REGIME
        algorithm_inputs = ("k",)
    elif method_id == "independent_iae_topk":
        result = run_independent_iae_topk(
            oracle,
            instance.k,
            config=method_configs.independent_iae,
            confidence=method_configs.failure_probability,
            target_angular_precision=precision,
        )
        outcome = _MethodOutcome(
            selected=tuple(result.selected),
            certified=result.certified,
            timeout=not result.certified,
            status=result.status,
            failure_reason=result.unresolved_reason,
            query_counts=result.query_counts,
            backend=result.backend,
            method_claim_scope=result.claim_scope,
        )
        target_precision = precision
        information_matched_to_qgapselect = False
        information_regime = INDEPENDENT_IAE_INFORMATION_REGIME
        algorithm_inputs = ("k", "public_gap_floor")
    elif method_id == "known_threshold_iae_scan":
        result = IterativeAEThresholdScan(
            oracle,
            threshold=instance.public_threshold,
            expected_count=instance.k,
            relation="above",
            confidence=method_configs.failure_probability,
            target_angular_precision=precision,
            config=method_configs.known_threshold_iae,
            seed=measurement_seed,
        ).run()
        certified = result.complete and result.verified and len(result.outputs) == instance.k
        outcome = _MethodOutcome(
            selected=tuple(result.outputs),
            certified=certified,
            timeout=not result.complete,
            status=result.status,
            failure_reason=result.failure_reason,
            query_counts=result.resources.query_counts,
            backend=result.backend,
            method_claim_scope=result.claim_status,
        )
        target_precision = precision
        information_matched_to_qgapselect = False
        information_regime = KNOWN_THRESHOLD_IAE_INFORMATION_REGIME
        algorithm_inputs = ("k", "public_gap_floor", "public_threshold")
    else:  # pragma: no cover - validated by FrozenQuantumMethodConfigs
        raise ValueError(f"unknown method_id {method_id!r}")

    ledger = _immutable_counts(outcome.query_counts)
    if dict(ledger) != oracle.query_snapshot().flat():
        raise RuntimeError("method result and fresh-oracle query ledgers differ")
    outcome = _MethodOutcome(
        selected=outcome.selected,
        certified=outcome.certified,
        timeout=outcome.timeout,
        status=outcome.status,
        failure_reason=outcome.failure_reason,
        query_counts=ledger,
        backend=outcome.backend,
        method_claim_scope=outcome.method_claim_scope,
    )
    return (
        outcome,
        target_precision,
        information_matched_to_qgapselect,
        information_regime,
        algorithm_inputs,
    )


def _build_run(
    *,
    instance: FrozenQuantumReferenceInstance,
    repetition: int,
    method_id: str,
    method_configs: FrozenQuantumMethodConfigs,
    measurement_seed: int,
) -> FrozenQuantumReferenceRun:
    (
        outcome,
        precision,
        information_matched_to_qgapselect,
        information_regime,
        inputs,
    ) = _run_method(
        instance=instance,
        method_id=method_id,
        method_configs=method_configs,
        measurement_seed=measurement_seed,
    )
    candidate_ids = instance.fixture.tensor.graph.candidate_ids
    if any(not 0 <= arm < len(candidate_ids) for arm in outcome.selected):
        raise RuntimeError("method selected an arm outside the frozen candidate set")
    if len(set(outcome.selected)) != len(outcome.selected):
        raise RuntimeError("method selected duplicate arms")

    truth = _reference_top_k(instance)
    exact_recovery = len(outcome.selected) == instance.k and set(outcome.selected) == set(truth)
    certified_exact_recovery = outcome.certified and exact_recovery
    ledger = outcome.query_counts
    return FrozenQuantumReferenceRun(
        family_id=instance.family_id,
        instance_id=instance.instance_id,
        repetition=repetition,
        method_id=method_id,
        measurement_seed=measurement_seed,
        fixture_manifest_hash=instance.fixture.manifest_hash,
        k=instance.k,
        public_threshold=instance.public_threshold,
        public_gap_floor=instance.public_gap_floor,
        target_angular_precision=precision,
        information_matched_to_qgapselect=information_matched_to_qgapselect,
        information_regime=information_regime,
        algorithm_inputs=inputs,
        selected=outcome.selected,
        selected_candidate_ids=tuple(candidate_ids[arm] for arm in outcome.selected),
        reference_top_k=truth,
        reference_top_k_candidate_ids=tuple(candidate_ids[arm] for arm in truth),
        certified=outcome.certified,
        exact_recovery=exact_recovery,
        certified_exact_recovery=certified_exact_recovery,
        heuristic_only=not outcome.certified,
        timeout=outcome.timeout,
        status=outcome.status,
        failure_reason=outcome.failure_reason,
        coherent_query_ledger=ledger,
        coherent_queries=int(ledger["coherent_total"]),
        classical_queries=int(ledger["classical_total"]),
        total_queries=int(ledger["total"]),
        backend=outcome.backend,
        method_claim_scope=outcome.method_claim_scope,
    )


def _aggregate_runs(
    runs: tuple[FrozenQuantumReferenceRun, ...],
) -> tuple[FrozenQuantumReferenceAggregate, ...]:
    grouped: dict[tuple[str, str], list[FrozenQuantumReferenceRun]] = defaultdict(list)
    for run in runs:
        grouped[(run.family_id, run.method_id)].append(run)

    aggregates: list[FrozenQuantumReferenceAggregate] = []
    for (family_id, method_id), group in grouped.items():
        coherent_queries = [run.coherent_queries for run in group]
        certified_count = sum(run.certified for run in group)
        exact_count = sum(run.exact_recovery for run in group)
        certified_exact_count = sum(run.certified_exact_recovery for run in group)
        timeout_count = sum(run.timeout for run in group)
        run_count = len(group)
        regimes = {(run.information_matched_to_qgapselect, run.information_regime) for run in group}
        if len(regimes) != 1:
            raise RuntimeError("a method changed information regimes within one family")
        information_matched_to_qgapselect, information_regime = regimes.pop()
        certified_exact_interval = wilson_score_interval(
            certified_exact_count,
            run_count,
        )
        aggregates.append(
            FrozenQuantumReferenceAggregate(
                family_id=family_id,
                method_id=method_id,
                information_matched_to_qgapselect=information_matched_to_qgapselect,
                information_regime=information_regime,
                run_count=run_count,
                certified_count=certified_count,
                exact_recovery_count=exact_count,
                certified_exact_recovery_count=certified_exact_count,
                timeout_count=timeout_count,
                certified_rate=certified_count / run_count,
                exact_recovery_rate=exact_count / run_count,
                certified_exact_recovery_rate=certified_exact_interval.estimate,
                certified_exact_wilson_lower=certified_exact_interval.lower,
                certified_exact_wilson_upper=certified_exact_interval.upper,
                timeout_rate=timeout_count / run_count,
                mean_coherent_queries=statistics.fmean(coherent_queries),
                median_coherent_queries=float(statistics.median(coherent_queries)),
                std_coherent_queries=statistics.pstdev(coherent_queries),
                mean_total_queries=statistics.fmean(run.total_queries for run in group),
            )
        )
    return tuple(aggregates)


def _instance_chunks(
    instances: Iterable[FrozenQuantumReferenceInstance],
    chunk_size: int,
) -> Iterator[tuple[FrozenQuantumReferenceInstance, ...]]:
    """Pull at most ``chunk_size`` fixture-bearing records from ``instances``.

    Keeping this helper inside the benchmark layer prevents a caller-provided
    generator from being materialized accidentally.  Only lightweight run
    records and public manifest rows survive after a chunk is processed.
    """

    iterator = iter(instances)
    while True:
        chunk = tuple(islice(iterator, chunk_size))
        if not chunk:
            return
        yield chunk
        # Release the prior tensors before asking the producer for another
        # chunk; assigning a new tuple directly would briefly retain both.
        del chunk


def run_frozen_quantum_reference_benchmark(
    instances: Iterable[FrozenQuantumReferenceInstance],
    method_configs: FrozenQuantumMethodConfigs,
    *,
    repetitions: int = 1,
    master_seed: int = 0,
    instance_chunk_size: int = 1,
) -> FrozenQuantumReferenceBenchmarkReport:
    """Execute the complete frozen empirical reference-method matrix.

    ``instances`` is consumed once in bounded chunks.  The default keeps only
    one complete fixture-bearing record in the runner at a time; increasing
    ``instance_chunk_size`` trades a bounded amount of memory for producer
    batching without changing seeds, ordering, manifests, or result schema.
    """

    if not isinstance(method_configs, FrozenQuantumMethodConfigs):
        raise TypeError("method_configs must be a FrozenQuantumMethodConfigs")
    repetition_count = _integer(repetitions, "repetitions", minimum=1)
    seed = _integer(master_seed, "master_seed", minimum=0)
    chunk_size = _integer(instance_chunk_size, "instance_chunk_size", minimum=1)

    runs: list[FrozenQuantumReferenceRun] = []
    instance_documents: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str]] = set()
    for chunk in _instance_chunks(instances, chunk_size):
        for instance in chunk:
            if not isinstance(instance, FrozenQuantumReferenceInstance):
                raise TypeError("instances must contain FrozenQuantumReferenceInstance objects")
            key = (instance.family_id, instance.instance_id)
            if key in seen_keys:
                raise ValueError("(family_id, instance_id) pairs must be unique")
            seen_keys.add(key)
            instance_documents.append(instance.public_document())
            for repetition in range(repetition_count):
                for method_id in method_configs.method_ids:
                    measurement_seed = _derived_seed(
                        seed,
                        instance.family_id,
                        instance.instance_id,
                        repetition,
                        method_id,
                    )
                    runs.append(
                        _build_run(
                            instance=instance,
                            repetition=repetition,
                            method_id=method_id,
                            method_configs=method_configs,
                            measurement_seed=measurement_seed,
                        )
                    )
            del instance
        # The tuple and its complete frozen tensors become unreachable before
        # the producer is asked for the next bounded chunk.
        del chunk

    if not instance_documents:
        raise ValueError("instances cannot be empty")

    manifest_document: dict[str, object] = {
        "schema": "qgapselect.frozen-quantum-reference-benchmark.v1",
        "claim_scope": CLAIM_SCOPE,
        "master_seed": seed,
        "repetitions": repetition_count,
        "instances": instance_documents,
        "method_configs": method_configs.as_dict(),
        "claim_boundaries": {key: list(values) for key, values in CLAIM_BOUNDARIES.items()},
    }
    run_tuple = tuple(runs)
    immutable_instance_documents = tuple(
        MappingProxyType(
            {
                **document,
                "structure_metrics": MappingProxyType(dict(document["structure_metrics"])),
            }
        )
        for document in instance_documents
    )
    return FrozenQuantumReferenceBenchmarkReport(
        manifest_hash=_canonical_hash(manifest_document),
        master_seed=seed,
        repetitions=repetition_count,
        method_ids=method_configs.method_ids,
        instances=immutable_instance_documents,
        runs=run_tuple,
        aggregates=_aggregate_runs(run_tuple),
        claim_boundaries=CLAIM_BOUNDARIES,
    )


__all__ = [
    "CLAIM_BOUNDARIES",
    "CLAIM_SCOPE",
    "DEFAULT_METHOD_IDS",
    "INDEPENDENT_IAE_INFORMATION_REGIME",
    "KNOWN_THRESHOLD_IAE_INFORMATION_REGIME",
    "QGAPSELECT_INFORMATION_REGIME",
    "FrozenQuantumMethodConfigs",
    "FrozenQuantumReferenceAggregate",
    "FrozenQuantumReferenceBenchmarkReport",
    "FrozenQuantumReferenceInstance",
    "FrozenQuantumReferenceRun",
    "run_frozen_quantum_reference_benchmark",
]
