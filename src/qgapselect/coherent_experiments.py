"""Reproducible experiments for the executable exact-state coherent backend.

This module keeps three evidence classes mechanically separate:

* exact-state executions of the coherent controller;
* executable independent-per-arm and classical baselines; and
* conjectural candidate-complexity expressions.

An exact state-vector execution is still a classical simulation.  Resource
fields describe the circuit IR that was actually built and applied; they are
not hardware measurements and they do not activate an asymptotic theorem.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from .complexity import candidate_layer_profile
from .models import TopKInstance
from .oracles import CanonicalBernoulliOracleSimulator

SCHEMA_VERSION = 1
COHERENT_BACKEND = "canonical_ry_exact_statevector"
COHERENT_CLAIM_STATUS = (
    "certificate_compiled_exact_state_not_quantum_discovery_not_hardware_or_advantage"
)
INDEPENDENT_BACKEND = "exact_state_independent_per_arm_boundary"
CLASSICAL_BACKEND = "basis_state_bernoulli_sampling"

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "experiment_name",
    "master_seed",
    "trials",
    "quantiles",
    "controller",
    "baselines",
    "scenarios",
    "notes",
}
_SCENARIO_FIELDS = {"name", "means", "k", "metadata"}
_BASELINE_FIELDS = {"independent_per_arm", "classical_uniform"}
_CLASSICAL_FIELDS = {"samples_per_arm", "confidence"}
_INDEPENDENT_FIELDS = {
    "confidence",
    "shots_per_round",
    "max_rounds",
}
_CONTROLLER_FIELDS = {
    "confidence",
    "shots_per_round",
    "max_boundary_rounds",
    "batch_strategy",
    "max_steps",
}


@dataclass(frozen=True, slots=True)
class CoherentScenario:
    """One strict Top-k mean vector used by every compared method."""

    name: str
    means: tuple[float, ...]
    k: int
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("scenario names cannot be empty")
        instance = TopKInstance(self.means, self.k)
        if self.k >= len(self.means):
            raise ValueError("coherent scenarios require 1 <= k < n")
        if not instance.identifiable:
            raise ValueError(
                f"scenario {self.name!r} requires a strict Top-k boundary"
            )

    @property
    def truth(self) -> tuple[int, ...]:
        return TopKInstance(self.means, self.k).top_k

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "means": list(self.means),
            "k": self.k,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ClassicalUniformConfig:
    """Fixed-budget classical baseline with an explicit Hoeffding certificate."""

    samples_per_arm: int
    confidence: float

    def __post_init__(self) -> None:
        if self.samples_per_arm <= 0:
            raise ValueError("classical samples_per_arm must be positive")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("classical confidence must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class IndependentPerArmConfig:
    """Configuration for the executed exact-state boundary baseline."""

    confidence: float
    shots_per_round: int
    max_rounds: int

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("independent confidence must lie in (0, 1)")
        if self.shots_per_round <= 0:
            raise ValueError("independent shots_per_round must be positive")
        if self.max_rounds <= 0:
            raise ValueError("independent max_rounds must be positive")


@dataclass(frozen=True, slots=True)
class CoherentControllerConfig:
    """Configuration passed to the dovetailed exact-state controller."""

    confidence: float
    shots_per_round: int
    max_boundary_rounds: int
    batch_strategy: str
    max_steps: int | None

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("controller confidence must lie in (0, 1)")
        if self.shots_per_round <= 0:
            raise ValueError("controller shots_per_round must be positive")
        if self.max_boundary_rounds <= 0:
            raise ValueError("controller max_boundary_rounds must be positive")
        if self.batch_strategy not in {"known", "bbht"}:
            raise ValueError("controller batch_strategy must be 'known' or 'bbht'")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("controller max_steps must be positive or null")


@dataclass(frozen=True, slots=True)
class CoherentExperimentConfig:
    """Fully resolved configuration hashed into every raw record."""

    experiment_name: str
    master_seed: int
    trials: int
    quantiles: tuple[float, ...]
    controller: CoherentControllerConfig
    independent_per_arm: IndependentPerArmConfig
    classical_uniform: ClassicalUniformConfig
    scenarios: tuple[CoherentScenario, ...]
    notes: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported coherent schema_version {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        if not self.experiment_name or not self.experiment_name.strip():
            raise ValueError("experiment_name cannot be empty")
        if self.trials <= 0:
            raise ValueError("trials must be positive")
        if not self.quantiles:
            raise ValueError("at least one quantile is required")
        if tuple(sorted(set(self.quantiles))) != self.quantiles or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.quantiles
        ):
            raise ValueError("quantiles must be finite, sorted, unique, and in [0, 1]")
        if not self.scenarios:
            raise ValueError("at least one scenario is required")
        names = tuple(scenario.name for scenario in self.scenarios)
        if len(names) != len(set(names)):
            raise ValueError("scenario names must be unique")
    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "master_seed": self.master_seed,
            "trials": self.trials,
            "quantiles": list(self.quantiles),
            "controller": asdict(self.controller),
            "baselines": {
                "independent_per_arm": asdict(self.independent_per_arm),
                "classical_uniform": asdict(self.classical_uniform),
            },
            "scenarios": [scenario.as_dict() for scenario in self.scenarios],
            "notes": list(self.notes),
        }

    @property
    def sha256(self) -> str:
        return resolved_config_hash(self.as_dict())


def resolved_config_hash(document: Mapping[str, object]) -> str:
    """Return the SHA-256 of canonical UTF-8 JSON."""

    encoded = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _reject_unknown(
    document: Mapping[str, object], allowed: set[str], context: str
) -> None:
    unknown = set(document) - allowed
    if unknown:
        raise ValueError(f"unknown {context} fields: " + ", ".join(sorted(unknown)))


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def resolve_coherent_config(
    document: Mapping[str, object],
    *,
    trials_override: int | None = None,
    seed_override: int | None = None,
    scenario_names: Sequence[str] | None = None,
) -> CoherentExperimentConfig:
    """Validate a coherent experiment document and apply explicit overrides."""

    _reject_unknown(document, _TOP_LEVEL_FIELDS, "top-level")
    controller_document = _object(document.get("controller"), "controller")
    _reject_unknown(controller_document, _CONTROLLER_FIELDS, "controller")
    controller = CoherentControllerConfig(
        confidence=_number(
            controller_document.get("confidence", 0.05),
            "controller.confidence",
        ),
        shots_per_round=_integer(
            controller_document.get("shots_per_round", 32),
            "controller.shots_per_round",
        ),
        max_boundary_rounds=_integer(
            controller_document.get("max_boundary_rounds", 8),
            "controller.max_boundary_rounds",
        ),
        batch_strategy=_string(
            controller_document.get("batch_strategy", "known"),
            "controller.batch_strategy",
        ),
        max_steps=(
            None
            if controller_document.get("max_steps") is None
            else _integer(controller_document["max_steps"], "controller.max_steps")
        ),
    )

    baseline_document = _object(document.get("baselines"), "baselines")
    _reject_unknown(baseline_document, _BASELINE_FIELDS, "baselines")
    independent_document = _object(
        baseline_document.get("independent_per_arm"),
        "baselines.independent_per_arm",
    )
    classical_document = _object(
        baseline_document.get("classical_uniform"),
        "baselines.classical_uniform",
    )
    _reject_unknown(
        independent_document,
        _INDEPENDENT_FIELDS,
        "baselines.independent_per_arm",
    )
    _reject_unknown(
        classical_document,
        _CLASSICAL_FIELDS,
        "baselines.classical_uniform",
    )
    independent = IndependentPerArmConfig(
        confidence=_number(
            independent_document.get("confidence", 0.05),
            "baselines.independent_per_arm.confidence",
        ),
        shots_per_round=_integer(
            independent_document.get("shots_per_round", 32),
            "baselines.independent_per_arm.shots_per_round",
        ),
        max_rounds=_integer(
            independent_document.get("max_rounds", 8),
            "baselines.independent_per_arm.max_rounds",
        ),
    )
    classical = ClassicalUniformConfig(
        samples_per_arm=_integer(
            classical_document.get("samples_per_arm"),
            "baselines.classical_uniform.samples_per_arm",
        ),
        confidence=_number(
            classical_document.get("confidence", 0.05),
            "baselines.classical_uniform.confidence",
        ),
    )

    raw_scenarios = document.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("scenarios must be a non-empty JSON array")
    scenarios: list[CoherentScenario] = []
    for index, item in enumerate(raw_scenarios):
        scenario_document = _object(item, f"scenarios[{index}]")
        _reject_unknown(scenario_document, _SCENARIO_FIELDS, f"scenarios[{index}]")
        means = scenario_document.get("means")
        if not isinstance(means, list):
            raise ValueError(f"scenarios[{index}].means must be a JSON array")
        metadata = scenario_document.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"scenarios[{index}].metadata must be a JSON object")
        if "name" not in scenario_document:
            raise ValueError(f"scenarios[{index}].name is required")
        scenarios.append(
            CoherentScenario(
                name=_string(scenario_document["name"], f"scenarios[{index}].name"),
                means=tuple(
                    _number(value, f"scenarios[{index}].means[{mean_index}]")
                    for mean_index, value in enumerate(means)
                ),
                k=_integer(scenario_document.get("k"), f"scenarios[{index}].k"),
                metadata=dict(metadata),
            )
        )

    if scenario_names is not None:
        requested = tuple(dict.fromkeys(str(name) for name in scenario_names))
        available = {scenario.name for scenario in scenarios}
        missing = set(requested) - available
        if missing:
            raise ValueError("unknown scenarios: " + ", ".join(sorted(missing)))
        requested_set = set(requested)
        scenarios = [scenario for scenario in scenarios if scenario.name in requested_set]

    trials = _integer(document.get("trials", 1), "trials")
    if trials_override is not None:
        trials = _integer(trials_override, "trials override")
    master_seed = _integer(document.get("master_seed", 20260714), "master_seed")
    if seed_override is not None:
        master_seed = _integer(seed_override, "seed override")
    quantiles = document.get("quantiles", [0.5, 0.9, 0.95])
    if not isinstance(quantiles, list):
        raise ValueError("quantiles must be a JSON array")
    notes = document.get("notes", [])
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        raise ValueError("notes must be an array of strings")

    return CoherentExperimentConfig(
        schema_version=_integer(
            document.get("schema_version", SCHEMA_VERSION), "schema_version"
        ),
        experiment_name=_string(
            document.get("experiment_name", "coherent-exact-state"),
            "experiment_name",
        ),
        master_seed=master_seed,
        trials=trials,
        quantiles=tuple(
            _number(value, f"quantiles[{index}]")
            for index, value in enumerate(quantiles)
        ),
        controller=controller,
        independent_per_arm=independent,
        classical_uniform=classical,
        scenarios=tuple(scenarios),
        notes=tuple(notes),
    )


def load_coherent_config(
    path: str | Path,
    *,
    trials_override: int | None = None,
    seed_override: int | None = None,
    scenario_names: Sequence[str] | None = None,
) -> CoherentExperimentConfig:
    """Load and resolve one coherent experiment JSON file."""

    config_path = Path(path)
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON config {config_path}: {error}") from error
    return resolve_coherent_config(
        _object(document, "coherent config"),
        trials_override=trials_override,
        seed_override=seed_override,
        scenario_names=scenario_names,
    )


def derive_trial_seed(
    master_seed: int, scenario_name: str, trial_index: int, method: str
) -> int:
    """Derive stable method-separated seeds without Python's salted hash."""

    if trial_index < 0:
        raise ValueError("trial_index cannot be negative")
    material = (
        f"{int(master_seed)}\0{scenario_name}\0{trial_index}\0{method}".encode()
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _candidate_theory(scenario: CoherentScenario) -> dict[str, object]:
    profile = candidate_layer_profile(scenario.means, scenario.k)
    return {
        "data_source": "analytic_expression_from_ground_truth",
        "unit": "conjectural_normalized_layer_charge",
        "proof_status": profile.theorem_status,
        "chosen_orientation": profile.representation,
        "value": profile.value,
        "alternative_value": profile.alternative_value,
    }


def _failure_reason(success: bool, certificate_valid: bool) -> str | None:
    if not certificate_valid:
        return "certificate_not_obtained"
    if not success:
        return "incorrect_certified_output"
    return None


def _jsonable(value: object) -> object:
    """Convert backend dataclasses, enums, and tuples to strict JSON values."""

    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)) or enum_value is None:
        if enum_value is not None:
            return enum_value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _certificate_valid(certificate: object, *, complete: bool) -> bool:
    """Read a backend certificate's explicit validity flag conservatively."""

    if not complete or certificate is None:
        return False
    if isinstance(certificate, Mapping):
        for key in ("valid", "verified", "complete"):
            if key in certificate:
                return bool(certificate[key])
    for attribute in ("valid", "verified", "complete"):
        if hasattr(certificate, attribute):
            return bool(getattr(certificate, attribute))
    if isinstance(certificate, Sequence) and not isinstance(certificate, (str, bytes)):
        return bool(certificate) and any(
            _certificate_valid(item, complete=True) for item in certificate
        )
    # A complete result with an opaque non-null certificate is accepted only as
    # a controller assertion; its serialized details remain in the raw record.
    return True


def _primitive_resources(resources: object) -> dict[str, object]:
    """Normalize the backend's executed PrimitiveResources without imputation."""

    raw_counts = resources.query_counts
    counts_value = _jsonable(raw_counts)
    if not isinstance(counts_value, Mapping):
        raise TypeError("PrimitiveResources.query_counts must be a mapping")
    counts = {str(key): int(value) for key, value in counts_value.items()}
    if "coherent_total" in counts:
        oracle_queries = counts["coherent_total"]
    elif "total" in counts:
        oracle_queries = counts["total"]
    else:
        oracle_queries = sum(
            counts.get(key, 0)
            for key in (
                "forward",
                "inverse",
                "controlled_forward",
                "controlled_inverse",
            )
        )
    return {
        "oracle_queries": oracle_queries,
        "classical_queries": 0,
        "query_counts": counts,
        "phase_oracle_queries": int(resources.phase_oracle_queries),
        "gates": int(resources.gates),
        "depth": int(resources.depth),
        "total_qubits": int(resources.qubits),
        "workspace_qubits": int(resources.workspace_qubits),
        "uncompute_residual": float(resources.uncompute_residual),
        "resource_model": (
            "logical_operations_executed_by_numpy_exact_state_simulator_"
            "not_hardware_synthesis_cost"
        ),
        "backend": str(resources.backend),
    }


def _classical_record(
    config: CoherentExperimentConfig,
    scenario: CoherentScenario,
    trial_index: int,
) -> dict[str, object]:
    method = "classical_uniform"
    seed = derive_trial_seed(config.master_seed, scenario.name, trial_index, method)
    oracle = CanonicalBernoulliOracleSimulator(scenario.means, seed=seed)
    shots = config.classical_uniform.samples_per_arm
    estimates = tuple(
        oracle.sample(arm, shots, tag="coherent_experiment_classical") / shots
        for arm in range(len(scenario.means))
    )
    selected = tuple(
        sorted(
            sorted(range(len(estimates)), key=lambda arm: (-estimates[arm], arm))[
                : scenario.k
            ]
        )
    )
    selected_set = frozenset(selected)
    radius = math.sqrt(
        math.log(2.0 * len(estimates) / config.classical_uniform.confidence)
        / (2.0 * shots)
    )
    selected_lower = min(max(0.0, estimates[arm] - radius) for arm in selected)
    rejected_upper = max(
        min(1.0, estimates[arm] + radius)
        for arm in range(len(estimates))
        if arm not in selected_set
    )
    certificate_valid = selected_lower > rejected_upper
    success = selected == scenario.truth
    snapshot = oracle.query_snapshot()
    return _base_record(
        config=config,
        scenario=scenario,
        trial_index=trial_index,
        method=method,
        seed=seed,
        backend=CLASSICAL_BACKEND,
        selected=selected,
        success=success,
        certificate_valid=certificate_valid,
        certificate={
            "kind": "simultaneous_hoeffding_separation",
            "valid": certificate_valid,
            "radius": radius,
            "selected_minimum_lower": selected_lower,
            "rejected_maximum_upper": rejected_upper,
        },
        resources={
            "oracle_queries": 0,
            "classical_queries": snapshot.classical_total,
            "query_counts": snapshot.flat(),
            "phase_oracle_queries": 0,
            "gates": 0,
            "depth": 0,
            "total_qubits": 0,
            "workspace_qubits": 0,
            "uncompute_residual": 0.0,
            "resource_model": "executed_basis_state_samples",
        },
        failure_reason=_failure_reason(success, certificate_valid),
    )


def _independent_record(
    config: CoherentExperimentConfig,
    scenario: CoherentScenario,
    trial_index: int,
) -> dict[str, object]:
    from .coherent import CanonicalRyStatevectorOracle
    from .primitives import qboundary

    method = "independent_per_arm_boundary"
    seed = derive_trial_seed(config.master_seed, scenario.name, trial_index, method)
    oracle = CanonicalRyStatevectorOracle(scenario.means, seed=seed)
    result = qboundary(
        oracle,
        scenario.k,
        confidence=config.independent_per_arm.confidence,
        shots_per_round=config.independent_per_arm.shots_per_round,
        max_rounds=config.independent_per_arm.max_rounds,
        seed=seed,
    )
    selected = tuple(sorted(result.candidate_selected))
    success = selected == scenario.truth
    certificate_valid = _certificate_valid(
        result.certificate,
        complete=bool(result.complete),
    )
    return _base_record(
        config=config,
        scenario=scenario,
        trial_index=trial_index,
        method=method,
        seed=seed,
        backend=INDEPENDENT_BACKEND,
        selected=selected,
        success=success,
        certificate_valid=certificate_valid,
        certificate={
            "kind": "independent_per_arm_exact_state_boundary",
            "valid": certificate_valid,
            "details": _jsonable(result.certificate),
            "candidate_rejected": list(result.candidate_rejected),
            "rounds": len(result.rounds),
        },
        resources=_primitive_resources(result.resources),
        failure_reason=_failure_reason(success, certificate_valid),
    )


def _base_record(
    *,
    config: CoherentExperimentConfig,
    scenario: CoherentScenario,
    trial_index: int,
    method: str,
    seed: int,
    backend: str,
    selected: Sequence[int],
    success: bool,
    certificate_valid: bool,
    certificate: Mapping[str, object],
    resources: Mapping[str, object],
    failure_reason: str | None,
) -> dict[str, object]:
    record_claim_status = (
        "executed_classical_baseline_not_quantum_evidence"
        if backend == CLASSICAL_BACKEND
        else COHERENT_CLAIM_STATUS
    )
    return {
        "scenario": scenario.name,
        "trial_index": trial_index,
        "trial_seed": seed,
        "method": method,
        "backend": backend,
        "n": len(scenario.means),
        "k": scenario.k,
        "truth": list(scenario.truth),
        "selected": list(selected),
        "heuristic_output_match": bool(success),
        "certified_success": bool(success and certificate_valid),
        # ``success`` is retained as the primary endpoint and is deliberately
        # certificate-gated.  Consumers must use ``heuristic_output_match`` for
        # the weaker diagnostic that includes uncertified output guesses.
        "success": bool(success and certificate_valid),
        "certificate": dict(certificate),
        "certificate_valid": bool(certificate_valid),
        "failure": {
            "occurred": failure_reason is not None,
            "reason": failure_reason,
        },
        "executed_resources": dict(resources),
        "claim_status": record_claim_status,
        "config_hash": config.sha256,
    }


def _coherent_record(
    config: CoherentExperimentConfig,
    scenario: CoherentScenario,
    trial_index: int,
) -> dict[str, object]:
    """Execute the exact-state controller.

    The import is local so configuration validation and baseline-only tooling do
    not disguise a missing coherent implementation.  The concrete adapter is
    kept narrow and is tested against the backend's public result dataclasses.
    """

    from .coherent import CanonicalRyStatevectorOracle
    from .primitives import DovetailTopKController

    # Boundary sampling has already produced the complete classical certificate.
    # The coherent stage tests reversible flag/enumeration semantics; it is not a
    # coherent discovery of the unknown Top-k set.
    method = "coherent_certificate_enumeration"
    seed = derive_trial_seed(config.master_seed, scenario.name, trial_index, method)
    oracle = CanonicalRyStatevectorOracle(scenario.means, seed=seed)
    controller = DovetailTopKController(
        oracle,
        scenario.k,
        confidence=config.controller.confidence,
        shots_per_round=config.controller.shots_per_round,
        max_boundary_rounds=config.controller.max_boundary_rounds,
        batch_strategy=config.controller.batch_strategy,
        seed=seed,
    )
    result = controller.run(max_steps=config.controller.max_steps)
    selected = tuple(sorted(result.selected))
    success = selected == scenario.truth
    certificate_valid = _certificate_valid(
        result.certificates,
        complete=bool(result.complete),
    )
    return _base_record(
        config=config,
        scenario=scenario,
        trial_index=trial_index,
        method=method,
        seed=seed,
        backend=COHERENT_BACKEND,
        selected=selected,
        success=success,
        certificate_valid=certificate_valid,
        certificate={
            "kind": "dovetailed_selected_or_rejected_exact_state_certificate",
            "valid": certificate_valid,
            "orientation": _jsonable(result.winning_orientation),
            "details": _jsonable(result.certificates),
            "steps": int(result.steps),
            "status": _jsonable(result.status),
        },
        resources=_primitive_resources(result.resources),
        failure_reason=(
            _failure_reason(success, certificate_valid)
            if bool(result.complete)
            else f"controller_{_jsonable(result.status)}"
        ),
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(
    values: Sequence[float], quantiles: tuple[float, ...]
) -> dict[str, object] | None:
    if not values:
        return None
    return {
        "observations": len(values),
        "mean": fmean(values),
        "min": min(values),
        "max": max(values),
        "quantiles": {
            format(probability, ".12g"): _quantile(values, probability)
            for probability in quantiles
        },
    }


_RESOURCE_FIELDS = (
    "oracle_queries",
    "phase_oracle_queries",
    "classical_queries",
    "gates",
    "depth",
    "total_qubits",
    "workspace_qubits",
    "uncompute_residual",
)


def aggregate_execution_records(
    records: Sequence[Mapping[str, object]], quantiles: tuple[float, ...]
) -> dict[str, object]:
    """Aggregate only raw executed-resource fields, never theory proxies."""

    if not records:
        raise ValueError("cannot aggregate an empty record collection")
    certified_success_count = sum(
        bool(record["certified_success"]) for record in records
    )
    heuristic_match_count = sum(
        bool(record["heuristic_output_match"]) for record in records
    )
    certificate_count = sum(bool(record["certificate_valid"]) for record in records)
    failures = Counter(
        str(_object(record["failure"], "failure")["reason"])
        for record in records
        if bool(_object(record["failure"], "failure")["occurred"])
    )
    resource_summaries: dict[str, object] = {}
    for field in _RESOURCE_FIELDS:
        values = []
        for record in records:
            value = _object(record["executed_resources"], "executed_resources").get(
                field
            )
            if value is not None:
                values.append(float(value))
        resource_summaries[field] = _summary(values, quantiles)
    count = len(records)
    return {
        "executions": count,
        "success_count": certified_success_count,
        "success_rate": certified_success_count / count,
        "certified_success_count": certified_success_count,
        "certified_success_rate": certified_success_count / count,
        "heuristic_output_match_count": heuristic_match_count,
        "heuristic_output_match_rate": heuristic_match_count / count,
        "certificate_valid_count": certificate_count,
        "certificate_valid_rate": certificate_count / count,
        "failure_reason_counts": dict(sorted(failures.items())),
        "executed_resources": resource_summaries,
        "aggregation_rule": "raw executed fields only; null resources are not imputed",
    }


def run_coherent_experiments(
    config: CoherentExperimentConfig,
) -> dict[str, object]:
    """Run coherent, independent-per-arm, and classical methods for every seed."""

    raw_records: list[dict[str, object]] = []
    for scenario in config.scenarios:
        for trial_index in range(config.trials):
            raw_records.extend(
                (
                    _coherent_record(config, scenario, trial_index),
                    _independent_record(config, scenario, trial_index),
                    _classical_record(config, scenario, trial_index),
                )
            )

    methods = tuple(sorted({str(record["method"]) for record in raw_records}))
    by_method = {
        method: aggregate_execution_records(
            [record for record in raw_records if record["method"] == method],
            config.quantiles,
        )
        for method in methods
    }
    by_scenario_method = {
        scenario.name: {
            method: aggregate_execution_records(
                [
                    record
                    for record in raw_records
                    if record["scenario"] == scenario.name
                    and record["method"] == method
                ],
                config.quantiles,
            )
            for method in methods
        }
        for scenario in config.scenarios
    }
    return {
        "artifact_type": "coherent_exact_state_execution",
        "schema_version": SCHEMA_VERSION,
        "claim_status": COHERENT_CLAIM_STATUS,
        "config_hash": config.sha256,
        "provenance": {
            "generator": "scripts/run_coherent.py",
            "backend": COHERENT_BACKEND,
            "master_seed": config.master_seed,
            "seed_derivation": "sha256(master_seed, scenario, trial, method)",
            "raw_records_are_aggregate_source": True,
            "resource_aggregation": (
                "executed numeric fields only; missing fields remain null"
            ),
        },
        "resolved_config": config.as_dict(),
        "raw_execution_records": raw_records,
        "aggregate_executions": {
            "by_method": by_method,
            "by_scenario_method": by_scenario_method,
        },
        "candidate_theory_reference": {
            scenario.name: _candidate_theory(scenario)
            for scenario in config.scenarios
        },
        "evidence_boundary": (
            "exact-state circuit simulation is executable evidence, not hardware "
            "execution, a complexity theorem, or quantum advantage"
        ),
    }


def write_coherent_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write deterministic, human-readable JSON without NaN values."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output
