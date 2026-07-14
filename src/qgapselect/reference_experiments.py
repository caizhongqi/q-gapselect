"""Reproducible multi-seed experiments for the executable analytic reference.

Unlike :mod:`qgapselect.experiments`, which evaluates complexity expressions,
this module actually runs :class:`~qgapselect.gapselect.QGapSelect` against the
analytic Bernoulli-oracle simulator.  The simulator samples the measurement
law of each requested Grover experiment and records logical ``A``/``A†``
calls.  It does not execute a coherent index-register batch circuit.

Executed-query accounting and conjectural layer charges are deliberately kept
in different JSON objects throughout raw and aggregate records.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any

from .gapselect import QGapSelect
from .models import GapSelectConfig, TerminationStatus, TopKInstance
from .oracles import CanonicalBernoulliOracleSimulator

REFERENCE_BACKEND = "all_active_analytic_iae_reference"
REFERENCE_CLAIM_STATUS = "simulation_regression_not_coherent_batch_execution"
SCHEMA_VERSION = 1

_ALLOWED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "experiment_name",
    "master_seed",
    "trials",
    "quantiles",
    "algorithm",
    "scenarios",
    "notes",
}
_ALLOWED_ALGORITHM_FIELDS = {
    "confidence",
    "initial_angular_epsilon",
    "epsilon_decay",
    "max_rounds",
    "shots_per_iae_round",
    "iae_max_rounds",
    "iae_max_grover_power",
    "iae_grid_points",
}
_ALLOWED_SCENARIO_FIELDS = {"name", "means", "k", "metadata"}


@dataclass(frozen=True, slots=True)
class ReferenceScenario:
    """One strict Top-k instance used in repeated simulator trials."""

    name: str
    means: tuple[float, ...]
    k: int
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("scenario names cannot be empty")
        instance = TopKInstance(self.means, self.k)
        if not instance.identifiable:
            raise ValueError(
                f"scenario {self.name!r} needs a strict Top-k boundary for exact recovery"
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
class ReferenceExperimentConfig:
    """Fully resolved executable reference configuration."""

    experiment_name: str
    master_seed: int
    trials: int
    quantiles: tuple[float, ...]
    algorithm: GapSelectConfig
    scenarios: tuple[ReferenceScenario, ...]
    notes: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported reference schema_version {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        if not self.experiment_name or not self.experiment_name.strip():
            raise ValueError("experiment_name cannot be empty")
        if self.trials <= 0:
            raise ValueError("trials must be positive")
        if not self.quantiles:
            raise ValueError("at least one query quantile is required")
        if any(
            not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0
            for quantile in self.quantiles
        ):
            raise ValueError("quantiles must be finite and lie in [0, 1]")
        if tuple(sorted(set(self.quantiles))) != self.quantiles:
            raise ValueError("quantiles must be sorted and unique")
        if not self.scenarios:
            raise ValueError("at least one scenario is required")
        names = tuple(scenario.name for scenario in self.scenarios)
        if len(set(names)) != len(names):
            raise ValueError("scenario names must be unique")

    def as_dict(self) -> dict[str, object]:
        """Return the canonical, hashable resolved experiment document."""

        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "master_seed": self.master_seed,
            "trials": self.trials,
            "quantiles": list(self.quantiles),
            "algorithm": asdict(self.algorithm),
            "scenarios": [scenario.as_dict() for scenario in self.scenarios],
            "notes": list(self.notes),
        }

    @property
    def sha256(self) -> str:
        return resolved_config_hash(self.as_dict())


def resolved_config_hash(document: Mapping[str, object]) -> str:
    """Hash a resolved config using canonical UTF-8 JSON."""

    encoded = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _reject_unknown(
    document: Mapping[str, object], allowed: set[str], context: str
) -> None:
    unknown = set(document) - allowed
    if unknown:
        raise ValueError(
            f"unknown {context} fields: " + ", ".join(sorted(unknown))
        )


def resolve_reference_config(
    document: Mapping[str, object],
    *,
    trials_override: int | None = None,
    seed_override: int | None = None,
    scenario_names: Sequence[str] | None = None,
) -> ReferenceExperimentConfig:
    """Validate JSON data and apply explicit command-line overrides."""

    _reject_unknown(document, _ALLOWED_TOP_LEVEL_FIELDS, "top-level")
    algorithm_document = _require_object(document.get("algorithm"), "algorithm")
    _reject_unknown(algorithm_document, _ALLOWED_ALGORITHM_FIELDS, "algorithm")
    if "initial_epsilon" in algorithm_document:
        raise ValueError(
            "initial_epsilon is a mean-space legacy field; use "
            "initial_angular_epsilon"
        )
    if "initial_angular_epsilon" not in algorithm_document:
        raise ValueError("algorithm.initial_angular_epsilon is required")
    algorithm = GapSelectConfig(**dict(algorithm_document))

    raw_scenarios = document.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("scenarios must be a non-empty JSON array")
    scenarios: list[ReferenceScenario] = []
    for index, raw_scenario in enumerate(raw_scenarios):
        scenario_document = _require_object(
            raw_scenario, f"scenarios[{index}]"
        )
        _reject_unknown(
            scenario_document,
            _ALLOWED_SCENARIO_FIELDS,
            f"scenarios[{index}]",
        )
        if "name" not in scenario_document:
            raise ValueError(f"scenarios[{index}].name is required")
        means = scenario_document.get("means")
        if not isinstance(means, list):
            raise ValueError(f"scenarios[{index}].means must be a JSON array")
        metadata = scenario_document.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"scenarios[{index}].metadata must be a JSON object")
        scenarios.append(
            ReferenceScenario(
                name=str(scenario_document["name"]),
                means=tuple(float(mean) for mean in means),
                k=int(scenario_document.get("k", 0)),
                metadata=dict(metadata),
            )
        )

    if scenario_names is not None:
        requested = tuple(dict.fromkeys(str(name) for name in scenario_names))
        available = {scenario.name for scenario in scenarios}
        missing = set(requested) - available
        if missing:
            raise ValueError(
                "unknown requested scenarios: " + ", ".join(sorted(missing))
            )
        scenarios = [
            scenario for scenario in scenarios if scenario.name in set(requested)
        ]

    trials = int(document.get("trials", 500))
    if trials_override is not None:
        trials = int(trials_override)
    master_seed = int(document.get("master_seed", 20260714))
    if seed_override is not None:
        master_seed = int(seed_override)

    raw_quantiles = document.get("quantiles", [0.5, 0.9, 0.95])
    if not isinstance(raw_quantiles, list):
        raise ValueError("quantiles must be a JSON array")
    raw_notes = document.get("notes", [])
    if not isinstance(raw_notes, list) or any(
        not isinstance(note, str) for note in raw_notes
    ):
        raise ValueError("notes must be an array of strings")

    return ReferenceExperimentConfig(
        schema_version=int(document.get("schema_version", SCHEMA_VERSION)),
        experiment_name=str(
            document.get("experiment_name", "q-gapselect-analytic-reference")
        ),
        master_seed=master_seed,
        trials=trials,
        quantiles=tuple(float(quantile) for quantile in raw_quantiles),
        algorithm=algorithm,
        scenarios=tuple(scenarios),
        notes=tuple(raw_notes),
    )


def load_reference_config(
    path: str | Path,
    *,
    trials_override: int | None = None,
    seed_override: int | None = None,
    scenario_names: Sequence[str] | None = None,
) -> ReferenceExperimentConfig:
    """Load and resolve one reference JSON config."""

    config_path = Path(path)
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON config {config_path}: {error}") from error
    return resolve_reference_config(
        _require_object(document, "reference config"),
        trials_override=trials_override,
        seed_override=seed_override,
        scenario_names=scenario_names,
    )


def derive_trial_seed(master_seed: int, scenario_name: str, trial_index: int) -> int:
    """Derive a stable per-scenario seed without Python's salted ``hash``."""

    if trial_index < 0:
        raise ValueError("trial_index cannot be negative")
    material = f"{int(master_seed)}\0{scenario_name}\0{trial_index}".encode()
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _run_trial(
    config: ReferenceExperimentConfig,
    scenario: ReferenceScenario,
    trial_index: int,
) -> dict[str, object]:
    trial_seed = derive_trial_seed(
        config.master_seed,
        scenario.name,
        trial_index,
    )
    oracle = CanonicalBernoulliOracleSimulator(
        scenario.means,
        seed=trial_seed,
    )
    result = QGapSelect(config.algorithm).run(oracle, scenario.k)
    if result.backend != REFERENCE_BACKEND:
        raise RuntimeError(
            f"unexpected Q-GapSelect backend {result.backend!r}; "
            f"expected {REFERENCE_BACKEND!r}"
        )
    executed_counts = {
        key: int(value) for key, value in result.executed_query_counts.items()
    }
    theory = result.candidate_theory_accounting
    return {
        "scenario": scenario.name,
        "trial_index": trial_index,
        "trial_seed": trial_seed,
        "n": len(scenario.means),
        "k": scenario.k,
        "truth": list(scenario.truth),
        "selected": list(result.selected),
        "heuristic_inclusive_exact_recovery": result.selected == scenario.truth,
        "certified_exact_recovery": (
            result.interval_resolved and result.selected == scenario.truth
        ),
        "interval_resolved": result.interval_resolved,
        "termination_status": result.status.value,
        "timeout": result.status is TerminationStatus.MAX_ROUNDS,
        "executed_query_accounting": {
            "unit": "logical_oracle_calls_in_analytic_measurement_simulator",
            "coherent_queries": executed_counts["coherent_total"],
            "classical_queries": executed_counts["classical_total"],
            "all_counts": executed_counts,
        },
        "candidate_theory_accounting": {
            "unit": "conjectural_normalized_layer_charge",
            "proof_status": theory.proof_status,
            "comparison_status": theory.comparison_status,
            "orientations": {
                orientation: {
                    "complete": theory.orientation_completion[orientation],
                    "partial_charge": theory.orientation_partial_charges[
                        orientation
                    ],
                }
                for orientation in ("selected", "rejected_complement")
            },
            "chosen_orientation": theory.chosen_representation,
            "chosen_charge": theory.total_candidate_charge,
            "alternative_orientation": theory.alternative_representation,
            "alternative_charge": theory.alternative_candidate_charge,
            "expression": theory.expression,
        },
        "rounds": len(result.rounds),
        "warnings": list(result.warnings),
        "backend": REFERENCE_BACKEND,
        "claim_status": REFERENCE_CLAIM_STATUS,
        "config_hash": config.sha256,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    """Linearly interpolated empirical quantile (NumPy's default convention)."""

    if not values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _numeric_summary(
    values: Sequence[float], quantiles: tuple[float, ...]
) -> dict[str, object]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "mean": fmean(values),
        "quantiles": {
            format(probability, ".12g"): _quantile(values, probability)
            for probability in quantiles
        },
    }


def aggregate_raw_records(
    raw_records: Sequence[Mapping[str, object]],
    quantiles: tuple[float, ...],
) -> dict[str, object]:
    """Recompute all aggregate metrics from raw trial records only."""

    if not raw_records:
        raise ValueError("cannot aggregate an empty raw-record collection")
    heuristic_exact_count = sum(
        bool(record["heuristic_inclusive_exact_recovery"])
        for record in raw_records
    )
    certified_exact_count = sum(
        bool(record["certified_exact_recovery"]) for record in raw_records
    )
    resolved_count = sum(bool(record["interval_resolved"]) for record in raw_records)
    timeout_count = sum(bool(record["timeout"]) for record in raw_records)
    coherent_queries = [
        float(
            _require_object(
                record["executed_query_accounting"],
                "executed_query_accounting",
            )["coherent_queries"]
        )
        for record in raw_records
    ]
    theory_records = [
        _require_object(
            record["candidate_theory_accounting"],
            "candidate_theory_accounting",
        )
        for record in raw_records
    ]
    comparable_theory_records = [
        record for record in theory_records if record["chosen_charge"] is not None
    ]
    chosen_charges = [
        float(record["chosen_charge"]) for record in comparable_theory_records
    ]
    orientation_counts = Counter(
        str(record["chosen_orientation"]) for record in comparable_theory_records
    )
    count = len(raw_records)
    return {
        "trials": count,
        "heuristic_inclusive_exact_recovery_count": heuristic_exact_count,
        "heuristic_inclusive_exact_recovery_rate": heuristic_exact_count / count,
        "certified_exact_recovery_count": certified_exact_count,
        "certified_exact_recovery_rate": certified_exact_count / count,
        "interval_resolved_count": resolved_count,
        "interval_resolved_rate": resolved_count / count,
        "timeout_count": timeout_count,
        "chosen_orientation_counts": {
            "selected": orientation_counts.get("selected", 0),
            "rejected_complement": orientation_counts.get(
                "rejected_complement", 0
            ),
        },
        "executed_query_accounting": {
            "coherent_queries": _numeric_summary(coherent_queries, quantiles),
            "unit": "logical_oracle_calls_in_analytic_measurement_simulator",
        },
        "candidate_theory_accounting": {
            "comparable_complete_certificate_count": len(chosen_charges),
            "incomplete_trace_count": count - len(chosen_charges),
            "chosen_charge": (
                _numeric_summary(chosen_charges, quantiles)
                if chosen_charges
                else None
            ),
            "unit": "conjectural_normalized_layer_charge",
            "proof_status": "conjectural_not_a_query_bound",
            "aggregation_rule": (
                "chosen charges summarize complete certificate traces only"
            ),
        },
        "backend": REFERENCE_BACKEND,
        "claim_status": REFERENCE_CLAIM_STATUS,
    }


def run_reference_experiments(
    config: ReferenceExperimentConfig,
) -> dict[str, object]:
    """Run all scenario/seed pairs and return a deterministic JSON document."""

    raw_records = [
        _run_trial(config, scenario, trial_index)
        for scenario in config.scenarios
        for trial_index in range(config.trials)
    ]
    by_scenario = {
        scenario.name: aggregate_raw_records(
            [
                record
                for record in raw_records
                if record["scenario"] == scenario.name
            ],
            config.quantiles,
        )
        for scenario in config.scenarios
    }
    overall = aggregate_raw_records(raw_records, config.quantiles)
    for aggregate in (*by_scenario.values(), overall):
        aggregate["config_hash"] = config.sha256
    return {
        "artifact_type": "analytic_reference_multiseed_regression",
        "schema_version": SCHEMA_VERSION,
        "backend": REFERENCE_BACKEND,
        "claim_status": REFERENCE_CLAIM_STATUS,
        "config_hash": config.sha256,
        "resolved_config": config.as_dict(),
        "raw_records": raw_records,
        "aggregate": {
            "overall": overall,
            "by_scenario": by_scenario,
        },
    }


def write_reference_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write a report as canonical, human-readable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def with_trial_override(
    config: ReferenceExperimentConfig, trials: int
) -> ReferenceExperimentConfig:
    """Programmatic trial-count override for notebooks and diagnostics."""

    return replace(config, trials=int(trials))
