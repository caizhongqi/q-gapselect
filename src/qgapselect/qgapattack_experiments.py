"""Machine-readable experiment design and audit for Q-GapAttack.

The design deliberately separates three questions that are easy to conflate:

* whether the quantum selector is correct and resource competitive;
* whether it selects a better attack portfolio from a frozen candidate pool;
* whether the complete attack pipeline transfers to held-out tasks and models.

This module audits a preregistration manifest.  It does not execute an LLM,
instantiate a coherent LLM oracle, or turn a planned experiment into evidence.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CLAIM_STATUS = "preregistered_experiment_design_no_empirical_superiority_claim"
READINESS = "design_complete_execution_and_theorem_gates_open"

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "design_name",
    "frozen_on",
    "claim_status",
    "baseline_registry",
    "benchmark_registry",
    "metric_registry",
    "experiment_panels",
    "fairness_contract",
    "statistics_plan",
    "model_protocol",
    "budget_protocol",
    "execution_requirements",
    "notes",
}
_BASELINE_FIELDS = {
    "id",
    "name",
    "track",
    "stage",
    "family",
    "role",
    "oracle_model",
    "comparison_scope",
    "primary_eligible",
    "implementation_status",
    "citation",
    "budget_dimensions",
    "notes",
}
_BENCHMARK_FIELDS = {
    "id",
    "name",
    "track",
    "primary",
    "task_unit",
    "split_keys",
    "validator",
    "citation",
    "notes",
}
_METRIC_FIELDS = {
    "id",
    "name",
    "track",
    "primary",
    "direction",
    "unit",
    "aggregation",
    "definition",
}
_PANEL_FIELDS = {
    "id",
    "name",
    "track",
    "hypothesis",
    "status",
    "baseline_ids",
    "benchmark_ids",
    "metric_ids",
    "factors",
    "repetitions",
    "seed_policy",
    "claim_gate",
}

_REQUIRED_FAIRNESS = {
    "candidate_pool_frozen_for_selector_track",
    "source_victim_model_split",
    "held_out_task_repository_cwe_split",
    "equal_source_budget_within_panel",
    "equal_victim_budget_within_panel",
    "same_validator_and_seed_schedule",
    "timeouts_and_indeterminate_in_denominator",
    "oracle_mismatches_reported_separately",
    "no_victim_tuning",
    "no_free_qram",
    "full_resource_accounting",
}
_REQUIRED_STATISTICS = {
    "confidence_level",
    "cluster_unit",
    "bootstrap_repetitions",
    "paired_binary_test",
    "paired_continuous_test",
    "multiple_comparison_correction",
    "effect_sizes_required",
    "negative_results_required",
}
_REQUIRED_EXECUTION = {
    "dataset_manifests_frozen",
    "model_revisions_frozen",
    "validator_revisions_frozen",
    "all_primary_baselines_executable",
    "raw_records_persisted",
    "independent_reproduction_complete",
}
_REQUIRED_COVERAGE = {
    "quantum_core": {
        "proposed",
        "classical_fixed_confidence",
        "independent_quantum",
        "quantum_composition",
        "input_model_diagnostic",
        "invalid_negative_control",
    },
    "attack_application": {
        "proposed",
        "clean_control",
        "random_selector",
        "exhaustive_selector",
        "classical_adaptive_selector",
        "independent_quantum_selector",
        "code_security_attack",
        "general_llm_attack",
    },
}


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _strings(value: object, name: str, *, nonempty: bool = True) -> tuple[str, ...]:
    result = tuple(
        _string(item, f"{name}[{index}]")
        for index, item in enumerate(_sequence(value, name))
    )
    if nonempty and not result:
        raise ValueError(f"{name} cannot be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def _only_fields(
    value: Mapping[str, object], allowed: set[str], name: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    """One comparison method with its access model and eligibility."""

    baseline_id: str
    name: str
    track: str
    stage: str
    family: str
    role: str
    oracle_model: str
    comparison_scope: str
    primary_eligible: bool
    implementation_status: str
    citation: str
    budget_dimensions: tuple[str, ...]
    notes: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], index: int) -> BaselineSpec:
        _only_fields(value, _BASELINE_FIELDS, f"baseline_registry[{index}]")
        spec = cls(
            baseline_id=_string(value.get("id"), f"baseline[{index}].id"),
            name=_string(value.get("name"), f"baseline[{index}].name"),
            track=_string(value.get("track"), f"baseline[{index}].track"),
            stage=_string(value.get("stage"), f"baseline[{index}].stage"),
            family=_string(value.get("family"), f"baseline[{index}].family"),
            role=_string(value.get("role"), f"baseline[{index}].role"),
            oracle_model=_string(
                value.get("oracle_model"), f"baseline[{index}].oracle_model"
            ),
            comparison_scope=_string(
                value.get("comparison_scope"),
                f"baseline[{index}].comparison_scope",
            ),
            primary_eligible=_boolean(
                value.get("primary_eligible"),
                f"baseline[{index}].primary_eligible",
            ),
            implementation_status=_string(
                value.get("implementation_status"),
                f"baseline[{index}].implementation_status",
            ),
            citation=_string(
                value.get("citation"), f"baseline[{index}].citation"
            ),
            budget_dimensions=_strings(
                value.get("budget_dimensions"),
                f"baseline[{index}].budget_dimensions",
            ),
            notes=_string(value.get("notes"), f"baseline[{index}].notes"),
        )
        if spec.track not in _REQUIRED_COVERAGE:
            raise ValueError(f"unsupported baseline track {spec.track!r}")
        if spec.primary_eligible and (
            spec.comparison_scope == "diagnostic_only"
            or spec.role in {"diagnostic", "invalid_negative_control"}
            or spec.oracle_model.startswith("mismatched_")
        ):
            raise ValueError(
                f"baseline {spec.baseline_id!r} is not eligible for a primary win"
            )
        return spec


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    benchmark_id: str
    name: str
    track: str
    primary: bool
    task_unit: str
    split_keys: tuple[str, ...]
    validator: str
    citation: str
    notes: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], index: int) -> BenchmarkSpec:
        _only_fields(value, _BENCHMARK_FIELDS, f"benchmark_registry[{index}]")
        return cls(
            benchmark_id=_string(value.get("id"), f"benchmark[{index}].id"),
            name=_string(value.get("name"), f"benchmark[{index}].name"),
            track=_string(value.get("track"), f"benchmark[{index}].track"),
            primary=_boolean(value.get("primary"), f"benchmark[{index}].primary"),
            task_unit=_string(
                value.get("task_unit"), f"benchmark[{index}].task_unit"
            ),
            split_keys=_strings(
                value.get("split_keys"), f"benchmark[{index}].split_keys"
            ),
            validator=_string(
                value.get("validator"), f"benchmark[{index}].validator"
            ),
            citation=_string(
                value.get("citation"), f"benchmark[{index}].citation"
            ),
            notes=_string(value.get("notes"), f"benchmark[{index}].notes"),
        )


@dataclass(frozen=True, slots=True)
class MetricSpec:
    metric_id: str
    name: str
    track: str
    primary: bool
    direction: str
    unit: str
    aggregation: str
    definition: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], index: int) -> MetricSpec:
        _only_fields(value, _METRIC_FIELDS, f"metric_registry[{index}]")
        direction = _string(
            value.get("direction"), f"metric[{index}].direction"
        )
        if direction not in {"higher", "lower", "calibration"}:
            raise ValueError(f"invalid metric direction {direction!r}")
        return cls(
            metric_id=_string(value.get("id"), f"metric[{index}].id"),
            name=_string(value.get("name"), f"metric[{index}].name"),
            track=_string(value.get("track"), f"metric[{index}].track"),
            primary=_boolean(value.get("primary"), f"metric[{index}].primary"),
            direction=direction,
            unit=_string(value.get("unit"), f"metric[{index}].unit"),
            aggregation=_string(
                value.get("aggregation"), f"metric[{index}].aggregation"
            ),
            definition=_string(
                value.get("definition"), f"metric[{index}].definition"
            ),
        )


@dataclass(frozen=True, slots=True)
class PanelSpec:
    panel_id: str
    name: str
    track: str
    hypothesis: str
    status: str
    baseline_ids: tuple[str, ...]
    benchmark_ids: tuple[str, ...]
    metric_ids: tuple[str, ...]
    factors: Mapping[str, object]
    repetitions: int
    seed_policy: str
    claim_gate: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], index: int) -> PanelSpec:
        _only_fields(value, _PANEL_FIELDS, f"experiment_panels[{index}]")
        return cls(
            panel_id=_string(value.get("id"), f"panel[{index}].id"),
            name=_string(value.get("name"), f"panel[{index}].name"),
            track=_string(value.get("track"), f"panel[{index}].track"),
            hypothesis=_string(
                value.get("hypothesis"), f"panel[{index}].hypothesis"
            ),
            status=_string(value.get("status"), f"panel[{index}].status"),
            baseline_ids=_strings(
                value.get("baseline_ids"), f"panel[{index}].baseline_ids"
            ),
            benchmark_ids=_strings(
                value.get("benchmark_ids"), f"panel[{index}].benchmark_ids"
            ),
            metric_ids=_strings(
                value.get("metric_ids"), f"panel[{index}].metric_ids"
            ),
            factors=dict(_mapping(value.get("factors"), f"panel[{index}].factors")),
            repetitions=_integer(
                value.get("repetitions"), f"panel[{index}].repetitions", minimum=1
            ),
            seed_policy=_string(
                value.get("seed_policy"), f"panel[{index}].seed_policy"
            ),
            claim_gate=_string(
                value.get("claim_gate"), f"panel[{index}].claim_gate"
            ),
        )


@dataclass(frozen=True, slots=True)
class ExperimentDesign:
    design_name: str
    frozen_on: str
    baselines: tuple[BaselineSpec, ...]
    benchmarks: tuple[BenchmarkSpec, ...]
    metrics: tuple[MetricSpec, ...]
    panels: tuple[PanelSpec, ...]
    fairness_contract: Mapping[str, bool]
    statistics_plan: Mapping[str, object]
    model_protocol: Mapping[str, object]
    budget_protocol: Mapping[str, object]
    execution_requirements: Mapping[str, bool]
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExperimentDesignAudit:
    """Audit result; design validity is separate from empirical readiness."""

    design_name: str
    frozen_on: str
    baseline_count: int
    primary_eligible_baseline_count: int
    benchmark_count: int
    metric_count: int
    panel_count: int
    baseline_track_counts: Mapping[str, int]
    baseline_family_counts: Mapping[str, int]
    panel_status_counts: Mapping[str, int]
    fairness_checks: Mapping[str, bool]
    statistics_checks: Mapping[str, bool]
    execution_checks: Mapping[str, bool]
    coverage_checks: Mapping[str, bool]
    blockers: tuple[str, ...]
    design_valid: bool
    empirical_ready: bool
    ccf_a_claimable: bool
    claim_status: str = CLAIM_STATUS
    readiness: str = READINESS

    def as_dict(self) -> dict[str, object]:
        return {
            "design_name": self.design_name,
            "frozen_on": self.frozen_on,
            "baseline_count": self.baseline_count,
            "primary_eligible_baseline_count": self.primary_eligible_baseline_count,
            "benchmark_count": self.benchmark_count,
            "metric_count": self.metric_count,
            "panel_count": self.panel_count,
            "baseline_track_counts": dict(self.baseline_track_counts),
            "baseline_family_counts": dict(self.baseline_family_counts),
            "panel_status_counts": dict(self.panel_status_counts),
            "fairness_checks": dict(self.fairness_checks),
            "statistics_checks": dict(self.statistics_checks),
            "execution_checks": dict(self.execution_checks),
            "coverage_checks": dict(self.coverage_checks),
            "blockers": list(self.blockers),
            "design_valid": self.design_valid,
            "empirical_ready": self.empirical_ready,
            "ccf_a_claimable": self.ccf_a_claimable,
            "claim_status": self.claim_status,
            "readiness": self.readiness,
        }


def _unique_ids(values: Sequence[object], attribute: str, name: str) -> None:
    identifiers = [getattr(value, attribute) for value in values]
    if len(set(identifiers)) != len(identifiers):
        duplicates = sorted(
            identifier for identifier, count in Counter(identifiers).items() if count > 1
        )
        raise ValueError(f"duplicate {name} IDs: {duplicates}")


def parse_experiment_design(document: Mapping[str, object]) -> ExperimentDesign:
    """Parse and cross-check a strict experiment preregistration document."""

    _only_fields(document, _TOP_LEVEL_FIELDS, "top-level")
    if _integer(document.get("schema_version"), "schema_version", minimum=1) != 1:
        raise ValueError("schema_version must be 1")
    if _string(document.get("claim_status"), "claim_status") != CLAIM_STATUS:
        raise ValueError(f"claim_status must be {CLAIM_STATUS!r}")

    baselines = tuple(
        BaselineSpec.from_mapping(_mapping(item, f"baseline[{index}]"), index)
        for index, item in enumerate(
            _sequence(document.get("baseline_registry"), "baseline_registry")
        )
    )
    benchmarks = tuple(
        BenchmarkSpec.from_mapping(_mapping(item, f"benchmark[{index}]"), index)
        for index, item in enumerate(
            _sequence(document.get("benchmark_registry"), "benchmark_registry")
        )
    )
    metrics = tuple(
        MetricSpec.from_mapping(_mapping(item, f"metric[{index}]"), index)
        for index, item in enumerate(
            _sequence(document.get("metric_registry"), "metric_registry")
        )
    )
    panels = tuple(
        PanelSpec.from_mapping(_mapping(item, f"panel[{index}]"), index)
        for index, item in enumerate(
            _sequence(document.get("experiment_panels"), "experiment_panels")
        )
    )
    if not baselines or not benchmarks or not metrics or not panels:
        raise ValueError("baseline, benchmark, metric, and panel registries are required")
    _unique_ids(baselines, "baseline_id", "baseline")
    _unique_ids(benchmarks, "benchmark_id", "benchmark")
    _unique_ids(metrics, "metric_id", "metric")
    _unique_ids(panels, "panel_id", "panel")

    baseline_map = {item.baseline_id: item for item in baselines}
    benchmark_map = {item.benchmark_id: item for item in benchmarks}
    metric_map = {item.metric_id: item for item in metrics}
    for panel in panels:
        missing_baselines = sorted(set(panel.baseline_ids) - set(baseline_map))
        missing_benchmarks = sorted(set(panel.benchmark_ids) - set(benchmark_map))
        missing_metrics = sorted(set(panel.metric_ids) - set(metric_map))
        if missing_baselines or missing_benchmarks or missing_metrics:
            raise ValueError(
                f"panel {panel.panel_id!r} has missing references: "
                f"baselines={missing_baselines}, benchmarks={missing_benchmarks}, "
                f"metrics={missing_metrics}"
            )
        if any(baseline_map[item].track != panel.track for item in panel.baseline_ids):
            raise ValueError(f"panel {panel.panel_id!r} mixes baseline tracks")
        if any(benchmark_map[item].track != panel.track for item in panel.benchmark_ids):
            raise ValueError(f"panel {panel.panel_id!r} mixes benchmark tracks")
        if any(metric_map[item].track != panel.track for item in panel.metric_ids):
            raise ValueError(f"panel {panel.panel_id!r} mixes metric tracks")
        primary_methods = [
            baseline_map[item]
            for item in panel.baseline_ids
            if baseline_map[item].primary_eligible
        ]
        if len(primary_methods) < 2:
            raise ValueError(
                f"panel {panel.panel_id!r} needs at least two primary-eligible methods"
            )

    fairness_raw = _mapping(document.get("fairness_contract"), "fairness_contract")
    fairness = {
        key: _boolean(fairness_raw.get(key), f"fairness_contract.{key}")
        for key in sorted(_REQUIRED_FAIRNESS)
    }
    unknown_fairness = set(fairness_raw) - _REQUIRED_FAIRNESS
    if unknown_fairness:
        raise ValueError(f"unknown fairness checks: {sorted(unknown_fairness)}")

    statistics = dict(_mapping(document.get("statistics_plan"), "statistics_plan"))
    missing_statistics = _REQUIRED_STATISTICS - set(statistics)
    if missing_statistics:
        raise ValueError(f"missing statistics fields: {sorted(missing_statistics)}")
    confidence = _number(statistics["confidence_level"], "confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    _integer(statistics["bootstrap_repetitions"], "bootstrap_repetitions", minimum=1000)

    execution_raw = _mapping(
        document.get("execution_requirements"), "execution_requirements"
    )
    execution = {
        key: _boolean(execution_raw.get(key), f"execution_requirements.{key}")
        for key in sorted(_REQUIRED_EXECUTION)
    }
    unknown_execution = set(execution_raw) - _REQUIRED_EXECUTION
    if unknown_execution:
        raise ValueError(f"unknown execution checks: {sorted(unknown_execution)}")

    return ExperimentDesign(
        design_name=_string(document.get("design_name"), "design_name"),
        frozen_on=_string(document.get("frozen_on"), "frozen_on"),
        baselines=baselines,
        benchmarks=benchmarks,
        metrics=metrics,
        panels=panels,
        fairness_contract=fairness,
        statistics_plan=statistics,
        model_protocol=dict(
            _mapping(document.get("model_protocol"), "model_protocol")
        ),
        budget_protocol=dict(
            _mapping(document.get("budget_protocol"), "budget_protocol")
        ),
        execution_requirements=execution,
        notes=_strings(document.get("notes"), "notes", nonempty=False),
    )


def audit_experiment_design(design: ExperimentDesign) -> ExperimentDesignAudit:
    """Check baseline coverage and distinguish design from executed evidence."""

    family_by_track = {
        track: {item.family for item in design.baselines if item.track == track}
        for track in _REQUIRED_COVERAGE
    }
    coverage_checks = {
        f"{track}:{family}": family in family_by_track[track]
        for track, families in _REQUIRED_COVERAGE.items()
        for family in sorted(families)
    }
    fairness_checks = dict(design.fairness_contract)
    statistics_checks = {
        "confidence_level_is_95_percent": (
            float(design.statistics_plan["confidence_level"]) == 0.95
        ),
        "cluster_bootstrap_at_least_10000": (
            int(design.statistics_plan["bootstrap_repetitions"]) >= 10_000
        ),
        "paired_binary_test_declared": bool(
            design.statistics_plan["paired_binary_test"]
        ),
        "paired_continuous_test_declared": bool(
            design.statistics_plan["paired_continuous_test"]
        ),
        "multiplicity_control_declared": bool(
            design.statistics_plan["multiple_comparison_correction"]
        ),
        "effect_sizes_required": bool(
            design.statistics_plan["effect_sizes_required"]
        ),
        "negative_results_required": bool(
            design.statistics_plan["negative_results_required"]
        ),
    }
    execution_checks = dict(design.execution_requirements)
    design_valid = (
        all(coverage_checks.values())
        and all(fairness_checks.values())
        and all(statistics_checks.values())
    )
    panel_statuses = Counter(panel.status for panel in design.panels)
    all_panels_executed = bool(design.panels) and all(
        panel.status == "executed_with_frozen_artifact" for panel in design.panels
    )
    primary_statuses = {
        item.implementation_status
        for item in design.baselines
        if item.primary_eligible
    }
    all_primary_executable = primary_statuses <= {
        "implemented",
        "external_reproduction_frozen",
    }
    empirical_ready = (
        design_valid
        and all(execution_checks.values())
        and all_panels_executed
        and all_primary_executable
    )

    blockers: list[str] = []
    if not design_valid:
        blockers.append(
            "The preregistered design has missing fairness, statistics, or coverage gates."
        )
    if not all_primary_executable:
        blockers.append(
            "At least one primary baseline is not implemented or frozen for reproduction."
        )
    if not all_panels_executed:
        blockers.append(
            "Experiment panels are preregistered but not all have immutable executed artifacts."
        )
    for key, passed in execution_checks.items():
        if not passed:
            blockers.append(f"Execution requirement remains open: {key}.")
    blockers.extend(
        (
            "The Layer-P reversible LLM reward sampler and its cleanup/resource theorem are open.",
            "The Q-GapSelect upper bound, composition separation, and matching "
            "lower bound are open.",
            "Closed victim APIs can only provide classical transfer evidence, "
            "not coherent queries.",
        )
    )

    return ExperimentDesignAudit(
        design_name=design.design_name,
        frozen_on=design.frozen_on,
        baseline_count=len(design.baselines),
        primary_eligible_baseline_count=sum(
            item.primary_eligible for item in design.baselines
        ),
        benchmark_count=len(design.benchmarks),
        metric_count=len(design.metrics),
        panel_count=len(design.panels),
        baseline_track_counts=Counter(item.track for item in design.baselines),
        baseline_family_counts=Counter(item.family for item in design.baselines),
        panel_status_counts=panel_statuses,
        fairness_checks=fairness_checks,
        statistics_checks=statistics_checks,
        execution_checks=execution_checks,
        coverage_checks=coverage_checks,
        blockers=tuple(blockers),
        design_valid=design_valid,
        empirical_ready=empirical_ready,
        ccf_a_claimable=False,
    )


def load_and_audit_experiment_design(
    path: str | Path,
) -> tuple[ExperimentDesign, ExperimentDesignAudit]:
    """Load one JSON design, then return its parsed form and audit."""

    document = _mapping(
        json.loads(Path(path).read_text(encoding="utf-8")), "experiment design"
    )
    design = parse_experiment_design(document)
    return design, audit_experiment_design(design)


def experiment_design_markdown(
    design: ExperimentDesign, audit: ExperimentDesignAudit
) -> str:
    """Render a compact, reviewable view of the preregistration."""

    lines = [
        "# Q-GapAttack experiment-design audit",
        "",
        f"- Design: `{design.design_name}`",
        f"- Frozen on: `{design.frozen_on}`",
        f"- Claim status: `{audit.claim_status}`",
        f"- Design valid: `{str(audit.design_valid).lower()}`",
        f"- Empirically ready: `{str(audit.empirical_ready).lower()}`",
        f"- CCF-A claimable: `{str(audit.ccf_a_claimable).lower()}`",
        "",
        "## Baselines",
        "",
        "| ID | Track | Stage | Family | Primary | Status |",
        "|---|---|---|---|---:|---|",
    ]
    lines.extend(
        f"| {item.baseline_id} | {item.track} | {item.stage} | {item.family} | "
        f"{str(item.primary_eligible).lower()} | {item.implementation_status} |"
        for item in design.baselines
    )
    lines.extend(
        [
            "",
            "## Experiment panels",
            "",
            "| ID | Track | Baselines | Benchmarks | Repetitions | Status |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    lines.extend(
        f"| {panel.panel_id} | {panel.track} | {len(panel.baseline_ids)} | "
        f"{len(panel.benchmark_ids)} | {panel.repetitions} | {panel.status} |"
        for panel in design.panels
    )
    lines.extend(["", "## Open blockers", ""])
    lines.extend(f"- {blocker}" for blocker in audit.blockers)
    lines.append("")
    return "\n".join(lines)
