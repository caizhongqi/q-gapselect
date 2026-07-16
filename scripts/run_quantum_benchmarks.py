#!/usr/bin/env python3
"""Run the reproducible Q-GapSelect quantum-core audit suites."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import operator
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from qgapselect.coherent import CanonicalRyStatevectorOracle
from qgapselect.composition_audit import (
    composition_audit_sweep,
    composition_loglog_slope,
)
from qgapselect.direct_baselines import (
    ClassicalThresholdScan,
    IndependentQPEThresholdScan,
)
from qgapselect.direct_phase import DirectAmplitudeThresholdFlag
from qgapselect.direct_search import FullWorkspaceBBHT
from qgapselect.direct_topk import CalibratedDirectTopKController
from qgapselect.iterative_ae_baseline import IterativeAEThresholdScan
from qgapselect.models import IAEConfig
from qgapselect.oracles import CanonicalBernoulliOracleSimulator
from qgapselect.quantum_benchmarking import (
    FAMILIES,
    BenchmarkRecord,
    QuantumBenchmarkConfig,
    QuantumBenchmarkInstance,
    QuantumBenchmarkRunner,
    aggregate_benchmark_records,
    aggregate_paired_query_ratios,
    make_benchmark_instance,
    paired_query_ratios,
)
from qgapselect.quantum_diagnostics import (
    make_threshold_angular_gap_instance,
    run_diffusion_ablation,
    run_phase_grid_sweep,
    run_qpe_acceptance_sweep,
)
from qgapselect.quantum_validation import (
    run_unitary_validation,
    run_verifier_calibration,
)
from qgapselect.theory_falsification import (
    descriptive_loglog_slope,
    orientation_separation_sweep,
)
from qgapselect.unknown_boundary_history import (
    unknown_boundary_history_loglog_slope,
    unknown_boundary_history_sweep,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs" / "quantum_benchmarks.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "quantum_benchmark_diagnostic.json"
ARTIFACT_TYPE = "q_gapselect_executed_quantum_core_audit"
CLAIM_STATUS = "finite_exact_state_and_analytic_diagnostics_no_advantage_theorem"
SUITES = (
    "unitary_validation",
    "phase_grid",
    "qpe_resolution",
    "verifier_calibration",
    "random_benchmarks",
    "topk_comparison",
    "orientation_separation",
    "generic_composition_audit",
    "unknown_boundary_history",
    "iterative_ae",
    "scheduler_sweep",
    "diffusion_ablation",
    "failure_semantics",
)

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "experiment_name",
    "master_seed",
    "quantiles",
    "common",
    *SUITES,
    "notes",
}


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a JSON number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a JSON number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        if strict_minimum and result <= minimum:
            raise ValueError(f"{name} must exceed {minimum}")
        if not strict_minimum and result < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"all keys in {name} must be strings")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _only_keys(value: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")


def _numbers(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> list[float]:
    result = [
        _number(item, f"{name}[{index}]", minimum=minimum, maximum=maximum)
        for index, item in enumerate(_sequence(value, name))
    ]
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _integers(value: object, name: str, *, minimum: int = 1) -> list[int]:
    result = [
        _integer(item, f"{name}[{index}]", minimum=minimum)
        for index, item in enumerate(_sequence(value, name))
    ]
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _strings(value: object, name: str) -> list[str]:
    result = [
        _string(item, f"{name}[{index}]")
        for index, item in enumerate(_sequence(value, name))
    ]
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _cases(value: object, name: str) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    for index, item in enumerate(_sequence(value, name)):
        case = _mapping(item, f"{name}[{index}]")
        _only_keys(case, {"n", "k"}, f"{name}[{index}]")
        n = _integer(case.get("n"), f"{name}[{index}].n", minimum=2)
        k = _integer(case.get("k"), f"{name}[{index}].k", minimum=1)
        if k >= n:
            raise ValueError(f"{name}[{index}].k must be less than n")
        result.append({"n": n, "k": k})
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    raise TypeError(f"cannot serialize value of type {type(value).__name__}")


def _config_hash(config: Mapping[str, object]) -> str:
    encoded = json.dumps(
        _jsonable(config),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _derived_seed(master_seed: int, suite: str, *parts: object) -> int:
    payload = ":".join([str(master_seed), suite, *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def _git_provenance() -> dict[str, object]:
    def command(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = command("rev-parse", "HEAD")
    tree = command("rev-parse", "HEAD^{tree}")
    status = command("status", "--porcelain")
    return {
        "git_commit": commit or "unknown",
        "git_tree": tree or "unknown",
        "source_tree_dirty_at_execution": bool(status),
    }


def _runtime_provenance() -> dict[str, object]:
    """Return the execution environment without implying byte-stable wall times."""

    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine() or "unknown",
        "byte_for_byte_reproduction_expected": False,
        "volatile_fields": ["suite_results.*.simulator_wall_seconds"],
        "reproduction_scope": (
            "scientific records should match for the same source, configuration, "
            "runtime, and seeds after excluding declared volatile fields"
        ),
    }


def _resolved_config(document: object) -> dict[str, object]:
    root = _mapping(document, "configuration")
    _only_keys(root, _TOP_LEVEL_FIELDS, "top-level configuration")
    schema = _integer(root.get("schema_version"), "schema_version", minimum=1)
    if schema != 2:
        raise ValueError("schema_version must be 2")
    resolved: dict[str, object] = {
        "schema_version": schema,
        "experiment_name": _string(root.get("experiment_name"), "experiment_name"),
        "master_seed": _integer(root.get("master_seed"), "master_seed", minimum=0),
    }
    quantiles = _numbers(root.get("quantiles"), "quantiles", minimum=0.0, maximum=1.0)
    if quantiles != sorted(set(quantiles)):
        raise ValueError("quantiles must be sorted and unique")
    resolved["quantiles"] = quantiles

    common = _mapping(root.get("common"), "common")
    common_fields = {
        "threshold_angle",
        "phase_qubits",
        "max_phase_qubits",
        "verification_confidence",
        "verification_shots",
        "max_attempts_per_output",
        "max_statevector_dimension",
        "classical_shots_per_arm",
        "boundary_shots_per_round",
        "max_boundary_rounds",
    }
    _only_keys(common, common_fields, "common")
    resolved["common"] = {
        "threshold_angle": _number(
            common.get("threshold_angle"),
            "common.threshold_angle",
            minimum=0.0,
            maximum=math.pi / 2.0,
        ),
        "phase_qubits": _integer(
            common.get("phase_qubits"), "common.phase_qubits", minimum=1
        ),
        "max_phase_qubits": _integer(
            common.get("max_phase_qubits"),
            "common.max_phase_qubits",
            minimum=1,
        ),
        "verification_confidence": _number(
            common.get("verification_confidence"),
            "common.verification_confidence",
            minimum=0.0,
            maximum=1.0,
            strict_minimum=True,
        ),
        "verification_shots": _integer(
            common.get("verification_shots"),
            "common.verification_shots",
            minimum=1,
        ),
        "max_attempts_per_output": _integer(
            common.get("max_attempts_per_output"),
            "common.max_attempts_per_output",
            minimum=1,
        ),
        "max_statevector_dimension": _integer(
            common.get("max_statevector_dimension"),
            "common.max_statevector_dimension",
            minimum=1,
        ),
        "classical_shots_per_arm": _integer(
            common.get("classical_shots_per_arm"),
            "common.classical_shots_per_arm",
            minimum=1,
        ),
        "boundary_shots_per_round": _integer(
            common.get("boundary_shots_per_round"),
            "common.boundary_shots_per_round",
            minimum=1,
        ),
        "max_boundary_rounds": _integer(
            common.get("max_boundary_rounds"),
            "common.max_boundary_rounds",
            minimum=1,
        ),
    }
    resolved_common = resolved["common"]
    if not isinstance(resolved_common, Mapping):
        raise RuntimeError("resolved common configuration is invalid")
    if int(resolved_common["max_phase_qubits"]) < int(
        resolved_common["phase_qubits"]
    ):
        raise ValueError(
            "common.max_phase_qubits must be at least common.phase_qubits"
        )
    if int(resolved_common["max_phase_qubits"]) > 12:
        raise ValueError("common.max_phase_qubits cannot exceed 12")

    unitary = _mapping(root.get("unitary_validation"), "unitary_validation")
    _only_keys(unitary, {"trials", "cases"}, "unitary_validation")
    unitary_cases: list[dict[str, object]] = []
    for index, item in enumerate(_sequence(unitary.get("cases"), "unitary_validation.cases")):
        case = _mapping(item, f"unitary_validation.cases[{index}]")
        _only_keys(case, {"name", "means", "phase_qubits"}, f"unitary_validation.cases[{index}]")
        unitary_cases.append(
            {
                "name": _string(case.get("name"), f"unitary_validation.cases[{index}].name"),
                "means": _numbers(
                    case.get("means"),
                    f"unitary_validation.cases[{index}].means",
                    minimum=0.0,
                    maximum=1.0,
                ),
                "phase_qubits": _integer(
                    case.get("phase_qubits"),
                    f"unitary_validation.cases[{index}].phase_qubits",
                    minimum=1,
                ),
            }
        )
    if not unitary_cases:
        raise ValueError("unitary_validation.cases cannot be empty")
    resolved["unitary_validation"] = {
        "trials": _integer(unitary.get("trials"), "unitary_validation.trials", minimum=1),
        "cases": unitary_cases,
    }

    phase_grid = _mapping(root.get("phase_grid"), "phase_grid")
    _only_keys(
        phase_grid,
        {"phase_qubits", "threshold_angles", "relations"},
        "phase_grid",
    )
    phase_relations = _strings(phase_grid.get("relations"), "phase_grid.relations")
    if any(value not in {"above", "below"} for value in phase_relations):
        raise ValueError("phase_grid.relations must contain only above/below")
    resolved["phase_grid"] = {
        "phase_qubits": _integers(phase_grid.get("phase_qubits"), "phase_grid.phase_qubits"),
        "threshold_angles": _numbers(
            phase_grid.get("threshold_angles"),
            "phase_grid.threshold_angles",
            minimum=0.0,
            maximum=math.pi / 2.0,
        ),
        "relations": phase_relations,
    }

    qpe = _mapping(root.get("qpe_resolution"), "qpe_resolution")
    _only_keys(qpe, {"phase_qubits", "angular_gaps", "relations"}, "qpe_resolution")
    qpe_relations = _strings(qpe.get("relations"), "qpe_resolution.relations")
    if any(value not in {"above", "below"} for value in qpe_relations):
        raise ValueError("qpe_resolution.relations must contain only above/below")
    resolved["qpe_resolution"] = {
        "phase_qubits": _integers(qpe.get("phase_qubits"), "qpe_resolution.phase_qubits"),
        "angular_gaps": _numbers(
            qpe.get("angular_gaps"),
            "qpe_resolution.angular_gaps",
            minimum=0.0,
        ),
        "relations": qpe_relations,
    }

    verifier = _mapping(root.get("verifier_calibration"), "verifier_calibration")
    _only_keys(
        verifier,
        {"trials", "phase_qubits", "shots", "confidence_values", "angular_offsets"},
        "verifier_calibration",
    )
    resolved["verifier_calibration"] = {
        "trials": _integer(verifier.get("trials"), "verifier_calibration.trials", minimum=1),
        "phase_qubits": _integers(
            verifier.get("phase_qubits"), "verifier_calibration.phase_qubits"
        ),
        "shots": _integers(verifier.get("shots"), "verifier_calibration.shots"),
        "confidence_values": _numbers(
            verifier.get("confidence_values"),
            "verifier_calibration.confidence_values",
            minimum=0.0,
            maximum=1.0,
        ),
        "angular_offsets": _numbers(
            verifier.get("angular_offsets"), "verifier_calibration.angular_offsets"
        ),
    }

    for suite_name, allowed_methods, topk in (
        (
            "random_benchmarks",
            {"direct_bbht", "independent_qpe_scan", "classical_threshold_scan"},
            False,
        ),
        (
            "topk_comparison",
            {
                "boundary_only_negative_control",
                "refined_boundary_only_negative_control",
                "calibrated_direct_topk",
                "adaptive_calibrated_direct_topk",
                "fixed_max_precision_topk",
            },
            True,
        ),
    ):
        section = _mapping(root.get(suite_name), suite_name)
        fields = {"trials", "families", "cases", "methods"}
        if not topk:
            fields.add("relations")
        _only_keys(section, fields, suite_name)
        families = _strings(section.get("families"), f"{suite_name}.families")
        if any(family not in FAMILIES for family in families):
            raise ValueError(f"{suite_name}.families contains an unknown family")
        methods = _strings(section.get("methods"), f"{suite_name}.methods")
        if any(method not in allowed_methods for method in methods):
            raise ValueError(f"{suite_name}.methods contains an unsupported method")
        item: dict[str, object] = {
            "trials": _integer(section.get("trials"), f"{suite_name}.trials", minimum=1),
            "families": families,
            "cases": _cases(section.get("cases"), f"{suite_name}.cases"),
            "methods": methods,
        }
        if not topk:
            relations = _strings(section.get("relations"), f"{suite_name}.relations")
            if any(relation not in {"above", "below"} for relation in relations):
                raise ValueError(f"{suite_name}.relations must contain only above/below")
            item["relations"] = relations
        resolved[suite_name] = item

    separation = _mapping(root.get("orientation_separation"), "orientation_separation")
    _only_keys(
        separation,
        {"m_values", "gamma_exponent", "beta", "far_offset"},
        "orientation_separation",
    )
    separation_m_values = _integers(
        separation.get("m_values"),
        "orientation_separation.m_values",
        minimum=2,
    )
    if separation_m_values != sorted(set(separation_m_values)):
        raise ValueError("orientation_separation.m_values must be sorted and unique")
    separation_beta = _number(
        separation.get("beta"),
        "orientation_separation.beta",
        minimum=0.0,
        maximum=math.pi / 2.0,
        strict_minimum=True,
    )
    if separation_beta >= math.pi / 2.0:
        raise ValueError("orientation_separation.beta must be strictly below pi/2")
    separation_far_offset = _number(
        separation.get("far_offset"),
        "orientation_separation.far_offset",
        minimum=0.0,
        strict_minimum=True,
    )
    if separation_far_offset >= separation_beta:
        raise ValueError(
            "orientation_separation.far_offset must be strictly below beta"
        )
    resolved["orientation_separation"] = {
        "m_values": separation_m_values,
        "gamma_exponent": _number(
            separation.get("gamma_exponent"),
            "orientation_separation.gamma_exponent",
            minimum=0.0,
            strict_minimum=True,
        ),
        "beta": separation_beta,
        "far_offset": separation_far_offset,
    }

    composition = _mapping(
        root.get("generic_composition_audit"), "generic_composition_audit"
    )
    _only_keys(
        composition,
        {"m_values", "gamma_exponent", "beta", "far_offset"},
        "generic_composition_audit",
    )
    composition_m_values = _integers(
        composition.get("m_values"),
        "generic_composition_audit.m_values",
        minimum=2,
    )
    if composition_m_values != sorted(set(composition_m_values)):
        raise ValueError(
            "generic_composition_audit.m_values must be sorted and unique"
        )
    composition_beta = _number(
        composition.get("beta"),
        "generic_composition_audit.beta",
        minimum=0.0,
        maximum=math.pi / 2.0,
        strict_minimum=True,
    )
    if composition_beta >= math.pi / 2.0:
        raise ValueError(
            "generic_composition_audit.beta must be strictly below pi/2"
        )
    composition_far_offset = _number(
        composition.get("far_offset"),
        "generic_composition_audit.far_offset",
        minimum=0.0,
        strict_minimum=True,
    )
    if composition_far_offset >= composition_beta:
        raise ValueError(
            "generic_composition_audit.far_offset must be strictly below beta"
        )
    resolved["generic_composition_audit"] = {
        "m_values": composition_m_values,
        "gamma_exponent": _number(
            composition.get("gamma_exponent"),
            "generic_composition_audit.gamma_exponent",
            minimum=0.0,
            strict_minimum=True,
        ),
        "beta": composition_beta,
        "far_offset": composition_far_offset,
    }

    history = _mapping(
        root.get("unknown_boundary_history"), "unknown_boundary_history"
    )
    _only_keys(
        history,
        {
            "m_values",
            "n_exponent",
            "level_exponent",
            "active_exponent",
            "gamma_exponent",
            "output_births_per_level",
            "epsilon_growth_exponent",
            "activity_decay_exponent",
            "predicate_cost_exponent",
            "baseline_match_tolerance",
        },
        "unknown_boundary_history",
    )
    history_m_values = _integers(
        history.get("m_values"),
        "unknown_boundary_history.m_values",
        minimum=2,
    )
    if history_m_values != sorted(set(history_m_values)):
        raise ValueError("unknown_boundary_history.m_values must be sorted and unique")
    resolved["unknown_boundary_history"] = {
        "m_values": history_m_values,
        "n_exponent": _number(
            history.get("n_exponent"),
            "unknown_boundary_history.n_exponent",
            minimum=0.0,
            strict_minimum=True,
        ),
        "level_exponent": _number(
            history.get("level_exponent"),
            "unknown_boundary_history.level_exponent",
            minimum=0.0,
            strict_minimum=True,
        ),
        "active_exponent": _number(
            history.get("active_exponent"),
            "unknown_boundary_history.active_exponent",
            minimum=0.0,
            strict_minimum=True,
        ),
        "gamma_exponent": _number(
            history.get("gamma_exponent"),
            "unknown_boundary_history.gamma_exponent",
            minimum=0.0,
            strict_minimum=True,
        ),
        "output_births_per_level": _integer(
            history.get("output_births_per_level"),
            "unknown_boundary_history.output_births_per_level",
            minimum=1,
        ),
        "epsilon_growth_exponent": _number(
            history.get("epsilon_growth_exponent"),
            "unknown_boundary_history.epsilon_growth_exponent",
            minimum=0.0,
        ),
        "activity_decay_exponent": _number(
            history.get("activity_decay_exponent"),
            "unknown_boundary_history.activity_decay_exponent",
            minimum=0.0,
        ),
        "predicate_cost_exponent": _number(
            history.get("predicate_cost_exponent"),
            "unknown_boundary_history.predicate_cost_exponent",
            minimum=0.0,
        ),
        "baseline_match_tolerance": _number(
            history.get("baseline_match_tolerance"),
            "unknown_boundary_history.baseline_match_tolerance",
            minimum=1.0,
        ),
    }

    iae = _mapping(root.get("iterative_ae"), "iterative_ae")
    _only_keys(
        iae,
        {
            "trials",
            "families",
            "cases",
            "relations",
            "target_angular_precision",
            "shots_per_round",
            "max_rounds",
            "max_grover_power",
            "grid_points",
        },
        "iterative_ae",
    )
    iae_families = _strings(iae.get("families"), "iterative_ae.families")
    if any(family not in FAMILIES for family in iae_families):
        raise ValueError("iterative_ae.families contains an unknown family")
    iae_relations = _strings(iae.get("relations"), "iterative_ae.relations")
    if any(relation not in {"above", "below"} for relation in iae_relations):
        raise ValueError("iterative_ae.relations must contain only above/below")
    resolved["iterative_ae"] = {
        "trials": _integer(iae.get("trials"), "iterative_ae.trials", minimum=1),
        "families": iae_families,
        "cases": _cases(iae.get("cases"), "iterative_ae.cases"),
        "relations": iae_relations,
        "target_angular_precision": _number(
            iae.get("target_angular_precision"),
            "iterative_ae.target_angular_precision",
            minimum=0.0,
            strict_minimum=True,
        ),
        "shots_per_round": _integer(
            iae.get("shots_per_round"), "iterative_ae.shots_per_round", minimum=1
        ),
        "max_rounds": _integer(iae.get("max_rounds"), "iterative_ae.max_rounds", minimum=1),
        "max_grover_power": _integer(
            iae.get("max_grover_power"), "iterative_ae.max_grover_power", minimum=0
        ),
        "grid_points": _integer(iae.get("grid_points"), "iterative_ae.grid_points", minimum=257),
    }

    scheduler = _mapping(root.get("scheduler_sweep"), "scheduler_sweep")
    _only_keys(
        scheduler,
        {"trials", "phase_qubits", "growth_values", "cases"},
        "scheduler_sweep",
    )
    growth_values = _numbers(
        scheduler.get("growth_values"), "scheduler_sweep.growth_values", minimum=1.0
    )
    if any(value <= 1.0 for value in growth_values):
        raise ValueError("scheduler_sweep.growth_values must exceed one")
    resolved["scheduler_sweep"] = {
        "trials": _integer(scheduler.get("trials"), "scheduler_sweep.trials", minimum=1),
        "phase_qubits": _integer(
            scheduler.get("phase_qubits"), "scheduler_sweep.phase_qubits", minimum=1
        ),
        "growth_values": growth_values,
        "cases": _cases(scheduler.get("cases"), "scheduler_sweep.cases"),
    }

    diffusion = _mapping(root.get("diffusion_ablation"), "diffusion_ablation")
    _only_keys(
        diffusion,
        {"phase_qubits", "iterations", "means", "threshold", "relation"},
        "diffusion_ablation",
    )
    diffusion_relation = _string(diffusion.get("relation"), "diffusion_ablation.relation")
    if diffusion_relation not in {"above", "below"}:
        raise ValueError("diffusion_ablation.relation must be above/below")
    resolved["diffusion_ablation"] = {
        "phase_qubits": _integers(
            diffusion.get("phase_qubits"), "diffusion_ablation.phase_qubits"
        ),
        "iterations": _integers(
            diffusion.get("iterations"), "diffusion_ablation.iterations", minimum=0
        ),
        "means": _numbers(
            diffusion.get("means"),
            "diffusion_ablation.means",
            minimum=0.0,
            maximum=1.0,
        ),
        "threshold": _number(
            diffusion.get("threshold"),
            "diffusion_ablation.threshold",
            minimum=0.0,
            maximum=1.0,
        ),
        "relation": diffusion_relation,
    }

    failures = _mapping(root.get("failure_semantics"), "failure_semantics")
    _only_keys(failures, {"enabled"}, "failure_semantics")
    enabled = failures.get("enabled")
    if not isinstance(enabled, bool):
        raise TypeError("failure_semantics.enabled must be bool")
    resolved["failure_semantics"] = {"enabled": enabled}
    resolved["notes"] = _strings(root.get("notes"), "notes")
    return resolved


def load_config(
    path: Path,
    *,
    trials_override: int | None = None,
    seed_override: int | None = None,
) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        resolved = _resolved_config(json.load(stream))
    if seed_override is not None:
        resolved["master_seed"] = _integer(seed_override, "seed", minimum=0)
    if trials_override is not None:
        trials = _integer(trials_override, "trials", minimum=1)
        for name in (
            "unitary_validation",
            "verifier_calibration",
            "random_benchmarks",
            "topk_comparison",
            "iterative_ae",
            "scheduler_sweep",
        ):
            section = resolved[name]
            if not isinstance(section, dict):
                raise TypeError(f"resolved {name} configuration is invalid")
            section["trials"] = trials
    return resolved


def _common(config: Mapping[str, object]) -> Mapping[str, object]:
    value = config["common"]
    if not isinstance(value, Mapping):
        raise TypeError("resolved common configuration is invalid")
    return value


def _section(config: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = config[name]
    if not isinstance(value, Mapping):
        raise TypeError(f"resolved {name} configuration is invalid")
    return value


def _benchmark_config(config: Mapping[str, object]) -> QuantumBenchmarkConfig:
    common = _common(config)
    return QuantumBenchmarkConfig(
        phase_qubits=int(common["phase_qubits"]),
        max_phase_qubits=int(common["max_phase_qubits"]),
        verification_shots=int(common["verification_shots"]),
        confidence=float(common["verification_confidence"]),
        max_attempts_per_output=int(common["max_attempts_per_output"]),
        max_statevector_dimension=int(common["max_statevector_dimension"]),
        classical_shots_per_arm=int(common["classical_shots_per_arm"]),
        boundary_shots_per_round=int(common["boundary_shots_per_round"]),
        max_boundary_rounds=int(common["max_boundary_rounds"]),
    )


def _run_to_terminal(search: Any) -> Any:
    result = search.run()
    while str(result.status) == "paused_resumable":
        result = search.resume()
    return result


def _unitary_suite(config: Mapping[str, object]) -> dict[str, object]:
    section = _section(config, "unitary_validation")
    common = _common(config)
    master = int(config["master_seed"])
    records: list[dict[str, object]] = []
    for case in section["cases"]:  # type: ignore[union-attr]
        if not isinstance(case, Mapping):
            raise TypeError("resolved unitary case is invalid")
        seed = _derived_seed(master, "unitary_validation", case["name"])
        result = run_unitary_validation(
            case["means"],
            threshold=math.sin(float(common["threshold_angle"])) ** 2,
            phase_qubits=int(case["phase_qubits"]),
            trials=int(section["trials"]),
            seed=seed,
        )
        for trial in result.trials:
            record = _jsonable(trial)
            record.update(suite="unitary_validation", case=case["name"])
            records.append(record)
    return {
        "raw_records": records,
        "summary": {
            "trials": len(records),
            "passed": sum(bool(record["passed"]) for record in records),
            "max_compute_inverse_residual": max(
                float(record["compute_inverse_residual"]) for record in records
            ),
            "max_reflection_involution_residual": max(
                float(record["reflection_involution_residual"]) for record in records
            ),
            "query_formula_failures": sum(
                not bool(record["query_formula_exact"]) for record in records
            ),
        },
    }


def _phase_grid_suite(config: Mapping[str, object]) -> dict[str, object]:
    section = _section(config, "phase_grid")
    master = int(config["master_seed"])
    records: list[dict[str, object]] = []
    for bits in section["phase_qubits"]:  # type: ignore[union-attr]
        for threshold_angle in section["threshold_angles"]:  # type: ignore[union-attr]
            threshold = math.sin(float(threshold_angle)) ** 2
            for relation in section["relations"]:  # type: ignore[union-attr]
                sweep = run_phase_grid_sweep(
                    phase_qubits=int(bits),
                    threshold=threshold,
                    relation=str(relation),
                    seed=_derived_seed(
                        master,
                        "phase_grid",
                        bits,
                        threshold_angle,
                        relation,
                    ),
                )
                for point in sweep.points:
                    expected = 1.0 if point.truth_accept else 0.0
                    records.append(
                        {
                            "suite": "phase_grid",
                            "phase_qubits": sweep.phase_qubits,
                            "phase_bins": sweep.phase_bins,
                            "threshold_angle": threshold_angle,
                            "threshold": threshold,
                            "relation": relation,
                            "grid_index": point.grid_index,
                            "angle": point.angle,
                            "mean": point.mean,
                            "mirrored_peak_bins": list(point.mirrored_peak_bins),
                            "truth_accept": point.truth_accept,
                            "joint_acceptance_probability": (
                                point.joint_acceptance_probability
                            ),
                            "absolute_classification_error": abs(
                                point.joint_acceptance_probability - expected
                            ),
                            "oracle_queries": point.resources.oracle_queries,
                            "peak_complex_amplitudes": (
                                point.resources.estimated_peak_complex_amplitudes
                            ),
                        }
                    )
    errors = [float(record["absolute_classification_error"]) for record in records]
    return {
        "raw_records": records,
        "summary": {
            "points": len(records),
            "max_absolute_classification_error": max(errors),
            "mirror_symmetry_boundary_regression_included": True,
        },
    }


def _qpe_resolution_suite(config: Mapping[str, object]) -> dict[str, object]:
    section = _section(config, "qpe_resolution")
    common = _common(config)
    master = int(config["master_seed"])
    threshold_angle = float(common["threshold_angle"])
    threshold = math.sin(threshold_angle) ** 2
    records: list[dict[str, object]] = []
    for gap in section["angular_gaps"]:  # type: ignore[union-attr]
        instance = make_threshold_angular_gap_instance(
            n_below=1,
            n_above=1,
            threshold=threshold,
            angular_gap=float(gap),
            seed=_derived_seed(master, "qpe_resolution", gap),
        )
        for relation in section["relations"]:  # type: ignore[union-attr]
            sweep = run_qpe_acceptance_sweep(
                instance.means,
                threshold=threshold,
                phase_qubits=section["phase_qubits"],  # type: ignore[arg-type]
                relation=str(relation),
                seed=_derived_seed(master, "qpe_resolution", gap, relation),
            )
            for point in sweep.points:
                probability = point.joint_acceptance_probability
                error = 1.0 - probability if point.truth_accept else probability
                records.append(
                    {
                        "suite": "qpe_resolution",
                        "angular_gap": gap,
                        "arm": point.arm,
                        "angle": point.angle,
                        "mean": point.mean,
                        "phase_qubits": point.phase_qubits,
                        "phase_bins": point.phase_bins,
                        "relation": relation,
                        "truth_accept": point.truth_accept,
                        "joint_acceptance_probability": probability,
                        "classification_error_mass": error,
                        "phase_support_size": point.phase_support_size,
                        "oracle_queries": point.resources.oracle_queries,
                        "peak_complex_amplitudes": (
                            point.resources.estimated_peak_complex_amplitudes
                        ),
                    }
                )
    grouped: dict[str, list[float]] = {}
    for record in records:
        key = (
            f"m={record['phase_qubits']}:gap={record['angular_gap']}:"
            f"relation={record['relation']}"
        )
        grouped.setdefault(key, []).append(float(record["classification_error_mass"]))
    return {
        "raw_records": records,
        "summary": {
            "points": len(records),
            "mean_error_by_cell": {
                key: float(np.mean(values)) for key, values in sorted(grouped.items())
            },
            "pointwise_monotonicity_claimed": False,
        },
    }


def _verifier_suite(config: Mapping[str, object]) -> dict[str, object]:
    section = _section(config, "verifier_calibration")
    common = _common(config)
    master = int(config["master_seed"])
    threshold_angle = float(common["threshold_angle"])
    threshold = math.sin(threshold_angle) ** 2
    records: list[dict[str, object]] = []
    for bits in section["phase_qubits"]:  # type: ignore[union-attr]
        for shots in section["shots"]:  # type: ignore[union-attr]
            for confidence in section["confidence_values"]:  # type: ignore[union-attr]
                for offset in section["angular_offsets"]:  # type: ignore[union-attr]
                    angle = threshold_angle + float(offset)
                    if not 0.0 <= angle <= math.pi / 2.0:
                        continue
                    seed = _derived_seed(
                        master,
                        "verifier_calibration",
                        bits,
                        shots,
                        confidence,
                        offset,
                    )
                    result = run_verifier_calibration(
                        math.sin(angle) ** 2,
                        threshold=threshold,
                        phase_qubits=int(bits),
                        shots=int(shots),
                        confidence=float(confidence),
                        trials=int(section["trials"]),
                        seed=seed,
                    )
                    records.append(
                        {
                            "suite": "verifier_calibration",
                            "phase_qubits": bits,
                            "shots": shots,
                            "confidence": confidence,
                            "angular_offset": offset,
                            "mean": result.mean,
                            "exact_qpe_acceptance_probability": (
                                result.exact_qpe_acceptance_probability
                            ),
                            "exact_decision_side": result.exact_decision_side,
                            "trials": len(result.trials),
                            "status_counts": dict(result.status_counts),
                            "interval_coverage_rate": result.interval_coverage_rate,
                            "coverage_wilson_interval": list(
                                result.coverage_wilson_interval
                            ),
                            "wrong_resolved_count": result.wrong_resolved_count,
                            "wrong_resolved_rate": result.wrong_resolved_rate,
                            "wrong_resolved_wilson_interval": list(
                                result.wrong_resolved_wilson_interval
                            ),
                            "evaluation_only_oracle_queries": (
                                result.evaluation_only_oracle_queries
                            ),
                            "procedure_oracle_queries": result.procedure_oracle_queries,
                        }
                    )
    return {
        "raw_records": records,
        "summary": {
            "cells": len(records),
            "total_trials": sum(int(record["trials"]) for record in records),
            "wrong_resolved_decisions": sum(
                int(record["wrong_resolved_count"]) for record in records
            ),
            "minimum_empirical_interval_coverage": min(
                float(record["interval_coverage_rate"]) for record in records
            ),
        },
    }


def _budget_matched_controls(
    config: Mapping[str, object],
    *,
    instance: QuantumBenchmarkInstance,
    relation: str,
    trial_seed: int,
    direct_record: BenchmarkRecord,
) -> list[dict[str, object]]:
    """Run per-arm baselines under the paired direct run's logical-call cap."""

    common = _common(config)
    budget = direct_record.total_queries
    expected = instance.k if relation == "above" else instance.n_arms - instance.k
    truth = instance.truth_above if relation == "above" else instance.truth_below
    phase_qubits = int(common["phase_qubits"])
    qpe_shot_cost = 2 * (1 << phase_qubits) - 1
    qpe_shots = max(1, budget // max(1, instance.n_arms * qpe_shot_cost))
    classical_shots = max(1, budget // instance.n_arms)
    base = {
        "suite": "random_benchmarks_budget_matched",
        "family": instance.family,
        "instance_seed": instance.instance_seed,
        "trial_seed": trial_seed,
        "n_arms": instance.n_arms,
        "k": instance.k,
        "relation": relation,
        "truth": list(truth),
        "logical_query_cap": budget,
        "comparison_unit": "logical_A_family_calls_cap_not_wall_time",
    }
    rows: list[dict[str, object]] = [
        {
            **base,
            "method": "direct_bbht_reference",
            "outputs": direct_record.outputs,
            "complete": direct_record.complete,
            "exact": direct_record.exact,
            "certified": direct_record.certified,
            "status": direct_record.status,
            "oracle_queries": direct_record.total_queries,
            "query_counts_available_by_stage": True,
        }
    ]

    qpe_oracle = CanonicalRyStatevectorOracle(instance.means, seed=trial_seed)
    qpe = IndependentQPEThresholdScan(
        qpe_oracle,
        instance.threshold,
        expected,
        phase_qubits=phase_qubits,
        relation=relation,
        verification_shots=qpe_shots,
        confidence=float(common["verification_confidence"]),
        max_oracle_queries=budget,
        seed=trial_seed,
    ).run()
    rows.append(
        {
            **base,
            "method": "independent_qpe_scan_budget_matched",
            "shots_per_arm": qpe_shots,
            "outputs": list(qpe.outputs),
            "complete": qpe.complete,
            "exact": qpe.complete and set(qpe.outputs) == set(truth),
            "certified": qpe.verified,
            "status": qpe.status,
            "oracle_queries": qpe.resources.oracle_queries,
            "query_counts": dict(qpe.resources.query_counts),
            "cap_respected": qpe.resources.oracle_queries <= budget,
        }
    )

    classical_oracle = CanonicalRyStatevectorOracle(instance.means, seed=trial_seed)
    classical = ClassicalThresholdScan(
        classical_oracle,
        instance.threshold,
        expected,
        relation=relation,
        shots_per_arm=classical_shots,
        confidence=float(common["verification_confidence"]),
        max_oracle_queries=budget,
        seed=trial_seed,
    ).run()
    rows.append(
        {
            **base,
            "method": "classical_threshold_scan_budget_matched",
            "shots_per_arm": classical_shots,
            "outputs": list(classical.outputs),
            "complete": classical.complete,
            "exact": classical.complete and set(classical.outputs) == set(truth),
            "certified": classical.verified,
            "status": classical.status,
            "oracle_queries": classical.resources.oracle_queries,
            "query_counts": dict(classical.resources.query_counts),
            "cap_respected": classical.resources.oracle_queries <= budget,
        }
    )
    return rows


def _benchmark_suite(
    config: Mapping[str, object],
    *,
    suite_name: str,
) -> dict[str, object]:
    section = _section(config, suite_name)
    master = int(config["master_seed"])
    quantiles = config["quantiles"]
    runner = QuantumBenchmarkRunner(_benchmark_config(config))
    records: list[BenchmarkRecord] = []
    budget_matched_records: list[dict[str, object]] = []
    for case in section["cases"]:  # type: ignore[union-attr]
        if not isinstance(case, Mapping):
            raise TypeError(f"resolved {suite_name} case is invalid")
        n = int(case["n"])
        k = int(case["k"])
        for family in section["families"]:  # type: ignore[union-attr]
            for trial in range(int(section["trials"])):
                instance_seed = _derived_seed(
                    master,
                    suite_name,
                    "instance",
                    family,
                    n,
                    k,
                    trial,
                )
                trial_seed = _derived_seed(
                    master,
                    suite_name,
                    "algorithm",
                    family,
                    n,
                    k,
                    trial,
                )
                instance = make_benchmark_instance(
                    str(family),
                    n_arms=n,
                    k=k,
                    seed=instance_seed,
                )
                relations = (
                    ("topk",)
                    if suite_name == "topk_comparison"
                    else tuple(section["relations"])  # type: ignore[arg-type]
                )
                for relation in relations:
                    local: dict[str, BenchmarkRecord] = {}
                    for method in section["methods"]:  # type: ignore[union-attr]
                        record = runner.run(
                            str(method),
                            instance,
                            trial_seed=trial_seed,
                            relation=(
                                "above" if relation == "topk" else str(relation)
                            ),
                        )
                        records.append(record)
                        local[record.method] = record
                    if (
                        suite_name == "random_benchmarks"
                        and "direct_bbht" in local
                    ):
                        budget_matched_records.extend(
                            _budget_matched_controls(
                                config,
                                instance=instance,
                                relation=str(relation),
                                trial_seed=trial_seed,
                                direct_record=local["direct_bbht"],
                            )
                        )
    aggregates = aggregate_benchmark_records(
        records,
        quantiles=quantiles,  # type: ignore[arg-type]
    )
    pairs: dict[str, object] = {}
    comparisons = (
        (
            ("adaptive_calibrated_direct_topk", "calibrated_direct_topk"),
            ("adaptive_calibrated_direct_topk", "fixed_max_precision_topk"),
            (
                "adaptive_calibrated_direct_topk",
                "refined_boundary_only_negative_control",
            ),
            ("adaptive_calibrated_direct_topk", "boundary_only_negative_control"),
        )
        if suite_name == "topk_comparison"
        else (
            ("direct_bbht", "independent_qpe_scan"),
            ("direct_bbht", "classical_threshold_scan"),
        )
    )
    available = {record.method for record in records}
    for numerator, denominator in comparisons:
        if numerator not in available or denominator not in available:
            continue
        paired = paired_query_ratios(records, numerator, denominator)
        if paired:
            pairs[f"{numerator}_over_{denominator}"] = {
                "raw_pairs": _jsonable(paired),
                "aggregate": _jsonable(
                    aggregate_paired_query_ratios(
                        paired,
                        quantiles=quantiles,  # type: ignore[arg-type]
                    )
                ),
            }
    summary: dict[str, object] = {
        "records": len(records),
        "successes": sum(record.success for record in records),
        "aggregates": _jsonable(aggregates),
        "paired_query_ratios": pairs,
        "paired_query_ratio_eligibility": (
            "ratio is finite only when both methods are certified successes"
        ),
        "fixed_parameter_records_present": True,
        "accuracy_matched_advantage_claimed": False,
        "budget_matched_controls": len(budget_matched_records),
        "budget_match_semantics": (
            "baseline total logical oracle calls are capped by the paired "
            "direct run; access categories remain reported separately"
        ),
    }
    if suite_name == "topk_comparison":
        summary["selected_phase_qubit_counts"] = {
            method: dict(
                sorted(
                    Counter(
                        str(record.phase_qubits)
                        for record in records
                        if record.method == method
                        and record.phase_qubits is not None
                    ).items()
                )
            )
            for method in sorted(available)
        }
        summary["status_counts_by_method"] = {
            method: dict(
                sorted(
                    Counter(
                        record.status
                        for record in records
                        if record.method == method
                    ).items()
                )
            )
            for method in sorted(available)
        }
        summary["adaptive_precision_semantics"] = (
            "the measured boundary margin selects one QPE precision before "
            "coherent search; candidate levels are not executed quantum stages"
        )
        summary["boundary_certificate_discovery_advantage_claimed"] = False
    else:
        multi_output = [
            record
            for record in records
            if record.method == "direct_bbht" and record.expected_count > 1
        ]
        summary["direct_multi_output_records"] = len(multi_output)
        summary["direct_multi_output_certified_successes"] = sum(
            record.success for record in multi_output
        )
        summary["direct_multi_output_semantics"] = (
            "sequential full-workspace BBHT with measured exclusion and fresh "
            "verification; not a one-shot multi-output theorem"
        )
    return {
        "raw_records": [record.as_flat_dict() for record in records],
        "budget_matched_records": budget_matched_records,
        "summary": summary,
    }


def _orientation_separation_suite(config: Mapping[str, object]) -> dict[str, object]:
    section = _section(config, "orientation_separation")
    records = orientation_separation_sweep(
        section["m_values"],  # type: ignore[arg-type]
        gamma_exponent=float(section["gamma_exponent"]),
        beta=float(section["beta"]),
        far_offset=float(section["far_offset"]),
    )
    raw_records = [dataclasses.asdict(record) for record in records]
    slope_fields = (
        "orientation_candidate_proxy",
        "independent_all_arm_proxy",
        "worst_gap_marked_extraction_proxy",
        "selected_candidate_proxy",
        "rejected_candidate_proxy",
    )
    slopes = {
        field: descriptive_loglog_slope(records, field)
        for field in slope_fields
    }
    ratios = {
        "candidate_over_independent_last": (
            raw_records[-1]["orientation_candidate_proxy"]
            / raw_records[-1]["independent_all_arm_proxy"]
        ),
        "candidate_over_worst_gap_marked_last": (
            raw_records[-1]["orientation_candidate_proxy"]
            / raw_records[-1]["worst_gap_marked_extraction_proxy"]
        ),
        "candidate_over_rejected_orientation_last": (
            raw_records[-1]["orientation_candidate_proxy"]
            / raw_records[-1]["rejected_candidate_proxy"]
        ),
    }
    return {
        "raw_records": raw_records,
        "summary": {
            "records": len(raw_records),
            "m_values": [record["m"] for record in raw_records],
            "chosen_orientation_counts": dict(
                sorted(
                    Counter(record["chosen_orientation"] for record in raw_records).items()
                )
            ),
            "descriptive_loglog_slopes": slopes,
            "last_point_ratios": ratios,
            "executed_quantum_algorithm": False,
            "asymptotic_theorem_claimed": False,
            "generic_composition_audit_status": "failed_explicit_family",
            "interpretation": (
                "analytic separation-family diagnostic for the declared "
                "orientation proxy; the configured family has been falsified "
                "by a matching same-relation composition"
            ),
        },
    }


def _generic_composition_audit_suite(
    config: Mapping[str, object],
) -> dict[str, object]:
    section = _section(config, "generic_composition_audit")
    records = composition_audit_sweep(
        section["m_values"],  # type: ignore[arg-type]
        gamma_exponent=float(section["gamma_exponent"]),
        beta=float(section["beta"]),
        far_offset=float(section["far_offset"]),
    )
    raw_records = [dataclasses.asdict(record) for record in records]
    slope_fields = (
        "orientation_candidate_proxy",
        "direct_layered_multi_output_proxy",
        "variable_time_rms_proxy",
        "coarse_partition_plus_bai_proxy",
        "independent_all_arm_proxy",
        "worst_gap_marked_extraction_proxy",
        "exceptional_search_lower_proxy",
    )
    slopes = {
        field: composition_loglog_slope(records, field) for field in slope_fields
    }
    failures = sum(
        record.novelty_gate == "failed_explicit_family" for record in records
    )
    return {
        "raw_records": raw_records,
        "summary": {
            "records": len(raw_records),
            "descriptive_loglog_slopes": slopes,
            "outer_composition_match_count": sum(
                record.outer_composition_matches_candidate for record in records
            ),
            "explicit_family_novelty_failure_count": failures,
            "explicit_family_separation_survives": all(
                record.explicit_family_separation_survives for record in records
            ),
            "direct_multi_output_status": (
                "known_boundary_outer_layer_matched_by_all_marked_extraction"
            ),
            "variable_time_status": (
                "generic_rms_and_loop_composition_are_mandatory_known_baselines"
            ),
            "upper_bound_status": (
                "conditional_outer_cost_only_unknown_boundary_and_activity_open"
            ),
            "lower_bound_status": (
                "explicit_family_scaling_matched_general_all_algorithm_bound_open"
            ),
            "executed_quantum_algorithm": False,
            "asymptotic_theorem_claimed": False,
            "novelty_gate": (
                "failed_explicit_family"
                if failures == len(records)
                else "requires_further_audit"
            ),
        },
    }


def _unknown_boundary_history_suite(
    config: Mapping[str, object],
) -> dict[str, object]:
    section = _section(config, "unknown_boundary_history")
    records = unknown_boundary_history_sweep(
        section["m_values"],  # type: ignore[arg-type]
        n_exponent=float(section["n_exponent"]),
        level_exponent=float(section["level_exponent"]),
        active_exponent=float(section["active_exponent"]),
        gamma_exponent=float(section["gamma_exponent"]),
        output_births_per_level=int(section["output_births_per_level"]),
        epsilon_growth_exponent=float(section["epsilon_growth_exponent"]),
        activity_decay_exponent=float(section["activity_decay_exponent"]),
        predicate_cost_exponent=float(section["predicate_cost_exponent"]),
        baseline_match_tolerance=float(section["baseline_match_tolerance"]),
    )
    raw_records = [dataclasses.asdict(record) for record in records]
    slope_fields = (
        "boundary_localization_proxy",
        "coherent_activity_history_proxy",
        "direct_multi_output_quadrature_proxy",
        "candidate_total_proxy",
        "known_boundary_free_history_layered_proxy",
        "rebuild_history_scan_layered_proxy",
        "variable_time_rebuild_rms_proxy",
        "grover_activity_layered_proxy",
        "independent_all_arm_proxy",
        "coarse_partition_bai_proxy",
        "adversary_lower_target_proxy",
        "min_encoded_valid_baseline_proxy",
    )
    slopes = {
        field: unknown_boundary_history_loglog_slope(records, field)
        for field in slope_fields
    }
    gates = Counter(record.novelty_gate for record in records)
    strongest = Counter(record.min_encoded_valid_baseline_name for record in records)
    last = records[-1]
    return {
        "raw_records": raw_records,
        "summary": {
            "records": len(raw_records),
            "m_values": [record.m for record in records],
            "descriptive_loglog_slopes": slopes,
            "novelty_gate_counts": dict(sorted(gates.items())),
            "strongest_encoded_valid_baseline_counts": dict(
                sorted(strongest.items())
            ),
            "last_point_min_valid_baseline_over_candidate": (
                last.min_valid_baseline_over_candidate
            ),
            "last_point_lower_target_over_candidate": (
                last.lower_target_over_candidate
            ),
            "no_free_qram_assumption": last.no_free_qram_assumption,
            "known_boundary_free_history_is_valid_baseline": False,
            "executed_quantum_algorithm": False,
            "asymptotic_theorem_claimed": False,
            "upper_bound_status": last.matching_upper_bound_status,
            "lower_bound_status": last.matching_lower_bound_status,
            "interpretation": (
                "analytic candidate-core audit for unknown-boundary coherent "
                "activity histories; open records mean no encoded valid "
                "baseline matched under the configured tolerance, not that an "
                "advantage theorem has been proved"
            ),
        },
    }


def _iterative_ae_suite(config: Mapping[str, object]) -> dict[str, object]:
    section = _section(config, "iterative_ae")
    common = _common(config)
    master = int(config["master_seed"])
    records: list[dict[str, object]] = []
    iae_config = IAEConfig(
        target_angular_precision=float(section["target_angular_precision"]),
        confidence=float(common["verification_confidence"]),
        shots_per_round=int(section["shots_per_round"]),
        max_rounds=int(section["max_rounds"]),
        max_grover_power=int(section["max_grover_power"]),
        grid_points=int(section["grid_points"]),
    )
    for case in section["cases"]:  # type: ignore[union-attr]
        if not isinstance(case, Mapping):
            raise TypeError("resolved iterative_ae case is invalid")
        n = int(case["n"])
        k = int(case["k"])
        for family in section["families"]:  # type: ignore[union-attr]
            for trial in range(int(section["trials"])):
                instance_seed = _derived_seed(
                    master, "iterative_ae", "instance", family, n, k, trial
                )
                instance = make_benchmark_instance(
                    str(family), n_arms=n, k=k, seed=instance_seed
                )
                for relation in section["relations"]:  # type: ignore[union-attr]
                    seed = _derived_seed(
                        master,
                        "iterative_ae",
                        "algorithm",
                        family,
                        n,
                        k,
                        trial,
                        relation,
                    )
                    expected = k if relation == "above" else n - k
                    truth = (
                        instance.truth_above
                        if relation == "above"
                        else instance.truth_below
                    )
                    oracle = CanonicalBernoulliOracleSimulator(
                        instance.means,
                        seed=seed,
                    )
                    result = IterativeAEThresholdScan(
                        oracle,
                        instance.threshold,
                        expected,
                        relation=str(relation),
                        confidence=float(common["verification_confidence"]),
                        target_angular_precision=float(
                            section["target_angular_precision"]
                        ),
                        config=iae_config,
                        seed=seed,
                    ).run()
                    exact = result.complete and set(result.outputs) == set(truth)
                    records.append(
                        {
                            "suite": "iterative_ae",
                            "backend": result.backend,
                            "family": family,
                            "instance_seed": instance_seed,
                            "trial_seed": seed,
                            "n_arms": n,
                            "k": k,
                            "relation": relation,
                            "expected_count": expected,
                            "outputs": list(result.outputs),
                            "truth": list(truth),
                            "complete": result.complete,
                            "exact": exact,
                            "status": result.status,
                            "failure_reason": result.failure_reason,
                            "oracle_queries": result.resources.oracle_queries,
                            "arms_examined": result.resources.arms_examined,
                            "grover_experiments": (
                                result.resources.grover_experiments
                            ),
                            "measurement_shots": result.resources.measurement_shots,
                            "query_counts": dict(result.resources.query_counts),
                            "gate_or_qubit_resources_available": False,
                            "claim_status": result.claim_status,
                        }
                    )
    return {
        "raw_records": records,
        "summary": {
            "records": len(records),
            "complete": sum(bool(record["complete"]) for record in records),
            "exact": sum(bool(record["exact"]) for record in records),
            "mean_queries": float(
                np.mean([float(record["oracle_queries"]) for record in records])
            ),
            "separate_backend_not_mixed_with_exact_state_resources": True,
        },
    }


def _scheduler_suite(config: Mapping[str, object]) -> dict[str, object]:
    section = _section(config, "scheduler_sweep")
    common = _common(config)
    master = int(config["master_seed"])
    threshold_angle = float(common["threshold_angle"])
    threshold = math.sin(threshold_angle) ** 2
    records: list[dict[str, object]] = []
    for case in section["cases"]:  # type: ignore[union-attr]
        if not isinstance(case, Mapping):
            raise TypeError("resolved scheduler case is invalid")
        n = int(case["n"])
        k = int(case["k"])
        for trial in range(int(section["trials"])):
            endpoint_gap = math.nextafter(
                min(threshold_angle, math.pi / 2.0 - threshold_angle),
                0.0,
            )
            instance = make_threshold_angular_gap_instance(
                n_below=n - k,
                n_above=k,
                threshold=threshold,
                angular_gap=endpoint_gap,
                seed=_derived_seed(master, "scheduler_sweep", "instance", n, k, trial),
            )
            for growth in section["growth_values"]:  # type: ignore[union-attr]
                seed = _derived_seed(
                    master,
                    "scheduler_sweep",
                    "algorithm",
                    n,
                    k,
                    trial,
                    growth,
                )
                oracle = CanonicalRyStatevectorOracle(instance.means, seed=seed)
                search = FullWorkspaceBBHT(
                    oracle,
                    threshold,
                    k,
                    phase_qubits=int(section["phase_qubits"]),
                    verification_shots=int(common["verification_shots"]),
                    verification_confidence=float(
                        common["verification_confidence"]
                    ),
                    max_attempts_per_output=int(common["max_attempts_per_output"]),
                    bbht_growth=float(growth),
                    max_statevector_dimension=int(
                        common["max_statevector_dimension"]
                    ),
                    seed=seed,
                )
                result = _run_to_terminal(search)
                records.append(
                    {
                        "suite": "scheduler_sweep",
                        "n_arms": n,
                        "marked_count": k,
                        "marked_fraction": k / n,
                        "trial": trial,
                        "seed": seed,
                        "bbht_growth": growth,
                        "complete": result.complete,
                        "exact": set(result.outputs) == set(instance.truth_above),
                        "status": result.status,
                        "attempts": result.attempts,
                        "amplitude_amplification_iterations": (
                            result.resources.amplitude_amplification_iterations
                        ),
                        "oracle_queries": result.resources.oracle_queries,
                        "peak_complex_amplitudes": (
                            result.resources.estimated_peak_complex_amplitudes
                        ),
                    }
                )
    grouped: dict[str, list[float]] = {}
    for record in records:
        key = (
            f"n={record['n_arms']}:t={record['marked_count']}:"
            f"growth={record['bbht_growth']}"
        )
        grouped.setdefault(key, []).append(float(record["oracle_queries"]))
    return {
        "raw_records": records,
        "summary": {
            "records": len(records),
            "successes": sum(
                bool(record["complete"] and record["exact"]) for record in records
            ),
            "mean_queries_by_cell": {
                key: float(np.mean(values)) for key, values in sorted(grouped.items())
            },
            "finite_size_scheduler_diagnostic_only": True,
        },
    }


def _diffusion_suite(config: Mapping[str, object]) -> dict[str, object]:
    section = _section(config, "diffusion_ablation")
    master = int(config["master_seed"])
    records: list[dict[str, object]] = []
    for bits in section["phase_qubits"]:  # type: ignore[union-attr]
        for iterations in section["iterations"]:  # type: ignore[union-attr]
            result = run_diffusion_ablation(
                section["means"],  # type: ignore[arg-type]
                threshold=float(section["threshold"]),
                phase_qubits=int(bits),
                relation=str(section["relation"]),
                grover_iterations=int(iterations),
                seed=_derived_seed(master, "diffusion_ablation", bits, iterations),
            )
            records.append(
                {
                    "suite": "diffusion_ablation",
                    "phase_qubits": bits,
                    "iterations": iterations,
                    "state_distance_up_to_global_phase": (
                        result.state_distance_up_to_global_phase
                    ),
                    "finite_phase_leakage_detected": (
                        result.finite_phase_leakage_detected
                    ),
                    "valid_full_acceptance_probability": (
                        result.full_workspace.joint_acceptance_probability
                    ),
                    "invalid_index_only_acceptance_probability": (
                        result.invalid_index_only.joint_acceptance_probability
                    ),
                    "invalid_output_eligible": result.invalid_index_only.output_eligible,
                    "invalid_warning": result.invalid_index_only.warning,
                    "full_oracle_queries": (
                        result.full_workspace.resources.oracle_queries
                    ),
                    "invalid_oracle_queries": (
                        result.invalid_index_only.resources.oracle_queries
                    ),
                }
            )
    return {
        "raw_records": records,
        "summary": {
            "records": len(records),
            "invalid_branch_output_eligible_count": sum(
                bool(record["invalid_output_eligible"]) for record in records
            ),
            "nonzero_iteration_divergence_count": sum(
                int(record["iterations"]) > 0
                and float(record["state_distance_up_to_global_phase"]) > 1e-10
                for record in records
            ),
        },
    }


def _failure_suite(config: Mapping[str, object]) -> dict[str, object]:
    section = _section(config, "failure_semantics")
    if not bool(section["enabled"]):
        return {"raw_records": [], "summary": {"disabled": True}}
    common = _common(config)
    master = int(config["master_seed"])
    parameters = {
        "phase_qubits": 3,
        "verification_shots": int(common["verification_shots"]),
        "verification_confidence": float(common["verification_confidence"]),
        "max_attempts_per_output": 3,
    }
    cases = (
        ("zero_target", (0.4, 0.6), 0, None),
        ("zero_marked_request_one", (0.0, 0.0), 1, None),
        ("target_exceeds_true_marked", (1.0, 0.0), 2, None),
        ("statevector_budget_block", (1.0, 0.0), 1, 1),
    )
    records: list[dict[str, object]] = []
    for name, means, target, budget in cases:
        seed = _derived_seed(master, "failure_semantics", name)
        oracle = CanonicalRyStatevectorOracle(means, seed=seed)
        search = FullWorkspaceBBHT(
            oracle,
            0.5,
            target,
            max_statevector_dimension=(
                int(common["max_statevector_dimension"])
                if budget is None
                else budget
            ),
            seed=seed,
            **parameters,
        )
        result = _run_to_terminal(search)
        records.append(
            {
                "suite": "failure_semantics",
                "case": name,
                "status": result.status,
                "complete": result.complete,
                "outputs": list(result.outputs),
                "target_count": target,
                "absence_certified": result.absence_certified,
                "failure_reason": result.failure_reason,
                "attempts": result.attempts,
                "oracle_queries": result.resources.oracle_queries,
            }
        )

    equality_oracle = CanonicalRyStatevectorOracle([0.5])
    above = DirectAmplitudeThresholdFlag(
        equality_oracle, 0.5, phase_qubits=3, relation="above"
    ).acceptance_probability(0, tag="failure_semantics_threshold_equality")
    below = DirectAmplitudeThresholdFlag(
        equality_oracle, 0.5, phase_qubits=3, relation="below"
    ).acceptance_probability(0, tag="failure_semantics_threshold_equality")
    records.append(
        {
            "suite": "failure_semantics",
            "case": "exact_threshold_mirror_boundary",
            "status": "explicit_equality_semantics",
            "above_acceptance_probability": above,
            "below_acceptance_probability": below,
            "mirror_symmetric": abs(above - 1.0) < 1e-12 and abs(below) < 1e-12,
            "oracle_queries": equality_oracle.query_snapshot().coherent_total,
            "absence_certified": False,
        }
    )

    topk_oracle = CanonicalRyStatevectorOracle([1.0, 0.0], seed=master)
    topk = CalibratedDirectTopKController(
        topk_oracle,
        1,
        phase_qubits=1,
        confidence=float(common["verification_confidence"]),
        boundary_shots_per_round=int(common["boundary_shots_per_round"]),
        max_boundary_rounds=int(common["max_boundary_rounds"]),
        verification_shots=int(common["verification_shots"]),
        max_attempts_per_output=3,
        max_statevector_dimension=int(common["max_statevector_dimension"]),
        seed=master,
    ).run()
    records.append(
        {
            "suite": "failure_semantics",
            "case": "topk_phase_guard",
            "status": topk.status,
            "complete": topk.complete,
            "phase_guard_passed": topk.phase_guard_passed,
            "phase_resolution": topk.phase_resolution,
            "angular_margin": topk.angular_margin,
            "oracle_queries": topk.resources.oracle_queries,
            "absence_certified": False,
        }
    )
    return {
        "raw_records": records,
        "summary": {
            "records": len(records),
            "all_budget_failures_avoid_absence_claim": all(
                not bool(record.get("absence_certified", False)) for record in records
            ),
            "statevector_block_queries": next(
                record["oracle_queries"]
                for record in records
                if record["case"] == "statevector_budget_block"
            ),
            "threshold_mirror_bug_fixed": next(
                record["mirror_symmetric"]
                for record in records
                if record["case"] == "exact_threshold_mirror_boundary"
            ),
        },
    }


def run(
    config: Mapping[str, object],
    *,
    suites: Sequence[str] | None = None,
) -> dict[str, object]:
    execution_provenance = {
        **_git_provenance(),
        **_runtime_provenance(),
        "source_status_capture": "before_suite_execution",
    }
    selected = SUITES if suites is None else tuple(suites)
    if not selected:
        raise ValueError("at least one suite must be selected")
    unknown = sorted(set(selected) - set(SUITES))
    if unknown:
        raise ValueError(f"unknown suites: {unknown}")
    if len(set(selected)) != len(selected):
        raise ValueError("suite selections must be unique")
    runners = {
        "unitary_validation": _unitary_suite,
        "phase_grid": _phase_grid_suite,
        "qpe_resolution": _qpe_resolution_suite,
        "verifier_calibration": _verifier_suite,
        "random_benchmarks": lambda value: _benchmark_suite(
            value, suite_name="random_benchmarks"
        ),
        "topk_comparison": lambda value: _benchmark_suite(
            value, suite_name="topk_comparison"
        ),
        "orientation_separation": _orientation_separation_suite,
        "generic_composition_audit": _generic_composition_audit_suite,
        "unknown_boundary_history": _unknown_boundary_history_suite,
        "iterative_ae": _iterative_ae_suite,
        "scheduler_sweep": _scheduler_suite,
        "diffusion_ablation": _diffusion_suite,
        "failure_semantics": _failure_suite,
    }
    results: dict[str, object] = {}
    for name in selected:
        started = time.perf_counter()
        suite_result = runners[name](config)
        suite_result["simulator_wall_seconds"] = time.perf_counter() - started
        suite_result["wall_time_interpretation"] = (
            "classical simulator performance only; never quantum speedup evidence"
        )
        results[name] = suite_result
    return {
        "schema_version": 2,
        "artifact_type": ARTIFACT_TYPE,
        "experiment_name": config["experiment_name"],
        "claim_status": CLAIM_STATUS,
        "config_hash": _config_hash(config),
        "selected_suites": list(selected),
        "resolved_config": _jsonable(config),
        "provenance": {
            **execution_provenance,
            "means_used_only_for_oracle_construction_and_evaluation": True,
            "direct_search_receives_no_explicit_membership_tuple": True,
            "post_qpe_validation_uses_full_boundary_intervals": True,
            "boundary_certificate_discovery_advantage_claimed": False,
            "query_ledgers_are_executed_logical_calls": True,
            "dense_qft_analytic_array_proxy_is_reported": True,
            "measured_numpy_peak_memory_is_reported": False,
            "statevector_simulation_is_not_hardware_execution": True,
        },
        "claim_boundaries": {
            "supports": [
                "finite-dimensional unitary and query-ledger invariants",
                "finite-QPE resolution and verifier calibration diagnostics",
                "small exact-state BBHT behavior on randomized angular instances",
                "negative controls, failure semantics, and reproducibility",
                "analytic composition falsification of the declared orientation proxy",
                "analytic no-free-QRAM audit for the new unknown-boundary candidate",
            ],
            "does_not_support": [
                "a new QPE, BBHT, or iterative amplitude-estimation theorem",
                "asymptotic quantum advantage or a matching lower bound",
                "hardware feasibility, noise robustness, or simulator speedup",
                "an application-domain performance or advantage claim",
                "Top-k advantage over boundary-only calibration",
                "a generic-composition separation for the orientation candidate",
                "a proved unknown-boundary activity-history transducer theorem",
            ],
        },
        "suite_results": results,
    }


def write_report(report: Mapping[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--suite", action="append", dest="suites", choices=SUITES)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(
            args.config,
            trials_override=args.trials,
            seed_override=args.seed,
        )
        report = run(config, suites=args.suites)
        output = write_report(report, args.output)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"quantum benchmark failed: {error}") from error
    record_count = sum(
        len(result["raw_records"])
        for result in report["suite_results"].values()  # type: ignore[union-attr]
    )
    sys.stdout.write(
        f"wrote {record_count} executed diagnostic records to {output}\n"
        f"suites={','.join(report['selected_suites'])}\n"
        f"claim_status={CLAIM_STATUS}\n"
        "These exact-state and analytic measurement-law results are not a "
        "complexity theorem, hardware run, or application-domain experiment.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
