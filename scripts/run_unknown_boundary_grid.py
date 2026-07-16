#!/usr/bin/env python3
"""Run parameter-grid audits for the unknown-boundary history candidate."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qgapselect.unknown_boundary_history import (
    unknown_boundary_history_loglog_slope,
    unknown_boundary_history_sweep,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs" / "unknown_boundary_grid.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "unknown_boundary_grid.json"
ARTIFACT_TYPE = "q_gapselect_unknown_boundary_history_grid"
CLAIM_STATUS = "analytic_unknown_boundary_history_grid_no_theorem"

SLOPE_FIELDS = (
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


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be an integer") from error
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
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


def _integers(value: object, name: str, *, minimum: int = 1) -> list[int]:
    result = [
        _integer(item, f"{name}[{index}]", minimum=minimum)
        for index, item in enumerate(_sequence(value, name))
    ]
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _only_keys(value: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def _git_provenance() -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ("git", *args),
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return "unknown"
        return completed.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_tree": run("rev-parse", "HEAD^{tree}"),
        "source_tree_dirty_at_execution": bool(status),
        "source_status_capture": "before_grid_execution",
    }


def _config_hash(config: Mapping[str, object]) -> str:
    encoded = json.dumps(_jsonable(config), sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_config(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    root = _mapping(document, "root")
    _only_keys(root, {"schema_version", "experiment_name", "cases", "notes"}, "root")
    if _integer(root.get("schema_version"), "schema_version", minimum=1) != 1:
        raise ValueError("schema_version must be 1")
    cases: list[dict[str, object]] = []
    for index, raw_case in enumerate(_sequence(root.get("cases"), "cases")):
        case = _mapping(raw_case, f"cases[{index}]")
        _only_keys(
            case,
            {
                "name",
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
            f"cases[{index}]",
        )
        m_values = _integers(case.get("m_values"), f"cases[{index}].m_values", minimum=2)
        if m_values != sorted(set(m_values)):
            raise ValueError(f"cases[{index}].m_values must be sorted and unique")
        cases.append(
            {
                "name": _string(case.get("name"), f"cases[{index}].name"),
                "m_values": m_values,
                "n_exponent": _number(
                    case.get("n_exponent"),
                    f"cases[{index}].n_exponent",
                    minimum=0.0,
                    strict_minimum=True,
                ),
                "level_exponent": _number(
                    case.get("level_exponent"),
                    f"cases[{index}].level_exponent",
                    minimum=0.0,
                    strict_minimum=True,
                ),
                "active_exponent": _number(
                    case.get("active_exponent"),
                    f"cases[{index}].active_exponent",
                    minimum=0.0,
                    strict_minimum=True,
                ),
                "gamma_exponent": _number(
                    case.get("gamma_exponent"),
                    f"cases[{index}].gamma_exponent",
                    minimum=0.0,
                    strict_minimum=True,
                ),
                "output_births_per_level": _integer(
                    case.get("output_births_per_level"),
                    f"cases[{index}].output_births_per_level",
                    minimum=1,
                ),
                "epsilon_growth_exponent": _number(
                    case.get("epsilon_growth_exponent"),
                    f"cases[{index}].epsilon_growth_exponent",
                    minimum=0.0,
                ),
                "activity_decay_exponent": _number(
                    case.get("activity_decay_exponent"),
                    f"cases[{index}].activity_decay_exponent",
                    minimum=0.0,
                ),
                "predicate_cost_exponent": _number(
                    case.get("predicate_cost_exponent"),
                    f"cases[{index}].predicate_cost_exponent",
                    minimum=0.0,
                ),
                "baseline_match_tolerance": _number(
                    case.get("baseline_match_tolerance"),
                    f"cases[{index}].baseline_match_tolerance",
                    minimum=1.0,
                ),
            }
        )
    if not cases:
        raise ValueError("cases cannot be empty")
    notes = [
        _string(item, f"notes[{index}]")
        for index, item in enumerate(_sequence(root.get("notes", []), "notes"))
    ]
    return {
        "schema_version": 1,
        "experiment_name": _string(root.get("experiment_name"), "experiment_name"),
        "cases": cases,
        "notes": notes,
    }


def _run_case(case: Mapping[str, object]) -> dict[str, object]:
    started = time.perf_counter()
    records = unknown_boundary_history_sweep(
        case["m_values"],  # type: ignore[arg-type]
        n_exponent=float(case["n_exponent"]),
        level_exponent=float(case["level_exponent"]),
        active_exponent=float(case["active_exponent"]),
        gamma_exponent=float(case["gamma_exponent"]),
        output_births_per_level=int(case["output_births_per_level"]),
        epsilon_growth_exponent=float(case["epsilon_growth_exponent"]),
        activity_decay_exponent=float(case["activity_decay_exponent"]),
        predicate_cost_exponent=float(case["predicate_cost_exponent"]),
        baseline_match_tolerance=float(case["baseline_match_tolerance"]),
    )
    slopes = {
        field: unknown_boundary_history_loglog_slope(records, field)
        for field in SLOPE_FIELDS
    }
    gates = Counter(record.novelty_gate for record in records)
    strongest = Counter(record.min_encoded_valid_baseline_name for record in records)
    last = records[-1]
    return {
        "name": case["name"],
        "parameters": dict(case),
        "raw_records": [dataclasses.asdict(record) for record in records],
        "summary": {
            "records": len(records),
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
            "last_point_free_history_over_candidate": (
                last.free_history_over_candidate
            ),
            "all_points_open": all(
                not record.encoded_baseline_match_found for record in records
            ),
            "executed_quantum_algorithm": False,
            "asymptotic_theorem_claimed": False,
            "wall_time_interpretation": (
                "analytic proxy runtime only; never quantum speedup evidence"
            ),
        },
        "simulator_wall_seconds": time.perf_counter() - started,
    }


def run_grid(config: Mapping[str, object]) -> dict[str, object]:
    case_results = [_run_case(case) for case in config["cases"]]  # type: ignore[index]
    gate_totals: Counter[str] = Counter()
    baseline_totals: Counter[str] = Counter()
    for result in case_results:
        summary = result["summary"]
        if not isinstance(summary, Mapping):
            raise TypeError("internal case summary is invalid")
        gate_totals.update(summary["novelty_gate_counts"])  # type: ignore[arg-type]
        baseline_totals.update(
            summary["strongest_encoded_valid_baseline_counts"]  # type: ignore[arg-type]
        )
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "experiment_name": config["experiment_name"],
        "claim_status": CLAIM_STATUS,
        "config_hash": _config_hash(config),
        "resolved_config": _jsonable(config),
        "provenance": {
            **_git_provenance(),
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "byte_for_byte_reproduction_expected": False,
            "volatile_fields": ["case_results.*.simulator_wall_seconds"],
        },
        "claim_boundaries": {
            "supports": [
                "analytic parameter-grid audit for the unknown-boundary history candidate",
                "encoded legal-baseline comparisons under the no-free-QRAM interface",
            ],
            "does_not_support": [
                "a proved unknown-boundary activity-history transducer theorem",
                "a matching all-algorithms lower bound",
                "hardware feasibility or simulator speedup",
            ],
        },
        "summary": {
            "case_count": len(case_results),
            "total_records": sum(
                int(result["summary"]["records"])  # type: ignore[index]
                for result in case_results
            ),
            "novelty_gate_counts": dict(sorted(gate_totals.items())),
            "strongest_encoded_valid_baseline_counts": dict(
                sorted(baseline_totals.items())
            ),
            "open_case_count": sum(
                bool(result["summary"]["all_points_open"])  # type: ignore[index]
                for result in case_results
            ),
            "executed_quantum_algorithm": False,
            "asymptotic_theorem_claimed": False,
        },
        "case_results": case_results,
    }


def write_report(report: Mapping[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        report = run_grid(config)
        output = write_report(report, args.output)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"unknown-boundary grid error: {error}") from error
    print(
        "wrote "
        f"{report['summary']['total_records']} unknown-boundary grid records "
        f"to {output}"
    )
    print(f"claim_status={report['claim_status']}")
    print(
        "These analytic grid results are not an upper-bound theorem, lower-bound "
        "theorem, or hardware run."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
