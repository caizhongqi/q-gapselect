#!/usr/bin/env python3
"""Run charged finite-phase activity-history prototype audits."""

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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from qgapselect.charged_activity_history import (
    CLAIM_STATUS,
    ChargedPhaseHistoryTransducer,
    deterministic_boundary_phases,
    logarithmic_precision_schedule,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs" / "charged_activity_history.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "charged_activity_history.json"
ARTIFACT_TYPE = "q_gapselect_charged_activity_history"


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


def _integers(value: object, name: str, *, minimum: int = 1) -> list[int]:
    values = [
        _integer(item, f"{name}[{index}]", minimum=minimum)
        for index, item in enumerate(_sequence(value, name))
    ]
    if not values:
        raise ValueError(f"{name} cannot be empty")
    return values


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
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
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
        "source_status_capture": "before_charged_activity_execution",
    }


def _config_hash(config: Mapping[str, object]) -> str:
    encoded = json.dumps(_jsonable(config), sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _power_count(m: int, exponent: float) -> int:
    return max(1, int(math.ceil(float(m) ** exponent)))


def _loglog_slope(records: Sequence[Mapping[str, object]], field: str) -> float | None:
    if len(records) < 2:
        return None
    x = np.log(np.asarray([record["m"] for record in records], dtype=np.float64))
    y = np.log(np.asarray([record[field] for record in records], dtype=np.float64))
    return float(np.polyfit(x, y, 1)[0])


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
                "boundary_phase",
                "near_boundary_fraction",
                "base_bits",
                "growth_period",
                "guard_band",
                "max_statevector_dimension",
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
                "boundary_phase": _number(
                    case.get("boundary_phase"),
                    f"cases[{index}].boundary_phase",
                    minimum=0.0,
                    maximum=1.0,
                ),
                "near_boundary_fraction": _number(
                    case.get("near_boundary_fraction"),
                    f"cases[{index}].near_boundary_fraction",
                    minimum=0.0,
                    maximum=1.0,
                    strict_minimum=True,
                ),
                "base_bits": _integer(
                    case.get("base_bits"), f"cases[{index}].base_bits", minimum=1
                ),
                "growth_period": _integer(
                    case.get("growth_period"),
                    f"cases[{index}].growth_period",
                    minimum=1,
                ),
                "guard_band": _number(
                    case.get("guard_band", 0.0),
                    f"cases[{index}].guard_band",
                    minimum=0.0,
                ),
                "max_statevector_dimension": _integer(
                    case.get("max_statevector_dimension", 131072),
                    f"cases[{index}].max_statevector_dimension",
                    minimum=16,
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


def _trace_exact_state(
    transducer: ChargedPhaseHistoryTransducer,
    *,
    max_statevector_dimension: int,
) -> dict[str, object]:
    if transducer.statevector_dimension > max_statevector_dimension:
        return {
            "executed": False,
            "reason": "statevector_dimension_exceeds_configured_cap",
            "statevector_dimension": transducer.statevector_dimension,
        }
    state = transducer.uniform_history_state()
    restored = transducer.apply_compute(transducer.apply_compute(state))
    residual = float(np.linalg.norm(restored - state))
    result = transducer.apply_phase(state, phase_on="output")
    flag_mass = float(
        np.sum(np.abs(result.state.reshape(transducer.shape)[:, :, 1:, :]) ** 2)
        + np.sum(np.abs(result.state.reshape(transducer.shape)[:, :, :, 1:]) ** 2)
    )
    return {
        "executed": True,
        "statevector_dimension": transducer.statevector_dimension,
        "compute_uncompute_residual_l2": residual,
        "flag_garbage_probability_after_phase": flag_mass,
        "active_probability": result.active_probability,
        "output_probability": result.output_probability,
        "query_counts_after_trace": dict(result.resources.query_counts),
    }


def _run_record(case: Mapping[str, object], m: int) -> dict[str, object]:
    n_arms = _power_count(m, float(case["n_exponent"]))
    level_count = _power_count(m, float(case["level_exponent"]))
    phases = deterministic_boundary_phases(
        n_arms,
        boundary_phase=float(case["boundary_phase"]),
        near_boundary_fraction=float(case["near_boundary_fraction"]),
    )
    bits = logarithmic_precision_schedule(
        level_count,
        base_bits=int(case["base_bits"]),
        growth_period=int(case["growth_period"]),
    )
    transducer = ChargedPhaseHistoryTransducer(
        phases,
        bits,
        boundary_phase=float(case["boundary_phase"]),
        guard_band=float(case["guard_band"]),
    )
    predicate_summary = transducer.summarize_predicates()
    exact_trace = _trace_exact_state(
        transducer,
        max_statevector_dimension=int(case["max_statevector_dimension"]),
    )
    average_active_fraction = predicate_summary.total_active_pairs / (
        n_arms * level_count
    )
    average_output_fraction = predicate_summary.total_output_pairs / (
        n_arms * level_count
    )
    return {
        "m": m,
        "n_arms": n_arms,
        "level_count": level_count,
        "statevector_dimension": transducer.statevector_dimension,
        "boundary_phase": case["boundary_phase"],
        "precision_bits_by_level": bits,
        "predicate_summary": dataclasses.asdict(predicate_summary),
        "average_active_fraction": average_active_fraction,
        "average_output_fraction": average_output_fraction,
        "serial_qpe_query_units_per_compute": (
            predicate_summary.serial_qpe_query_units_per_compute
        ),
        "max_level_qpe_query_units_per_compute": (
            predicate_summary.max_level_qpe_query_units_per_compute
        ),
        "no_supplied_predicate_rows": True,
        "exact_state_trace": exact_trace,
    }


def _run_case(case: Mapping[str, object]) -> dict[str, object]:
    started = time.perf_counter()
    records = [_run_record(case, int(m)) for m in case["m_values"]]  # type: ignore[index]
    slopes = {
        field: _loglog_slope(records, field)
        for field in (
            "serial_qpe_query_units_per_compute",
            "max_level_qpe_query_units_per_compute",
            "average_active_fraction",
            "average_output_fraction",
        )
    }
    return {
        "name": case["name"],
        "parameters": dict(case),
        "raw_records": records,
        "summary": {
            "records": len(records),
            "m_values": [record["m"] for record in records],
            "descriptive_loglog_slopes": slopes,
            "all_output_subset_active": all(
                record["predicate_summary"]["output_subset_active"]  # type: ignore[index]
                for record in records
            ),
            "all_no_supplied_predicate_rows": all(
                bool(record["no_supplied_predicate_rows"]) for record in records
            ),
            "exact_state_trace_count": sum(
                bool(record["exact_state_trace"]["executed"])  # type: ignore[index]
                for record in records
            ),
            "executed_quantum_algorithm": False,
            "asymptotic_theorem_claimed": False,
            "interpretation": (
                "charged finite-phase predicate-generation prototype; not a "
                "variable-time upper-bound theorem"
            ),
        },
        "simulator_wall_seconds": time.perf_counter() - started,
    }


def run_charged_activity(config: Mapping[str, object]) -> dict[str, object]:
    case_results = [_run_case(case) for case in config["cases"]]  # type: ignore[index]
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
                "predicate generation from finite phase windows rather than supplied rows",
                "exact-state compute/uncompute traces for small charged fixtures",
                "explicit QPE query-unit ledger for classifier calls",
            ],
            "does_not_support": [
                "unknown Top-k boundary localization by itself",
                "a variable-time coherent-search upper bound",
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
            "all_output_subset_active": all(
                bool(result["summary"]["all_output_subset_active"])  # type: ignore[index]
                for result in case_results
            ),
            "all_no_supplied_predicate_rows": all(
                bool(result["summary"]["all_no_supplied_predicate_rows"])  # type: ignore[index]
                for result in case_results
            ),
            "exact_state_trace_count": sum(
                int(result["summary"]["exact_state_trace_count"])  # type: ignore[index]
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
        report = run_charged_activity(config)
        output = write_report(report, args.output)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"charged activity-history error: {error}") from error
    print(
        "wrote "
        f"{report['summary']['total_records']} charged activity-history records "
        f"to {output}"
    )
    print(f"claim_status={report['claim_status']}")
    print(
        "This is a charged predicate-generation prototype, not a proved "
        "CCF-A-level quantum advantage theorem."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
