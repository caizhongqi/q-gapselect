#!/usr/bin/env python3
"""Generate stopping-time unitary lemma scaffold artifacts."""

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

import numpy as np

from qgapselect.charged_activity_history import (
    deterministic_boundary_phases,
    logarithmic_precision_schedule,
)
from qgapselect.stopping_time_theorem import (
    CLAIM_STATUS,
    build_stopping_unitary_lemma_scaffold,
)
from qgapselect.stopping_time_transducer import VariableTimeStoppingTransducer

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs" / "stopping_unitary_theorem.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "stopping_unitary_theorem.json"
DEFAULT_MARKDOWN = REPOSITORY / "docs" / "stopping_unitary_theorem.md"
ARTIFACT_TYPE = "q_gapselect_stopping_unitary_theorem_scaffold"


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
        "source_status_capture": "before_stopping_unitary_theorem_execution",
    }


def _config_hash(config: Mapping[str, object]) -> str:
    encoded = json.dumps(_jsonable(config), sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _power_count(m: int, exponent: float) -> int:
    return max(1, int(math.ceil(float(m) ** exponent)))


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
                "phase_on",
            },
            f"cases[{index}]",
        )
        m_values = _integers(case.get("m_values"), f"cases[{index}].m_values", minimum=2)
        if m_values != sorted(set(m_values)):
            raise ValueError(f"cases[{index}].m_values must be sorted and unique")
        phase_on = _string(case.get("phase_on", "output"), f"cases[{index}].phase_on")
        if phase_on not in {"active", "output", "active_output"}:
            raise ValueError(f"cases[{index}].phase_on is invalid")
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
                "phase_on": phase_on,
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
    transducer = VariableTimeStoppingTransducer(
        phases,
        bits,
        boundary_phase=float(case["boundary_phase"]),
        guard_band=float(case["guard_band"]),
    )
    scaffold = build_stopping_unitary_lemma_scaffold(
        transducer,
        phase_on=str(case["phase_on"]),
    )
    return {
        "m": m,
        "n_arms": n_arms,
        "level_count": level_count,
        "precision_bits_by_level": bits,
        "scaffold": dataclasses.asdict(scaffold),
    }


def _run_case(case: Mapping[str, object]) -> dict[str, object]:
    started = time.perf_counter()
    records = [_run_record(case, int(m)) for m in case["m_values"]]  # type: ignore[index]
    status_counter: Counter[str] = Counter()
    for record in records:
        for check in record["scaffold"]["checks"]:  # type: ignore[index]
            status_counter[str(check["status"])] += 1
    last = records[-1]["scaffold"]  # type: ignore[index]
    return {
        "name": case["name"],
        "parameters": dict(case),
        "raw_records": records,
        "summary": {
            "records": len(records),
            "m_values": [record["m"] for record in records],
            "check_status_counts": dict(sorted(status_counter.items())),
            "all_execution_checks_passed": all(
                int(record["scaffold"]["failed_count"]) == 0  # type: ignore[index]
                for record in records
            ),
            "last_branch_rms_over_serial": last["branch_rms_over_serial"],
            "last_proof_obligation_count": last["proof_obligation_count"],
            "executed_quantum_algorithm": False,
            "asymptotic_theorem_claimed": False,
            "interpretation": (
                "paper-facing stopping-unitary lemma scaffold; executable checks "
                "pass but proof obligations remain"
            ),
        },
        "simulator_wall_seconds": time.perf_counter() - started,
    }


def run_theorem_scaffold(config: Mapping[str, object]) -> dict[str, object]:
    case_results = [_run_case(case) for case in config["cases"]]  # type: ignore[index]
    status_totals: Counter[str] = Counter()
    for result in case_results:
        summary = result["summary"]
        if not isinstance(summary, Mapping):
            raise TypeError("internal case summary is invalid")
        status_totals.update(summary["check_status_counts"])  # type: ignore[arg-type]
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
                "formalized stopping-unitary lemma obligations",
                "exact-state execution checks for involution, phase equivalence, "
                "cleanup, and RMS ledgers",
            ],
            "does_not_support": [
                "a completed variable-time upper-bound proof",
                "a matching all-algorithms lower bound",
                "composition-frontier novelty against all prior variable-time theorems",
            ],
        },
        "summary": {
            "case_count": len(case_results),
            "total_records": sum(
                int(result["summary"]["records"])  # type: ignore[index]
                for result in case_results
            ),
            "check_status_counts": dict(sorted(status_totals.items())),
            "all_execution_checks_passed": all(
                bool(result["summary"]["all_execution_checks_passed"])  # type: ignore[index]
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


def write_markdown(report: Mapping[str, object], path: Path) -> Path:
    case_results = report.get("case_results")
    if not isinstance(case_results, list) or not case_results:
        raise ValueError("report has no case results")
    first_case = case_results[0]
    if not isinstance(first_case, Mapping):
        raise TypeError("first case is invalid")
    raw_records = first_case.get("raw_records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("first case has no records")
    first_record = raw_records[0]
    if not isinstance(first_record, Mapping):
        raise TypeError("first record is invalid")
    scaffold_dict = first_record.get("scaffold")
    if not isinstance(scaffold_dict, Mapping):
        raise TypeError("first scaffold is invalid")
    # The markdown renderer expects the dataclass; rebuild only the short note
    # directly from the first already-jsonable record to avoid hidden work.
    lines = [
        "# Variable-time stopping-unitary theorem scaffold",
        "",
        f"Claim status: `{report['claim_status']}`",
        "",
        "The JSON artifact contains the full per-record check table.  The first "
        "record is summarized here as the canonical proof template.",
        "",
        f"- readiness: `{scaffold_dict['readiness']}`",
        f"- passed checks: `{scaffold_dict['passed_count']}`",
        f"- failed checks: `{scaffold_dict['failed_count']}`",
        f"- proof obligations: `{scaffold_dict['proof_obligation_count']}`",
        "",
        "## Checks",
        "",
    ]
    checks = scaffold_dict.get("checks")
    if not isinstance(checks, (list, tuple)):
        raise TypeError("scaffold checks are invalid")
    for check in checks:
        if not isinstance(check, Mapping):
            raise TypeError("scaffold check is invalid")
        lines.extend(
            [
                f"### {check['check_id']}: {check['statement']}",
                "",
                f"- status: `{check['status']}`",
                f"- evidence: {check['evidence']}",
                f"- missing for theorem: {check['missing_for_theorem']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        report = run_theorem_scaffold(config)
        output = write_report(report, args.output)
        markdown = write_markdown(report, args.markdown)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"stopping-unitary theorem scaffold error: {error}") from error
    print(
        "wrote "
        f"{report['summary']['total_records']} stopping-unitary theorem records "
        f"to {output}"
    )
    print(f"wrote theorem scaffold Markdown to {markdown}")
    print(f"claim_status={report['claim_status']}")
    print(
        "This is a theorem scaffold with proof obligations, not a completed "
        "CCF-A-level upper-bound theorem."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
