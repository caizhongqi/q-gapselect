#!/usr/bin/env python3
"""Run the lower-bound proof-program scaffold."""

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

from qgapselect.lower_bound_program import (
    CLAIM_STATUS,
    lower_bound_program_loglog_slope,
    lower_bound_program_sweep,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs" / "lower_bound_program.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "lower_bound_program.json"
DEFAULT_MARKDOWN = REPOSITORY / "docs" / "lower_bound_program.md"
ARTIFACT_TYPE = "q_gapselect_lower_bound_program"

SLOPE_FIELDS = (
    "candidate_proxy",
    "adversary_target_proxy",
    "strongest_block_proxy",
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
        "source_status_capture": "before_lower_bound_program_execution",
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
    allowed_case_keys = {
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
        "precision_multiplier",
        "min_precision_bits",
        "max_precision_bits",
        "baseline_match_tolerance",
    }
    for index, raw_case in enumerate(_sequence(root.get("cases"), "cases")):
        case = _mapping(raw_case, f"cases[{index}]")
        _only_keys(case, allowed_case_keys, f"cases[{index}]")
        m_values = _integers(case.get("m_values"), f"cases[{index}].m_values", minimum=2)
        if m_values != sorted(set(m_values)):
            raise ValueError(f"cases[{index}].m_values must be sorted and unique")
        max_bits_value = case.get("max_precision_bits")
        max_bits = (
            None
            if max_bits_value is None
            else _integer(
                max_bits_value,
                f"cases[{index}].max_precision_bits",
                minimum=1,
            )
        )
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
                    case.get("predicate_cost_exponent", 0.0),
                    f"cases[{index}].predicate_cost_exponent",
                    minimum=0.0,
                ),
                "precision_multiplier": _number(
                    case.get("precision_multiplier", 2.0),
                    f"cases[{index}].precision_multiplier",
                    minimum=0.0,
                    strict_minimum=True,
                ),
                "min_precision_bits": _integer(
                    case.get("min_precision_bits", 1),
                    f"cases[{index}].min_precision_bits",
                    minimum=1,
                ),
                "max_precision_bits": max_bits,
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
    records = lower_bound_program_sweep(
        case["m_values"],  # type: ignore[arg-type]
        n_exponent=float(case["n_exponent"]),
        level_exponent=float(case["level_exponent"]),
        active_exponent=float(case["active_exponent"]),
        gamma_exponent=float(case["gamma_exponent"]),
        output_births_per_level=int(case["output_births_per_level"]),
        epsilon_growth_exponent=float(case["epsilon_growth_exponent"]),
        activity_decay_exponent=float(case["activity_decay_exponent"]),
        predicate_cost_exponent=float(case["predicate_cost_exponent"]),
        precision_multiplier=float(case["precision_multiplier"]),
        min_precision_bits=int(case["min_precision_bits"]),
        max_precision_bits=case["max_precision_bits"],  # type: ignore[arg-type]
        baseline_match_tolerance=float(case["baseline_match_tolerance"]),
    )
    slopes = {
        field: lower_bound_program_loglog_slope(records, field)
        for field in SLOPE_FIELDS
    }
    strongest = Counter(record.strongest_block_id for record in records)
    proof_obligations = sum(record.proof_obligation_count for record in records)
    local_facts = sum(record.local_facts_count for record in records)
    last = records[-1]
    return {
        "name": case["name"],
        "parameters": dict(case),
        "raw_records": [dataclasses.asdict(record) for record in records],
        "summary": {
            "records": len(records),
            "m_values": [record.m for record in records],
            "descriptive_loglog_slopes": slopes,
            "strongest_block_counts": dict(sorted(strongest.items())),
            "proof_obligation_count": proof_obligations,
            "local_facts_count": local_facts,
            "last_point_target_over_candidate": last.target_over_candidate,
            "last_point_strongest_block_id": last.strongest_block_id,
            "executed_quantum_algorithm": False,
            "asymptotic_lower_bound_claimed": False,
            "interpretation": (
                "symbolic lower-bound proof program; blocks identify what an "
                "adversary or polynomial-method proof still has to establish"
            ),
        },
        "simulator_wall_seconds": time.perf_counter() - started,
    }


def run_lower_bound_program(config: Mapping[str, object]) -> dict[str, object]:
    case_results = [_run_case(case) for case in config["cases"]]  # type: ignore[index]
    strongest_totals: Counter[str] = Counter()
    for result in case_results:
        summary = result["summary"]
        if not isinstance(summary, Mapping):
            raise TypeError("internal case summary is invalid")
        strongest_totals.update(summary["strongest_block_counts"])  # type: ignore[arg-type]
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
                "paper-facing enumeration of lower-bound hard-family blocks",
                "explicit separation between local facts and open proof obligations",
            ],
            "does_not_support": [
                "a proved adversary lower bound",
                "a proved polynomial-method lower bound",
                "a CCF-A-ready optimality theorem",
            ],
        },
        "summary": {
            "case_count": len(case_results),
            "total_records": sum(
                int(result["summary"]["records"])  # type: ignore[index]
                for result in case_results
            ),
            "strongest_block_counts": dict(sorted(strongest_totals.items())),
            "total_proof_obligations": sum(
                int(result["summary"]["proof_obligation_count"])  # type: ignore[index]
                for result in case_results
            ),
            "total_local_facts": sum(
                int(result["summary"]["local_facts_count"])  # type: ignore[index]
                for result in case_results
            ),
            "executed_quantum_algorithm": False,
            "asymptotic_lower_bound_claimed": False,
        },
        "case_results": case_results,
    }


def lower_bound_program_markdown(report: Mapping[str, object]) -> str:
    summary = _mapping(report.get("summary"), "summary")
    lines = [
        "# Q-GapSelect lower-bound proof program",
        "",
        f"Claim status: `{report.get('claim_status')}`",
        "",
        "This document is a proof scaffold. It records target blocks and "
        "missing arguments; it is not an adversary lower-bound proof.",
        "",
        "## Summary",
        "",
        f"- cases: `{summary.get('case_count')}`",
        f"- records: `{summary.get('total_records')}`",
        f"- proof obligations: `{summary.get('total_proof_obligations')}`",
        f"- local facts: `{summary.get('total_local_facts')}`",
        "",
        "## Proof blocks",
        "",
    ]
    case_results = _sequence(report.get("case_results"), "case_results")
    seen: set[str] = set()
    for case in case_results:
        case_map = _mapping(case, "case")
        raw_records = _sequence(case_map.get("raw_records"), "raw_records")
        if not raw_records:
            continue
        first = _mapping(raw_records[0], "raw_records[0]")
        for raw_block in _sequence(first.get("blocks"), "blocks"):
            block = _mapping(raw_block, "block")
            block_id = str(block.get("block_id"))
            if block_id in seen:
                continue
            seen.add(block_id)
            lines.extend(
                [
                    f"### {block_id}",
                    "",
                    f"- description: {block.get('description')}",
                    f"- role: `{block.get('target_role')}`",
                    f"- status: `{block.get('proof_status')}`",
                    f"- missing argument: {block.get('missing_argument')}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Activation condition",
            "",
            "The L-07 claim can only be activated after the proof-obligation "
            "blocks are replaced by a formal adversary or polynomial-method "
            "argument under the same no-free-QRAM oracle interface.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: Mapping[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path.resolve()


def write_markdown(report: Mapping[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(lower_bound_program_markdown(report), encoding="utf-8")
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
        report = run_lower_bound_program(config)
        output = write_report(report, args.output)
        markdown = write_markdown(report, args.markdown)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"lower-bound program error: {error}") from error
    print(
        "wrote "
        f"{report['summary']['total_records']} lower-bound program records "
        f"to {output}"
    )
    print(f"wrote lower-bound Markdown to {markdown}")
    print(f"claim_status={report['claim_status']}")
    print("This is a lower-bound proof program, not a proved lower bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
