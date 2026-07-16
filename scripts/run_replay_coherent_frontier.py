#!/usr/bin/env python3
"""Run replay-preserving coherent frontier exact-state audits."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from qgapselect.replay_coherent_frontier import (  # noqa: E402
    CLAIM_SCOPE,
    ReplayFrontierSchedule,
    ReplayPreservingCoherentFrontier,
)

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "q_gapselect_replay_preserving_coherent_frontier"
DEFAULT_CONFIG = REPOSITORY / "configs" / "replay_coherent_frontier.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "replay_coherent_frontier.json"


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object with string keys")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _only_keys(value: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")


def _index_rows(value: object, name: str) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    for row_index, raw_row in enumerate(_sequence(value, name)):
        rows.append(
            tuple(
                _integer(item, f"{name}[{row_index}] index")
                for item in _sequence(raw_row, f"{name}[{row_index}]")
            )
        )
    if not rows:
        raise ValueError(f"{name} cannot be empty")
    return tuple(rows)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.name != "state"
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


def load_config(path: Path) -> dict[str, object]:
    root = _mapping(json.loads(path.read_text(encoding="utf-8")), "root")
    _only_keys(root, {"schema_version", "experiment_name", "cases", "notes"}, "root")
    if _integer(root.get("schema_version"), "schema_version", minimum=1) != 1:
        raise ValueError("schema_version must be 1")
    cases: list[dict[str, object]] = []
    for case_index, raw_case in enumerate(_sequence(root.get("cases"), "cases")):
        case = _mapping(raw_case, f"cases[{case_index}]")
        _only_keys(
            case,
            {
                "case_id",
                "n_arms",
                "active_indices_by_level",
                "output_births_by_level",
                "prefix_levels",
                "cleanup_tolerance",
                "max_statevector_dimension",
            },
            f"cases[{case_index}]",
        )
        prefixes = tuple(
            _integer(item, f"cases[{case_index}].prefix_levels", minimum=1)
            for item in _sequence(
                case.get("prefix_levels"), f"cases[{case_index}].prefix_levels"
            )
        )
        if not prefixes or tuple(sorted(set(prefixes))) != prefixes:
            raise ValueError("prefix_levels must be non-empty, sorted, and unique")
        tolerance = case.get("cleanup_tolerance", 1e-12)
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise TypeError("cleanup_tolerance must be numeric")
        cases.append(
            {
                "case_id": _string(case.get("case_id"), f"cases[{case_index}].case_id"),
                "n_arms": _integer(
                    case.get("n_arms"), f"cases[{case_index}].n_arms", minimum=2
                ),
                "active_indices_by_level": _index_rows(
                    case.get("active_indices_by_level"),
                    f"cases[{case_index}].active_indices_by_level",
                ),
                "output_births_by_level": _index_rows(
                    case.get("output_births_by_level"),
                    f"cases[{case_index}].output_births_by_level",
                ),
                "prefix_levels": prefixes,
                "cleanup_tolerance": float(tolerance),
                "max_statevector_dimension": _integer(
                    case.get("max_statevector_dimension", 8_388_608),
                    f"cases[{case_index}].max_statevector_dimension",
                    minimum=1,
                ),
            }
        )
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("cases must be non-empty with unique case_id values")
    notes = tuple(
        _string(item, f"notes[{index}]")
        for index, item in enumerate(_sequence(root.get("notes", []), "notes"))
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": _string(root.get("experiment_name"), "experiment_name"),
        "cases": tuple(cases),
        "notes": notes,
    }


def _git_provenance() -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ("git", *args),
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else "unknown"

    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_tree": run("rev-parse", "HEAD^{tree}"),
        "source_tree_dirty_at_execution": bool(run("status", "--porcelain")),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def _config_hash(config: Mapping[str, object]) -> str:
    payload = json.dumps(_jsonable(config), sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def run_experiment(config: Mapping[str, object]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for case in config["cases"]:
        schedule = ReplayFrontierSchedule(
            n_arms=int(case["n_arms"]),
            active_indices_by_level=tuple(case["active_indices_by_level"]),
            output_births_by_level=tuple(case["output_births_by_level"]),
        )
        for prefix in case["prefix_levels"]:
            frontier = ReplayPreservingCoherentFrontier(
                schedule,
                prefix_levels=int(prefix),
                cleanup_tolerance=float(case["cleanup_tolerance"]),
                max_statevector_dimension=int(case["max_statevector_dimension"]),
            )
            result = frontier.apply(frontier.uniform_index_state())
            result_document = _jsonable(result)
            result_document["invariants"]["passed"] = result.invariants.passed
            result_document["resources"]["cleanup"]["passed"] = (
                result.resources.cleanup.passed
            )
            records.append(
                {
                    "case_id": case["case_id"],
                    "prefix_levels": prefix,
                    "schedule_fingerprint": schedule.fingerprint,
                    "n_arms": schedule.n_arms,
                    "level_count": schedule.level_count,
                    "result": result_document,
                }
            )
    cleanup_ledgers = [record["result"]["resources"]["cleanup"] for record in records]
    resource_rows = [record["result"]["resources"] for record in records]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "claim_scope": CLAIM_SCOPE,
        "experiment_name": config["experiment_name"],
        "config_hash": _config_hash(config),
        "provenance": _git_provenance(),
        "resolved_config": _jsonable(config),
        "summary": {
            "record_count": len(records),
            "all_cleanup_passed": all(row["passed"] for row in cleanup_ledgers),
            "all_invariants_passed": all(
                record["result"]["invariants"]["passed"] for record in records
            ),
            "all_execution_traces_replayed": all(
                row["execution_trace_replayed"] for row in cleanup_ledgers
            ),
            "maximum_cleanup_residual_l2": max(
                row["expected_durable_output_residual_l2"] for row in cleanup_ledgers
            ),
            "maximum_transient_nonzero_probability": max(
                row["transient_nonzero_probability"] for row in cleanup_ledgers
            ),
            "maximum_qubits": max(row["qubits"] for row in resource_rows),
            "maximum_statevector_dimension": max(
                row["statevector_dimension"] for row in resource_rows
            ),
            "qram_assumed": any(row["qram_assumed"] for row in resource_rows),
            "coherent_index_execution_performed": True,
            "durable_output_copy_executed": True,
            "coherent_boundary_discovery_executed": False,
            "direct_multi_output_complete": False,
            "new_variable_time_upper_bound_proved": False,
            "same_interface_composition_separated": False,
            "matching_lower_bound_proved": False,
            "quantum_advantage_claimable": False,
        },
        "records": records,
        "notes": list(config["notes"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_config(args.config)
    artifact = run_experiment(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = artifact["summary"]
    print(
        "Replay coherent frontier code-sanity: "
        f"records={summary['record_count']}, cleanup={summary['all_cleanup_passed']}, "
        f"invariants={summary['all_invariants_passed']}."
    )
    print(
        "The public schedule is not coherent boundary discovery; no new upper "
        "bound, composition separation, lower bound, or CCF-A advantage is claimed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
