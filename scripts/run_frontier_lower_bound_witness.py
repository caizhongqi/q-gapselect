#!/usr/bin/env python3
"""Run the finite S3 lower-bound and composition-falsification panel."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
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

from qgapselect.frontier_lower_bound_witness import (  # noqa: E402
    CANONICAL_INTERFACE_ID,
    composition_falsification_certificate,
    johnson_adversary_certificate,
    paired_rotation_hybrid_certificate,
)
from qgapselect.strong_composition_registry import (  # noqa: E402
    audit_strong_composition_registry,
    load_strong_composition_registry,
    strong_composition_registry_sha256,
)

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "q_gapselect_frontier_lower_bound_witness_s3"
CLAIM_STATUS = "finite_witnesses_only_no_matching_lower_bound_or_composition_theorem"
DEFAULT_CONFIG = REPOSITORY / "configs" / "frontier_lower_bound_witness.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "frontier_lower_bound_witness.json"


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


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _strict_keys(
    value: Mapping[str, object],
    allowed: set[str],
    name: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")


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
    _strict_keys(
        root,
        {
            "schema_version",
            "experiment_name",
            "pair_hybrid_cases",
            "johnson_cases",
            "composition_cases",
            "notes",
        },
        "root",
    )
    if _integer(root.get("schema_version"), "schema_version", minimum=1) != 1:
        raise ValueError("schema_version must equal 1")

    pair_cases: list[dict[str, object]] = []
    pair_allowed = {
        "witness_id",
        "n",
        "k",
        "low_angle",
        "high_angle",
        "error_probability",
    }
    for index, raw in enumerate(_sequence(root.get("pair_hybrid_cases"), "pair cases")):
        row = _mapping(raw, f"pair_hybrid_cases[{index}]")
        _strict_keys(row, pair_allowed, f"pair_hybrid_cases[{index}]")
        pair_cases.append(
            {
                "witness_id": _string(row.get("witness_id"), "witness_id"),
                "n": _integer(row.get("n"), "n", minimum=2),
                "k": _integer(row.get("k"), "k", minimum=1),
                "low_angle": _number(row.get("low_angle"), "low_angle"),
                "high_angle": _number(row.get("high_angle"), "high_angle"),
                "error_probability": _number(row.get("error_probability"), "error_probability"),
            }
        )

    johnson_cases: list[dict[str, object]] = []
    johnson_allowed = {"witness_id", "n", "k"}
    for index, raw in enumerate(_sequence(root.get("johnson_cases"), "johnson cases")):
        row = _mapping(raw, f"johnson_cases[{index}]")
        _strict_keys(row, johnson_allowed, f"johnson_cases[{index}]")
        johnson_cases.append(
            {
                "witness_id": _string(row.get("witness_id"), "witness_id"),
                "n": _integer(row.get("n"), "n", minimum=2),
                "k": _integer(row.get("k"), "k", minimum=1),
            }
        )

    composition_cases: list[dict[str, object]] = []
    composition_allowed = {
        "witness_id",
        "means",
        "k",
        "phase_qubits",
        "match_tolerance",
        "cleanup_tolerance",
        "max_statevector_dimension",
    }
    for index, raw in enumerate(_sequence(root.get("composition_cases"), "composition cases")):
        row = _mapping(raw, f"composition_cases[{index}]")
        _strict_keys(row, composition_allowed, f"composition_cases[{index}]")
        means = tuple(
            _number(mean, f"composition_cases[{index}].means")
            for mean in _sequence(row.get("means"), "means")
        )
        if len(means) != 2:
            raise ValueError("composition cases require exactly two means for S3")
        k = _integer(row.get("k"), "k", minimum=1)
        if k != 1:
            raise ValueError("composition cases require k=1 for the tiny S3 candidate")
        composition_cases.append(
            {
                "witness_id": _string(row.get("witness_id"), "witness_id"),
                "means": means,
                "k": k,
                "phase_qubits": _integer(row.get("phase_qubits"), "phase_qubits", minimum=1),
                "match_tolerance": _number(row.get("match_tolerance"), "match_tolerance"),
                "cleanup_tolerance": _number(row.get("cleanup_tolerance"), "cleanup_tolerance"),
                "max_statevector_dimension": _integer(
                    row.get("max_statevector_dimension"),
                    "max_statevector_dimension",
                    minimum=1,
                ),
            }
        )
    if not pair_cases or not johnson_cases or not composition_cases:
        raise ValueError("all three case panels must be non-empty")
    witness_ids = [
        str(row["witness_id"])
        for panel in (pair_cases, johnson_cases, composition_cases)
        for row in panel
    ]
    if len(witness_ids) != len(set(witness_ids)):
        raise ValueError("witness_id values must be unique across every panel")
    notes = tuple(
        _string(note, f"notes[{index}]")
        for index, note in enumerate(_sequence(root.get("notes", ()), "notes"))
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": _string(root.get("experiment_name"), "experiment_name"),
        "pair_hybrid_cases": tuple(pair_cases),
        "johnson_cases": tuple(johnson_cases),
        "composition_cases": tuple(composition_cases),
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

    status = run("status", "--porcelain")
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_tree": run("rev-parse", "HEAD^{tree}"),
        "source_tree_dirty_at_execution": bool(status),
        "source_status_capture": "before_s3_artifact_write",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }


def run_panel(config: Mapping[str, object]) -> dict[str, object]:
    pair_records = [
        paired_rotation_hybrid_certificate(**dict(row))
        for row in config["pair_hybrid_cases"]  # type: ignore[union-attr]
    ]
    johnson_records = [
        johnson_adversary_certificate(**dict(row))
        for row in config["johnson_cases"]  # type: ignore[union-attr]
    ]
    composition_records = []
    for raw in config["composition_cases"]:  # type: ignore[union-attr]
        composition_records.append(
            composition_falsification_certificate(**dict(raw))
        )

    registry = load_strong_composition_registry()
    registry_audit = audit_strong_composition_registry(registry)
    if registry.canonical_interface.interface_id != CANONICAL_INTERFACE_ID:
        raise ValueError("frontier witness interface does not match strong registry")
    local_bounds = [record.computed_quantity for record in pair_records]
    upper_counts = [record.baseline_query_count for record in composition_records]
    all_local_verified = all(record.verification_passed for record in pair_records)
    all_johnson_verified = all(record.verification_passed for record in johnson_records)
    composition_kill_count = sum(record.composition_kill_flag for record in composition_records)
    summary = {
        "pair_hybrid_witness_count": len(pair_records),
        "pair_hybrid_all_verified": all_local_verified,
        "johnson_witness_count": len(johnson_records),
        "johnson_all_verified": all_johnson_verified,
        "composition_witness_count": len(composition_records),
        "composition_match_count": sum(record.composition_match for record in composition_records),
        "composition_kill_count": composition_kill_count,
        "finite_fixture_query_dominance_count": sum(
            record.finite_fixture_query_dominance_verified
            for record in composition_records
        ),
        "largest_local_rotation_lower_bound": max(local_bounds),
        "smallest_finite_composition_upper_count": min(upper_counts),
        "local_lower_bound_matches_finite_upper": False,
        "continuous_angle_johnson_composition_proved": False,
        "activity_history_direct_sum_proved": False,
        "registered_strongest_composition_coverage_complete": (
            registry_audit.strongest_composition_coverage_complete
        ),
        "uncovered_required_baseline_ids": list(registry_audit.uncovered_required_baseline_ids),
        "matching_lower_bound_claimable": False,
        "strongest_composition_claimable": False,
        "quantum_advantage_claimable": False,
        "ccf_a_claimable": False,
        "claim_status": CLAIM_STATUS,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "experiment_name": config["experiment_name"],
        "claim_status": CLAIM_STATUS,
        "registry_binding": {
            "registry_id": registry.registry_id,
            "registry_sha256": strong_composition_registry_sha256(registry),
            "canonical_interface_id": registry.canonical_interface.interface_id,
            "strongest_composition_claimable": (registry_audit.strongest_composition_claimable),
        },
        "summary": summary,
        "pair_hybrid_witnesses": _jsonable(pair_records),
        "johnson_adversary_witnesses": _jsonable(johnson_records),
        "composition_falsification_witnesses": _jsonable(composition_records),
        "notes": list(config["notes"]),  # type: ignore[arg-type]
        "provenance": _git_provenance(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    config_bytes = arguments.config.read_bytes()
    config = load_config(arguments.config)
    artifact = run_panel(config)
    artifact["config"] = {
        "path": str(arguments.config),
        "sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
