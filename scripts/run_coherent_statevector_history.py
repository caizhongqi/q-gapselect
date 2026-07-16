#!/usr/bin/env python3
"""Run the tiny exact-state coherent activity-history semantics panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import operator
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from qgapselect.coherent import CanonicalRyStatevectorOracle  # noqa: E402
from qgapselect.coherent_activity_history_statevector import (  # noqa: E402
    CLAIM_SCOPE,
    CoherentHistoryStatevectorConfig,
    run_coherent_activity_history_statevector,
)

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "q_gapselect_exact_state_coherent_activity_history"
DEFAULT_CONFIG = REPOSITORY / "configs" / "coherent_statevector_history.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "coherent_statevector_history.json"


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


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


def load_config(path: Path) -> dict[str, object]:
    document = _mapping(json.loads(path.read_text(encoding="utf-8")), "config")
    if _integer(document.get("schema_version"), "schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    levels = tuple(
        _integer(value, "phase qubits", minimum=1)
        for value in _sequence(
            document.get("phase_qubits_by_level"), "phase_qubits_by_level"
        )
    )
    cases: list[dict[str, object]] = []
    for raw in _sequence(document.get("cases"), "cases"):
        case = _mapping(raw, "case")
        means = tuple(
            _number(value, "case mean")
            for value in _sequence(case.get("means"), "case means")
        )
        if not means or any(not 0.0 <= value <= 1.0 for value in means):
            raise ValueError("case means must be non-empty and lie in [0, 1]")
        cases.append(
            {
                "case_id": _string(case.get("case_id"), "case_id"),
                "means": means,
                "k": _integer(case.get("k"), "case k", minimum=1),
            }
        )
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("cases must be non-empty with unique IDs")
    notes = tuple(
        _string(value, "note")
        for value in _sequence(document.get("notes"), "notes")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": _string(document.get("experiment_name"), "experiment_name"),
        "master_seed": _integer(document.get("master_seed"), "master_seed"),
        "repetitions": _integer(document.get("repetitions"), "repetitions", minimum=1),
        "phase_qubits_by_level": levels,
        "boundary_phase_qubits": _integer(
            document.get("boundary_phase_qubits"),
            "boundary_phase_qubits",
            minimum=1,
        ),
        "boundary_shots": _integer(
            document.get("boundary_shots"), "boundary_shots", minimum=1
        ),
        "minimum_boundary_samples_per_arm": _integer(
            document.get("minimum_boundary_samples_per_arm"),
            "minimum_boundary_samples_per_arm",
            minimum=1,
        ),
        "cleanup_tolerance": _number(
            document.get("cleanup_tolerance"), "cleanup_tolerance"
        ),
        "max_statevector_dimension": _integer(
            document.get("max_statevector_dimension"),
            "max_statevector_dimension",
            minimum=1,
        ),
        "cases": tuple(cases),
        "notes": notes,
    }


def _derived_seed(master_seed: int, case_id: str, repetition: int) -> int:
    payload = (
        f"qgapselect.coherent-statevector-history.v1\0{master_seed}\0"
        f"{case_id}\0{repetition}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


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


def _result_document(case_id: str, repetition: int, seed: int, result: object) -> dict[str, object]:
    return {
        "case_id": case_id,
        "repetition": repetition,
        "measurement_seed": seed,
        "boundary": {
            "lower": result.boundary.lower,
            "upper": result.boundary.upper,
            "center": result.boundary.center,
            "complete": result.boundary.complete,
            "status": result.boundary.status,
            "finite_sample_confidence_proved": (
                result.boundary.finite_sample_confidence_proved
            ),
            "query_counts": dict(result.boundary.query_counts),
            "arm_histograms": [
                {
                    "arm": row.arm,
                    "samples": row.samples,
                    "folded_bin_counts": {
                        str(key): int(value)
                        for key, value in row.folded_bin_counts.items()
                    },
                    "modal_folded_bin": (
                        None if row.modal_folded_bin is None else int(row.modal_folded_bin)
                    ),
                    "modal_amplitude": (
                        None if row.modal_amplitude is None else float(row.modal_amplitude)
                    ),
                    "empirical_radius": (
                        None if row.empirical_radius is None else float(row.empirical_radius)
                    ),
                }
                for row in result.boundary.arm_histograms
            ],
        },
        "layers": [
            {
                "level": row.level,
                "phase_qubits": row.phase_qubits,
                "active_probability_before": row.active_probability_before,
                "active_probability_after": row.active_probability_after,
                "newly_stopped_probability": row.newly_stopped_probability,
                "existing_stopped_branch_residual": (
                    row.existing_stopped_branch_residual
                ),
                "predicate_workspace_residual": row.predicate_workspace_residual,
                "control_workspace_residual": row.control_workspace_residual,
                "phase_reward_workspace_residual": (
                    row.phase_reward_workspace_residual
                ),
                "norm_error": row.norm_error,
                "cleanup_passed": row.cleanup_passed,
                "query_counts": dict(row.query_counts),
                "depth": row.depth,
                "gate_counts": dict(row.gate_counts),
            }
            for row in result.layers
        ],
        "resources": {
            "query_counts": dict(result.resources.query_counts),
            "boundary_query_counts": dict(result.resources.boundary_query_counts),
            "history_query_counts": dict(result.resources.history_query_counts),
            "gate_counts": dict(result.resources.gate_counts),
            "depth": result.resources.depth,
            "qubits": result.resources.qubits,
            "register_dimensions": dict(result.resources.register_dimensions),
            "retained_statevector_dimension": (
                result.resources.retained_statevector_dimension
            ),
            "peak_statevector_dimension": result.resources.peak_statevector_dimension,
        },
        "active_probability": result.active_probability,
        "stop_probabilities": [float(value) for value in result.stop_probabilities],
        "output_mask_probabilities": [
            float(value) for value in result.output_mask_probabilities
        ],
        "direct_output_write_executed": result.direct_output_write_executed,
        "direct_multi_output_complete": result.direct_multi_output_complete,
        "certificate_issued": result.certificate_issued,
        "blockers": list(result.blockers),
        "status": result.status,
        "quantum_advantage_claimable": result.quantum_advantage_claimable,
    }


def run_experiment(config: Mapping[str, object]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for case in config["cases"]:
        for repetition in range(int(config["repetitions"])):
            seed = _derived_seed(int(config["master_seed"]), str(case["case_id"]), repetition)
            oracle = CanonicalRyStatevectorOracle(case["means"], seed=seed)
            execution_config = CoherentHistoryStatevectorConfig(
                phase_qubits_by_level=tuple(config["phase_qubits_by_level"]),
                boundary_phase_qubits=int(config["boundary_phase_qubits"]),
                boundary_shots=int(config["boundary_shots"]),
                minimum_boundary_samples_per_arm=int(
                    config["minimum_boundary_samples_per_arm"]
                ),
                measurement_seed=seed,
                cleanup_tolerance=float(config["cleanup_tolerance"]),
                max_statevector_dimension=int(config["max_statevector_dimension"]),
            )
            result = run_coherent_activity_history_statevector(
                oracle,
                int(case["k"]),
                config=execution_config,
            )
            records.append(
                _result_document(str(case["case_id"]), repetition, seed, result)
            )
    completed = [row for row in records if row["boundary"]["complete"]]
    layers = [layer for row in records for layer in row["layers"]]
    resolved_config = {
        **dict(config),
        "phase_qubits_by_level": list(config["phase_qubits_by_level"]),
        "cases": [
            {**dict(case), "means": list(case["means"])} for case in config["cases"]
        ],
        "notes": list(config["notes"]),
    }
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "experiment_name": config["experiment_name"],
        "claim_status": CLAIM_SCOPE,
        "resolved_config": resolved_config,
        "summary": {
            "record_count": len(records),
            "boundary_complete_count": len(completed),
            "executed_history_layer_count": len(layers),
            "all_executed_layers_cleanup_passed": all(
                bool(layer["cleanup_passed"]) for layer in layers
            ),
            "actual_coherent_index_execution_performed": True,
            "analytic_per_arm_iae_used": False,
            "direct_output_write_executed": all(
                bool(row["direct_output_write_executed"]) for row in completed
            ),
            "complete_direct_multi_output_count": sum(
                bool(row["direct_multi_output_complete"]) for row in records
            ),
            "certificate_count": sum(bool(row["certificate_issued"]) for row in records),
            "quantum_advantage_claimable": False,
            "llm_execution_performed": False,
            "hardware_execution_performed": False,
        },
        "records": records,
        "provenance": _git_provenance(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.repetitions is not None:
        config["repetitions"] = _integer(args.repetitions, "--repetitions", minimum=1)
    artifact = run_experiment(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    summary = artifact["summary"]
    sys.stdout.write(
        f"wrote exact-state coherent history to {args.output}\n"
        f"records={summary['record_count']} certificates={summary['certificate_count']}\n"
        "scope=coherent-index code-sanity; complete multi-output theorem remains blocked\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
