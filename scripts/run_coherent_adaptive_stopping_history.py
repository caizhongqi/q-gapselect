#!/usr/bin/env python3
"""Run the tiny true-coherent two-level stopping-history S3 panel."""

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

from qgapselect.coherent import CanonicalRyStatevectorOracle  # noqa: E402
from qgapselect.coherent_adaptive_stopping_history import (  # noqa: E402
    CLAIM_SCOPE,
    METHOD_ID,
    OUTPUT_INCONCLUSIVE,
    OUTPUT_MASK,
    TinyCoherentStoppingHistoryConfig,
    run_tiny_coherent_adaptive_stopping_history,
    run_tiny_inactive_level_subspace_audit,
)

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "q_gapselect_tiny_true_coherent_stopping_history_s3_panel"
DEFAULT_CONFIG = REPOSITORY / "configs" / "coherent_adaptive_stopping_history.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "coherent_adaptive_stopping_history.json"
REQUIRED_ROLES = frozenset(
    {
        "exact_grid_first_stop",
        "exact_grid_second_stop",
        "exact_grid_arm1_winner",
        "exact_grid_tie",
        "off_grid_fail_closed",
    }
)
EXPECTED_HISTORIES = frozenset({"10", "01", "00", "mixed"})


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


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        document = {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
        if hasattr(value, "passed") and "passed" not in document:
            document["passed"] = bool(value.passed)
        return document
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
    allowed_root = {
        "schema_version",
        "experiment_name",
        "algorithm_parameters",
        "cases",
        "notes",
    }
    unknown_root = sorted(set(root) - allowed_root)
    if unknown_root:
        raise ValueError(f"unknown root fields: {unknown_root}")
    if _integer(root.get("schema_version"), "schema_version", minimum=1) != 1:
        raise ValueError("schema_version must be 1")

    parameters = _mapping(root.get("algorithm_parameters"), "algorithm_parameters")
    allowed_parameters = {"cleanup_tolerance", "max_statevector_dimension"}
    unknown_parameters = sorted(set(parameters) - allowed_parameters)
    if unknown_parameters:
        raise ValueError(f"unknown algorithm_parameters fields: {unknown_parameters}")
    resolved_parameters = {
        "cleanup_tolerance": _number(
            parameters.get("cleanup_tolerance"),
            "algorithm_parameters.cleanup_tolerance",
        ),
        "max_statevector_dimension": _integer(
            parameters.get("max_statevector_dimension"),
            "algorithm_parameters.max_statevector_dimension",
            minimum=1,
        ),
    }
    TinyCoherentStoppingHistoryConfig(**resolved_parameters)

    allowed_case = {
        "case_id",
        "role",
        "means",
        "expected_output_status",
        "expected_history",
    }
    cases: list[dict[str, object]] = []
    for index, raw_case in enumerate(_sequence(root.get("cases"), "cases")):
        case = _mapping(raw_case, f"cases[{index}]")
        unknown_case = sorted(set(case) - allowed_case)
        if unknown_case:
            raise ValueError(f"unknown cases[{index}] fields: {unknown_case}")
        means = tuple(
            _number(mean, f"cases[{index}].means")
            for mean in _sequence(case.get("means"), f"cases[{index}].means")
        )
        if len(means) != 2 or any(not 0.0 <= mean <= 1.0 for mean in means):
            raise ValueError("each case requires exactly two means in [0, 1]")
        role = _string(case.get("role"), f"cases[{index}].role")
        if role not in REQUIRED_ROLES:
            raise ValueError(f"cases[{index}].role is not registered")
        status = _string(
            case.get("expected_output_status"),
            f"cases[{index}].expected_output_status",
        )
        if status not in {OUTPUT_MASK, OUTPUT_INCONCLUSIVE}:
            raise ValueError(f"cases[{index}].expected_output_status is invalid")
        expected_history = _string(
            case.get("expected_history"), f"cases[{index}].expected_history"
        )
        if expected_history not in EXPECTED_HISTORIES:
            raise ValueError(f"cases[{index}].expected_history is invalid")
        cases.append(
            {
                "case_id": _string(case.get("case_id"), f"cases[{index}].case_id"),
                "role": role,
                "means": means,
                "expected_output_status": status,
                "expected_history": expected_history,
            }
        )
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("cases must be non-empty with unique case_id values")
    roles = [str(case["role"]) for case in cases]
    if len(set(roles)) != len(roles) or set(roles) != REQUIRED_ROLES:
        raise ValueError("cases must contain each required role exactly once")
    notes = tuple(
        _string(note, f"notes[{index}]")
        for index, note in enumerate(_sequence(root.get("notes", []), "notes"))
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": _string(root.get("experiment_name"), "experiment_name"),
        "algorithm_parameters": resolved_parameters,
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

    status = run("status", "--porcelain")
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_tree": run("rev-parse", "HEAD^{tree}"),
        "source_tree_dirty_at_execution": bool(status),
        "source_status_capture": "before_true_coherent_s3_artifact_write",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }


def _truth(means: tuple[float, float]) -> dict[str, object]:
    strict = means[0] != means[1]
    winner = 0 if means[0] > means[1] else 1
    return {
        "strict_boundary": strict,
        "membership_mask": (1 << winner) if strict else None,
    }


def _history_matches(expected: str, probabilities: Mapping[str, float]) -> bool:
    if expected == "mixed":
        return sum(value > 1e-6 for value in probabilities.values()) >= 2
    return probabilities[expected] >= 1.0 - 1e-10


def run_experiment(config: Mapping[str, object]) -> dict[str, object]:
    parameters = _mapping(config["algorithm_parameters"], "algorithm_parameters")
    records: list[dict[str, object]] = []
    for raw_case in config["cases"]:
        case = _mapping(raw_case, "case")
        means = tuple(float(mean) for mean in case["means"])
        oracle = CanonicalRyStatevectorOracle(means)
        result = run_tiny_coherent_adaptive_stopping_history(
            oracle,
            config=TinyCoherentStoppingHistoryConfig(
                cleanup_tolerance=float(parameters["cleanup_tolerance"]),
                max_statevector_dimension=int(parameters["max_statevector_dimension"]),
            ),
        )
        truth = _truth(means)
        exact = (
            result.membership_mask == truth["membership_mask"]
            if result.output_status == OUTPUT_MASK
            else None
        )
        expected_matched = (
            result.output_status == case["expected_output_status"]
            and _history_matches(case["expected_history"], result.history.probabilities)
            and (exact is True if result.output_status == OUTPUT_MASK else True)
        )
        records.append(
            {
                "case_id": case["case_id"],
                "role": case["role"],
                "expected_output_status": case["expected_output_status"],
                "expected_history": case["expected_history"],
                "algorithm_inputs": {
                    "oracle": "opaque_canonical_ry_statevector_oracle_handle",
                    "cleanup_tolerance": parameters["cleanup_tolerance"],
                    "max_statevector_dimension": parameters[
                        "max_statevector_dimension"
                    ],
                    "answer_gap_boundary_family_schedule_history_supplied": False,
                },
                "trusted_fixture_and_scoring": {
                    "means": list(means),
                    "truth": truth,
                    "diagnostic_mask_exact": exact,
                },
                "result": _jsonable(result),
                "expected_behavior_matched": expected_matched,
            }
        )

    by_role = {str(row["role"]): row for row in records}
    off_grid_means = tuple(
        float(mean)
        for mean in by_role["off_grid_fail_closed"]["trusted_fixture_and_scoring"][
            "means"
        ]
    )
    inactive_subspace_audit = run_tiny_inactive_level_subspace_audit(
        CanonicalRyStatevectorOracle(off_grid_means),
        config=TinyCoherentStoppingHistoryConfig(
            cleanup_tolerance=float(parameters["cleanup_tolerance"]),
            max_statevector_dimension=int(parameters["max_statevector_dimension"]),
        ),
    )
    assertions = {
        "required_panel_roles_present_exactly_once": (
            set(by_role) == REQUIRED_ROLES and len(by_role) == len(records)
        ),
        "all_preregistered_behaviors_matched": all(
            row["expected_behavior_matched"] for row in records
        ),
        "all_exact_grid_winner_cases_complete_clean_and_exact": all(
            by_role[role]["result"]["output_status"] == OUTPUT_MASK
            and by_role[role]["result"]["resources"]["cleanup"]["passed"]
            and by_role[role]["trusted_fixture_and_scoring"][
                "diagnostic_mask_exact"
            ]
            for role in (
                "exact_grid_first_stop",
                "exact_grid_second_stop",
                "exact_grid_arm1_winner",
            )
        ),
        "exact_tie_cleans_and_fails_closed": (
            by_role["exact_grid_tie"]["result"]["output_status"]
            == OUTPUT_INCONCLUSIVE
            and by_role["exact_grid_tie"]["result"]["resources"]["cleanup"][
                "passed"
            ]
        ),
        "off_grid_entangles_and_fails_closed": (
            by_role["off_grid_fail_closed"]["result"]["output_status"]
            == OUTPUT_INCONCLUSIVE
            and not by_role["off_grid_fail_closed"]["result"]["resources"][
                "cleanup"
            ]["passed"]
        ),
        "all_query_ledgers_reconcile_to_176": all(
            row["result"]["resources"]["query_ledger"]["reconciled"]
            and row["result"]["resources"]["query_ledger"]["query_counts"][
                "coherent_total"
            ]
            == 176
            for row in records
        ),
        "runtime_tag_ledger_reconciles_level0_28_level1_60_one_way": all(
            len(row["result"]["resources"]["query_ledger"][
                "per_level_runtime_records"
            ])
            == 2
            and row["result"]["resources"]["query_ledger"][
                "per_level_runtime_records"
            ][0]["runtime_derived_one_way_counts"]["coherent_total"]
            == 28
            and row["result"]["resources"]["query_ledger"][
                "per_level_runtime_records"
            ][1]["runtime_derived_one_way_counts"]["coherent_total"]
            == 60
            and all(
                level["full_replay_reconciled"]
                and level["one_way_reconciled"]
                for level in row["result"]["resources"]["query_ledger"][
                    "per_level_runtime_records"
                ]
            )
            for row in records
        ),
        "all_later_level_oracles_active_controlled": all(
            row["result"]["history"][
                "later_level_oracles_controlled_by_active_flag"
            ]
            for row in records
        ),
        "all_durable_copies_and_full_replays_executed": all(
            row["result"]["durable_output"]["scratch_to_durable_copy_executed"]
            and row["result"]["durable_output"]["full_history_replay_executed"]
            for row in records
        ),
        "branch_rms_never_reported_as_executed_saving": all(
            not row["result"]["resources"]["query_ledger"][
                "branch_rms_is_executed_saving"
            ]
            for row in records
        ),
        "inactive_clean_basis_identity_witness_passes": (
            inactive_subspace_audit.clean_identity_witness_passed
            and inactive_subspace_audit.clean_query_ledger_reconciled
        ),
        "inactive_dirty_work_negative_control_activates": (
            inactive_subspace_audit.dirty_negative_control_activated
            and inactive_subspace_audit.dirty_query_ledger_reconciled
        ),
        "no_certificate_issued": all(
            not row["result"]["certificate"]["issued"] for row in records
        ),
        "no_quantum_advantage_claimed": all(
            not row["result"]["quantum_advantage_claimable"] for row in records
        ),
    }
    assertions["all_assertions_passed"] = all(assertions.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "method_id": METHOD_ID,
        "claim_scope": CLAIM_SCOPE,
        "claim_status": (
            "tiny_true_coherent_history_copy_replay_supported_generic_off_grid_"
            "and_variable_time_theory_blocked"
        ),
        "experiment_name": config["experiment_name"],
        "resolved_config": _jsonable(config),
        "summary": {
            "case_count": len(records),
            "exact_grid_complete_count": sum(
                row["result"]["output_status"] == OUTPUT_MASK for row in records
            ),
            "fail_closed_count": sum(
                row["result"]["output_status"] == OUTPUT_INCONCLUSIVE
                for row in records
            ),
            "executed_queries_per_case": 176,
            "runtime_tag_derived_one_way_level_queries": [28, 60],
            "true_coherent_stopping_history_unitary_implemented": True,
            "later_level_active_control_implemented": True,
            "durable_copy_and_full_replay_implemented": True,
            "branch_rms_is_theorem_target_only": True,
            "generic_off_grid_cleanup_proved": False,
            "variable_time_query_speedup_proved": False,
            "new_query_upper_bound_proved": False,
            "matching_lower_bound_proved": False,
            "quantum_advantage_claimable": False,
            "ccf_a_claimable": False,
        },
        "aggregate_assertions": assertions,
        "records": records,
        "inactive_level_clean_dirty_subspace_audit": _jsonable(
            inactive_subspace_audit
        ),
        "notes": list(config["notes"]),
        "claim_boundaries": (
            records[0]["result"]["claim_boundary"] if records else {}
        ),
    }


def build_artifact(config_path: Path) -> dict[str, object]:
    config_bytes = config_path.read_bytes()
    artifact = run_experiment(load_config(config_path))
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    artifact["config_hash"] = config_hash
    artifact["provenance"] = {
        "config_path": str(config_path.resolve()),
        "config_sha256": config_hash,
        **_git_provenance(),
    }
    return artifact


def write_artifact(artifact: Mapping[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            artifact,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        artifact = build_artifact(args.config)
        output = write_artifact(artifact, args.output)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"true-coherent stopping-history S3 error: {error}") from error
    summary = _mapping(artifact["summary"], "summary")
    assertions = _mapping(artifact["aggregate_assertions"], "assertions")
    print(f"wrote true-coherent stopping-history S3 panel to {output}")
    print(
        f"cases={summary['case_count']} "
        f"exact_grid_complete={summary['exact_grid_complete_count']} "
        f"fail_closed={summary['fail_closed_count']} "
        f"queries_per_case={summary['executed_queries_per_case']}"
    )
    print(
        f"all_assertions_passed={assertions['all_assertions_passed']} "
        f"true_coherent_history="
        f"{summary['true_coherent_stopping_history_unitary_implemented']} "
        f"ccf_a_claimable={summary['ccf_a_claimable']}"
    )
    print(
        "Branch-RMS is a theorem target only; executed worst-case queries remain 176."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
