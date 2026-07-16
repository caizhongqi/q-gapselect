#!/usr/bin/env python3
"""Run the fail-closed adaptive unknown-boundary Top-k S3 audit panel."""

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

from qgapselect.adaptive_unknown_boundary_topk import (  # noqa: E402
    CLAIM_SCOPE,
    METHOD_ID,
    OUTPUT_INCONCLUSIVE,
    OUTPUT_MASK,
    AdaptiveUnknownBoundaryTopKConfig,
    run_adaptive_unknown_boundary_topk,
)
from qgapselect.coherent import CanonicalRyStatevectorOracle  # noqa: E402

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "q_gapselect_adaptive_unknown_boundary_topk_s3_panel"
DEFAULT_CONFIG = REPOSITORY / "configs" / "adaptive_unknown_boundary_topk.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "adaptive_unknown_boundary_topk.json"
REQUIRED_ROLES = frozenset(
    {
        "on_grid_fast",
        "off_grid_mid_precision",
        "off_grid_deep_precision",
        "exact_tie",
        "query_cap",
    }
)


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
    allowed_parameters = {
        "minimum_phase_qubits",
        "maximum_phase_qubits",
        "target_diagnostic_error_probability",
        "numerical_cleanup_tolerance",
        "max_statevector_dimension",
    }
    unknown_parameters = sorted(set(parameters) - allowed_parameters)
    if unknown_parameters:
        raise ValueError(f"unknown algorithm_parameters fields: {unknown_parameters}")
    resolved_parameters = {
        "minimum_phase_qubits": _integer(
            parameters.get("minimum_phase_qubits"),
            "algorithm_parameters.minimum_phase_qubits",
            minimum=1,
        ),
        "maximum_phase_qubits": _integer(
            parameters.get("maximum_phase_qubits"),
            "algorithm_parameters.maximum_phase_qubits",
            minimum=1,
        ),
        "target_diagnostic_error_probability": _number(
            parameters.get("target_diagnostic_error_probability"),
            "algorithm_parameters.target_diagnostic_error_probability",
        ),
        "numerical_cleanup_tolerance": _number(
            parameters.get("numerical_cleanup_tolerance"),
            "algorithm_parameters.numerical_cleanup_tolerance",
        ),
        "max_statevector_dimension": _integer(
            parameters.get("max_statevector_dimension"),
            "algorithm_parameters.max_statevector_dimension",
            minimum=1,
        ),
    }
    # Reuse the implementation validator for coupled bounds and probabilities.
    AdaptiveUnknownBoundaryTopKConfig(
        **resolved_parameters,
        max_canonical_oracle_queries=0,
    )

    allowed_case = {
        "case_id",
        "role",
        "means",
        "k",
        "max_canonical_oracle_queries",
        "expected_output_status",
        "expected_selected_phase_qubits",
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
        if not 2 <= len(means) <= 3 or any(not 0.0 <= mean <= 1.0 for mean in means):
            raise ValueError("each case requires two or three means in [0, 1]")
        k = _integer(case.get("k"), f"cases[{index}].k", minimum=1)
        if k >= len(means):
            raise ValueError(f"cases[{index}].k must satisfy 1 <= k < n")
        role = _string(case.get("role"), f"cases[{index}].role")
        if role not in REQUIRED_ROLES:
            raise ValueError(f"cases[{index}].role is not a registered panel role")
        expected_status = _string(
            case.get("expected_output_status"),
            f"cases[{index}].expected_output_status",
        )
        if expected_status not in {OUTPUT_MASK, OUTPUT_INCONCLUSIVE}:
            raise ValueError(f"cases[{index}].expected_output_status is invalid")
        selected = case.get("expected_selected_phase_qubits")
        if selected is not None:
            selected = _integer(
                selected,
                f"cases[{index}].expected_selected_phase_qubits",
                minimum=1,
            )
        if (expected_status == OUTPUT_MASK) != (selected is not None):
            raise ValueError("MASK requires a selected precision; INCONCLUSIVE requires null")
        cases.append(
            {
                "case_id": _string(case.get("case_id"), f"cases[{index}].case_id"),
                "role": role,
                "means": means,
                "k": k,
                "max_canonical_oracle_queries": _integer(
                    case.get("max_canonical_oracle_queries"),
                    f"cases[{index}].max_canonical_oracle_queries",
                ),
                "expected_output_status": expected_status,
                "expected_selected_phase_qubits": selected,
            }
        )
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("cases must be non-empty with unique case_id values")
    roles = [str(case["role"]) for case in cases]
    if len(set(roles)) != len(roles) or set(roles) != REQUIRED_ROLES:
        raise ValueError("cases must contain each required panel role exactly once")

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
        "source_status_capture": "before_s3_artifact_write",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }


def _trusted_truth(means: tuple[float, ...], k: int) -> dict[str, object]:
    ranking = tuple(sorted(range(len(means)), key=lambda arm: (-means[arm], arm)))
    strict = means[ranking[k - 1]] > means[ranking[k]]
    indices = tuple(sorted(ranking[:k])) if strict else None
    mask = sum(1 << arm for arm in indices) if indices is not None else None
    return {
        "strict_boundary": strict,
        "top_k_indices": list(indices) if indices is not None else None,
        "membership_mask": mask,
    }


def run_experiment(config: Mapping[str, object]) -> dict[str, object]:
    parameters = _mapping(config["algorithm_parameters"], "algorithm_parameters")
    records: list[dict[str, object]] = []
    for raw_case in config["cases"]:
        case = _mapping(raw_case, "case")
        means = tuple(float(mean) for mean in case["means"])
        k = int(case["k"])
        oracle = CanonicalRyStatevectorOracle(means)
        result = run_adaptive_unknown_boundary_topk(
            oracle,
            k,
            config=AdaptiveUnknownBoundaryTopKConfig(
                minimum_phase_qubits=int(parameters["minimum_phase_qubits"]),
                maximum_phase_qubits=int(parameters["maximum_phase_qubits"]),
                target_diagnostic_error_probability=float(
                    parameters["target_diagnostic_error_probability"]
                ),
                numerical_cleanup_tolerance=float(
                    parameters["numerical_cleanup_tolerance"]
                ),
                max_statevector_dimension=int(parameters["max_statevector_dimension"]),
                max_canonical_oracle_queries=int(
                    case["max_canonical_oracle_queries"]
                ),
            ),
        )
        truth = _trusted_truth(means, k)
        exact = (
            result.membership_mask == truth["membership_mask"]
            if result.output_status == OUTPUT_MASK and truth["strict_boundary"]
            else None
        )
        expected_matched = (
            result.output_status == case["expected_output_status"]
            and result.stopping_history.first_stop_phase_qubits
            == case["expected_selected_phase_qubits"]
            and (exact is True if result.output_status == OUTPUT_MASK else True)
        )
        records.append(
            {
                "case_id": case["case_id"],
                "role": case["role"],
                "expected_output_status": case["expected_output_status"],
                "expected_selected_phase_qubits": case[
                    "expected_selected_phase_qubits"
                ],
                "algorithm_inputs": {
                    "oracle": "opaque_canonical_ry_statevector_oracle_handle",
                    "k": k,
                    "public_precision_bounds": [
                        parameters["minimum_phase_qubits"],
                        parameters["maximum_phase_qubits"],
                    ],
                    "public_target_diagnostic_error_probability": parameters[
                        "target_diagnostic_error_probability"
                    ],
                    "public_numerical_cleanup_tolerance": parameters[
                        "numerical_cleanup_tolerance"
                    ],
                    "public_statevector_dimension_cap": parameters[
                        "max_statevector_dimension"
                    ],
                    "public_canonical_query_hard_cap": case[
                        "max_canonical_oracle_queries"
                    ],
                    "answer_gap_boundary_family_schedule_history_supplied": False,
                },
                "trusted_fixture_and_scoring": {
                    "means": list(means),
                    "truth": truth,
                    "diagnostic_mask_exact_on_strict_relation": exact,
                },
                "result": _jsonable(result),
                "expected_behavior_matched": expected_matched,
            }
        )

    by_role = {str(row["role"]): row for row in records}
    assertions = {
        "required_panel_roles_present_exactly_once": (
            set(by_role) == REQUIRED_ROLES and len(by_role) == len(records)
        ),
        "all_preregistered_behaviors_matched": all(
            bool(row["expected_behavior_matched"]) for row in records
        ),
        "off_grid_mid_stops_at_phase_three": (
            by_role["off_grid_mid_precision"]["result"]["stopping_history"][
                "first_stop_phase_qubits"
            ]
            == 3
        ),
        "off_grid_deep_stops_at_phase_four": (
            by_role["off_grid_deep_precision"]["result"]["stopping_history"][
                "first_stop_phase_qubits"
            ]
            == 4
        ),
        "tie_and_query_cap_fail_closed": all(
            by_role[role]["result"]["output_status"] == OUTPUT_INCONCLUSIVE
            for role in ("exact_tie", "query_cap")
        ),
        "all_executed_query_ledgers_reconciled": all(
            row["result"]["query_budget"]["budget_valid"] for row in records
        ),
        "all_hard_caps_respected": all(
            row["result"]["hard_cap_respected"] for row in records
        ),
        "no_answer_gap_boundary_family_schedule_history_input": all(
            not row["algorithm_inputs"][
                "answer_gap_boundary_family_schedule_history_supplied"
            ]
            for row in records
        ),
        "all_controllers_explicitly_classical": all(
            row["result"]["stopping_history"]["controller_is_classical"]
            for row in records
        ),
        "no_single_coherent_variable_time_unitary_claimed": all(
            not row["result"]["stopping_history"][
                "single_coherent_variable_time_unitary_implemented"
            ]
            for row in records
        ),
        "no_certificate_issued": all(
            not row["result"]["certificate"]["issued"] for row in records
        ),
        "no_qram_assumed": all(
            not row["result"]["query_budget"]["qram_assumed"] for row in records
        ),
        "no_quantum_advantage_claimed": all(
            not row["result"]["quantum_advantage_claimable"] for row in records
        ),
    }
    assertions["all_assertions_passed"] = all(assertions.values())
    mask_records = [row for row in records if row["result"]["output_status"] == OUTPUT_MASK]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "method_id": METHOD_ID,
        "claim_scope": CLAIM_SCOPE,
        "claim_status": (
            "adaptive_exact_state_diagnostic_supported_coherent_variable_time_"
            "unitary_and_theory_not_implemented"
        ),
        "experiment_name": config["experiment_name"],
        "resolved_config": _jsonable(config),
        "summary": {
            "case_count": len(records),
            "diagnostic_mask_count": len(mask_records),
            "inconclusive_count": len(records) - len(mask_records),
            "off_grid_diagnostic_mask_count": sum(
                str(row["role"]).startswith("off_grid_")
                and row["result"]["output_status"] == OUTPUT_MASK
                for row in records
            ),
            "all_mask_outputs_exact_under_trusted_scoring": all(
                row["trusted_fixture_and_scoring"][
                    "diagnostic_mask_exact_on_strict_relation"
                ]
                is True
                for row in mask_records
            ),
            "certificate_issued_count": 0,
            "independently_coherent_level_unitaries_executed": True,
            "single_coherent_variable_time_unitary_implemented": False,
            "coherent_history_cleanup_proved": False,
            "observable_stop_estimator_implemented": False,
            "generic_off_grid_correctness_proved": False,
            "new_query_upper_bound_proved": False,
            "matching_lower_bound_proved": False,
            "quantum_advantage_claimable": False,
            "ccf_a_claimable": False,
        },
        "aggregate_assertions": assertions,
        "records": records,
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
        raise SystemExit(f"adaptive unknown-boundary S3 error: {error}") from error
    summary = _mapping(artifact["summary"], "summary")
    assertions = _mapping(artifact["aggregate_assertions"], "assertions")
    print(f"wrote adaptive unknown-boundary S3 panel to {output}")
    print(
        f"cases={summary['case_count']} "
        f"diagnostic_masks={summary['diagnostic_mask_count']} "
        f"inconclusive={summary['inconclusive_count']}"
    )
    print(
        f"all_assertions_passed={assertions['all_assertions_passed']} "
        f"single_coherent_variable_time_unitary_implemented="
        f"{summary['single_coherent_variable_time_unitary_implemented']} "
        f"ccf_a_claimable={summary['ccf_a_claimable']}"
    )
    print(
        "The adaptive controller is exact-state classical simulation, not a "
        "coherent variable-time unitary or correctness certificate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
