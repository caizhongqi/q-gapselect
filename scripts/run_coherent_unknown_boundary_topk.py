#!/usr/bin/env python3
"""Run the exact-state coherent unknown-boundary Top-k S2 audit panel."""

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
from qgapselect.coherent_unknown_boundary_topk import (  # noqa: E402
    CLAIM_SCOPE,
    CoherentUnknownBoundaryTopKConfig,
    run_coherent_unknown_boundary_topk,
)

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "q_gapselect_coherent_unknown_boundary_topk_s2_panel"
DEFAULT_CONFIG = REPOSITORY / "configs" / "coherent_unknown_boundary_topk.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "coherent_unknown_boundary_topk.json"
REQUIRED_ROLES = frozenset(
    {
        "on_grid_two_arm",
        "on_grid_three_arm",
        "off_grid_two_arm",
        "exact_tie",
    }
)
EXPECTED_BEHAVIORS = frozenset({"complete", "entanglement_fail_closed", "tie_fail_closed"})


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
    allowed_root = {"schema_version", "experiment_name", "cases", "notes"}
    unknown_root = sorted(set(root) - allowed_root)
    if unknown_root:
        raise ValueError(f"unknown root fields: {unknown_root}")
    if _integer(root.get("schema_version"), "schema_version", minimum=1) != 1:
        raise ValueError("schema_version must be 1")

    allowed_case = {
        "case_id",
        "role",
        "expected_behavior",
        "means",
        "k",
        "phase_qubits",
        "cleanup_tolerance",
        "max_statevector_dimension",
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
        expected_behavior = _string(
            case.get("expected_behavior"), f"cases[{index}].expected_behavior"
        )
        if expected_behavior not in EXPECTED_BEHAVIORS:
            raise ValueError(f"cases[{index}].expected_behavior is not a registered behavior")
        cleanup_tolerance = _number(
            case.get("cleanup_tolerance", 1e-10),
            f"cases[{index}].cleanup_tolerance",
        )
        if cleanup_tolerance <= 0.0:
            raise ValueError(f"cases[{index}].cleanup_tolerance must be positive")
        cases.append(
            {
                "case_id": _string(case.get("case_id"), f"cases[{index}].case_id"),
                "role": role,
                "expected_behavior": expected_behavior,
                "means": means,
                "k": k,
                "phase_qubits": _integer(
                    case.get("phase_qubits"),
                    f"cases[{index}].phase_qubits",
                    minimum=1,
                ),
                "cleanup_tolerance": cleanup_tolerance,
                "max_statevector_dimension": _integer(
                    case.get("max_statevector_dimension", 8_388_608),
                    f"cases[{index}].max_statevector_dimension",
                    minimum=1,
                ),
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
        "source_status_capture": "before_s2_artifact_write",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }


def _trusted_truth(means: tuple[float, ...], k: int) -> dict[str, object]:
    ranking = tuple(sorted(range(len(means)), key=lambda arm: (-means[arm], arm)))
    strict = means[ranking[k - 1]] > means[ranking[k]]
    top_k = tuple(sorted(ranking[:k])) if strict else None
    mask = sum(1 << arm for arm in top_k) if top_k is not None else None
    return {
        "strict_boundary": strict,
        "top_k_indices": list(top_k) if top_k is not None else None,
        "membership_mask": mask,
    }


def _grid_witness(mean: float, phase_qubits: int, tolerance: float) -> int | None:
    phase_bins = 1 << phase_qubits
    coordinate = math.asin(math.sqrt(mean)) * phase_bins / math.pi
    nearest = round(coordinate)
    reconstructed = math.sin(nearest * math.pi / phase_bins) ** 2
    return nearest if abs(reconstructed - mean) <= 10.0 * tolerance else None


def _output_distribution(
    probabilities: Mapping[int, float], n_arms: int
) -> list[dict[str, object]]:
    return [
        {
            "membership_mask": int(mask),
            "arm_order_membership_bits": [(int(mask) >> arm) & 1 for arm in range(n_arms)],
            "probability": float(probability),
        }
        for mask, probability in sorted(probabilities.items())
    ]


def _query_reconciliation(
    *, n_arms: int, phase_qubits: int, observed: Mapping[str, int]
) -> dict[str, object]:
    phase_bins = 1 << phase_qubits
    controlled_per_direction = 2 * n_arms * (phase_bins - 1)
    expected = {
        "forward": n_arms,
        "inverse": n_arms,
        "controlled_forward": controlled_per_direction,
        "controlled_inverse": controlled_per_direction,
        "coherent_total": 2 * n_arms * (2 * phase_bins - 1),
        "qram_queries": 0,
    }
    selected_observed = {key: int(observed.get(key, -1)) for key in expected}
    return {
        "formula": "Q = 2*n*(2^(phase_qubits+1)-1)",
        "derivation": (
            "n forward preparations + n inverse preparations + two QPE "
            "directions, each charging forward and inverse oracle calls for "
            "n*(2^phase_qubits-1) controlled Grover iterations"
        ),
        "parameters": {
            "n": n_arms,
            "phase_qubits": phase_qubits,
            "phase_bins": phase_bins,
        },
        "expected_query_counts": expected,
        "observed_query_counts": selected_observed,
        "reconciled": selected_observed == expected,
    }


def _behavior_matches(record: Mapping[str, object]) -> bool:
    result = _mapping(record["result"], "result")
    scoring = _mapping(record["trusted_scoring"], "trusted_scoring")
    cleanup = _mapping(record["cleanup_identity"], "cleanup_identity")
    expected = record["expected_behavior"]
    if expected == "complete":
        return bool(
            result["direct_multi_output_complete"]
            and cleanup["passed"]
            and scoring["output_exact_on_strict_relation"]
        )
    if expected == "entanglement_fail_closed":
        return bool(
            not result["direct_multi_output_complete"]
            and not cleanup["passed"]
            and cleanup["executed_garbage_probability"] > cleanup["cleanup_tolerance"]
            and result["status"] == "finite_qpe_output_entanglement_fail_closed"
        )
    return bool(
        not result["direct_multi_output_complete"]
        and cleanup["passed"]
        and not scoring["truth"]["strict_boundary"]
        and result["status"] == "coherent_discrete_boundary_not_strict_fail_closed"
    )


def run_experiment(config: Mapping[str, object]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for raw_case in config["cases"]:
        case = _mapping(raw_case, "case")
        means = tuple(float(mean) for mean in case["means"])
        k = int(case["k"])
        phase_qubits = int(case["phase_qubits"])
        tolerance = float(case["cleanup_tolerance"])
        oracle = CanonicalRyStatevectorOracle(means)
        result = run_coherent_unknown_boundary_topk(
            oracle,
            k,
            config=CoherentUnknownBoundaryTopKConfig(
                phase_qubits=phase_qubits,
                cleanup_tolerance=tolerance,
                max_statevector_dimension=int(case["max_statevector_dimension"]),
            ),
        )
        truth = _trusted_truth(means, k)
        output_indices = (
            [
                arm
                for arm in range(len(means))
                if result.membership_mask is not None and (result.membership_mask >> arm) & 1
            ]
            if result.membership_mask is not None
            else None
        )
        output_exact = (
            output_indices == truth["top_k_indices"] if truth["strict_boundary"] else None
        )
        cleanup = result.resources.cleanup
        result_document = _jsonable(result)
        result_document["resources"]["cleanup"]["passed"] = cleanup.passed
        record: dict[str, object] = {
            "case_id": case["case_id"],
            "role": case["role"],
            "expected_behavior": case["expected_behavior"],
            "algorithm_inputs": {
                "oracle": "opaque_canonical_ry_statevector_oracle_handle",
                "k": k,
                "phase_qubits": phase_qubits,
                "cleanup_tolerance": tolerance,
                "max_statevector_dimension": case["max_statevector_dimension"],
                "answer_dependent_inputs_supplied": False,
            },
            "trusted_fixture_and_scoring": {
                "means": list(means),
                "grid_coordinates": [
                    _grid_witness(mean, phase_qubits, tolerance) for mean in means
                ],
                "truth": truth,
                "returned_top_k_indices": output_indices,
                "output_exact_on_strict_relation": output_exact,
                "fail_closed": not result.direct_multi_output_complete,
            },
            "output_distribution": _output_distribution(
                result.boundary.output_mask_probabilities, len(means)
            ),
            "cleanup_identity": {
                "identity": "P_garbage = 1 - sum_y p_y^2",
                "predicted_garbage_probability": (cleanup.predicted_transient_nonzero_probability),
                "executed_garbage_probability": (cleanup.executed_transient_nonzero_probability),
                "prediction_residual": cleanup.prediction_residual,
                "output_collision_probability": cleanup.output_collision_probability,
                "output_reduced_purity": cleanup.output_reduced_purity,
                "purity_residual": cleanup.purity_residual,
                "cleanup_tolerance": cleanup.tolerance,
                "passed": cleanup.passed,
            },
            "query_formula_reconciliation": _query_reconciliation(
                n_arms=len(means),
                phase_qubits=phase_qubits,
                observed=result.resources.query_counts,
            ),
            "result": result_document,
        }
        # Use the shorter key throughout the executable assertions while keeping
        # the artifact label explicit about its trusted-only status.
        record["trusted_scoring"] = record.pop("trusted_fixture_and_scoring")
        record["expected_behavior_matched"] = _behavior_matches(record)
        records.append(record)

    records_by_role = {str(row["role"]): row for row in records}
    cleanup_identity_holds = all(
        float(row["cleanup_identity"]["prediction_residual"])
        <= 10.0 * float(row["cleanup_identity"]["cleanup_tolerance"])
        and float(row["cleanup_identity"]["purity_residual"])
        <= 10.0 * float(row["cleanup_identity"]["cleanup_tolerance"])
        for row in records
    )
    assertions = {
        "required_panel_roles_present_exactly_once": (
            set(records_by_role) == REQUIRED_ROLES and len(records_by_role) == len(records)
        ),
        "both_on_grid_cases_complete_clean_and_score_exact": all(
            bool(records_by_role[role]["expected_behavior_matched"])
            for role in ("on_grid_two_arm", "on_grid_three_arm")
        ),
        "off_grid_case_entangles_and_fails_closed": bool(
            records_by_role["off_grid_two_arm"]["expected_behavior_matched"]
        ),
        "exact_tie_cleans_and_fails_closed": bool(
            records_by_role["exact_tie"]["expected_behavior_matched"]
        ),
        "cleanup_collision_identity_holds_within_tolerance": cleanup_identity_holds,
        "all_query_formulas_reconciled": all(
            bool(row["query_formula_reconciliation"]["reconciled"]) for row in records
        ),
        "all_cases_match_preregistered_behavior": all(
            bool(row["expected_behavior_matched"]) for row in records
        ),
        "no_answer_dependent_algorithm_input": all(
            not bool(row["algorithm_inputs"]["answer_dependent_inputs_supplied"]) for row in records
        ),
        "no_qram_assumed": all(
            not bool(row["result"]["resources"]["qram_assumed"]) for row in records
        ),
        "no_elementary_gate_ledger_claimed": all(
            not bool(
                row["result"]["resources"]["elementary_gate_ledger_available"]
            )
            for row in records
        ),
        "no_transpiled_depth_claimed": all(
            not bool(row["result"]["resources"]["transpiled_depth_available"])
            and row["result"]["resources"]["transpiled_depth"] is None
            for row in records
        ),
        "no_compiled_ancilla_count_claimed": all(
            not bool(
                row["result"]["resources"]["compiled_ancilla_qubits_available"]
            )
            for row in records
        ),
        "no_certificate_issued": all(
            not bool(row["result"]["certificate_issued"]) for row in records
        ),
        "no_quantum_advantage_claimed": all(
            not bool(row["result"]["quantum_advantage_claimable"]) for row in records
        ),
    }
    assertions["all_assertions_passed"] = all(assertions.values())
    strict_records = [row for row in records if row["trusted_scoring"]["truth"]["strict_boundary"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "claim_scope": CLAIM_SCOPE,
        "claim_status": (
            "circuit_semantics_and_exact_queries_supported_no_transpilation_"
            "theory_advantage_not_claimable"
        ),
        "experiment_name": config["experiment_name"],
        "resolved_config": _jsonable(config),
        "summary": {
            "case_count": len(records),
            "strict_case_count": len(strict_records),
            "on_grid_complete_count": sum(
                bool(row["result"]["direct_multi_output_complete"])
                for row in records
                if str(row["role"]).startswith("on_grid_")
            ),
            "cleanup_pass_count": sum(bool(row["cleanup_identity"]["passed"]) for row in records),
            "fail_closed_count": sum(
                bool(row["trusted_scoring"]["fail_closed"]) for row in records
            ),
            "exact_output_rate_on_strict_cases": sum(
                row["trusted_scoring"]["output_exact_on_strict_relation"] is True
                for row in strict_records
            )
            / len(strict_records),
            "certificate_issued_count": sum(
                bool(row["result"]["certificate_issued"]) for row in records
            ),
            "end_to_end_coherent_tiny_reference": True,
            "generic_rounding_robustness_proved": False,
            "new_query_upper_bound_proved": False,
            "same_interface_composition_separation_proved": False,
            "matching_lower_bound_proved": False,
            "hardware_evidence": False,
            "elementary_gate_ledger_available": False,
            "transpiled_depth_available": False,
            "compiled_ancilla_qubits_available": False,
            "quantum_advantage_claimable": False,
            "ccf_a_claimable": False,
        },
        "aggregate_assertions": assertions,
        "records": records,
        "notes": list(config["notes"]),
        "claim_boundaries": {
            "supports": [
                "tiny exact-state end-to-end coherent rank-copy-uncompute semantics",
                "complete durable Top-k masks on two exact-grid strict fixtures",
                "the rank-copy cleanup collision identity on all four fixtures",
                "fail-closed behavior for generic off-grid entanglement and an exact tie",
                "an exact executed canonical-oracle query ledger with formula reconciliation",
                "executed NumPy kernel-macro counts, not circuit gate counts",
                "undecomposed logical macro counts and unsynthesised structural rank proxies",
                "declared register qubits and analytic statevector-size proxies",
            ],
            "does_not_support": [
                "a generic finite-QPE rounding theorem",
                "a correctness or confidence certificate",
                "a new query-complexity upper bound",
                "a same-interface composition separation",
                "a matching oracle lower bound",
                "an elementary-gate count or target gate-set decomposition",
                "logical, transpiled, scheduled, or hardware circuit depth",
                "post-decomposition ancilla or physical-qubit requirements",
                "hardware performance",
                "a new quantum advantage or CCF-A publication claim",
            ],
        },
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
        raise SystemExit(f"coherent unknown-boundary S2 error: {error}") from error
    summary = _mapping(artifact["summary"], "summary")
    assertions = _mapping(artifact["aggregate_assertions"], "assertions")
    print(f"wrote coherent unknown-boundary S2 panel to {output}")
    print(
        f"cases={summary['case_count']} "
        f"on_grid_complete={summary['on_grid_complete_count']} "
        f"cleanup_pass={summary['cleanup_pass_count']} "
        f"fail_closed={summary['fail_closed_count']}"
    )
    print(
        f"all_assertions_passed={assertions['all_assertions_passed']} "
        f"quantum_advantage_claimable={summary['quantum_advantage_claimable']} "
        f"ccf_a_claimable={summary['ccf_a_claimable']}"
    )
    print(
        "This is tiny exact-state circuit-semantic evidence, not a certificate, "
        "new quantum-advantage result, or CCF-A claim."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
