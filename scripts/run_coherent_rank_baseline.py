#!/usr/bin/env python3
"""Run the tiny measured-QPE direct Top-k semantic baseline panel."""

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
from qgapselect.coherent_rank_baseline import (  # noqa: E402
    CLAIM_SCOPE,
    CoherentRankBaselineConfig,
    run_coherent_rank_baseline,
)

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "q_gapselect_measured_qpe_direct_topk_semantic_baseline"
DEFAULT_CONFIG = REPOSITORY / "configs" / "coherent_rank_baseline.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "coherent_rank_baseline.json"


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
    allowed_root = {"schema_version", "experiment_name", "master_seed", "cases", "notes"}
    unknown = sorted(set(root) - allowed_root)
    if unknown:
        raise ValueError(f"unknown root fields: {unknown}")
    if _integer(root.get("schema_version"), "schema_version", minimum=1) != 1:
        raise ValueError("schema_version must be 1")
    cases: list[dict[str, object]] = []
    allowed_case = {
        "case_id",
        "means",
        "k",
        "phase_qubits",
        "shots_per_arm",
        "repetitions",
        "cleanup_tolerance",
        "max_statevector_dimension",
    }
    for index, raw in enumerate(_sequence(root.get("cases"), "cases")):
        case = _mapping(raw, f"cases[{index}]")
        unknown_case = sorted(set(case) - allowed_case)
        if unknown_case:
            raise ValueError(f"unknown cases[{index}] fields: {unknown_case}")
        means = tuple(
            _number(value, f"cases[{index}].means")
            for value in _sequence(case.get("means"), f"cases[{index}].means")
        )
        if not 2 <= len(means) <= 4 or any(not 0.0 <= value <= 1.0 for value in means):
            raise ValueError("each case requires 2--4 means in [0, 1]")
        tolerance = _number(
            case.get("cleanup_tolerance", 1e-12),
            f"cases[{index}].cleanup_tolerance",
        )
        if tolerance <= 0.0:
            raise ValueError("cleanup_tolerance must be positive")
        cases.append(
            {
                "case_id": _string(case.get("case_id"), f"cases[{index}].case_id"),
                "means": means,
                "k": _integer(case.get("k"), f"cases[{index}].k", minimum=1),
                "phase_qubits": _integer(
                    case.get("phase_qubits"), f"cases[{index}].phase_qubits", minimum=1
                ),
                "shots_per_arm": _integer(
                    case.get("shots_per_arm"),
                    f"cases[{index}].shots_per_arm",
                    minimum=1,
                ),
                "repetitions": _integer(
                    case.get("repetitions"), f"cases[{index}].repetitions", minimum=1
                ),
                "cleanup_tolerance": tolerance,
                "max_statevector_dimension": _integer(
                    case.get("max_statevector_dimension", 8_388_608),
                    f"cases[{index}].max_statevector_dimension",
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
        "master_seed": _integer(root.get("master_seed"), "master_seed"),
        "cases": tuple(cases),
        "notes": notes,
    }


def _seed(master_seed: int, case_id: str, repetition: int) -> int:
    payload = f"qgapselect.coherent-rank.v1\0{master_seed}\0{case_id}\0{repetition}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _git_provenance() -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ("git", *args), cwd=REPOSITORY, check=False, capture_output=True, text=True
        )
        return completed.stdout.strip() if completed.returncode == 0 else "unknown"

    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_tree": run("rev-parse", "HEAD^{tree}"),
        "source_tree_dirty_at_execution": bool(run("status", "--porcelain")),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def _truth(means: tuple[float, ...], k: int) -> tuple[bool, tuple[int, ...]]:
    ranking = tuple(sorted(range(len(means)), key=lambda arm: (-means[arm], arm)))
    strict = means[ranking[k - 1]] > means[ranking[k]]
    return strict, tuple(sorted(ranking[:k]))


def run_experiment(config: Mapping[str, object]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for case in config["cases"]:
        means = tuple(case["means"])
        k = int(case["k"])
        if not 1 <= k < len(means):
            raise ValueError(f"case {case['case_id']!r} must satisfy 1 <= k < n")
        truth_strict, truth_set = _truth(means, k)
        for repetition in range(int(case["repetitions"])):
            seed = _seed(int(config["master_seed"]), str(case["case_id"]), repetition)
            oracle = CanonicalRyStatevectorOracle(means, seed=seed)
            result = run_coherent_rank_baseline(
                oracle,
                k,
                config=CoherentRankBaselineConfig(
                    phase_qubits=int(case["phase_qubits"]),
                    shots_per_arm=int(case["shots_per_arm"]),
                    measurement_seed=seed,
                    cleanup_tolerance=float(case["cleanup_tolerance"]),
                    max_statevector_dimension=int(case["max_statevector_dimension"]),
                ),
            )
            output_set = tuple(
                index for index, selected in enumerate(result.membership_bits) if selected
            )
            output_exact = truth_strict and output_set == truth_set
            result_document = _jsonable(result)
            result_document["resources"]["cleanup"]["passed"] = (
                result.resources.cleanup.passed
            )
            expected_queries = len(means) * int(case["shots_per_arm"]) * (
                2 ** (int(case["phase_qubits"]) + 1) - 1
            )
            records.append(
                {
                    "case_id": case["case_id"],
                    "repetition": repetition,
                    "seed": seed,
                    "n": len(means),
                    "k": k,
                    "truth_strict": truth_strict,
                    "truth_top_k": list(truth_set) if truth_strict else None,
                    "output_exact": output_exact,
                    "certified_exact": output_exact and result.certificate_issued,
                    "query_formula_reconciled": (
                        result.resources.oracle_queries == expected_queries
                    ),
                    "result": result_document,
                }
            )
    strict_rows = [row for row in records if row["truth_strict"]]
    tie_rows = [row for row in records if not row["truth_strict"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "claim_scope": CLAIM_SCOPE,
        "experiment_name": config["experiment_name"],
        "provenance": _git_provenance(),
        "resolved_config": _jsonable(config),
        "summary": {
            "record_count": len(records),
            "strict_attempt_count": len(strict_rows),
            "tie_negative_control_count": len(tie_rows),
            "direct_multi_output_complete_rate_on_strict": sum(
                row["result"]["direct_multi_output_complete"] for row in strict_rows
            )
            / len(strict_rows),
            "exact_output_rate_on_strict": sum(row["output_exact"] for row in strict_rows)
            / len(strict_rows),
            "certified_exact_rate_on_strict": sum(
                row["certified_exact"] for row in strict_rows
            )
            / len(strict_rows),
            "tie_fail_closed_rate": sum(
                not row["result"]["direct_multi_output_complete"] for row in tie_rows
            )
            / len(tie_rows),
            "all_rank_cleanup_passed": all(
                row["result"]["resources"]["cleanup"]["passed"] for row in records
            ),
            "all_query_formulas_reconciled": all(
                row["query_formula_reconciled"] for row in records
            ),
            "qram_assumed": any(
                row["result"]["resources"]["qram_assumed"] for row in records
            ),
            "end_to_end_coherent": False,
            "finite_sample_confidence_proved": False,
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
    artifact = run_experiment(load_config(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = artifact["summary"]
    print(
        "Measured-QPE rank semantic baseline: "
        f"records={summary['record_count']}, "
        f"exact={summary['exact_output_rate_on_strict']:.3f}, "
        f"tie_reject={summary['tie_fail_closed_rate']:.3f}."
    )
    print(
        "This measured/reset baseline has no finite-shot certificate or end-to-end "
        "coherence and cannot support a quantum-advantage claim."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
