from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from qgapselect.coherent_experiments import (
    COHERENT_BACKEND,
    COHERENT_CLAIM_STATUS,
    aggregate_execution_records,
    derive_trial_seed,
    resolve_coherent_config,
    run_coherent_experiments,
)


def _small_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_name": "coherent-test",
        "master_seed": 91,
        "trials": 2,
        "quantiles": [0.5, 0.9],
        "controller": {
            "confidence": 0.1,
            "shots_per_round": 64,
            "max_boundary_rounds": 3,
            "batch_strategy": "known",
            "max_steps": 3,
        },
        "baselines": {
            "independent_per_arm": {
                "confidence": 0.1,
                "shots_per_round": 64,
                "max_rounds": 3,
            },
            "classical_uniform": {
                "samples_per_arm": 32,
                "confidence": 0.1,
            },
        },
        "scenarios": [
            {
                "name": "deterministic_best",
                "means": [1.0, 0.0],
                "k": 1,
                "metadata": {"purpose": "exact_state_smoke"},
            }
        ],
        "notes": ["small deterministic test"],
    }


def test_config_resolution_is_strict_hashable_and_override_aware() -> None:
    document = _small_document()
    first = resolve_coherent_config(document)
    second = resolve_coherent_config(document)
    overridden = resolve_coherent_config(document, trials_override=1, seed_override=7)

    assert first.sha256 == second.sha256
    assert first.as_dict() == second.as_dict()
    assert overridden.trials == 1
    assert overridden.master_seed == 7
    assert overridden.sha256 != first.sha256

    invalid = dict(document)
    invalid["unexpected"] = True
    with pytest.raises(ValueError, match="unknown top-level"):
        resolve_coherent_config(invalid)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("controller", "confidence"), True, "JSON number"),
        (("controller", "batch_strategy"), 1, "non-empty string"),
        (("scenarios", 0, "name"), 7, "non-empty string"),
        (("scenarios", 0, "means", 0), "1.0", "JSON number"),
        (("quantiles", 0), False, "JSON number"),
        (("experiment_name",), 99, "non-empty string"),
    ],
)
def test_config_rejects_implicit_scalar_coercions(
    path: tuple[str | int, ...], value: object, message: str
) -> None:
    document = deepcopy(_small_document())
    target: object = document
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=message):
        resolve_coherent_config(document)


def test_trial_seeds_are_stable_and_method_separated() -> None:
    one = derive_trial_seed(4, "case", 0, "coherent")
    assert one == derive_trial_seed(4, "case", 0, "coherent")
    assert one != derive_trial_seed(4, "case", 1, "coherent")
    assert one != derive_trial_seed(4, "case", 0, "classical")


def test_aggregate_uses_executed_fields_and_does_not_impute_null_resources() -> None:
    records = [
        {
            "success": True,
            "certified_success": True,
            "heuristic_output_match": True,
            "certificate_valid": True,
            "failure": {"occurred": False, "reason": None},
            "executed_resources": {
                "oracle_queries": 4,
                "classical_queries": 0,
                "gates": 8,
                "depth": 3,
                "total_qubits": 4,
                "workspace_qubits": 1,
                "uncompute_residual": 0.0,
            },
        },
        {
            "success": False,
            "certified_success": False,
            "heuristic_output_match": True,
            "certificate_valid": False,
            "failure": {"occurred": True, "reason": "certificate_not_obtained"},
            "executed_resources": {
                "oracle_queries": 2,
                "classical_queries": 0,
                "gates": None,
                "depth": None,
                "total_qubits": None,
                "workspace_qubits": None,
                "uncompute_residual": None,
            },
        },
    ]

    aggregate = aggregate_execution_records(records, (0.5, 0.9))

    assert aggregate["success_rate"] == 0.5
    assert aggregate["certified_success_rate"] == 0.5
    assert aggregate["heuristic_output_match_rate"] == 1.0
    assert aggregate["executed_resources"]["oracle_queries"]["mean"] == 3.0
    assert aggregate["executed_resources"]["oracle_queries"]["observations"] == 2
    assert aggregate["executed_resources"]["gates"]["observations"] == 1
    assert aggregate["failure_reason_counts"] == {"certificate_not_obtained": 1}
    assert "candidate" not in json.dumps(aggregate).lower()


def test_exact_state_pipeline_executes_all_methods_and_separates_theory() -> None:
    config = resolve_coherent_config(_small_document())
    report = run_coherent_experiments(config)
    raw = report["raw_execution_records"]

    assert report["artifact_type"] == "coherent_exact_state_execution"
    assert report["claim_status"] == COHERENT_CLAIM_STATUS
    assert report["config_hash"] == config.sha256
    assert report["provenance"]["raw_records_are_aggregate_source"]
    assert len(raw) == config.trials * len(config.scenarios) * 3
    assert {record["method"] for record in raw} == {
        "coherent_certificate_enumeration",
        "independent_per_arm_boundary",
        "classical_uniform",
    }
    assert all(record["config_hash"] == config.sha256 for record in raw)
    assert all("candidate_theory" not in record for record in raw)
    assert set(report["candidate_theory_reference"]) == {"deterministic_best"}

    coherent = [
        record
        for record in raw
        if record["method"] == "coherent_certificate_enumeration"
    ]
    independent = [
        record for record in raw if record["method"] == "independent_per_arm_boundary"
    ]
    classical = [record for record in raw if record["method"] == "classical_uniform"]
    assert all(record["backend"] == COHERENT_BACKEND for record in coherent)
    assert all(record["executed_resources"]["oracle_queries"] > 0 for record in coherent)
    assert all(record["executed_resources"]["gates"] > 0 for record in coherent)
    assert all(record["executed_resources"]["uncompute_residual"] <= 1e-9 for record in coherent)
    assert all(record["executed_resources"]["oracle_queries"] > 0 for record in independent)
    assert all(record["executed_resources"]["classical_queries"] == 64 for record in classical)
    assert all(record["success"] for record in coherent + independent + classical)
    assert all(record["certificate_valid"] for record in coherent + independent + classical)


def test_exact_state_report_is_reproducible_for_fixed_config() -> None:
    config = resolve_coherent_config(_small_document(), trials_override=1)

    assert run_coherent_experiments(config) == run_coherent_experiments(config)


def test_coherent_cli_writes_hashed_execution_artifact(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "coherent.json"
    output_path = tmp_path / "report.json"
    config_path.write_text(json.dumps(_small_document()), encoding="utf-8")
    environment = dict(os.environ)
    source = str(repository / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_coherent.py"),
            "--config",
            str(config_path),
            "--trials",
            "1",
            "--output",
            str(output_path),
        ],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact["config_hash"]
    assert artifact["resolved_config"]["trials"] == 1
    assert len(artifact["raw_execution_records"]) == 3
    assert "not_hardware" in artifact["claim_status"]
    assert "candidate_theory_reference" in artifact
    assert "wrote 3 execution records" in completed.stdout
