from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path


def _small_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_name": "direct-search-script-test",
        "master_seed": 71,
        "trials": 1,
        "search": {
            "phase_qubits": 2,
            "max_attempts_per_output": 32,
            "verification_shots": 16,
            "verification_confidence": 0.1,
            "max_statevector_dimension": 256,
        },
        "scenarios": [
            {
                "name": "one_exact_above",
                "means": [1.0, 0.0],
                "threshold": 0.5,
                "expected_count": 1,
                "relation": "above",
                "metadata": {"purpose": "deterministic smoke"},
            }
        ],
        "notes": ["test configuration"],
    }


def _environment(repository: Path) -> dict[str, str]:
    environment = dict(os.environ)
    source = str(repository / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )
    return environment


def test_direct_search_cli_writes_unknown_oracle_resource_audit(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "direct.json"
    output_path = tmp_path / "result.json"
    config_path.write_text(json.dumps(_small_document()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_direct_search.py"),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        cwd=repository,
        env=_environment(repository),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "direct_unknown_oracle_threshold_search_execution"
    assert artifact["provenance"] == {
        "means_used_only_to_construct_oracle": True,
        "query_resources_are_executed_ledger_counts": True,
        "search_receives_no_marked_index_set": True,
        "statevector_simulation_is_not_hardware_evidence": True,
    }
    assert artifact["config_hash"]
    assert len(artifact["raw_execution_records"]) == 1
    record = artifact["raw_execution_records"][0]
    assert record["direct_unknown_oracle_search"]
    assert record["selected_indices"] == [0]
    assert "complete" in record["status"]
    assert record["executed_resources"]
    assert record["oracle_query_ledger"]["coherent_total"] > 0
    assert record["config_hash"] == artifact["config_hash"]
    assert "unknown canonical reward oracle" in completed.stdout
    assert "not hardware" in completed.stdout


def test_direct_search_cli_overrides_trials_seed_and_scenario(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    document = _small_document()
    second = deepcopy(document["scenarios"][0])  # type: ignore[index]
    second["name"] = "one_exact_below"
    second["means"] = [1.0, 0.0]
    second["relation"] = "below"
    document["scenarios"].append(second)  # type: ignore[union-attr]
    config_path = tmp_path / "direct.json"
    output_path = tmp_path / "result.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_direct_search.py"),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--trials",
            "2",
            "--seed",
            "123",
            "--scenario",
            "one_exact_below",
        ],
        cwd=repository,
        env=_environment(repository),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["resolved_config"]["trials"] == 2
    assert artifact["resolved_config"]["master_seed"] == 123
    assert [scenario["name"] for scenario in artifact["resolved_config"]["scenarios"]] == [
        "one_exact_below"
    ]
    assert len(artifact["raw_execution_records"]) == 2
    assert all(
        record["selected_indices"] == [1]
        for record in artifact["raw_execution_records"]
    )


def test_direct_search_cli_resumes_multi_output_search_to_terminal(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    document = _small_document()
    document["search"]["max_attempts_per_output"] = 1  # type: ignore[index]
    document["scenarios"] = [
        {
            "name": "two_exact_above",
            "means": [1.0, 1.0, 0.0],
            "threshold": 0.5,
            "expected_count": 2,
            "relation": "above",
        }
    ]
    config_path = tmp_path / "direct.json"
    output_path = tmp_path / "result.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_direct_search.py"),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        cwd=repository,
        env=_environment(repository),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output_path.read_text(encoding="utf-8"))[
        "raw_execution_records"
    ][0]
    assert record["runner_reached_terminal"]
    assert record["status"] == "complete_fixed_confidence_qpe_predicate"
    assert sorted(record["selected_indices"]) == [0, 1]
    assert record["result"]["attempts"] == 2


def test_direct_search_cli_rejects_unknown_fields_and_bool_numbers(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    for position, document in enumerate(
        (
            {**_small_document(), "unregistered": True},
            {**_small_document(), "trials": True},
        )
    ):
        config_path = tmp_path / f"invalid-{position}.json"
        config_path.write_text(json.dumps(document), encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "run_direct_search.py"),
                "--config",
                str(config_path),
                "--output",
                str(tmp_path / f"unused-{position}.json"),
            ],
            cwd=repository,
            env=_environment(repository),
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
        assert "direct-search experiment failed" in completed.stderr
