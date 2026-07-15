from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _small_document(repository: Path) -> dict[str, object]:
    document = json.loads(
        (repository / "configs" / "quantum_benchmarks.json").read_text(
            encoding="utf-8"
        )
    )
    document["common"].update(  # type: ignore[union-attr]
        {
            "phase_qubits": 2,
            "verification_shots": 32,
            "max_attempts_per_output": 8,
            "max_statevector_dimension": 4096,
            "classical_shots_per_arm": 128,
            "boundary_shots_per_round": 32,
            "max_boundary_rounds": 3,
        }
    )
    document["unitary_validation"] = {
        "trials": 1,
        "cases": [{"name": "tiny", "means": [0.2, 0.8], "phase_qubits": 2}],
    }
    document["phase_grid"] = {
        "phase_qubits": [2],
        "threshold_angles": [0.7853981633974483],
        "relations": ["above"],
    }
    document["qpe_resolution"] = {
        "phase_qubits": [2],
        "angular_gaps": [0.19634954084936207],
        "relations": ["above"],
    }
    document["verifier_calibration"] = {
        "trials": 2,
        "phase_qubits": [2],
        "shots": [4],
        "confidence_values": [0.1],
        "angular_offsets": [-0.19634954084936207, 0.19634954084936207],
    }
    document["random_benchmarks"] = {
        "trials": 1,
        "families": ["endpoint_angular"],
        "cases": [{"n": 2, "k": 1}],
        "methods": [
            "direct_bbht",
            "independent_qpe_scan",
            "classical_threshold_scan",
        ],
        "relations": ["above"],
    }
    document["topk_comparison"] = {
        "trials": 1,
        "families": ["endpoint_angular"],
        "cases": [{"n": 2, "k": 1}],
        "methods": [
            "boundary_only_negative_control",
            "calibrated_direct_topk",
        ],
    }
    document["iterative_ae"].update(  # type: ignore[union-attr]
        {
            "trials": 1,
            "families": ["endpoint_angular"],
            "cases": [{"n": 2, "k": 1}],
            "relations": ["above"],
            "shots_per_round": 16,
            "max_rounds": 2,
            "max_grover_power": 1,
            "grid_points": 257,
        }
    )
    document["scheduler_sweep"] = {
        "trials": 1,
        "phase_qubits": 2,
        "growth_values": [1.2],
        "cases": [{"n": 2, "k": 1}],
    }
    document["diffusion_ablation"] = {
        "phase_qubits": [2],
        "iterations": [0, 1],
        "means": [0.8, 0.2],
        "threshold": 0.5,
        "relation": "above",
    }
    return document


def _environment(repository: Path) -> dict[str, str]:
    environment = dict(os.environ)
    source = str(repository / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )
    return environment


def test_quantum_benchmark_cli_runs_selected_suites_and_writes_claim_boundaries(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "quantum.json"
    output_path = tmp_path / "result.json"
    config_path.write_text(
        json.dumps(_small_document(repository)),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_quantum_benchmarks.py"),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--suite",
            "unitary_validation",
            "--suite",
            "random_benchmarks",
            "--suite",
            "failure_semantics",
            "--trials",
            "1",
            "--seed",
            "73",
        ],
        cwd=repository,
        env=_environment(repository),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "q_gapselect_executed_quantum_core_audit"
    assert artifact["selected_suites"] == [
        "unitary_validation",
        "random_benchmarks",
        "failure_semantics",
    ]
    assert artifact["resolved_config"]["master_seed"] == 73
    assert artifact["resolved_config"]["random_benchmarks"]["trials"] == 1
    assert artifact["config_hash"]
    provenance = artifact["provenance"]
    assert provenance["python_implementation"]
    assert provenance["python_version"]
    assert provenance["numpy_version"]
    assert provenance["platform"]
    assert provenance["byte_for_byte_reproduction_expected"] is False
    assert provenance["volatile_fields"] == [
        "suite_results.*.simulator_wall_seconds"
    ]
    assert provenance["source_status_capture"] == "before_suite_execution"
    assert artifact["suite_results"]["unitary_validation"]["summary"]["passed"] == 1
    random_records = artifact["suite_results"]["random_benchmarks"]["raw_records"]
    assert {record["method"] for record in random_records} == {
        "direct_bbht",
        "independent_qpe_scan",
        "classical_threshold_scan",
    }
    assert artifact["suite_results"]["failure_semantics"]["summary"][
        "statevector_block_queries"
    ] == 0
    assert "asymptotic quantum advantage" in " ".join(
        artifact["claim_boundaries"]["does_not_support"]
    )
    assert "not a complexity theorem" in completed.stdout


def test_quantum_benchmark_cli_rejects_unknown_fields_and_bool_trials(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    documents = []
    unknown = _small_document(repository)
    unknown["unregistered"] = True
    documents.append(unknown)
    boolean = _small_document(repository)
    boolean["unitary_validation"]["trials"] = True  # type: ignore[index]
    documents.append(boolean)

    for index, document in enumerate(documents):
        config_path = tmp_path / f"invalid-{index}.json"
        config_path.write_text(json.dumps(document), encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "run_quantum_benchmarks.py"),
                "--config",
                str(config_path),
                "--output",
                str(tmp_path / f"unused-{index}.json"),
                "--suite",
                "unitary_validation",
            ],
            cwd=repository,
            env=_environment(repository),
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
        assert "quantum benchmark failed" in completed.stderr
