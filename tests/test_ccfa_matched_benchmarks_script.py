from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path


def test_ccfa_matched_cli_runs_balanced_algorithm_only_smoke(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "matched.json.gz"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_ccfa_matched_benchmarks.py"),
            "--config",
            str(repository / "configs" / "ccfa_matched_benchmarks.json"),
            "--mixture-artifact",
            str(repository / "artifacts" / "frozen_quantum_reference_diagnostic.json"),
            "--theory-artifact",
            str(repository / "artifacts" / "theorem_closure_audit.json"),
            "--output",
            str(output),
            "--families",
            "equal_n16_k8_gap_pi64",
            "--anchors-per-family",
            "2",
            "--repetitions-per-fixture",
            "2",
            "--query-caps",
            "65536",
            "--bootstrap-repetitions",
            "31",
            "--workers",
            "2",
        ),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        artifact = json.load(handle)
    assert artifact["artifact_type"] == "q_gapselect_ccfa_matched_layer_c_campaign"
    assert artifact["summary"]["run_count"] == 2 * 2 * 1 * 5
    assert artifact["summary"]["information_matched_primary_panel"]
    assert artifact["summary"]["query_cap_matched_primary_panel"]
    assert artifact["summary"]["fixed_fixture_multiseed_calibration"]
    assert not artifact["summary"]["llm_execution_performed"]
    assert not artifact["summary"]["coherent_index_execution_performed"]
    assert not artifact["summary"]["ccf_a_quantum_advantage_claimable"]
    assert len(artifact["records"]) == artifact["summary"]["run_count"]
    assert all(row["budget_valid"] for row in artifact["records"])
    assert artifact["provenance"]["workers"] == 2
    assert artifact["provenance"]["executor"] == "process_pool_fork_one_task_per_fixture"
    assert "workers" not in artifact["resolved_config"]
    assert "no LLM, API, hardware, or security run" in completed.stdout
