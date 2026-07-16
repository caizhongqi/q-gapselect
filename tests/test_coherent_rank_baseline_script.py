from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _environment(repository: Path) -> dict[str, str]:
    environment = dict(os.environ)
    source = str(repository / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )
    return environment


def test_coherent_rank_baseline_cli_writes_semantic_artifact(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = tmp_path / "rank.json"
    output = tmp_path / "rank-out.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_name": "test-rank",
                "master_seed": 9,
                "cases": [
                    {
                        "case_id": "strict",
                        "means": [1.0, 0.0],
                        "k": 1,
                        "phase_qubits": 2,
                        "shots_per_arm": 1,
                        "repetitions": 1,
                    },
                    {
                        "case_id": "tie",
                        "means": [0.5, 0.5],
                        "k": 1,
                        "phase_qubits": 2,
                        "shots_per_arm": 1,
                        "repetitions": 1,
                    },
                ],
                "notes": ["test"],
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_coherent_rank_baseline.py"),
            "--config",
            str(config),
            "--output",
            str(output),
        ],
        cwd=repository,
        env=_environment(repository),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    summary = artifact["summary"]
    assert artifact["artifact_type"] == (
        "q_gapselect_measured_qpe_direct_topk_semantic_baseline"
    )
    assert summary["record_count"] == 2
    assert summary["strict_attempt_count"] == 1
    assert summary["tie_negative_control_count"] == 1
    assert summary["direct_multi_output_complete_rate_on_strict"] == 1.0
    assert summary["exact_output_rate_on_strict"] == 1.0
    assert summary["certified_exact_rate_on_strict"] == 0.0
    assert summary["tie_fail_closed_rate"] == 1.0
    assert summary["all_rank_cleanup_passed"]
    assert summary["all_query_formulas_reconciled"]
    assert not summary["qram_assumed"]
    assert not summary["end_to_end_coherent"]
    assert not summary["quantum_advantage_claimable"]
    assert "cannot support a quantum-advantage claim" in completed.stdout
