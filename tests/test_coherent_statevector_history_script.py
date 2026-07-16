from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_exact_state_history_cli_records_real_coherent_calls(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "statevector.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_coherent_statevector_history.py"),
            "--config",
            str(repository / "configs" / "coherent_statevector_history.json"),
            "--output",
            str(output),
            "--repetitions",
            "1",
        ),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "q_gapselect_exact_state_coherent_activity_history"
    assert artifact["summary"]["record_count"] == 3
    assert artifact["summary"]["actual_coherent_index_execution_performed"]
    assert not artifact["summary"]["analytic_per_arm_iae_used"]
    assert artifact["summary"]["complete_direct_multi_output_count"] == 0
    assert artifact["summary"]["certificate_count"] == 0
    assert not artifact["summary"]["quantum_advantage_claimable"]
    assert all(
        row["resources"]["query_counts"]["forward"] == 0
        and row["resources"]["query_counts"]["inverse"] == 0
        for row in artifact["records"]
    )
    assert any(
        row["resources"]["query_counts"]["controlled_forward"] > 0
        for row in artifact["records"]
    )
    assert "complete multi-output theorem remains blocked" in completed.stdout
