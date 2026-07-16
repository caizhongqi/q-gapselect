from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_matched_failure_analysis_script_reads_formal_artifact(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "failure.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_matched_failure_analysis.py"),
            "--input",
            str(repository / "artifacts" / "ccfa_matched_benchmark_diagnostic.json.gz"),
            "--output",
            str(output),
        ),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["analysis"]["attempt_count"] == 37500
    assert artifact["analysis"]["family_count"] == 3
    assert artifact["analysis"]["query_cap_count"] == 5
    assert artifact["ccf_a_quantum_advantage_claimable"] is False
    assert "attempts=37500" in completed.stdout
