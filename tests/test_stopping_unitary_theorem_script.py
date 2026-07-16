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


def test_stopping_unitary_theorem_cli_writes_scaffold_artifacts(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "theorem.json"
    output_path = tmp_path / "theorem-out.json"
    markdown_path = tmp_path / "theorem.md"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_name": "test-theorem",
                "cases": [
                    {
                        "name": "small",
                        "m_values": [4, 8],
                        "n_exponent": 1.5,
                        "level_exponent": 1.0,
                        "boundary_phase": 0.5,
                        "near_boundary_fraction": 0.5,
                        "base_bits": 3,
                        "growth_period": 1,
                        "guard_band": 0.0,
                        "phase_on": "output",
                    }
                ],
                "notes": ["test"],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_stopping_unitary_theorem.py"),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--markdown",
            str(markdown_path),
        ],
        cwd=repository,
        env=_environment(repository),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "q_gapselect_stopping_unitary_theorem_scaffold"
    assert artifact["claim_status"] == "stopping_time_unitary_lemma_scaffold_no_proof"
    assert artifact["summary"]["total_records"] == 2
    assert artifact["summary"]["all_execution_checks_passed"] is True
    assert artifact["summary"]["check_status_counts"]["passed"] == 10
    assert artifact["summary"]["check_status_counts"]["proof_obligation"] == 8
    assert artifact["summary"]["asymptotic_theorem_claimed"] is False
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "ST-P01" in markdown
    assert "proof_obligation" in markdown
    assert "not a completed CCF-A-level" in completed.stdout
