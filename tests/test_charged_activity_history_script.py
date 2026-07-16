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


def test_charged_activity_history_cli_writes_audited_artifact(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "charged.json"
    output_path = tmp_path / "charged-out.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_name": "test-charged",
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
                        "max_statevector_dimension": 65536,
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
            str(repository / "scripts" / "run_charged_activity_history.py"),
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
    assert artifact["artifact_type"] == "q_gapselect_charged_activity_history"
    assert artifact["claim_status"] == "charged_phase_history_prototype_no_upper_bound_theorem"
    assert artifact["summary"]["total_records"] == 2
    assert artifact["summary"]["all_no_supplied_predicate_rows"] is True
    assert artifact["summary"]["all_output_subset_active"] is True
    assert artifact["summary"]["exact_state_trace_count"] == 2
    assert artifact["summary"]["asymptotic_theorem_claimed"] is False
    assert "not a proved CCF-A-level quantum advantage theorem" in completed.stdout
