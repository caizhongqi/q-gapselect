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


def test_stopping_time_transducer_cli_writes_skeleton_artifact(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "stopping.json"
    output_path = tmp_path / "stopping-out.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_name": "test-stopping",
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
            str(repository / "scripts" / "run_stopping_time_transducer.py"),
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
    assert artifact["artifact_type"] == "q_gapselect_stopping_time_transducer"
    assert artifact["claim_status"] == "variable_time_stopping_transducer_skeleton_no_theorem"
    assert artifact["summary"]["total_records"] == 2
    assert artifact["summary"]["exact_state_trace_count"] == 2
    assert artifact["summary"]["all_variable_over_serial_at_most_one"] is True
    assert artifact["summary"]["asymptotic_theorem_claimed"] is False
    assert "stopping relation skeleton" in completed.stdout
