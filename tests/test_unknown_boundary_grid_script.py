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


def test_unknown_boundary_grid_cli_writes_open_and_failed_cases(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "grid.json"
    output_path = tmp_path / "grid-out.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_name": "test-grid",
                "cases": [
                        {
                            "name": "open",
                            "m_values": [8, 16],
                        "n_exponent": 3.0,
                        "level_exponent": 1.0,
                        "active_exponent": 1.0,
                        "gamma_exponent": 2.0,
                        "output_births_per_level": 1,
                        "epsilon_growth_exponent": 0.25,
                        "activity_decay_exponent": 0.0,
                        "predicate_cost_exponent": 0.0,
                        "baseline_match_tolerance": 1.2,
                    },
                    {
                        "name": "loose",
                        "m_values": [4, 8],
                        "n_exponent": 3.0,
                        "level_exponent": 1.0,
                        "active_exponent": 1.0,
                        "gamma_exponent": 2.0,
                        "output_births_per_level": 1,
                        "epsilon_growth_exponent": 0.25,
                        "activity_decay_exponent": 0.0,
                        "predicate_cost_exponent": 0.0,
                        "baseline_match_tolerance": 2.0,
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
            str(repository / "scripts" / "run_unknown_boundary_grid.py"),
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
    assert artifact["artifact_type"] == "q_gapselect_unknown_boundary_history_grid"
    assert artifact["summary"]["case_count"] == 2
    assert artifact["summary"]["total_records"] == 4
    gates = artifact["summary"]["novelty_gate_counts"]
    assert gates["open_no_encoded_baseline_match_requires_unitary_and_lower_bound"] == 2
    assert gates["failed_encoded_baseline_match"] == 2
    assert artifact["case_results"][0]["summary"]["all_points_open"] is True
    assert artifact["case_results"][1]["summary"]["all_points_open"] is False
    assert "not an upper-bound theorem" in completed.stdout
