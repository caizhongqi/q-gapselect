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


def test_variable_time_charged_history_cli_writes_alignment_artifact(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "vt-charged.json"
    output_path = tmp_path / "vt-charged-out.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_name": "test-vt-charged",
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
                        "precision_multiplier": 2.0,
                        "min_precision_bits": 1,
                        "max_precision_bits": None,
                        "baseline_match_tolerance": 1.2,
                    },
                    {
                        "name": "loose",
                        "m_values": [8, 16],
                        "n_exponent": 3.0,
                        "level_exponent": 1.0,
                        "active_exponent": 1.0,
                        "gamma_exponent": 2.0,
                        "output_births_per_level": 1,
                        "epsilon_growth_exponent": 0.25,
                        "activity_decay_exponent": 0.0,
                        "predicate_cost_exponent": 0.0,
                        "precision_multiplier": 2.0,
                        "min_precision_bits": 1,
                        "max_precision_bits": None,
                        "baseline_match_tolerance": 10.0,
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
            str(repository / "scripts" / "run_variable_time_charged_history.py"),
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
    assert artifact["artifact_type"] == "q_gapselect_variable_time_charged_history"
    assert artifact["claim_status"] == "variable_time_charged_history_alignment_no_theorem"
    assert artifact["summary"]["total_records"] == 4
    assert artifact["summary"]["open_case_count"] == 1
    assert artifact["summary"]["asymptotic_theorem_claimed"] is False
    gates = artifact["summary"]["novelty_gate_counts"]
    assert gates["open_charged_variable_time_gap_requires_upper_and_lower_bound"] == 2
    assert gates["failed_charged_baseline_match"] == 2
    assert "mainline alignment audit" in completed.stdout
