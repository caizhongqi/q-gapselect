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


def test_frozen_selector_cli_executes_algorithm_only_matrix(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = tmp_path / "config.json"
    output = tmp_path / "artifact.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_name": "test-frozen-selector",
                "master_seed": 17,
                "trials": 2,
                "k": 4,
                "samples_per_candidate": 32,
                "selector_ids": [
                    "random",
                    "uniform",
                    "successive_halving",
                    "cost_aware_racing",
                    "clucb_style",
                    "gcgs",
                ],
                "budgets": [
                    {"budget_id": "q32", "max_queries": 32, "max_cost": 48.0}
                ],
                "cases": [
                    {
                        "name": "gap",
                        "family": "rank_gap_ring",
                        "gap": 0.08,
                        "cost_profile": "unit",
                        "cost_jitter": 0.0,
                    },
                    {"name": "disconnected", "family": "disconnected_multipeak"},
                ],
                "notes": ["test"],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_frozen_selector_benchmarks.py"),
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
    assert artifact["artifact_type"] == (
        "q_gapselect_frozen_selector_algorithm_diagnostic"
    )
    assert artifact["summary"]["run_count"] == 2 * 1 * 2 * 6
    assert artifact["summary"]["all_runs_within_query_budget"]
    assert artifact["summary"]["all_runs_within_cost_budget"]
    assert artifact["summary"]["random_selector_zero_oracle_queries"]
    assert not artifact["summary"]["llm_execution_performed"]
    assert not artifact["summary"]["quantum_advantage_claimed"]
    assert "no_llm_execution" in artifact["claim_status"]
    assert "no LLM execution" in completed.stdout


def test_cli_trial_override_is_recorded_in_resolved_config(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "artifact.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_frozen_selector_benchmarks.py"),
            "--config",
            str(repository / "configs" / "frozen_selector_benchmarks.json"),
            "--output",
            str(output),
            "--trials",
            "1",
            "--seed",
            "9",
        ],
        cwd=repository,
        env=_environment(repository),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["resolved_config"]["trials"] == 1
    assert artifact["resolved_config"]["master_seed"] == 9
    assert artifact["report"]["trials"] == 1
    assert artifact["report"]["master_seed"] == 9
