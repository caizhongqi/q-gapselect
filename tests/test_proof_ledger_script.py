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


def _write_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    stopping = tmp_path / "stopping.json"
    composition = tmp_path / "composition.json"
    lower_bound = tmp_path / "lower-bound.json"
    stopping.write_text(
        json.dumps(
            {
                "summary": {
                    "check_status_counts": {"passed": 25, "proof_obligation": 20},
                    "all_execution_checks_passed": True,
                }
            }
        ),
        encoding="utf-8",
    )
    composition.write_text(
        json.dumps(
            {
                "summary": {
                    "novelty_gate_counts": {
                        "open_no_encoded_composition_match": 9,
                        "failed_encoded_composition_match": 4,
                    },
                    "strongest_valid_baseline_counts": {
                        "loop_variable_time_rebuild": 13
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    lower_bound.write_text(
        json.dumps(
            {
                "summary": {
                    "total_local_facts": 12,
                    "total_proof_obligations": 36,
                }
            }
        ),
        encoding="utf-8",
    )
    return stopping, composition, lower_bound


def test_proof_ledger_script_writes_json_and_markdown(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    stopping, composition, lower_bound = _write_artifacts(tmp_path)
    output = tmp_path / "proof-ledger.json"
    markdown = tmp_path / "proof-ledger.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_proof_ledger.py"),
            "--stopping-artifact",
            str(stopping),
            "--composition-artifact",
            str(composition),
            "--lower-bound-artifact",
            str(lower_bound),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
        ],
        cwd=repository,
        env=_environment(repository),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "q_gapselect_proof_ledger"
    assert artifact["ledger"]["ccf_a_claimable"] is False
    assert artifact["ledger"]["entry_count"] == 10
    assert "proof bookkeeping" in completed.stdout
    assert "Q-GapSelect proof ledger" in markdown.read_text(encoding="utf-8")
