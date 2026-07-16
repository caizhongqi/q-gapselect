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


def test_replay_coherent_frontier_cli_writes_fail_closed_artifact(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = tmp_path / "frontier.json"
    output = tmp_path / "frontier-out.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_name": "test-frontier",
                "cases": [
                    {
                        "case_id": "tiny",
                        "n_arms": 4,
                        "active_indices_by_level": [
                            [0, 1, 2, 3],
                            [1, 2, 3],
                            [2],
                        ],
                        "output_births_by_level": [[0], [1]],
                        "prefix_levels": [1, 2],
                        "cleanup_tolerance": 1e-12,
                        "max_statevector_dimension": 100000,
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
            str(repository / "scripts" / "run_replay_coherent_frontier.py"),
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
        "q_gapselect_replay_preserving_coherent_frontier"
    )
    assert summary["record_count"] == 2
    assert summary["all_cleanup_passed"]
    assert summary["all_invariants_passed"]
    assert summary["all_execution_traces_replayed"]
    assert summary["coherent_index_execution_performed"]
    assert summary["durable_output_copy_executed"]
    assert not summary["qram_assumed"]
    assert not summary["coherent_boundary_discovery_executed"]
    assert not summary["direct_multi_output_complete"]
    assert not summary["quantum_advantage_claimable"]
    assert "no new upper bound" in completed.stdout
