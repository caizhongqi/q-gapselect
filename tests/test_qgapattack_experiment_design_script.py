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


def test_qgapattack_design_script_writes_a_nonclaiming_artifact(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "design.json"
    markdown = tmp_path / "design.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_qgapattack_experiment_design.py"),
            "--config",
            str(repository / "configs" / "qgapattack_experiments.json"),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--strict-design",
        ],
        cwd=repository,
        env=_environment(repository),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "qgapattack_preregistered_experiment_design"
    assert artifact["audit"]["baseline_count"] == 38
    assert artifact["audit"]["panel_count"] == 11
    assert artifact["audit"]["design_valid"] is True
    assert artifact["audit"]["empirical_ready"] is False
    assert artifact["audit"]["ccf_a_claimable"] is False
    assert artifact["provenance"]["config_path"] == "configs/qgapattack_experiments.json"
    assert "Open blockers" in markdown.read_text(encoding="utf-8")
    assert "ccf_a_claimable=false" in completed.stdout
