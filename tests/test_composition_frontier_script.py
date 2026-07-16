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


def test_composition_frontier_script_writes_artifact(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "composition-frontier.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_composition_frontier.py"),
            "--config",
            str(repository / "configs" / "composition_frontier.json"),
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
    assert artifact["artifact_type"] == "q_gapselect_composition_frontier"
    assert artifact["summary"]["total_records"] == 13
    assert artifact["summary"]["open_case_count"] == 2
    assert "frontier audit" in completed.stdout
