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


def test_lower_bound_program_script_writes_artifact_and_markdown(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "lower-bound-program.json"
    markdown = tmp_path / "lower-bound-program.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_lower_bound_program.py"),
            "--config",
            str(repository / "configs" / "lower_bound_program.json"),
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
    assert artifact["artifact_type"] == "q_gapselect_lower_bound_program"
    assert artifact["summary"]["total_records"] == 12
    assert artifact["summary"]["total_proof_obligations"] == 36
    assert "lower-bound proof program" in markdown.read_text(encoding="utf-8")
    assert "not a proved lower bound" in completed.stdout
