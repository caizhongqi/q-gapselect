from __future__ import annotations

import gzip
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


def test_fixed_fixture_calibration_cli_is_algorithm_only_and_clustered(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "anchor-calibration.json.gz"
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_frozen_anchor_calibration.py"),
            "--config",
            str(repository / "configs" / "frozen_anchor_calibration.json"),
            "--mixture-artifact",
            str(repository / "artifacts" / "frozen_quantum_reference_diagnostic.json"),
            "--output",
            str(output),
            "--anchors-per-family",
            "1",
            "--repetitions-per-anchor",
            "2",
        ],
        cwd=repository,
        env=_environment(repository),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        artifact = json.load(handle)
    assert artifact["artifact_type"] == (
        "q_gapselect_fixed_fixture_multiseed_calibration"
    )
    assert artifact["summary"]["family_count"] == 3
    assert artifact["summary"]["anchor_count"] == 3
    assert artifact["summary"]["run_count"] == 3 * 2 * 3
    assert artifact["summary"]["fixture_is_independent_unit"]
    assert artifact["summary"]["fixed_fixture_multiseed_calibration_performed"]
    assert not artifact["summary"]["selection_uses_algorithm_outcomes"]
    assert not artifact["summary"]["llm_execution_performed"]
    assert not artifact["summary"]["hardware_execution_performed"]
    assert not artifact["summary"]["quantum_advantage_claimed"]
    assert not artifact["summary"]["worst_case_fixed_confidence_claimed"]
    assert len(artifact["anchor_selections"]) == 3
    assert len(artifact["calibration"]["records"]) == 3 * 3
    assert all(record["repetitions"] == 2 for record in artifact["calibration"]["records"])
    assert len(artifact["benchmark"]["runs"]) == 18
    assert "no LLM, hardware, or advantage claim" in completed.stdout
