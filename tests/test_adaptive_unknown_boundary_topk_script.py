from __future__ import annotations

import hashlib
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


def test_adaptive_unknown_boundary_cli_writes_fail_closed_s3_panel(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = repository / "configs" / "adaptive_unknown_boundary_topk.json"
    output = tmp_path / "adaptive-unknown-boundary.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_adaptive_unknown_boundary_topk.py"),
            "--config",
            str(config),
            "--output",
            str(output),
        ),
        cwd=repository,
        env=_environment(repository),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == (
        "q_gapselect_adaptive_unknown_boundary_topk_s3_panel"
    )
    assert artifact["config_hash"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert artifact["provenance"]["config_sha256"] == artifact["config_hash"]
    assert artifact["summary"]["case_count"] == 5
    assert artifact["summary"]["diagnostic_mask_count"] == 3
    assert artifact["summary"]["inconclusive_count"] == 2
    assert artifact["summary"]["off_grid_diagnostic_mask_count"] == 2
    assert artifact["summary"]["all_mask_outputs_exact_under_trusted_scoring"] is True
    assert artifact["summary"]["single_coherent_variable_time_unitary_implemented"] is False
    assert artifact["summary"]["coherent_history_cleanup_proved"] is False
    assert artifact["summary"]["observable_stop_estimator_implemented"] is False
    assert artifact["summary"]["quantum_advantage_claimable"] is False
    assert artifact["summary"]["ccf_a_claimable"] is False
    assert artifact["aggregate_assertions"]["all_assertions_passed"] is True

    records = {record["role"]: record for record in artifact["records"]}
    assert records["off_grid_mid_precision"]["result"]["stopping_history"][
        "first_stop_phase_qubits"
    ] == 3
    assert records["off_grid_deep_precision"]["result"]["stopping_history"][
        "first_stop_phase_qubits"
    ] == 4
    assert records["query_cap"]["result"]["query_budget"][
        "blocked_before_phase_qubits"
    ] == 3
    assert all(record["result"]["query_budget"]["budget_valid"] for record in records.values())
    assert all(not record["result"]["certificate"]["issued"] for record in records.values())
    assert all(
        record["result"]["stopping_history"]["controller_is_classical"]
        for record in records.values()
    )
    assert "exact-state classical simulation" in completed.stdout


def test_adaptive_unknown_boundary_cli_rejects_abbreviated_arguments(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "must-not-exist.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_adaptive_unknown_boundary_topk.py"),
            "--out",
            str(output),
        ),
        cwd=repository,
        env=_environment(repository),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "unrecognized arguments" in completed.stderr
    assert not output.exists()
