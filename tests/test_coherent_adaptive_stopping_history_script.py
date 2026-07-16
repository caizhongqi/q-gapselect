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


def test_true_coherent_history_cli_writes_s3_panel(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = repository / "configs" / "coherent_adaptive_stopping_history.json"
    output = tmp_path / "coherent-history.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_coherent_adaptive_stopping_history.py"),
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
        "q_gapselect_tiny_true_coherent_stopping_history_s3_panel"
    )
    assert artifact["config_hash"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert artifact["provenance"]["config_sha256"] == artifact["config_hash"]
    summary = artifact["summary"]
    assert summary["case_count"] == 5
    assert summary["exact_grid_complete_count"] == 3
    assert summary["fail_closed_count"] == 2
    assert summary["executed_queries_per_case"] == 176
    assert summary["runtime_tag_derived_one_way_level_queries"] == [28, 60]
    assert summary["true_coherent_stopping_history_unitary_implemented"] is True
    assert summary["later_level_active_control_implemented"] is True
    assert summary["durable_copy_and_full_replay_implemented"] is True
    assert summary["branch_rms_is_theorem_target_only"] is True
    assert summary["generic_off_grid_cleanup_proved"] is False
    assert summary["variable_time_query_speedup_proved"] is False
    assert summary["quantum_advantage_claimable"] is False
    assert summary["ccf_a_claimable"] is False
    assert artifact["aggregate_assertions"]["all_assertions_passed"] is True

    records = {record["role"]: record for record in artifact["records"]}
    assert records["exact_grid_first_stop"]["result"]["history"][
        "dominant_history"
    ] == "10"
    assert records["exact_grid_second_stop"]["result"]["history"][
        "dominant_history"
    ] == "01"
    assert records["exact_grid_arm1_winner"]["result"]["membership_mask"] == 0b10
    assert records["exact_grid_tie"]["result"]["history"]["dominant_history"] == "00"
    assert records["off_grid_fail_closed"]["result"]["resources"]["cleanup"][
        "passed"
    ] is False
    for record in records.values():
        ledger = record["result"]["resources"]["query_ledger"]
        assert ledger["query_counts"]["coherent_total"] == 176
        assert ledger["reconciled"] is True
        assert ledger["branch_rms_is_executed_saving"] is False
        assert [
            level["runtime_derived_one_way_counts"]["coherent_total"]
            for level in ledger["per_level_runtime_records"]
        ] == [28, 60]
        assert all(
            level["full_replay_reconciled"] and level["one_way_reconciled"]
            for level in ledger["per_level_runtime_records"]
        )
        assert record["result"]["certificate"]["issued"] is False
        assert record["result"]["fixed_expected_query_ledger_respected"] is True
    inactive = artifact["inactive_level_clean_dirty_subspace_audit"]
    assert inactive["clean_identity_witness_passed"] is True
    assert inactive["dirty_negative_control_activated"] is True
    assert inactive["clean_query_counts"]["coherent_total"] == 60
    assert inactive["dirty_query_counts"]["coherent_total"] == 60
    assert inactive["theorem_status"] == "basis_witness_only_not_a_subspace_proof"
    assert "Branch-RMS is a theorem target only" in completed.stdout


def test_true_coherent_history_cli_rejects_abbreviated_arguments(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "must-not-exist.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_coherent_adaptive_stopping_history.py"),
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
