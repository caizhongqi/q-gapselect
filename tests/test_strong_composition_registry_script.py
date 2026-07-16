from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_strong_composition_registry_cli_writes_fail_closed_audit(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = repository / "configs" / "strong_composition_registry.json"
    output = tmp_path / "strong-composition.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_strong_composition_registry.py"),
            "--config",
            str(config),
            "--output",
            str(output),
        ),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    summary = artifact["summary"]
    audit = artifact["machine_readable_audit"]

    assert artifact["artifact_type"] == (
        "q_gapselect_strong_composition_registry_audit"
    )
    assert len(artifact["registry_inventory"]["baselines"]) == 10
    assert artifact["runtime_fidelity_self_reports"] == []
    assert artifact["trusted_runtime_attestations"] == []
    assert summary["registry_row_count"] == 10
    assert summary["required_coverage_row_count"] == 9
    assert summary["runtime_self_report_row_count"] == 0
    assert summary["trusted_runtime_attestation_row_count"] == 0
    assert summary["uncovered_required_row_count"] == 9
    assert summary["inventory_complete"] is True
    assert summary["source_identity_pin_complete"] is True
    assert summary["source_version_locator_pin_complete"] is True
    assert summary["trusted_runtime_attestation_pipeline_implemented"] is False
    assert summary["strongest_composition_coverage_complete"] is False
    assert summary["strongest_composition_claimable"] is False
    assert summary["ccf_a_quantum_advantage_claimable"] is False
    assert summary["theorem_claimed"] is False
    assert len(audit["uncovered_required_baseline_ids"]) == 9
    assert all(
        row["runtime_self_report_supplied"] is False
        for row in audit["entries"]
        if row["coverage_required"]
    )
    assert audit["query_bound_templates_are_proofs"] is False
    assert audit["registry_strings_are_fidelity_evidence"] is False
    assert audit["evidence_file_integrity_is_theorem_fidelity"] is False
    assert audit["self_report_integrity_is_trusted_attestation"] is False
    assert all(
        "version_locator" in row["source"]
        for row in artifact["registry_inventory"]["baselines"]
    )
    assert artifact["provenance"]["config_sha256"] == hashlib.sha256(
        config.read_bytes()
    ).hexdigest()
    assert artifact["provenance"]["git_commit"]
    assert artifact["provenance"]["git_tree"]
    assert "not a theorem" in completed.stdout
    assert "uncovered=9" in completed.stdout


def test_strong_composition_registry_cli_rejects_abbreviated_arguments(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "must-not-exist.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_strong_composition_registry.py"),
            "--out",
            str(output),
        ),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "unrecognized arguments" in completed.stderr
    assert not output.exists()
