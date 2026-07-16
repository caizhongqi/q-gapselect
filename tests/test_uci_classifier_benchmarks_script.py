from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path


def test_uci_classifier_cli_runs_complete_offline_matched_panel(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "uci-digits.json.gz"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_uci_classifier_benchmarks.py"),
            "--config",
            str(repository / "configs" / "uci_classifier_benchmarks.json"),
            "--theory-artifact",
            str(repository / "artifacts" / "theorem_closure_audit.json"),
            "--output",
            str(output),
            "--repetitions-per-fixture",
            "2",
            "--query-caps",
            "65536",
            "--bootstrap-repetitions",
            "31",
            "--workers",
            "2",
        ),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        artifact = json.load(handle)

    summary = artifact["summary"]
    assert artifact["artifact_type"] == "q_gapselect_uci_classifier_selection_campaign"
    assert summary["dataset_id"] == "sklearn_digits_offline"
    assert not summary["official_source"]
    assert not summary["official_confirmatory"]
    assert summary["semi_synthetic_external_validity"]
    assert summary["accepted_fixture_count"] >= 2
    assert (
        summary["accepted_fixture_count"] + summary["boundary_failure_count"]
        == 5
    )
    assert summary["arm_count"] == 24 and summary["k"] == 8
    assert summary["run_count"] == summary["accepted_fixture_count"] * 2 * 1 * 5
    assert summary["information_matched_primary_panel"]
    assert summary["query_cap_matched_primary_panel"]
    assert summary["all_attempts_in_denominator"]
    assert not summary["coherent_index_execution_performed"]
    assert not summary["llm_execution_performed"]
    assert not summary["ccf_a_quantum_advantage_claimable"]
    assert len(artifact["records"]) == summary["run_count"]
    assert len(artifact["campaign_manifest_hash"]) == 64
    assert len(artifact["benchmark_manifest"]["manifest_hash"]) == 64
    manifest_text = json.dumps(artifact["benchmark_manifest"], sort_keys=True)
    assert "accuracy" not in manifest_text
    assert "test_labels" not in manifest_text
    assert artifact["provenance"]["workers"] == 2
    assert "workers" not in artifact["resolved_config"]
    assert "semi-synthetic external validity" in completed.stdout

    refreshed_output = tmp_path / "uci-digits-refreshed.json.gz"
    subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "refresh_ccfa_evidence_gate.py"),
            str(output),
            "--output",
            str(refreshed_output),
        ),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    with gzip.open(refreshed_output, "rt", encoding="utf-8") as handle:
        refreshed = json.load(handle)
    assert refreshed["records"] == artifact["records"]
    assert refreshed["campaign_manifest_hash"] == artifact["campaign_manifest_hash"]
    assert "strongest_baseline_fidelity_not_verified" in refreshed["evidence_gate"][
        "blockers"
    ]
    assert refreshed["provenance"]["evidence_gate_refreshed_from_raw_records"]
