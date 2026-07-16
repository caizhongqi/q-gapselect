from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import scripts.run_strong_composition_s3 as s3_runner

from qgapselect.hidden_frontier_fixtures import F_PUBLIC_PARTITION, FAMILY_IDS

BLIND_FAMILY_IDS = tuple(
    family_id for family_id in FAMILY_IDS if family_id != F_PUBLIC_PARTITION
)


def _small_config(path: Path) -> Path:
    document = {
        "schema_version": 1,
        "experiment_name": "small-s3-runner-regression",
        "families": list(BLIND_FAMILY_IDS),
        "execution_config": {
            "phase_powers": [0],
            "fixed_shots_per_power": 4,
            "unknown_time_shots_per_level": 4,
            "grid_points": 257,
        },
        "panels": [
            {
                "panel_id": "small-n4-k1",
                "n": 4,
                "k": 1,
                "design_gap": 0.02,
                "fixture_seeds": [1, 2],
                "measurement_seeds": [3, 5],
                "delta": 0.05,
                "atomic_query_caps": [64],
            }
        ],
        "claim_boundary": ["runner regression; no research claim"],
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_s3_cli_writes_a_fail_closed_all_attempt_artifact(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = _small_config(tmp_path / "s3-config.json")
    output = tmp_path / "s3-artifact.json.gz"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_strong_composition_s3.py"),
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

    with gzip.open(output, "rt", encoding="utf-8") as stream:
        artifact = json.load(stream)
    audit = artifact["aggregate_audit"]

    assert artifact["artifact_type"] == ("q_gapselect_same_interface_strong_composition_s3")
    assert audit["all_checks_passed"] is True
    assert all(audit["checks"].values())
    assert audit["attempt_count"] == 72
    assert audit["expected_attempt_count"] == 72
    assert audit["incorrect_certificate_count"] == 0
    assert audit["checks"]["no_incorrect_certificates"] is True
    assert audit["budget_violation_count"] == 0
    assert audit["all_attempt_denominator_used"] is True
    assert len(artifact["attempts"]) == 72
    assert len(artifact["all_attempt_aggregates"]) == 3
    assert all(row["all_attempt_count"] == 24 for row in artifact["all_attempt_aggregates"])
    assert len(artifact["per_family_all_attempt_aggregates"]) == 18
    assert all(
        row["all_attempt_count"] == 4
        for row in artifact["per_family_all_attempt_aggregates"]
    )

    contract = artifact["canonical_runtime_contract"]
    assert contract["fields"] == ["n", "k", "delta", "oracle", "atomic_query_cap"]
    assert all(fields == contract["fields"] for fields in contract["method_signatures"].values())
    forbidden = set(contract["forbidden_fields"])
    assert {"gap", "family", "truth", "schedule", "stop_levels"} <= forbidden
    assert contract["excluded_stronger_information_families"] == [
        F_PUBLIC_PARTITION
    ]

    for row in artifact["attempts"]:
        runtime = row["algorithm_runtime_input_audit"]
        result = row["result"]
        score = row["trusted_score"]
        assert runtime["fields"] == contract["fields"]
        assert runtime["private_fixture_object_supplied"] is False
        assert runtime["truth_supplied"] is False
        assert runtime["gap_supplied"] is False
        assert runtime["family_supplied"] is False
        assert runtime["stopping_schedule_supplied"] is False
        assert score["included_in_all_attempt_denominator"] is True
        assert score["exact_canonical_query_count"] <= score["atomic_query_cap"]
        assert score["budget_valid"] is True
        assert result["registry_coverage_activated"] is False
        assert result["official_literature_reproduction"] is False
        assert result["hardware_claimable"] is False
        assert result["quantum_advantage_claimable"] is False
        if not result["certified"]:
            assert result["output_relation"] == "INCONCLUSIVE"
            assert result["output_indices"] is None
            assert result["output_mask"] is None

    assert len(artifact["method_inventory"]) == 3
    assert all(
        row["strongest_registry_baseline_covered"] is False for row in artifact["method_inventory"]
    )
    reconciliation = artifact["registry_reconciliation"]
    assert reconciliation["all_comparison_targets_registered"] is True
    assert reconciliation["coverage_activated_by_s3_controls"] is False
    boundary = artifact["claim_boundary"]
    assert boundary["qpe_circuit_executed"] is False
    assert boundary["coherent_variable_time_search_executed"] is False
    assert boundary["official_literature_reproduction"] is False
    assert boundary["candidate_included_in_cer_panel"] is False
    assert boundary["paired_candidate_cer_superiority_verified"] is False
    assert boundary["claim_bearing_sample_size_met"] is False
    assert boundary["strongest_registry_coverage_claimed"] is False
    assert boundary["quantum_advantage_claimed"] is False
    assert boundary["ccf_a_claimable"] is False
    assert (
        artifact["provenance"]["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    )
    assert "attempts=72" in completed.stdout
    assert "no full literature reproduction" in completed.stdout


def test_default_s3_config_commits_the_n32_multiseed_matched_panel() -> None:
    repository = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repository / "configs" / "strong_composition_s3.json").read_text(encoding="utf-8")
    )
    panel = config["panels"][0]

    assert config["families"] == list(BLIND_FAMILY_IDS)
    assert panel["n"] == 32
    assert panel["k"] == 6
    assert panel["fixture_seeds"] == [1, 4, 6]
    assert panel["measurement_seeds"] == [11, 23, 37]
    assert panel["atomic_query_caps"] == [262144, 524288, 1048576]


def test_s3_cli_rejects_abbreviated_arguments(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "must-not-exist.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_strong_composition_s3.py"),
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


def test_s3_cli_rejects_a_private_stopping_schedule_in_config(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = _small_config(tmp_path / "invalid-s3-config.json")
    document = json.loads(config.read_text(encoding="utf-8"))
    document["execution_config"]["stop_levels"] = [1, 2, 3, 4]
    config.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "must-not-exist.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_strong_composition_s3.py"),
            "--config",
            str(config),
            "--output",
            str(output),
        ),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "unknown=['stop_levels']" in completed.stderr
    assert not output.exists()


def test_incorrect_certificate_forces_aggregate_audit_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _small_config(tmp_path / "forged-score-config.json")
    original = s3_runner.score_s3_attempt

    def forged_score(result, truth):
        return replace(original(result, truth), incorrect_certificate=True)

    monkeypatch.setattr(s3_runner, "score_s3_attempt", forged_score)
    artifact = s3_runner.build_artifact(config)
    audit = artifact["aggregate_audit"]

    assert audit["incorrect_certificate_count"] == audit["attempt_count"]
    assert audit["checks"]["no_incorrect_certificates"] is False
    assert audit["all_checks_passed"] is False


def test_public_partition_cannot_enter_the_blind_same_interface_panel(
    tmp_path: Path,
) -> None:
    config = _small_config(tmp_path / "public-partition-config.json")
    document = json.loads(config.read_text(encoding="utf-8"))
    document["families"].append(F_PUBLIC_PARTITION)
    config.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="stronger interface"):
        s3_runner.load_config(config)
