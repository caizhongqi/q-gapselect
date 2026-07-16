from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_frontier_lower_bound_witness_cli_is_falsifiable_and_fail_closed(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = repository / "configs" / "frontier_lower_bound_witness.json"
    output = tmp_path / "frontier-lower-bound.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_frontier_lower_bound_witness.py"),
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

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    summary = artifact["summary"]
    pairs = artifact["pair_hybrid_witnesses"]
    johnson = artifact["johnson_adversary_witnesses"]
    compositions = artifact["composition_falsification_witnesses"]

    assert artifact["artifact_type"] == "q_gapselect_frontier_lower_bound_witness_s3"
    assert artifact["config"]["sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert artifact["registry_binding"]["canonical_interface_id"] == (
        "canonical_blind_exact_topk_v1"
    )
    assert artifact["registry_binding"]["strongest_composition_claimable"] is False
    assert summary["pair_hybrid_witness_count"] == 3
    assert summary["pair_hybrid_all_verified"] is True
    assert summary["johnson_witness_count"] == 3
    assert summary["johnson_all_verified"] is True
    assert summary["composition_witness_count"] == 2
    assert summary["composition_match_count"] == 0
    assert summary["composition_kill_count"] == 0
    assert summary["finite_fixture_query_dominance_count"] == 2
    assert summary["local_lower_bound_matches_finite_upper"] is False
    assert summary["continuous_angle_johnson_composition_proved"] is False
    assert summary["activity_history_direct_sum_proved"] is False
    assert summary["registered_strongest_composition_coverage_complete"] is False
    assert len(summary["uncovered_required_baseline_ids"]) == 9
    assert summary["matching_lower_bound_claimable"] is False
    assert summary["strongest_composition_claimable"] is False
    assert summary["quantum_advantage_claimable"] is False
    assert summary["ccf_a_claimable"] is False

    common_fields = {
        "witness_type",
        "oracle_interface",
        "computed_quantity",
        "verified_local_statement",
        "explicit_non_theorem_boundary",
        "composition_match",
        "composition_kill_flag",
    }
    assert all(common_fields <= set(record) for record in pairs + johnson + compositions)
    assert all(record["verification_passed"] for record in pairs)
    assert all(record["verification_passed"] for record in johnson)
    assert all(not record["composition_match"] for record in compositions)
    assert all(not record["composition_kill_flag"] for record in compositions)
    assert all(record["finite_fixture_query_dominance_verified"] for record in compositions)
    assert all(record["same_oracle_model_verified"] for record in compositions)
    assert all(record["same_fixture_harness_verified"] for record in compositions)
    assert all(record["distinct_oracle_instances_verified"] for record in compositions)
    assert all(not record["same_public_algorithm_interface_verified"] for record in compositions)
    assert all(not record["same_certified_output_contract_verified"] for record in compositions)
    assert all(record["candidate_query_count"] == 176 for record in compositions)
    assert all(record["baseline_query_count"] == 28 for record in compositions)
    assert all(record["computed_quantity"] == 28 / 176 for record in compositions)
    assert all(
        record["registered_published_baseline_fidelity_verified"] is False
        for record in compositions
    )
    assert all(record["global_composition_frontier_closed"] is False for record in compositions)
    assert artifact["provenance"]["git_commit"]
    assert artifact["provenance"]["git_tree"]
    assert artifact["provenance"]["python_version"]


def test_frontier_lower_bound_witness_cli_rejects_abbreviated_arguments(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "must-not-exist.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_frontier_lower_bound_witness.py"),
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
