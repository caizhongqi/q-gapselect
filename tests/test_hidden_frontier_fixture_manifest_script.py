from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path


def _all_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key).lower())
            keys.update(_all_keys(child))
    elif isinstance(value, list | tuple):
        for child in value:
            keys.update(_all_keys(child))
    return keys


def test_hidden_frontier_fixture_manifest_cli_writes_strict_audit(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = repository / "configs" / "hidden_frontier_fixture_manifest.json"
    output = tmp_path / "hidden-frontier-fixtures.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_hidden_frontier_fixture_manifest.py"),
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
    audit = artifact["aggregate_audit"]
    orbit = artifact["orbit_hash_audit"]

    assert artifact["artifact_type"] == "q_gapselect_hidden_frontier_fixture_manifest"
    assert artifact["claim_status"] == ("fixture_isolation_audit_only_not_algorithm_evidence")
    assert artifact["resolved_config"]["families"] == [
        "F-EQ",
        "F-DYADIC",
        "F-CLUSTER",
        "F-HIDDEN-FRONTIER",
        "F-PUBLIC-PARTITION",
        "F-UNKNOWN-TIME-NC",
        "F-TIE-NC",
    ]
    assert artifact["resolved_config"]["panels"][0]["fixture_seeds"] == [1, 4, 6]
    assert audit["all_checks_passed"] is True
    assert all(audit["checks"].values())
    assert audit["panel_count"] == 1
    assert audit["family_count"] == 7
    assert audit["fixture_count"] == 21
    assert audit["algorithm_performance_measured"] is False
    assert audit["theorem_claimed"] is False
    assert audit["quantum_advantage_claimed"] is False
    assert audit["ccf_a_claimable"] is False

    assert orbit["fixture_count"] == 21
    assert orbit["unique_fixture_hash_count"] == 21
    assert orbit["raw_unique_orbit_hash_count"] == 18
    assert orbit["unique_algorithmic_deduplication_key_count"] == 21
    assert orbit["deduplicated_algorithmic_fixture_count"] == 21
    assert orbit["raw_duplicate_orbit_group_count"] == 3
    assert orbit["deduplication_key"] == ["orbit_hash", "interface_id"]
    assert all(orbit["checks"].values())
    assert orbit["checks"]["hidden_public_pair_members_are_both_retained"] is True
    assert all(
        {key.split("/")[1] for key in group["fixture_keys"]}
        == {"F-HIDDEN-FRONTIER", "F-PUBLIC-PARTITION"}
        for group in orbit["raw_duplicate_orbit_groups"]
    )

    public_views = artifact["public_algorithm_views"]
    assert len(public_views) == 4
    forbidden = {
        "angle",
        "beta",
        "center",
        "ranking",
        "membership",
        "active",
        "schedule",
        "seed",
        "family",
        "permutation",
        "stopping",
        "radius",
        "radii",
    }
    for row in public_views:
        view = row["algorithm_view"]
        assert row["interface_id"] == view["interface_id"]
        assert not any(token in key for key in _all_keys(view) for token in forbidden)
        assert "fixture_hash" not in view
        assert "orbit_hash" not in view

    trusted = artifact["trusted_fixture_summary"]
    assert len(trusted) == 21
    assert len({row["fixture_hash"] for row in trusted}) == 21
    assert all(row["fixture_hash"] == row["replay_fixture_hash"] for row in trusted)
    assert all(row["replay_passed"] for row in trusted)
    ties = [row for row in trusted if row["family_id"] == "F-TIE-NC"]
    assert len(ties) == 3
    assert all(row["non_unique_output"] for row in ties)
    assert all(row["tie_label_scope"] == "trusted_harness_only" for row in ties)

    assert len(artifact["hidden_public_pair_audits"]) == 3
    assert all(row["passed"] for row in artifact["hidden_public_pair_audits"])
    assert all(all(row["checks"].values()) for row in artifact["hidden_public_pair_audits"])
    assert len(artifact["tie_trusted_only_audits"]) == 3
    assert all(row["passed"] for row in artifact["tie_trusted_only_audits"])

    assert (
        artifact["provenance"]["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    )
    assert artifact["provenance"]["git_commit"]
    assert artifact["provenance"]["git_tree"]
    assert artifact["provenance"]["python_version"]
    assert artifact["provenance"]["platform"]
    assert "not a theorem" in completed.stdout
    assert "fixtures=21" in completed.stdout
    assert "raw_unique_orbits=18" in completed.stdout
    assert "algorithmic_fixtures_after_dedup=21" in completed.stdout


def test_hidden_frontier_fixture_manifest_cli_rejects_abbreviated_arguments(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "must-not-exist.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_hidden_frontier_fixture_manifest.py"),
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
