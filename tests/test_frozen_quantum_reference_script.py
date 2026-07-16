from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest


def _environment(repository: Path) -> dict[str, str]:
    environment = dict(os.environ)
    source = str(repository / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )
    return environment


def test_frozen_layer_c_cli_runs_without_llm_or_hardware(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "layer-c.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_frozen_quantum_reference_benchmarks.py"),
            "--config",
            str(repository / "configs" / "frozen_quantum_reference_benchmarks.json"),
            "--output",
            str(output),
            "--problem-instances",
            "1",
        ],
        cwd=repository,
        env=_environment(repository),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == (
        "q_gapselect_frozen_empirical_layer_c_reference_diagnostic"
    )
    assert artifact["summary"]["problem_instance_count"] == 10
    assert artifact["summary"]["run_count"] == 10 * 3
    assert artifact["summary"]["same_layer_c_oracle"]
    assert artifact["summary"]["same_output_set"]
    assert artifact["summary"]["independent_problem_instances"]
    assert artifact["summary"]["instance_distribution_generator"] == (
        "nonisomorphic_angular_gap_vector_v1"
    )
    assert not artifact["summary"]["permutation_only_repetition_detected"]
    assert artifact["summary"]["known_threshold_control_has_stronger_information"]
    assert not artifact["summary"]["hardware_execution_performed"]
    assert not artifact["summary"]["llm_execution_performed"]
    assert not artifact["summary"]["quantum_advantage_claimed"]
    assert not artifact["summary"]["fixed_confidence_calibration_claimed"]
    assert not artifact["summary"]["fully_information_matched_primary_baseline_available"]
    diagnostics = artifact["summary"]["paired_gap_aided_reference_query_diagnostic"]
    assert "paired_same_information_query_summary" not in artifact["summary"]
    assert len(diagnostics) == 10
    assert all(item["success_conditioned_descriptive_only"] for item in diagnostics)
    assert all(not item["information_matched"] for item in diagnostics)
    assert all(not item["resource_advantage_claimed"] for item in diagnostics)
    assert all("accuracy_matched_advantage_claimed" not in item for item in diagnostics)
    paired = artifact["summary"]["paired_gap_aided_reference_statistics"]
    assert len(paired["family_analyses"]) == 10
    assert len(paired["holm_fwer"]["adjustments"]) == 10
    assert not paired["comparison_information_matched"]
    assert paired["baseline_is_gap_aided"]
    assert all(
        item["analysis"]["resources"]["conditioning"]
        == "all_fixture_pairs_unconditional"
        for item in paired["family_analyses"]
    )
    assert artifact["resolved_config"]["problem_instances"] == 1
    assert artifact["resolved_config"]["paired_bootstrap_repetitions"] == 10000
    assert artifact["resolved_config"]["paired_bootstrap_confidence_level"] == 0.95
    assert artifact["resolved_config"]["paired_holm_alpha"] == 0.05
    assert artifact["resolved_config"]["instance_distribution"]["varied_axes"] == [
        "boundary_gap",
        "nonboundary_gaps",
        "selected_active_count",
        "rejected_active_count",
        "heterogeneity",
    ]
    public_instances = artifact["report"]["instances"]
    assert len(public_instances) == 10
    assert all(len(instance["difficulty_fingerprint"]) == 64 for instance in public_instances)
    assert all(instance["structure_metrics"]["n_arms"] >= 8 for instance in public_instances)
    assert all("difficulty_fingerprint" not in run for run in artifact["report"]["runs"])
    assert "no hardware, LLM, or advantage claim" in completed.stdout


def test_layer_c_artifact_keeps_all_three_information_regimes_separate(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "layer-c.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_frozen_quantum_reference_benchmarks.py"),
            "--output",
            str(output),
            "--problem-instances",
            "1",
        ],
        cwd=repository,
        env=_environment(repository),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    runs = artifact["report"]["runs"]
    qgapselect = [run for run in runs if run["method_id"] == "qgapselect"]
    gap_aided = [run for run in runs if run["method_id"] == "independent_iae_topk"]
    controls = [run for run in runs if run["method_id"] == "known_threshold_iae_scan"]
    assert qgapselect and gap_aided and controls
    assert all(run["information_matched_to_qgapselect"] for run in qgapselect)
    assert all(not run["information_matched_to_qgapselect"] for run in gap_aided)
    assert all(not run["information_matched_to_qgapselect"] for run in controls)
    assert {run["information_regime"] for run in qgapselect} == {"k_only"}
    assert {run["information_regime"] for run in gap_aided} == {"k_and_public_gap_floor"}
    assert {run["information_regime"] for run in controls} == {
        "k_public_gap_floor_and_public_threshold"
    }
    assert all("public_threshold" in run["algorithm_inputs"] for run in controls)
    assert all("public_threshold" not in run["algorithm_inputs"] for run in gap_aided)
    assert all("same_information" not in run for run in runs)


def test_script_chunking_preserves_the_complete_artifact_schema_and_values() -> None:
    repository = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(
        str(repository / "scripts" / "run_frozen_quantum_reference_benchmarks.py")
    )
    config = namespace["load_config"](
        repository / "configs" / "frozen_quantum_reference_benchmarks.json"
    )
    config["problem_instances"] = 2
    config["cases"] = config["cases"][:1]

    single_fixture = namespace["run_experiment"](config, instance_chunk_size=1)
    two_fixture_chunk = namespace["run_experiment"](config, instance_chunk_size=2)

    assert single_fixture == two_fixture_chunk
    assert single_fixture["report"] == two_fixture_chunk["report"]
    assert set(single_fixture) == {
        "artifact_type",
        "schema_version",
        "experiment_name",
        "claim_status",
        "config_sha256",
        "resolved_config",
        "summary",
        "report",
        "provenance",
    }


def test_instance_builder_is_deterministic_and_changes_difficulty_not_only_order() -> None:
    repository = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(
        str(repository / "scripts" / "run_frozen_quantum_reference_benchmarks.py")
    )
    config = namespace["load_config"](
        repository / "configs" / "frozen_quantum_reference_benchmarks.json"
    )
    config["problem_instances"] = 12
    config["cases"] = config["cases"][:1]

    def signatures() -> list[tuple[str, str, dict[str, object]]]:
        result = []
        for instance in namespace["build_instances"](config):
            result.append(
                (
                    instance.difficulty_fingerprint,
                    instance.fixture.manifest_hash,
                    dict(instance.structure_metrics),
                )
            )
        return result

    first = signatures()
    second = signatures()

    assert first == second
    assert len({fingerprint for fingerprint, _, _ in first}) >= 11
    assert len({round(metrics["heterogeneity"], 10) for _, _, metrics in first}) == 12


def test_paper_scale_fingerprint_gate_rejects_permutation_or_low_diversity() -> None:
    repository = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(
        str(repository / "scripts" / "run_frozen_quantum_reference_benchmarks.py")
    )
    diagnostic = namespace["_difficulty_fingerprint_diagnostic"]

    permutation_only = [
        {"family_id": "cell", "difficulty_fingerprint": "0" * 64} for _ in range(500)
    ]
    with pytest.raises(RuntimeError, match="arm permutations"):
        diagnostic(
            permutation_only,
            expected_instances_per_family=500,
            paper_scale_minimum_unique_fraction=0.95,
        )

    below_gate = [
        {
            "family_id": "cell",
            "difficulty_fingerprint": f"{index % 474:064x}",
        }
        for index in range(500)
    ]
    with pytest.raises(RuntimeError, match="474/500"):
        diagnostic(
            below_gate,
            expected_instances_per_family=500,
            paper_scale_minimum_unique_fraction=0.95,
        )

    at_gate = [
        {
            "family_id": "cell",
            "difficulty_fingerprint": f"{index % 475:064x}",
        }
        for index in range(500)
    ]
    result = diagnostic(
        at_gate,
        expected_instances_per_family=500,
        paper_scale_minimum_unique_fraction=0.95,
    )
    assert result["paper_scale_nonisomorphic_gate_passed"]
    assert result["families"][0]["unique_difficulty_fingerprint_count"] == 475
