from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qgapselect.experiments import (
    canonical_scaling_suite,
    equal_gap_instance,
    evaluate_instance,
    fit_loglog_slopes,
    heterogeneous_gap_instance,
    partition_direct_sum_instance,
)


def test_generators_are_seeded_and_preserve_declared_boundary() -> None:
    first = equal_gap_instance(20, 7, 0.125, seed=123)
    second = equal_gap_instance(20, 7, 0.125, seed=123)
    different = equal_gap_instance(20, 7, 0.125, seed=124)

    assert first.means == second.means
    assert first.means != different.means
    assert first.metadata["seed"] == 123

    heterogeneous = heterogeneous_gap_instance(
        20, 7, 0.03125, spread=8.0, seed=123
    )
    records = evaluate_instance(heterogeneous)
    assert all(record.min_mean_gap == pytest.approx(0.03125) for record in records)
    assert all(record.max_mean_gap > record.min_mean_gap for record in records)
    assert all(record.min_angular_gap > 0.0 for record in records)
    assert all(record.max_angular_gap > record.min_angular_gap for record in records)


def test_canonical_suite_covers_all_preregistered_cases_and_methods() -> None:
    records = canonical_scaling_suite((8, 16), gap=0.125, spread=4.0, seed=99)
    scenarios = {record.scenario for record in records}
    methods = {record.method for record in records}

    assert scenarios == {
        "equal_gap_k1",
        "equal_gap_half",
        "heterogeneous_k1",
        "heterogeneous_half",
        "partition_equal_gap_g4_s2",
        "partition_equal_gap_g8_s2",
    }
    assert methods == {
        "candidate_layer",
        "prior_uniform_ae_kmin",
        "repeated_qbai",
        "classical_information",
        "classical_uniform",
    }
    assert all(record.data_source == "analytic_expression" for record in records)
    assert all(record.claim_status for record in records)
    assert all("seed" in record.metadata for record in records)


def test_partition_benchmark_size_and_required_outputs() -> None:
    instance = partition_direct_sum_instance(6, 4, 0.125, seed=4)

    assert instance.n == 24
    assert instance.k == 6
    assert len(instance.groups) == 6


def test_descriptive_slopes_distinguish_k1_from_half_output() -> None:
    records = canonical_scaling_suite((16, 32, 64, 128), gap=0.125)
    slopes = {
        (item.scenario, item.method): item
        for item in fit_loglog_slopes(records)
    }

    k1 = slopes[("equal_gap_k1", "candidate_layer")]
    half = slopes[("equal_gap_half", "candidate_layer")]
    partition = slopes[("partition_equal_gap", "candidate_layer")]
    assert k1.slope == pytest.approx(0.5)
    assert half.slope > 0.85
    assert partition.slope == pytest.approx(1.0)
    assert k1.claim_status == "descriptive_fit_to_analytic_proxy"


def test_scaling_script_emits_machine_readable_claim_provenance(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "scaling.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 2718,
                "sizes": [8, 32],
                "regimes": ["best_arm"],
                    "mean_gap": 0.25,
                "spread": 2.0,
                "trials": 11,
                "delta": 0.1,
            }
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    source_path = str(repository / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not environment.get("PYTHONPATH")
        else source_path + os.pathsep + environment["PYTHONPATH"]
    )
    completed = subprocess.run(
        [
            sys._base_executable,
            str(repository / "scripts" / "run_scaling.py"),
            "--config",
            str(config_path),
            "--sizes",
            "8,16",
            "--format",
            "json",
            "--seed",
            "31415",
        ],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(completed.stdout)
    assert artifact["artifact_type"] == "analytic_complexity_proxy_table"
    assert artifact["provenance"]["seed"] == 31415
    resolved = artifact["provenance"]["resolved_executable_config"]
    assert resolved == {
        "sizes": [8, 16],
        "mean_gap": 0.25,
        "spread": 2.0,
        "seed": 31415,
        "regimes": ["best_arm"],
    }
    assert artifact["provenance"]["source_config"]["seed"] == 2718
    assert artifact["provenance"]["config_sha256"]
    assert artifact["provenance"]["declared_but_unused_by_analytic_run"] == {
        "trials": 11,
        "delta": 0.1,
    }
    assert "not observed quantum query counts" in artifact["provenance"][
        "claim_boundary"
    ]
    assert artifact["records"]
    assert all(record["claim_status"] for record in artifact["records"])
    assert {record["scenario"] for record in artifact["records"]} == {
        "equal_gap_k1",
        "heterogeneous_k1",
    }
