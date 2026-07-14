from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qgapselect.reference_experiments import (
    REFERENCE_BACKEND,
    REFERENCE_CLAIM_STATUS,
    aggregate_raw_records,
    load_reference_config,
    resolve_reference_config,
    run_reference_experiments,
)


def _small_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_name": "reference-unit-test",
        "master_seed": 91,
        "trials": 4,
        "quantiles": [0.5, 0.75],
        "algorithm": {
            "confidence": 0.1,
            "initial_angular_epsilon": 0.25,
            "epsilon_decay": 0.5,
            "max_rounds": 1,
            "shots_per_iae_round": 64,
            "iae_max_rounds": 1,
            "iae_max_grover_power": 0,
            "iae_grid_points": 257,
        },
        "scenarios": [
            {
                "name": "deterministic",
                "means": [1.0, 0.0, 0.0],
                "k": 1,
                "metadata": {"kind": "resolved"},
            },
            {
                "name": "near_boundary",
                "means": [0.51, 0.5, 0.49],
                "k": 1,
                "metadata": {"kind": "timeout_stress"},
            },
        ],
        "notes": ["test-only low-cost simulator settings"],
    }


def test_config_parsing_uses_angular_fields_and_explicit_overrides(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(_small_document()), encoding="utf-8")

    config = load_reference_config(
        path,
        trials_override=2,
        seed_override=1234,
        scenario_names=["deterministic"],
    )

    assert config.trials == 2
    assert config.master_seed == 1234
    assert config.algorithm.initial_angular_epsilon == pytest.approx(0.25)
    assert [scenario.name for scenario in config.scenarios] == ["deterministic"]
    assert len(config.sha256) == 64
    assert config.as_dict()["algorithm"]["initial_angular_epsilon"] == 0.25

    legacy = _small_document()
    algorithm = dict(legacy["algorithm"])
    algorithm.pop("initial_angular_epsilon")
    algorithm["initial_epsilon"] = 0.25
    legacy["algorithm"] = algorithm
    with pytest.raises(ValueError, match="initial_epsilon|unknown algorithm"):
        resolve_reference_config(legacy)


def test_multiseed_reference_is_exactly_reproducible() -> None:
    config = resolve_reference_config(_small_document(), trials_override=3)

    first = run_reference_experiments(config)
    second = run_reference_experiments(config)

    assert first == second
    assert first["config_hash"] == config.sha256
    assert first["resolved_config"] == config.as_dict()
    assert len(first["raw_records"]) == 6
    assert all(record["backend"] == REFERENCE_BACKEND for record in first["raw_records"])
    assert all(
        record["claim_status"] == REFERENCE_CLAIM_STATUS
        for record in first["raw_records"]
    )


def test_raw_and_aggregate_metrics_are_consistent_and_separately_accounted() -> None:
    config = resolve_reference_config(_small_document(), trials_override=3)
    report = run_reference_experiments(config)
    raw = report["raw_records"]
    overall = report["aggregate"]["overall"]

    assert overall == {
        **aggregate_raw_records(raw, config.quantiles),
        "config_hash": config.sha256,
    }
    assert overall["trials"] == len(raw)
    assert overall["heuristic_inclusive_exact_recovery_count"] == sum(
        record["heuristic_inclusive_exact_recovery"] for record in raw
    )
    assert overall["certified_exact_recovery_count"] == sum(
        record["certified_exact_recovery"] for record in raw
    )
    assert overall["interval_resolved_count"] == sum(
        record["interval_resolved"] for record in raw
    )
    assert overall["timeout_count"] == sum(record["timeout"] for record in raw)
    comparable_count = overall["candidate_theory_accounting"][
        "comparable_complete_certificate_count"
    ]
    assert sum(overall["chosen_orientation_counts"].values()) == comparable_count
    assert comparable_count == overall["interval_resolved_count"]
    assert overall["candidate_theory_accounting"]["incomplete_trace_count"] == (
        overall["timeout_count"]
    )
    assert all(
        not record["certified_exact_recovery"]
        for record in raw
        if record["timeout"]
    )

    executed = [
        record["executed_query_accounting"]["coherent_queries"] for record in raw
    ]
    assert overall["executed_query_accounting"]["coherent_queries"][
        "mean"
    ] == pytest.approx(sum(executed) / len(executed))
    assert "candidate_theory_accounting" in overall
    assert (
        overall["candidate_theory_accounting"]["proof_status"]
        == "conjectural_not_a_query_bound"
    )
    assert all(
        set(record).isdisjoint({"candidate_queries", "theoretical_queries"})
        for record in raw
    )
    for record in raw:
        theory = record["candidate_theory_accounting"]
        if record["timeout"]:
            assert theory["comparison_status"] == (
                "incomplete_trace_not_comparable"
            )
            assert theory["chosen_orientation"] is None
            assert theory["chosen_charge"] is None
        else:
            assert theory["comparison_status"] == (
                "complete_certificate_trace_proxy"
            )
            assert theory["chosen_orientation"] in {
                "selected",
                "rejected_complement",
            }
            assert all(
                orientation["complete"]
                for orientation in theory["orientations"].values()
            )


def test_reference_cli_writes_resolved_hashed_artifact(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    config_path.write_text(json.dumps(_small_document()), encoding="utf-8")
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
            str(repository / "scripts" / "run_reference.py"),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--trials",
            "2",
            "--scenario",
            "deterministic",
        ],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["backend"] == REFERENCE_BACKEND
    assert artifact["claim_status"] == REFERENCE_CLAIM_STATUS
    assert artifact["resolved_config"]["trials"] == 2
    assert len(artifact["raw_records"]) == 2
    assert artifact["config_hash"]
    assert "not coherent batch execution" in completed.stdout
