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


def test_coherent_unknown_boundary_topk_cli_writes_fail_closed_s2_panel(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = repository / "configs" / "coherent_unknown_boundary_topk.json"
    output = tmp_path / "coherent-unknown-boundary.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_coherent_unknown_boundary_topk.py"),
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
    summary = artifact["summary"]
    assertions = artifact["aggregate_assertions"]
    records = {record["role"]: record for record in artifact["records"]}

    assert artifact["artifact_type"] == ("q_gapselect_coherent_unknown_boundary_topk_s2_panel")
    assert artifact["config_hash"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert artifact["provenance"]["config_sha256"] == artifact["config_hash"]
    assert artifact["provenance"]["git_commit"]
    assert artifact["provenance"]["git_tree"]
    assert artifact["provenance"]["python_implementation"]
    assert artifact["provenance"]["python_version"]
    assert artifact["provenance"]["platform"]
    assert set(records) == {
        "on_grid_two_arm",
        "on_grid_three_arm",
        "off_grid_two_arm",
        "exact_tie",
    }
    assert summary["case_count"] == 4
    assert summary["strict_case_count"] == 3
    assert summary["on_grid_complete_count"] == 2
    assert summary["cleanup_pass_count"] == 3
    assert summary["fail_closed_count"] == 2
    assert summary["exact_output_rate_on_strict_cases"] == 2 / 3
    assert summary["certificate_issued_count"] == 0
    assert summary["end_to_end_coherent_tiny_reference"] is True
    assert summary["generic_rounding_robustness_proved"] is False
    assert summary["new_query_upper_bound_proved"] is False
    assert summary["same_interface_composition_separation_proved"] is False
    assert summary["matching_lower_bound_proved"] is False
    assert summary["elementary_gate_ledger_available"] is False
    assert summary["transpiled_depth_available"] is False
    assert summary["compiled_ancilla_qubits_available"] is False
    assert summary["quantum_advantage_claimable"] is False
    assert summary["ccf_a_claimable"] is False
    assert assertions["all_assertions_passed"] is True
    assert assertions["no_elementary_gate_ledger_claimed"] is True
    assert assertions["no_transpiled_depth_claimed"] is True
    assert assertions["no_compiled_ancilla_count_claimed"] is True
    assert all(record["query_formula_reconciliation"]["reconciled"] for record in records.values())
    assert all(
        record["cleanup_identity"]["prediction_residual"] <= 1e-9 for record in records.values()
    )
    assert all(record["result"]["certificate_issued"] is False for record in records.values())
    assert all(
        record["result"]["quantum_advantage_claimable"] is False for record in records.values()
    )
    for record in records.values():
        resources = record["result"]["resources"]
        assert "gate_counts" not in resources
        assert "depth" not in resources
        assert resources["elementary_gate_ledger_available"] is False
        assert resources["transpiled_depth_available"] is False
        assert resources["transpiled_depth"] is None
        assert resources["compiled_ancilla_qubits_available"] is False
        assert resources["query_counts"]["coherent_total"] > 0
        assert resources["executed_numpy_kernel_operation_counts"]
        assert resources["logical_circuit_macro_counts"]
        assert resources["rank_compilation_proxies"]

    supports = artifact["claim_boundaries"]["supports"]
    does_not_support = artifact["claim_boundaries"]["does_not_support"]
    assert any("exact executed canonical-oracle query ledger" in item for item in supports)
    assert not any("executed oracle, gate, depth" in item for item in supports)
    assert any("elementary-gate count" in item for item in does_not_support)
    assert any("transpiled" in item and "depth" in item for item in does_not_support)

    assert records["on_grid_two_arm"]["result"]["membership_mask"] == 0b01
    assert records["on_grid_three_arm"]["result"]["membership_mask"] == 0b011
    off_grid = records["off_grid_two_arm"]
    assert off_grid["result"]["membership_mask"] is None
    assert off_grid["cleanup_identity"]["passed"] is False
    assert off_grid["cleanup_identity"]["executed_garbage_probability"] > 0.03
    assert off_grid["cleanup_identity"]["predicted_garbage_probability"] > 0.03
    tie = records["exact_tie"]
    assert tie["result"]["membership_mask"] is None
    assert tie["cleanup_identity"]["passed"] is True
    assert tie["trusted_scoring"]["truth"]["strict_boundary"] is False
    assert "not a certificate" in completed.stdout


def test_coherent_unknown_boundary_topk_cli_rejects_abbreviated_arguments(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "must-not-exist.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_coherent_unknown_boundary_topk.py"),
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
