from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_theorem_closure_cli_materializes_blocked_chain(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "closure.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts" / "run_theorem_closure_audit.py"),
            "--config",
            str(repository / "configs" / "theorem_closure_audit.json"),
            "--output",
            str(output),
        ),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "q_gapselect_unified_theorem_closure_audit"
    assert artifact["summary"]["scale_count"] == 7
    assert artifact["summary"][
        "public_partition_composition_matches_every_scale"
    ]
    assert artifact["summary"]["hidden_partition_upper_bound_open_every_scale"]
    assert artifact["summary"]["weighted_matching_lower_bound_open_every_scale"]
    assert artifact["summary"][
        "public_composition_asymptotically_beats_candidate_proxy"
    ]
    assert not artifact["summary"]["theorem_chain_closed"]
    assert not artifact["summary"]["ccf_a_quantum_advantage_claimable"]
    assert all(
        not row["public_instantiation"]["strict_composition_advantage_established"]
        for row in artifact["audits"]
    )
    assert all(
        not row["lower_bound"]["matching_lower_bound_established"]
        for row in artifact["status_maps"]
    )
    assert "no quantum-advantage claim" in completed.stdout
