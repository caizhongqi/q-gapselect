from __future__ import annotations

import json
from pathlib import Path

from qgapselect.proof_ledger import (
    CLAIM_STATUS,
    READINESS,
    build_proof_ledger,
    proof_ledger_markdown,
)


def _write_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    stopping = tmp_path / "stopping.json"
    composition = tmp_path / "composition.json"
    lower_bound = tmp_path / "lower-bound.json"
    stopping.write_text(
        json.dumps(
            {
                "summary": {
                    "check_status_counts": {"passed": 25, "proof_obligation": 20},
                    "all_execution_checks_passed": True,
                }
            }
        ),
        encoding="utf-8",
    )
    composition.write_text(
        json.dumps(
            {
                "summary": {
                    "novelty_gate_counts": {
                        "open_no_encoded_composition_match": 9,
                        "failed_encoded_composition_match": 4,
                    },
                    "strongest_valid_baseline_counts": {
                        "loop_variable_time_rebuild": 13
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    lower_bound.write_text(
        json.dumps(
            {
                "summary": {
                    "total_local_facts": 12,
                    "total_proof_obligations": 36,
                }
            }
        ),
        encoding="utf-8",
    )
    return stopping, composition, lower_bound


def test_proof_ledger_records_non_claimable_theorem_stack(tmp_path: Path) -> None:
    stopping, composition, lower_bound = _write_artifacts(tmp_path)

    ledger = build_proof_ledger(
        stopping_artifact=stopping,
        composition_artifact=composition,
        lower_bound_artifact=lower_bound,
    )

    assert ledger.claim_status == CLAIM_STATUS
    assert ledger.readiness == READINESS
    assert ledger.theorem_claimable is False
    assert ledger.ccf_a_claimable is False
    assert ledger.entry_count == 10
    assert ledger.status_counts["proof_obligation"] == 4
    assert ledger.status_counts["manual_instantiation_required"] == 2
    assert ledger.status_counts["local_fact"] == 1
    assert any(entry.entry_id == "LB-B03" for entry in ledger.entries)


def test_proof_ledger_markdown_preserves_boundaries(tmp_path: Path) -> None:
    stopping, composition, lower_bound = _write_artifacts(tmp_path)
    ledger = build_proof_ledger(
        stopping_artifact=stopping,
        composition_artifact=composition,
        lower_bound_artifact=lower_bound,
    )

    markdown = proof_ledger_markdown(ledger)

    assert "CCF-A claimable: `False`" in markdown
    assert "LB-B04" in markdown
    assert "Next required manual proofs" in markdown
