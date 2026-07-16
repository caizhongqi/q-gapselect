from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from qgapselect.research_gap_audit import (
    READINESS,
    build_research_gap_audit,
    research_gap_markdown,
    summarize_charged_activity,
    summarize_composition_frontier,
    summarize_lower_bound_program,
    summarize_proof_ledger,
    summarize_stopping_transducer,
    summarize_stopping_unitary_theorem,
    summarize_variable_time_charged,
)


def _minimal_quantum_artifact(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "experiment_name": "minimal-quantum",
                "claim_status": "finite_exact_state_and_analytic_diagnostics",
                "selected_suites": [
                    "generic_composition_audit",
                    "unknown_boundary_history",
                ],
                "suite_results": {
                    "generic_composition_audit": {
                        "raw_records": [{"m": 8}, {"m": 16}],
                        "summary": {
                            "novelty_gate": "failed_explicit_family",
                            "explicit_family_novelty_failure_count": 2,
                        },
                    },
                    "unknown_boundary_history": {
                        "raw_records": [{"m": 8}],
                        "summary": {
                            "novelty_gate_counts": {
                                "open_no_encoded_baseline_match_requires_unitary_and_lower_bound": 1
                            },
                            "last_point_min_valid_baseline_over_candidate": 2.0,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _minimal_grid_artifact(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "experiment_name": "minimal-grid",
                "claim_status": "analytic_unknown_boundary_history_grid_no_theorem",
                "summary": {
                    "case_count": 2,
                    "total_records": 4,
                    "novelty_gate_counts": {
                        "open_no_encoded_baseline_match_requires_unitary_and_lower_bound": 3,
                        "failed_encoded_baseline_match": 1,
                    },
                    "strongest_encoded_valid_baseline_counts": {
                        "variable_time_rebuild_rms": 4
                    },
                },
                "case_results": [
                    {"name": "open", "summary": {"all_points_open": True}},
                    {"name": "failed", "summary": {"all_points_open": False}},
                ],
            }
        ),
        encoding="utf-8",
    )


def _minimal_charged_artifact(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "experiment_name": "minimal-charged",
                "claim_status": "charged_phase_history_prototype_no_upper_bound_theorem",
                "summary": {
                    "case_count": 1,
                    "total_records": 2,
                    "all_no_supplied_predicate_rows": True,
                    "all_output_subset_active": True,
                    "exact_state_trace_count": 2,
                },
            }
        ),
        encoding="utf-8",
    )


def _minimal_variable_time_artifact(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "experiment_name": "minimal-variable-time",
                "claim_status": "variable_time_charged_history_alignment_no_theorem",
                "summary": {
                    "case_count": 2,
                    "total_records": 4,
                    "novelty_gate_counts": {
                        "open_charged_variable_time_gap_requires_upper_and_lower_bound": 3,
                        "failed_charged_baseline_match": 1,
                    },
                    "strongest_valid_baseline_counts": {
                        "variable_time_rebuild_rms": 4
                    },
                    "open_case_count": 1,
                },
                "case_results": [
                    {"name": "open", "summary": {"all_points_open": True}},
                    {"name": "failed", "summary": {"all_points_open": False}},
                ],
            }
        ),
        encoding="utf-8",
    )


def _minimal_stopping_artifact(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "experiment_name": "minimal-stopping",
                "claim_status": "variable_time_stopping_transducer_skeleton_no_theorem",
                "summary": {
                    "case_count": 1,
                    "total_records": 2,
                    "exact_state_trace_count": 2,
                    "all_variable_over_serial_at_most_one": True,
                },
            }
        ),
        encoding="utf-8",
    )


def _minimal_theorem_artifact(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "experiment_name": "minimal-theorem",
                "claim_status": "stopping_time_unitary_lemma_scaffold_no_proof",
                "summary": {
                    "case_count": 1,
                    "total_records": 2,
                    "check_status_counts": {
                        "passed": 10,
                        "proof_obligation": 8,
                    },
                    "all_execution_checks_passed": True,
                },
            }
        ),
        encoding="utf-8",
    )


def _minimal_composition_artifact(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "experiment_name": "minimal-composition",
                "claim_status": "composition_frontier_audit_no_novelty_theorem",
                "summary": {
                    "case_count": 2,
                    "total_records": 4,
                    "novelty_gate_counts": {
                        "open_no_encoded_composition_match": 3,
                        "failed_encoded_composition_match": 1,
                    },
                    "strongest_valid_baseline_counts": {
                        "loop_variable_time_rebuild": 4
                    },
                    "open_case_count": 1,
                },
                "case_results": [
                    {"name": "open", "summary": {"all_points_open": True}},
                    {"name": "failed", "summary": {"all_points_open": False}},
                ],
            }
        ),
        encoding="utf-8",
    )


def _minimal_lower_bound_artifact(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "experiment_name": "minimal-lower-bound",
                "claim_status": "lower_bound_program_scaffold_no_adversary_proof",
                "summary": {
                    "case_count": 1,
                    "total_records": 2,
                    "strongest_block_counts": {"LB-B03": 2},
                    "total_proof_obligations": 6,
                    "total_local_facts": 2,
                },
            }
        ),
        encoding="utf-8",
    )


def _minimal_proof_ledger_artifact(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_type": "q_gapselect_proof_ledger",
                "ledger": {
                    "claim_status": "proof_ledger_started_no_quantum_advantage_theorem",
                    "readiness": "proof_program_structured_not_ccf_a_claimable",
                    "entry_count": 10,
                    "status_counts": {
                        "execution_checked": 1,
                        "local_fact": 1,
                        "manual_instantiation_required": 2,
                        "proof_obligation": 4,
                        "proof_outline_started": 2,
                    },
                    "theorem_claimable": False,
                    "ccf_a_claimable": False,
                },
            }
        ),
        encoding="utf-8",
    )


def _environment(repository: Path) -> dict[str, str]:
    environment = dict(os.environ)
    source = str(repository / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )
    return environment


def test_research_gap_audit_builds_prioritized_gap_records(tmp_path: Path) -> None:
    quantum = tmp_path / "quantum.json"
    grid = tmp_path / "grid.json"
    _minimal_quantum_artifact(quantum)
    _minimal_grid_artifact(grid)

    audit = build_research_gap_audit(quantum, grid)

    assert audit.readiness == READINESS
    assert audit.stage == "active_unknown_boundary_history_candidate_no_theorem"
    assert len(audit.artifact_evidence) == 2
    assert audit.artifact_evidence[0].record_count == 3
    assert audit.artifact_evidence[1].key_metrics["failed_cases"] == ["failed"]
    assert audit.top_gaps[0].gap_id == "P0-U08"
    assert any("H_orient" in item for item in audit.rejected_claims)
    markdown = research_gap_markdown(audit)
    assert "Highest-priority gaps" in markdown
    assert "activity-history transducer" in markdown


def test_research_gap_audit_promotes_charged_activity_checkpoint(tmp_path: Path) -> None:
    quantum = tmp_path / "quantum.json"
    grid = tmp_path / "grid.json"
    charged = tmp_path / "charged.json"
    _minimal_quantum_artifact(quantum)
    _minimal_grid_artifact(grid)
    _minimal_charged_artifact(charged)

    evidence = summarize_charged_activity(charged)
    audit = build_research_gap_audit(quantum, grid, charged)

    assert evidence.record_count == 2
    assert evidence.key_metrics["all_no_supplied_predicate_rows"] is True
    assert len(audit.artifact_evidence) == 3
    assert audit.top_gaps[0].current_status == "charged_phase_prototype_started"
    assert "finite-phase" in audit.top_gaps[0].evidence_now


def test_research_gap_audit_promotes_variable_time_mainline_alignment(
    tmp_path: Path,
) -> None:
    quantum = tmp_path / "quantum.json"
    grid = tmp_path / "grid.json"
    charged = tmp_path / "charged.json"
    variable_time = tmp_path / "variable-time.json"
    _minimal_quantum_artifact(quantum)
    _minimal_grid_artifact(grid)
    _minimal_charged_artifact(charged)
    _minimal_variable_time_artifact(variable_time)

    evidence = summarize_variable_time_charged(variable_time)
    audit = build_research_gap_audit(quantum, grid, charged, variable_time)

    assert evidence.record_count == 4
    assert evidence.key_metrics["open_case_count"] == 1
    assert len(audit.artifact_evidence) == 4
    assert audit.top_gaps[0].current_status == (
        "mainline_variable_time_alignment_started"
    )
    assert "finite-QPE level costs" in audit.top_gaps[0].evidence_now


def test_research_gap_audit_promotes_stopping_relation_skeleton(
    tmp_path: Path,
) -> None:
    quantum = tmp_path / "quantum.json"
    grid = tmp_path / "grid.json"
    charged = tmp_path / "charged.json"
    variable_time = tmp_path / "variable-time.json"
    stopping = tmp_path / "stopping.json"
    _minimal_quantum_artifact(quantum)
    _minimal_grid_artifact(grid)
    _minimal_charged_artifact(charged)
    _minimal_variable_time_artifact(variable_time)
    _minimal_stopping_artifact(stopping)

    evidence = summarize_stopping_transducer(stopping)
    audit = build_research_gap_audit(quantum, grid, charged, variable_time, stopping)

    assert evidence.record_count == 2
    assert evidence.key_metrics["all_variable_over_serial_at_most_one"] is True
    assert len(audit.artifact_evidence) == 5
    assert audit.top_gaps[0].current_status == "stopping_relation_skeleton_started"
    assert "stopping-register" in audit.top_gaps[0].evidence_now


def test_research_gap_audit_promotes_stopping_unitary_theorem_scaffold(
    tmp_path: Path,
) -> None:
    quantum = tmp_path / "quantum.json"
    grid = tmp_path / "grid.json"
    charged = tmp_path / "charged.json"
    variable_time = tmp_path / "variable-time.json"
    stopping = tmp_path / "stopping.json"
    theorem = tmp_path / "theorem.json"
    _minimal_quantum_artifact(quantum)
    _minimal_grid_artifact(grid)
    _minimal_charged_artifact(charged)
    _minimal_variable_time_artifact(variable_time)
    _minimal_stopping_artifact(stopping)
    _minimal_theorem_artifact(theorem)

    evidence = summarize_stopping_unitary_theorem(theorem)
    audit = build_research_gap_audit(
        quantum, grid, charged, variable_time, stopping, theorem
    )

    assert evidence.record_count == 2
    assert evidence.key_metrics["all_execution_checks_passed"] is True
    assert len(audit.artifact_evidence) == 6
    assert audit.top_gaps[0].current_status == (
        "stopping_unitary_lemma_scaffold_started"
    )
    assert "proof obligations" in audit.top_gaps[0].evidence_now


def test_research_gap_audit_promotes_composition_and_lower_bound_programs(
    tmp_path: Path,
) -> None:
    quantum = tmp_path / "quantum.json"
    grid = tmp_path / "grid.json"
    charged = tmp_path / "charged.json"
    variable_time = tmp_path / "variable-time.json"
    stopping = tmp_path / "stopping.json"
    theorem = tmp_path / "theorem.json"
    composition = tmp_path / "composition.json"
    lower_bound = tmp_path / "lower-bound.json"
    _minimal_quantum_artifact(quantum)
    _minimal_grid_artifact(grid)
    _minimal_charged_artifact(charged)
    _minimal_variable_time_artifact(variable_time)
    _minimal_stopping_artifact(stopping)
    _minimal_theorem_artifact(theorem)
    _minimal_composition_artifact(composition)
    _minimal_lower_bound_artifact(lower_bound)

    composition_evidence = summarize_composition_frontier(composition)
    lower_bound_evidence = summarize_lower_bound_program(lower_bound)
    audit = build_research_gap_audit(
        quantum,
        grid,
        charged,
        variable_time,
        stopping,
        theorem,
        composition,
        lower_bound,
    )
    gap_status = {gap.gap_id: gap.current_status for gap in audit.top_gaps}

    assert composition_evidence.key_metrics["open_case_count"] == 1
    assert lower_bound_evidence.key_metrics["total_proof_obligations"] == 6
    assert len(audit.artifact_evidence) == 8
    assert gap_status["P1-COMP"] == "composition_frontier_started"
    assert gap_status["P0-L07"] == "lower_bound_program_started"


def test_research_gap_audit_promotes_proof_ledger(tmp_path: Path) -> None:
    quantum = tmp_path / "quantum.json"
    grid = tmp_path / "grid.json"
    charged = tmp_path / "charged.json"
    variable_time = tmp_path / "variable-time.json"
    stopping = tmp_path / "stopping.json"
    theorem = tmp_path / "theorem.json"
    composition = tmp_path / "composition.json"
    lower_bound = tmp_path / "lower-bound.json"
    proof_ledger = tmp_path / "proof-ledger.json"
    _minimal_quantum_artifact(quantum)
    _minimal_grid_artifact(grid)
    _minimal_charged_artifact(charged)
    _minimal_variable_time_artifact(variable_time)
    _minimal_stopping_artifact(stopping)
    _minimal_theorem_artifact(theorem)
    _minimal_composition_artifact(composition)
    _minimal_lower_bound_artifact(lower_bound)
    _minimal_proof_ledger_artifact(proof_ledger)

    evidence = summarize_proof_ledger(proof_ledger)
    audit = build_research_gap_audit(
        quantum,
        grid,
        charged,
        variable_time,
        stopping,
        theorem,
        composition,
        lower_bound,
        proof_ledger,
    )
    gap_status = {gap.gap_id: gap.current_status for gap in audit.top_gaps}

    assert evidence.key_metrics["ccf_a_claimable"] is False
    assert evidence.record_count == 10
    assert len(audit.artifact_evidence) == 9
    assert gap_status["P1-COMP"] == "proof_ledger_started"
    assert gap_status["P0-L07"] == "proof_ledger_started"
    assert audit.recommended_next_steps[0].startswith("Convert proof-ledger")


def test_research_gap_audit_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    quantum = tmp_path / "quantum.json"
    grid = tmp_path / "grid.json"
    output = tmp_path / "audit.json"
    markdown = tmp_path / "audit.md"
    _minimal_quantum_artifact(quantum)
    _minimal_grid_artifact(grid)

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_research_gap_audit.py"),
            "--quantum-artifact",
            str(quantum),
            "--grid-artifact",
            str(grid),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
        ],
        cwd=repository,
        env=_environment(repository),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["readiness"] == READINESS
    assert artifact["top_gaps"][0]["priority"] == "P0"
    assert "pre_theorem" in completed.stdout
    assert "Q-GapSelect research gap audit" in markdown.read_text(encoding="utf-8")
