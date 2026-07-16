from __future__ import annotations

from copy import deepcopy

import pytest

from qgapselect.s3_evidence_audit import audit_s3_evidence


def _coherent_record(
    role: str,
    *,
    output_status: str,
    cleanup_passed: bool,
    diagnostic_mask_exact: bool | None,
) -> dict:
    counts = {
        "forward": 4,
        "inverse": 4,
        "controlled_forward": 84,
        "controlled_inverse": 84,
        "classical_sample": 0,
        "coherent_total": 176,
        "classical_total": 0,
        "total": 176,
        "qram_queries": 0,
    }
    levels = [
        {
            "runtime_derived_one_way_counts": {"coherent_total": query_count},
            "full_replay_reconciled": True,
            "one_way_reconciled": True,
        }
        for query_count in (28, 60)
    ]
    return {
        "role": role,
        "trusted_fixture_and_scoring": {
            "diagnostic_mask_exact": diagnostic_mask_exact,
        },
        "result": {
            "output_status": output_status,
            "fixed_expected_query_ledger_respected": True,
            "budget_valid": True,
            "quantum_advantage_claimable": False,
            "history": {
                "single_statevector_history_register": True,
                "later_level_oracles_controlled_by_active_flag": True,
            },
            "durable_output": {
                "scratch_to_durable_copy_executed": True,
                "full_history_replay_executed": True,
            },
            "certificate": {"issued": False},
            "resources": {
                "cleanup": {"passed": cleanup_passed},
                "query_ledger": {
                    "query_counts": counts,
                    "expected_query_counts": dict(counts),
                    "reconciled": True,
                    "qram_assumed": False,
                    "per_level_runtime_records": levels,
                },
            },
        },
    }


def _documents() -> tuple[dict, dict, dict, dict]:
    adaptive = {
        "schema_version": 1,
        "artifact_type": "q_gapselect_adaptive_unknown_boundary_topk_s3_panel",
        "summary": {
            "case_count": 5,
            "quantum_advantage_claimable": False,
            "ccf_a_claimable": False,
        },
        "aggregate_assertions": {"all_assertions_passed": True},
    }
    coherent = {
        "schema_version": 1,
        "artifact_type": (
            "q_gapselect_tiny_true_coherent_stopping_history_s3_panel"
        ),
        "summary": {
            "case_count": 5,
            "exact_grid_complete_count": 3,
            "fail_closed_count": 2,
            "executed_queries_per_case": 176,
            "true_coherent_stopping_history_unitary_implemented": True,
            "later_level_active_control_implemented": True,
            "durable_copy_and_full_replay_implemented": True,
            "generic_off_grid_cleanup_proved": False,
            "variable_time_query_speedup_proved": False,
            "new_query_upper_bound_proved": False,
            "quantum_advantage_claimable": False,
            "ccf_a_claimable": False,
        },
        "aggregate_assertions": {"all_assertions_passed": True},
        "records": [
            _coherent_record(
                "exact_grid_first_stop",
                output_status="MASK",
                cleanup_passed=True,
                diagnostic_mask_exact=True,
            ),
            _coherent_record(
                "exact_grid_second_stop",
                output_status="MASK",
                cleanup_passed=True,
                diagnostic_mask_exact=True,
            ),
            _coherent_record(
                "exact_grid_arm1_winner",
                output_status="MASK",
                cleanup_passed=True,
                diagnostic_mask_exact=True,
            ),
            _coherent_record(
                "exact_grid_tie",
                output_status="INCONCLUSIVE",
                cleanup_passed=True,
                diagnostic_mask_exact=None,
            ),
            _coherent_record(
                "off_grid_fail_closed",
                output_status="INCONCLUSIVE",
                cleanup_passed=False,
                diagnostic_mask_exact=None,
            ),
        ],
        "inactive_level_clean_dirty_subspace_audit": {
            "clean_identity_witness_passed": True,
            "dirty_negative_control_activated": True,
        },
        "config_hash": "a" * 64,
        "provenance": {
            "config_sha256": "a" * 64,
            "source_tree_dirty_at_execution": False,
        },
    }
    frontier = {
        "schema_version": 1,
        "artifact_type": "q_gapselect_frontier_lower_bound_witness_s3",
        "summary": {
            "pair_hybrid_witness_count": 3,
            "pair_hybrid_all_verified": True,
            "johnson_witness_count": 3,
            "johnson_all_verified": True,
            "composition_kill_count": 0,
            "finite_fixture_query_dominance_count": 2,
            "uncovered_required_baseline_ids": [f"baseline-{index}" for index in range(9)],
            "registered_strongest_composition_coverage_complete": False,
            "matching_lower_bound_claimable": False,
            "strongest_composition_claimable": False,
            "quantum_advantage_claimable": False,
            "ccf_a_claimable": False,
        },
    }
    composition = {
        "schema_version": 1,
        "artifact_type": "q_gapselect_same_interface_strong_composition_s3",
        "aggregate_audit": {
            "all_checks_passed": True,
            "attempt_count": 486,
            "certified_exact_count": 59,
            "inconclusive_count": 427,
            "incorrect_certificate_count": 0,
            "budget_violation_count": 0,
        },
        "claim_boundary": {
            "official_literature_reproduction": False,
            "candidate_included_in_cer_panel": False,
            "paired_candidate_cer_superiority_verified": False,
            "claim_bearing_sample_size_met": False,
            "quantum_advantage_claimed": False,
            "ccf_a_claimable": False,
        },
    }
    return adaptive, coherent, frontier, composition


def test_s3_stack_closes_only_the_tiny_circuit_gate() -> None:
    report = audit_s3_evidence(*_documents())

    assert report.stage == "s3_tiny_true_coherent_kernel_theorem_stack_open"
    assert report.satisfied_gate_count == 1
    assert report.open_gate_count == 6
    assert report.checkpoint_count == 1
    assert report.satisfied_checkpoint_count == 1
    assert report.claim_gate_count == 6
    assert report.satisfied_claim_gate_count == 0
    assert report.open_claim_gate_count == 6
    assert report.execution_audits_passed
    assert report.upstream_claims_remain_false
    assert not report.upstream_positive_claim_flags_detected
    assert not report.independent_theorem_verifier_available
    assert report.theorem_claim_activation_locked
    assert report.coherent_executed_queries_per_case == 176
    assert report.composition_attempt_count == 486
    assert report.composition_certified_exact_count == 59
    assert report.composition_inconclusive_count == 427
    assert report.composition_incorrect_certificate_count == 0
    assert report.composition_budget_violation_count == 0
    assert report.uncovered_required_baseline_count == 9
    assert report.finite_composition_kill_count == 0
    assert report.finite_comparator_dominance_count == 2
    assert report.coherent_record_audit_passed
    assert not report.quantum_advantage_claimable
    assert not report.ccf_a_claimable
    gates = {gate.gate_id: gate for gate in report.gates}
    assert gates["S3-C0-TINY-CIRCUIT-CHECKPOINT"].status == (
        "VERIFIED_TINY_PROMISE_DOMAIN"
    )
    assert gates["S3-G2-EMPIRICAL-CER"].status == (
        "OPEN_CANDIDATE_NOT_IN_CER_PANEL"
    )
    assert gates["S3-G5-COMPOSITION-SEPARATION"].status == (
        "OPEN_S3_FINITE_DIAGNOSTIC_DOMINATED"
    )


def test_upstream_overclaim_is_detected_but_cannot_open_the_gate() -> None:
    documents = list(_documents())
    forged = deepcopy(documents[0])
    forged["summary"]["quantum_advantage_claimable"] = True
    documents[0] = forged

    report = audit_s3_evidence(*documents)

    assert not report.upstream_claims_remain_false
    assert report.upstream_positive_claim_flags_detected
    assert not report.quantum_advantage_claimable
    assert not report.ccf_a_claimable


def test_forging_every_upstream_theorem_flag_cannot_open_claim_gates() -> None:
    adaptive, coherent, frontier, composition = deepcopy(_documents())
    coherent["summary"].update(
        {
            "generic_off_grid_cleanup_proved": True,
            "variable_time_query_speedup_proved": True,
            "new_query_upper_bound_proved": True,
        }
    )
    frontier["summary"].update(
        {
            "registered_strongest_composition_coverage_complete": True,
            "matching_lower_bound_claimable": True,
            "strongest_composition_claimable": True,
        }
    )
    composition["claim_boundary"]["official_literature_reproduction"] = True

    report = audit_s3_evidence(adaptive, coherent, frontier, composition)

    assert report.satisfied_gate_count == 1
    assert report.open_gate_count == 6
    assert report.satisfied_claim_gate_count == 0
    assert report.upstream_positive_claim_flags_detected
    assert not report.upstream_claims_remain_false
    assert all(not gate.satisfied for gate in report.gates[1:])
    theory_gate_ids = {
        "S3-G1-CIRCUIT-GENERIC",
        "S3-G3-VARIABLE-TIME-UPPER",
        "S3-G4-STRONG-COMPOSITION-FIDELITY",
        "S3-G5-COMPOSITION-SEPARATION",
        "S3-G6-MATCHING-LOWER-BOUND",
    }
    assert all(
        "UNVERIFIED_UPSTREAM_FLAG" in gate.status
        for gate in report.gates
        if gate.gate_id in theory_gate_ids
    )
    assert not report.quantum_advantage_claimable
    assert not report.ccf_a_claimable


def test_tiny_circuit_gate_recomputes_records_instead_of_trusting_aggregate_flag() -> None:
    adaptive, coherent, frontier, composition = deepcopy(_documents())
    coherent["records"][0]["result"]["resources"]["query_ledger"][
        "query_counts"
    ]["controlled_forward"] = 83

    report = audit_s3_evidence(adaptive, coherent, frontier, composition)

    assert coherent["aggregate_assertions"]["all_assertions_passed"] is True
    assert not report.coherent_record_audit_passed
    assert not report.gates[0].satisfied
    assert report.satisfied_gate_count == 0
    assert report.satisfied_claim_gate_count == 0
    assert not report.quantum_advantage_claimable


def test_wrong_artifact_type_and_non_boolean_claims_fail_closed() -> None:
    documents = list(_documents())
    documents[1] = {**documents[1], "artifact_type": "wrong"}
    with pytest.raises(ValueError, match="artifact_type"):
        audit_s3_evidence(*documents)

    documents = list(_documents())
    documents[2]["summary"]["matching_lower_bound_claimable"] = "PROVED"
    with pytest.raises(TypeError, match="matching lower-bound"):
        audit_s3_evidence(*documents)
