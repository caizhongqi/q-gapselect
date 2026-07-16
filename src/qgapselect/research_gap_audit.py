"""Paper-readiness and research-gap audit for the Q-GapSelect program.

The goal of this module is not to score a paper automatically.  It creates a
structured, reproducible checkpoint that ties the current artifacts to the
remaining proof and experiment obligations.  This makes the CCF-A gap explicit:
which claims are supported by code-sanity evidence, which claims are still
proof obligations, and which candidate lines were already rejected.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CURRENT_STAGE = "active_unknown_boundary_history_candidate_no_theorem"
READINESS = "pre_theorem_not_ccf_a_ready"


@dataclass(frozen=True, slots=True)
class ArtifactEvidence:
    """Compact evidence extracted from one artifact."""

    name: str
    path: str
    status: str
    record_count: int
    key_metrics: dict[str, Any]
    interpretation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "status": self.status,
            "record_count": self.record_count,
            "key_metrics": self.key_metrics,
            "interpretation": self.interpretation,
        }


@dataclass(frozen=True, slots=True)
class ResearchGap:
    """One remaining paper obligation."""

    gap_id: str
    priority: str
    title: str
    current_status: str
    evidence_now: str
    missing_evidence: str
    next_code_or_proof_artifact: str
    activation_target: str
    failure_mode: str

    def as_dict(self) -> dict[str, str]:
        return {
            "gap_id": self.gap_id,
            "priority": self.priority,
            "title": self.title,
            "current_status": self.current_status,
            "evidence_now": self.evidence_now,
            "missing_evidence": self.missing_evidence,
            "next_code_or_proof_artifact": self.next_code_or_proof_artifact,
            "activation_target": self.activation_target,
            "failure_mode": self.failure_mode,
        }


@dataclass(frozen=True, slots=True)
class ResearchGapAudit:
    """Complete current-state audit."""

    stage: str
    readiness: str
    artifact_evidence: tuple[ArtifactEvidence, ...]
    completed_evidence: tuple[str, ...]
    rejected_claims: tuple[str, ...]
    top_gaps: tuple[ResearchGap, ...]
    recommended_next_steps: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "readiness": self.readiness,
            "artifact_evidence": [item.as_dict() for item in self.artifact_evidence],
            "completed_evidence": list(self.completed_evidence),
            "rejected_claims": list(self.rejected_claims),
            "top_gaps": [item.as_dict() for item in self.top_gaps],
            "recommended_next_steps": list(self.recommended_next_steps),
        }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing artifact: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _suite_record_count(result: Any) -> int:
    if not isinstance(result, dict):
        return 0
    raw = result.get("raw_records", [])
    return len(raw) if isinstance(raw, list) else 0


def summarize_quantum_benchmark(path: Path) -> ArtifactEvidence:
    """Extract the current quantum-core evidence summary."""

    data = _load_json(path)
    suite_results = data.get("suite_results", {})
    if not isinstance(suite_results, dict):
        raise TypeError("quantum benchmark artifact missing suite_results object")
    record_count = sum(_suite_record_count(item) for item in suite_results.values())
    selected_suites = data.get("selected_suites", [])
    history = suite_results.get("unknown_boundary_history", {})
    composition = suite_results.get("generic_composition_audit", {})
    history_summary = history.get("summary", {}) if isinstance(history, dict) else {}
    composition_summary = (
        composition.get("summary", {}) if isinstance(composition, dict) else {}
    )
    metrics = {
        "experiment_name": data.get("experiment_name"),
        "suite_count": len(selected_suites) if isinstance(selected_suites, list) else 0,
        "claim_status": data.get("claim_status"),
        "unknown_boundary_gates": history_summary.get("novelty_gate_counts"),
        "unknown_boundary_last_baseline_over_candidate": history_summary.get(
            "last_point_min_valid_baseline_over_candidate"
        ),
        "orientation_composition_gate": composition_summary.get("novelty_gate"),
        "orientation_failure_count": composition_summary.get(
            "explicit_family_novelty_failure_count"
        ),
    }
    return ArtifactEvidence(
        name="quantum_benchmark_diagnostic",
        path=str(path),
        status=str(data.get("claim_status", "unknown")),
        record_count=record_count,
        key_metrics=metrics,
        interpretation=(
            "Executed finite exact-state and analytic diagnostics.  Supports "
            "implementation sanity, the rejected orientation witness, and an "
            "open unknown-boundary candidate gate; does not prove an advantage."
        ),
    )


def summarize_unknown_boundary_grid(path: Path) -> ArtifactEvidence:
    """Extract the parameter-grid evidence summary."""

    data = _load_json(path)
    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        raise TypeError("unknown-boundary grid artifact missing summary object")
    case_results = data.get("case_results", [])
    if not isinstance(case_results, list):
        raise TypeError("unknown-boundary grid artifact missing case_results list")
    open_cases = [
        item.get("name")
        for item in case_results
        if isinstance(item, dict)
        and isinstance(item.get("summary"), dict)
        and item["summary"].get("all_points_open") is True
    ]
    failed_cases = [
        item.get("name")
        for item in case_results
        if isinstance(item, dict)
        and isinstance(item.get("summary"), dict)
        and item["summary"].get("all_points_open") is False
    ]
    metrics = {
        "experiment_name": data.get("experiment_name"),
        "claim_status": data.get("claim_status"),
        "case_count": summary.get("case_count"),
        "total_records": summary.get("total_records"),
        "novelty_gate_counts": summary.get("novelty_gate_counts"),
        "strongest_encoded_valid_baseline_counts": summary.get(
            "strongest_encoded_valid_baseline_counts"
        ),
        "open_cases": open_cases,
        "failed_cases": failed_cases,
    }
    return ArtifactEvidence(
        name="unknown_boundary_grid",
        path=str(path),
        status=str(data.get("claim_status", "unknown")),
        record_count=int(summary.get("total_records", 0)),
        key_metrics=metrics,
        interpretation=(
            "Analytic parameter sweep for the no-free-QRAM candidate.  Open "
            "families are proof targets; failed families are rejected witnesses."
        ),
    )


def summarize_charged_activity(path: Path) -> ArtifactEvidence:
    """Extract the charged predicate-generation prototype summary."""

    data = _load_json(path)
    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        raise TypeError("charged activity artifact missing summary object")
    metrics = {
        "experiment_name": data.get("experiment_name"),
        "claim_status": data.get("claim_status"),
        "case_count": summary.get("case_count"),
        "total_records": summary.get("total_records"),
        "all_no_supplied_predicate_rows": summary.get(
            "all_no_supplied_predicate_rows"
        ),
        "all_output_subset_active": summary.get("all_output_subset_active"),
        "exact_state_trace_count": summary.get("exact_state_trace_count"),
    }
    return ArtifactEvidence(
        name="charged_activity_history",
        path=str(path),
        status=str(data.get("claim_status", "unknown")),
        record_count=int(summary.get("total_records", 0)),
        key_metrics=metrics,
        interpretation=(
            "Finite-phase predicate-generation prototype.  It removes supplied "
            "active/output rows from the toy relation and records charged "
            "compute/uncompute traces, but it is not a variable-time theorem."
        ),
    )


def summarize_variable_time_charged(path: Path) -> ArtifactEvidence:
    """Extract the variable-time charged mainline alignment summary."""

    data = _load_json(path)
    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        raise TypeError("variable-time charged artifact missing summary object")
    case_results = data.get("case_results", [])
    if not isinstance(case_results, list):
        raise TypeError("variable-time charged artifact missing case_results list")
    open_cases = [
        item.get("name")
        for item in case_results
        if isinstance(item, dict)
        and isinstance(item.get("summary"), dict)
        and item["summary"].get("all_points_open") is True
    ]
    failed_cases = [
        item.get("name")
        for item in case_results
        if isinstance(item, dict)
        and isinstance(item.get("summary"), dict)
        and item["summary"].get("all_points_open") is False
    ]
    metrics = {
        "experiment_name": data.get("experiment_name"),
        "claim_status": data.get("claim_status"),
        "case_count": summary.get("case_count"),
        "total_records": summary.get("total_records"),
        "novelty_gate_counts": summary.get("novelty_gate_counts"),
        "strongest_valid_baseline_counts": summary.get(
            "strongest_valid_baseline_counts"
        ),
        "open_case_count": summary.get("open_case_count"),
        "open_cases": open_cases,
        "failed_cases": failed_cases,
    }
    return ArtifactEvidence(
        name="variable_time_charged_history",
        path=str(path),
        status=str(data.get("claim_status", "unknown")),
        record_count=int(summary.get("total_records", 0)),
        key_metrics=metrics,
        interpretation=(
            "Mainline alignment audit: finite-QPE charged predicate costs are "
            "inserted into the unknown-boundary history target and compared "
            "with same-interface proxy baselines.  Open gates are proof targets."
        ),
    )


def summarize_stopping_transducer(path: Path) -> ArtifactEvidence:
    """Extract the variable-time stopping relation skeleton summary."""

    data = _load_json(path)
    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        raise TypeError("stopping transducer artifact missing summary object")
    metrics = {
        "experiment_name": data.get("experiment_name"),
        "claim_status": data.get("claim_status"),
        "case_count": summary.get("case_count"),
        "total_records": summary.get("total_records"),
        "exact_state_trace_count": summary.get("exact_state_trace_count"),
        "all_variable_over_serial_at_most_one": summary.get(
            "all_variable_over_serial_at_most_one"
        ),
    }
    return ArtifactEvidence(
        name="stopping_time_transducer",
        path=str(path),
        status=str(data.get("claim_status", "unknown")),
        record_count=int(summary.get("total_records", 0)),
        key_metrics=metrics,
        interpretation=(
            "Executable variable-time stopping relation skeleton.  It checks "
            "stop-register compute/phase/uncompute traces and serial versus "
            "branch-RMS ledgers, but it is not a circuit theorem."
        ),
    )


def summarize_stopping_unitary_theorem(path: Path) -> ArtifactEvidence:
    """Extract the stopping-unitary theorem scaffold summary."""

    data = _load_json(path)
    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        raise TypeError("stopping-unitary theorem artifact missing summary object")
    metrics = {
        "experiment_name": data.get("experiment_name"),
        "claim_status": data.get("claim_status"),
        "case_count": summary.get("case_count"),
        "total_records": summary.get("total_records"),
        "check_status_counts": summary.get("check_status_counts"),
        "all_execution_checks_passed": summary.get("all_execution_checks_passed"),
    }
    return ArtifactEvidence(
        name="stopping_unitary_theorem",
        path=str(path),
        status=str(data.get("claim_status", "unknown")),
        record_count=int(summary.get("total_records", 0)),
        key_metrics=metrics,
        interpretation=(
            "Paper-facing stopping-unitary lemma scaffold.  It separates "
            "execution-checked identities from proof obligations; it is not "
            "a completed upper-bound theorem."
        ),
    )


def summarize_composition_frontier(path: Path) -> ArtifactEvidence:
    """Extract the composition-frontier audit summary."""

    data = _load_json(path)
    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        raise TypeError("composition-frontier artifact missing summary object")
    case_results = data.get("case_results", [])
    if not isinstance(case_results, list):
        raise TypeError("composition-frontier artifact missing case_results list")
    open_cases = [
        item.get("name")
        for item in case_results
        if isinstance(item, dict)
        and isinstance(item.get("summary"), dict)
        and item["summary"].get("all_points_open") is True
    ]
    failed_cases = [
        item.get("name")
        for item in case_results
        if isinstance(item, dict)
        and isinstance(item.get("summary"), dict)
        and item["summary"].get("all_points_open") is False
    ]
    metrics = {
        "experiment_name": data.get("experiment_name"),
        "claim_status": data.get("claim_status"),
        "case_count": summary.get("case_count"),
        "total_records": summary.get("total_records"),
        "novelty_gate_counts": summary.get("novelty_gate_counts"),
        "strongest_valid_baseline_counts": summary.get(
            "strongest_valid_baseline_counts"
        ),
        "open_case_count": summary.get("open_case_count"),
        "open_cases": open_cases,
        "failed_cases": failed_cases,
    }
    return ArtifactEvidence(
        name="composition_frontier",
        path=str(path),
        status=str(data.get("claim_status", "unknown")),
        record_count=int(summary.get("total_records", 0)),
        key_metrics=metrics,
        interpretation=(
            "Known-composition frontier audit for loop composition, generated "
            "predicate extraction, QBAI, independent QPE, and forbidden "
            "free-history QRAM baselines.  It is an encoded novelty screen, "
            "not a theorem against all prior work."
        ),
    )


def summarize_lower_bound_program(path: Path) -> ArtifactEvidence:
    """Extract the lower-bound proof-program summary."""

    data = _load_json(path)
    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        raise TypeError("lower-bound artifact missing summary object")
    metrics = {
        "experiment_name": data.get("experiment_name"),
        "claim_status": data.get("claim_status"),
        "case_count": summary.get("case_count"),
        "total_records": summary.get("total_records"),
        "strongest_block_counts": summary.get("strongest_block_counts"),
        "total_proof_obligations": summary.get("total_proof_obligations"),
        "total_local_facts": summary.get("total_local_facts"),
    }
    return ArtifactEvidence(
        name="lower_bound_program",
        path=str(path),
        status=str(data.get("claim_status", "unknown")),
        record_count=int(summary.get("total_records", 0)),
        key_metrics=metrics,
        interpretation=(
            "Symbolic lower-bound proof program.  It enumerates hard-family "
            "blocks and missing proof arguments; it does not prove L-07."
        ),
    )


def summarize_proof_ledger(path: Path) -> ArtifactEvidence:
    """Extract the theorem-stack proof-ledger summary."""

    data = _load_json(path)
    ledger = data.get("ledger", {})
    if not isinstance(ledger, dict):
        raise TypeError("proof-ledger artifact missing ledger object")
    metrics = {
        "claim_status": ledger.get("claim_status"),
        "readiness": ledger.get("readiness"),
        "entry_count": ledger.get("entry_count"),
        "status_counts": ledger.get("status_counts"),
        "theorem_claimable": ledger.get("theorem_claimable"),
        "ccf_a_claimable": ledger.get("ccf_a_claimable"),
    }
    return ArtifactEvidence(
        name="proof_ledger",
        path=str(path),
        status=str(ledger.get("claim_status", "unknown")),
        record_count=int(ledger.get("entry_count", 0)),
        key_metrics=metrics,
        interpretation=(
            "Machine-readable proof ledger tying stopping-unitary, composition, "
            "and lower-bound obligations together.  It explicitly marks the "
            "theorem stack as not CCF-A-claimable yet."
        ),
    )


def _gap_records(
    *,
    charged_activity_present: bool,
    variable_time_present: bool,
    stopping_present: bool,
    theorem_present: bool,
    composition_present: bool,
    lower_bound_present: bool,
    proof_ledger_present: bool,
) -> tuple[ResearchGap, ...]:
    if theorem_present:
        u08_status = "stopping_unitary_lemma_scaffold_started"
    elif stopping_present:
        u08_status = "stopping_relation_skeleton_started"
    elif variable_time_present:
        u08_status = "mainline_variable_time_alignment_started"
    elif charged_activity_present:
        u08_status = "charged_phase_prototype_started"
    else:
        u08_status = "toy_relation_skeleton_started"
    u08_evidence = (
        "unknown_boundary_history.py defines the cost target and gate; "
        "charged_activity_history.py derives finite-phase predicates; "
        "variable_time_charged_history.py aligns charged finite-QPE costs; "
        "stopping_time_transducer.py gives an executable stopping-register "
        "skeleton; stopping_time_theorem.py now separates executable lemma "
        "checks from proof obligations."
        if theorem_present
        else (
            "unknown_boundary_history.py defines the cost target and gate; "
            "charged_activity_history.py derives finite-phase predicates; "
            "variable_time_charged_history.py aligns charged finite-QPE costs; "
            "stopping_time_transducer.py now gives an executable stopping-register "
            "compute/phase/uncompute skeleton."
            if stopping_present
            else (
                "unknown_boundary_history.py defines the cost target and gate; "
                "charged_activity_history.py derives finite-phase predicates; "
                "variable_time_charged_history.py now aligns charged finite-QPE level "
                "costs with the unknown-boundary candidate and encoded baselines."
                if variable_time_present
                else (
                    "unknown_boundary_history.py defines the cost target and gate; "
                    "activity_history_transducer.py supplies a toy reversible relation "
                    "skeleton; charged_activity_history.py now derives active/output "
                    "predicates from finite-phase windows; artifacts show open candidate "
                    "families."
                    if charged_activity_present
                    else (
                        "unknown_boundary_history.py defines the cost target and gate; "
                        "activity_history_transducer.py now supplies a toy reversible "
                        "relation skeleton; artifacts show open candidate families."
                    )
                )
            )
        )
    )
    u08_missing = (
        "Manual theorem proof for ST-P01 through ST-P04: compiled stopping "
        "unitary, bounded-error confidence composition, coherent boundary "
        "localization, and strongest-composition separation."
        if theorem_present
        else (
            "A full variable-time coherent transducer theorem: stopping-unitary "
            "construction, cleanup proof, confidence accounting, and query upper "
            "bound matching the charged alignment proxy."
            if stopping_present
            else (
                "A circuit-level variable-time coherent transducer proof, cleanup proof, "
                "and query upper bound matching the charged alignment proxy without "
                "hidden scans or free QRAM."
                if variable_time_present
                else (
                    "A reversible unknown-boundary unitary that localizes the boundary, "
                    "generates the activity history with variable-time charged QPE, and "
                    "achieves the quadrature proxy without supplied lists or hidden scans."
                    if charged_activity_present
                    else (
                        "A reversible unitary construction that generates active-history "
                        "predicates from charged canonical-oracle access without supplied "
                        "predicate rows or measured lists."
                    )
                )
            )
        )
    )
    u08_next = (
        "Write the manual proof for the stopping-time unitary lemma and connect "
        "it to the composition frontier."
        if theorem_present
        else (
            "Promote the stopping relation skeleton into a formal stopping-time "
            "unitary lemma with exact cleanup and resource accounting."
            if stopping_present
            else (
                "Turn the variable-time charged alignment proxy into a circuit theorem "
                "with explicit stopping-time unitaries and cleanup."
                if variable_time_present
                else (
                    "Extend the charged phase-window prototype into an unknown-boundary "
                    "variable-time transducer and prove its query bound."
                    if charged_activity_present
                    else (
                        "Replace the toy supplied predicates with a charged "
                        "unknown-boundary construction and add exact-state traces."
                    )
                )
            )
        )
    )
    return (
        ResearchGap(
            gap_id="P0-U08",
            priority="P0",
            title="Construct the no-free-QRAM activity-history transducer",
            current_status=u08_status,
            evidence_now=u08_evidence,
            missing_evidence=u08_missing,
            next_code_or_proof_artifact=u08_next,
            activation_target="Claim U-08 / N-03 upper-bound side",
            failure_mode=(
                "Any hidden scan over inactive arms or free QRAM lookup collapses "
                "the candidate into a known rebuilt-history baseline."
            ),
        ),
        ResearchGap(
            gap_id="P0-L07",
            priority="P0",
            title="Prove a matching all-algorithms lower bound",
            current_status=(
                "proof_ledger_started"
                if proof_ledger_present
                else (
                    "lower_bound_program_started"
                    if lower_bound_present
                    else "proof_obligation"
                )
            ),
            evidence_now=(
                "proof_ledger.py connects LB-B02 through LB-B04 to the upper-bound "
                "and composition-frontier obligations; it still marks them as open "
                "proof obligations."
                if proof_ledger_present
                else (
                    "lower_bound_program.py enumerates LB-B01 through LB-B04 and "
                    "marks only local boundary localization as a local fact; the "
                    "history, multi-output, and composition-exclusion blocks remain "
                    "proof obligations."
                    if lower_bound_present
                    else (
                        "Artifacts record adversary_lower_target_proxy and legal baseline "
                        "ratios, but no lower-bound theorem."
                    )
                )
            ),
            missing_evidence=(
                "Adversary or polynomial-method proof under the same "
                "unknown-boundary/no-free-QRAM oracle interface."
            ),
            next_code_or_proof_artifact=(
                "Replace LB-B02 through LB-B04 proof_obligation labels with a "
                "formal adversary matrix or polynomial-method proof."
                if lower_bound_present
                else (
                    "docs/lower_bound_program.md and symbolic hard-family generator "
                    "with adversary matrix dimensions/checks."
                )
            ),
            activation_target="Claim L-07 / N-03 lower-bound side",
            failure_mode=(
                "A proof that only covers estimate-then-sort algorithms will not "
                "support a CCF-A algorithmic claim."
            ),
        ),
        ResearchGap(
            gap_id="P1-COMP",
            priority="P1",
            title="Extend the strongest-composition audit",
            current_status=(
                "proof_ledger_started"
                if proof_ledger_present
                else (
                    "composition_frontier_started"
                    if composition_present
                    else "code_sanity_partial"
                )
            ),
            evidence_now=(
                "proof_ledger.py records CF-T01 and CF-T02 as manual theorem "
                "instantiation requirements; composition_frontier.py supplies the "
                "encoded screen but not the final proof."
                if proof_ledger_present
                else (
                    "composition_audit.py falsifies the original orientation witness; "
                    "composition_frontier.py now encodes same-interface loop, generated "
                    "predicate extraction, QBAI, independent QPE, serial rebuild, and "
                    "forbidden free-history baselines."
                    if composition_present
                    else (
                        "composition_audit.py falsifies the original orientation witness; "
                        "grid artifacts compare several rebuilt-history baselines."
                    )
                )
            ),
            missing_evidence=(
                "Manual theorem-by-theorem instantiation showing no published "
                "composition theorem matches the same interface, or a valid "
                "published theorem that kills this candidate."
                if composition_present
                else (
                    "Instantiations of loop composition, time-efficient k-minimum, "
                    "approximate k-minimum, and quantum BAI under the exact new interface."
                )
            ),
            next_code_or_proof_artifact=(
                "docs/composition_frontier_proof.md with explicit theorem "
                "instantiations and assumption mismatches."
                if composition_present
                else (
                    "src/qgapselect/composition_frontier.py with per-baseline "
                    "assumption flags and dominance tests."
                )
            ),
            activation_target="Prior-work novelty boundary for N-03",
            failure_mode=(
                "If a valid known composition matches the candidate, the new core "
                "must be rejected or reframed."
            ),
        ),
        ResearchGap(
            gap_id="P1-VERIFY",
            priority="P1",
            title="Build verifier and confidence accounting for the new relation",
            current_status="not_started_for_new_core",
            evidence_now=(
                "Existing direct Top-k and BBHT paths have verifier/query-ledger "
                "tests, but they still rely on measured calibration."
            ),
            missing_evidence=(
                "A verifier for unknown-boundary history outputs with duplicate, "
                "unresolved, and non-separating failure semantics."
            ),
            next_code_or_proof_artifact=(
                "tests/test_activity_history_transducer.py and verifier traces in "
                "a new diagnostic artifact."
            ),
            activation_target="Constructive algorithm correctness",
            failure_mode=(
                "If verification consumes membership-equivalent information, it "
                "cannot support discovery claims."
            ),
        ),
        ResearchGap(
            gap_id="P2-PAPER",
            priority="P2",
            title="Convert the code audit into paper-ready theorem statements",
            current_status="documentation_partial",
            evidence_now=(
                "claim_matrix.md and unknown_boundary_history_spec.md define "
                "the active problem and blocked claims."
            ),
            missing_evidence=(
                "Precise theorem statements, assumptions, proof sketches, and "
                "a table separating proved facts from finite artifacts."
            ),
            next_code_or_proof_artifact=(
                "docs/paper_readiness_plan.md generated from this audit plus "
                "manual proof sections in paper/main.tex."
            ),
            activation_target="Submission narrative discipline",
            failure_mode=(
                "A manuscript that states open gates as proved advantages will be "
                "rejected by informed reviewers."
            ),
        ),
    )


def build_research_gap_audit(
    quantum_artifact: Path,
    unknown_boundary_grid_artifact: Path,
    charged_activity_artifact: Path | None = None,
    variable_time_artifact: Path | None = None,
    stopping_artifact: Path | None = None,
    theorem_artifact: Path | None = None,
    composition_artifact: Path | None = None,
    lower_bound_artifact: Path | None = None,
    proof_ledger_artifact: Path | None = None,
) -> ResearchGapAudit:
    """Build the current CCF-A gap checkpoint from versioned artifacts."""

    evidence_items = [
        summarize_quantum_benchmark(quantum_artifact),
        summarize_unknown_boundary_grid(unknown_boundary_grid_artifact),
    ]
    charged_present = False
    if charged_activity_artifact is not None and charged_activity_artifact.exists():
        evidence_items.append(summarize_charged_activity(charged_activity_artifact))
        charged_present = True
    variable_time_present = False
    if variable_time_artifact is not None and variable_time_artifact.exists():
        evidence_items.append(summarize_variable_time_charged(variable_time_artifact))
        variable_time_present = True
    stopping_present = False
    if stopping_artifact is not None and stopping_artifact.exists():
        evidence_items.append(summarize_stopping_transducer(stopping_artifact))
        stopping_present = True
    theorem_present = False
    if theorem_artifact is not None and theorem_artifact.exists():
        evidence_items.append(summarize_stopping_unitary_theorem(theorem_artifact))
        theorem_present = True
    composition_present = False
    if composition_artifact is not None and composition_artifact.exists():
        evidence_items.append(summarize_composition_frontier(composition_artifact))
        composition_present = True
    lower_bound_present = False
    if lower_bound_artifact is not None and lower_bound_artifact.exists():
        evidence_items.append(summarize_lower_bound_program(lower_bound_artifact))
        lower_bound_present = True
    proof_ledger_present = False
    if proof_ledger_artifact is not None and proof_ledger_artifact.exists():
        evidence_items.append(summarize_proof_ledger(proof_ledger_artifact))
        proof_ledger_present = True
    evidence = tuple(evidence_items)
    gate_counter: Counter[str] = Counter()
    for item in evidence:
        gates = item.key_metrics.get("novelty_gate_counts") or item.key_metrics.get(
            "unknown_boundary_gates"
        )
        if isinstance(gates, dict):
            gate_counter.update({str(key): int(value) for key, value in gates.items()})
    completed_items = [
        "Old orientation witness is rejected by explicit composition audit.",
        "Unknown-boundary history candidate is encoded as a no-free-QRAM audit target.",
        "Parameter grid contains both open families and a failing negative control.",
        "Finite exact-state runner and analytic grid produce reproducible artifacts.",
    ]
    if charged_present:
        completed_items.append(
            "Charged finite-phase prototype removes supplied predicate rows from "
            "the activity-history relation fixture."
        )
    if variable_time_present:
        completed_items.append(
            "Variable-time charged alignment inserts finite-QPE predicate costs "
            "back into the main unknown-boundary candidate comparison."
        )
    if stopping_present:
        completed_items.append(
            "Stopping-time relation skeleton adds explicit stop-register "
            "compute/phase/uncompute traces and branch-RMS ledgers."
        )
    if theorem_present:
        completed_items.append(
            "Stopping-unitary theorem scaffold separates executable checks from "
            "manual proof obligations."
        )
    if composition_present:
        completed_items.append(
            "Composition-frontier audit encodes same-interface known-composition "
            "baselines and a forbidden free-history QRAM collapse."
        )
    if lower_bound_present:
        completed_items.append(
            "Lower-bound proof program enumerates local facts and adversary-proof "
            "obligations for L-07."
        )
    if proof_ledger_present:
        completed_items.append(
            "Proof ledger connects upper-bound, composition-frontier, and "
            "lower-bound obligations without activating theorem claims."
        )
    completed = tuple(completed_items)
    rejected = (
        "H_orient by itself is not a new quantum core.",
        "Boundary-only membership certificates are not discovery evidence.",
        "Simulator runtime or finite slopes are not asymptotic quantum advantage.",
    )
    first_recommendation = (
        "Write the manual proof for ST-P01 through ST-P04 and connect it to "
        "composition-frontier checks."
        if theorem_present
        else (
            "Promote the stopping relation skeleton into a formal stopping-time "
            "unitary lemma."
            if stopping_present
            else (
                "Turn the variable-time charged alignment proxy into an explicit "
                "circuit-level transducer theorem."
                if variable_time_present
                else (
                    "Extend the charged phase-window prototype into a full "
                    "unknown-boundary variable-time transducer."
                    if charged_present
                    else (
                        "Implement the toy activity-history transducer with explicit "
                        "compute/uncompute traces."
                    )
                )
            )
        )
    )
    recommended = (
        first_recommendation,
        (
            "Manually instantiate the composition frontier against published "
            "composition theorems."
            if composition_present
            else (
                "Add a composition_frontier audit for loop composition, "
                "k-minimum, and QBAI assumptions."
            )
        ),
        (
            "Convert LB-B02 through LB-B04 into a formal adversary or "
            "polynomial-method proof."
            if lower_bound_present
            else "Draft the L-07 lower-bound program before adding more application experiments."
        ),
        "Keep every open novelty gate labelled as a proof obligation in paper text.",
    )
    if proof_ledger_present:
        recommended = (
            "Convert proof-ledger entries ST-P01, ST-P02, CF-T01/CF-T02, and "
            "LB-B02 through LB-B04 into manual proof sections.",
            *recommended[1:],
        )
    return ResearchGapAudit(
        stage=CURRENT_STAGE,
        readiness=READINESS,
        artifact_evidence=evidence,
        completed_evidence=completed,
        rejected_claims=rejected,
        top_gaps=_gap_records(
            charged_activity_present=charged_present,
            variable_time_present=variable_time_present,
            stopping_present=stopping_present,
            theorem_present=theorem_present,
            composition_present=composition_present,
            lower_bound_present=lower_bound_present,
            proof_ledger_present=proof_ledger_present,
        ),
        recommended_next_steps=recommended,
    )


def research_gap_markdown(audit: ResearchGapAudit) -> str:
    """Render a compact planning document from an audit."""

    lines = [
        "# Q-GapSelect research gap audit",
        "",
        f"Current stage: `{audit.stage}`",
        "",
        f"Readiness: `{audit.readiness}`",
        "",
        "## Evidence now",
        "",
    ]
    for evidence in audit.artifact_evidence:
        lines.extend(
            [
                f"### {evidence.name}",
                "",
                f"- path: `{evidence.path}`",
                f"- status: `{evidence.status}`",
                f"- records: `{evidence.record_count}`",
                f"- interpretation: {evidence.interpretation}",
                "",
                "Key metrics:",
                "",
            ]
        )
        for key, value in evidence.key_metrics.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    lines.extend(["## Completed evidence", ""])
    for item in audit.completed_evidence:
        lines.append(f"- {item}")
    lines.extend(["", "## Rejected claims", ""])
    for item in audit.rejected_claims:
        lines.append(f"- {item}")
    lines.extend(["", "## Highest-priority gaps", ""])
    for gap in audit.top_gaps:
        lines.extend(
            [
                f"### {gap.gap_id}: {gap.title}",
                "",
                f"- priority: `{gap.priority}`",
                f"- current status: `{gap.current_status}`",
                f"- evidence now: {gap.evidence_now}",
                f"- missing evidence: {gap.missing_evidence}",
                f"- next artifact: `{gap.next_code_or_proof_artifact}`",
                f"- activation target: `{gap.activation_target}`",
                f"- failure mode: {gap.failure_mode}",
                "",
            ]
        )
    lines.extend(["## Recommended next steps", ""])
    for item in audit.recommended_next_steps:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)
