"""Machine-readable proof ledger for the active Q-GapSelect theorem stack.

The ledger is deliberately conservative.  It records which parts of the current
program are local facts, execution-checked identities, proof outlines, or still
open obligations.  It never upgrades a scaffold into a theorem.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLAIM_STATUS = "proof_ledger_started_no_quantum_advantage_theorem"
READINESS = "proof_program_structured_not_ccf_a_claimable"


@dataclass(frozen=True, slots=True)
class ProofLedgerEntry:
    """One proof or audit item in the theorem stack."""

    entry_id: str
    pillar: str
    status: str
    statement: str
    evidence: str
    missing_argument: str
    activation_condition: str
    artifact_refs: tuple[str, ...]
    permitted_wording: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "pillar": self.pillar,
            "status": self.status,
            "statement": self.statement,
            "evidence": self.evidence,
            "missing_argument": self.missing_argument,
            "activation_condition": self.activation_condition,
            "artifact_refs": list(self.artifact_refs),
            "permitted_wording": self.permitted_wording,
        }


@dataclass(frozen=True, slots=True)
class ProofLedger:
    """Complete proof-ledger checkpoint."""

    claim_status: str
    readiness: str
    theorem_claimable: bool
    ccf_a_claimable: bool
    entry_count: int
    status_counts: dict[str, int]
    artifact_summaries: dict[str, Any]
    entries: tuple[ProofLedgerEntry, ...]
    next_required_manual_proofs: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_status": self.claim_status,
            "readiness": self.readiness,
            "theorem_claimable": self.theorem_claimable,
            "ccf_a_claimable": self.ccf_a_claimable,
            "entry_count": self.entry_count,
            "status_counts": dict(self.status_counts),
            "artifact_summaries": self.artifact_summaries,
            "entries": [entry.as_dict() for entry in self.entries],
            "next_required_manual_proofs": list(self.next_required_manual_proofs),
        }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing artifact: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _summary(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get("summary", {})
    if not isinstance(value, dict):
        raise TypeError(f"{name} artifact missing summary object")
    return value


def build_proof_ledger(
    *,
    stopping_artifact: Path,
    composition_artifact: Path,
    lower_bound_artifact: Path,
) -> ProofLedger:
    """Build the current proof ledger from versioned scaffold artifacts."""

    stopping = _load_json(stopping_artifact)
    composition = _load_json(composition_artifact)
    lower_bound = _load_json(lower_bound_artifact)
    stopping_summary = _summary(stopping, "stopping")
    composition_summary = _summary(composition, "composition")
    lower_bound_summary = _summary(lower_bound, "lower_bound")

    entries = (
        ProofLedgerEntry(
            entry_id="ST-U-CLEANUP",
            pillar="upper_bound",
            status="execution_checked",
            statement=(
                "The tested stopping relation has compute involution, phase "
                "equivalence, cleanup, and RMS ledger identities."
            ),
            evidence=(
                "stopping_unitary_theorem artifact reports "
                f"{stopping_summary.get('check_status_counts')} with "
                f"all_execution_checks_passed="
                f"{stopping_summary.get('all_execution_checks_passed')}."
            ),
            missing_argument=(
                "Lift exact-state fixture checks to a symbolic unitary over all "
                "valid basis states and workspaces."
            ),
            activation_condition=(
                "A written lemma proves the same identities for the compiled "
                "controlled-QPE stopping circuit."
            ),
            artifact_refs=(str(stopping_artifact),),
            permitted_wording=(
                "Execution checks pass for the scaffold; no upper-bound theorem "
                "is claimed."
            ),
        ),
        ProofLedgerEntry(
            entry_id="ST-P01",
            pillar="upper_bound",
            status="proof_outline_started",
            statement="Compile a stopping-time unitary with charged QPE calls.",
            evidence="The stopping scaffold identifies the required registers and calls.",
            missing_argument=(
                "Formal circuit map, inverse-call accounting, padding convention, "
                "and exact cleanup proof."
            ),
            activation_condition="Manual proof of the stopping-unitary lemma.",
            artifact_refs=(str(stopping_artifact),),
            permitted_wording="This is a proof outline, not a theorem.",
        ),
        ProofLedgerEntry(
            entry_id="ST-P02",
            pillar="upper_bound",
            status="proof_outline_started",
            statement="Compose bounded-error finite-QPE predicates across levels.",
            evidence="The scaffold isolates deterministic predicates from proof obligations.",
            missing_argument=(
                "Confidence split, correlated-error accounting, and rejection of "
                "unresolved or duplicate outputs."
            ),
            activation_condition=(
                "A bounded-error lemma with explicit failure budget and verifier "
                "semantics."
            ),
            artifact_refs=(str(stopping_artifact),),
            permitted_wording="Confidence composition remains open.",
        ),
        ProofLedgerEntry(
            entry_id="ST-P03",
            pillar="upper_bound",
            status="proof_obligation",
            statement="Localize the Top-k boundary coherently rather than supplying it.",
            evidence="Current stopping fixtures still receive a numeric boundary.",
            missing_argument=(
                "A boundary-localization unitary that does not leak selected/rejected "
                "membership and composes with the stopping relation."
            ),
            activation_condition="Coherent boundary lemma plus cleanup proof.",
            artifact_refs=(str(stopping_artifact),),
            permitted_wording="The boundary-localization step is not solved.",
        ),
        ProofLedgerEntry(
            entry_id="CF-T01",
            pillar="composition_frontier",
            status="manual_instantiation_required",
            statement=(
                "Known loop and variable-time composition theorems do not match "
                "the same no-free-QRAM interface."
            ),
            evidence=(
                "composition_frontier artifact reports gate counts "
                f"{composition_summary.get('novelty_gate_counts')} and strongest "
                f"baselines {composition_summary.get('strongest_valid_baseline_counts')}."
            ),
            missing_argument=(
                "Manual theorem-by-theorem instantiation with exact oracle, output, "
                "memory, and stopping assumptions."
            ),
            activation_condition=(
                "Every applicable published theorem is either instantiated as weaker "
                "or shown to use a strictly stronger interface."
            ),
            artifact_refs=(str(composition_artifact),),
            permitted_wording="The encoded frontier is a screen, not an exhaustive proof.",
        ),
        ProofLedgerEntry(
            entry_id="CF-T02",
            pillar="composition_frontier",
            status="manual_instantiation_required",
            statement=(
                "Known k-minimum, marked extraction, and QBAI baselines do not "
                "dominate the active candidate under identical assumptions."
            ),
            evidence=(
                "The frontier artifact includes generated-predicate extraction, "
                "coarse+QBAI, independent QPE, and serial rebuild rows."
            ),
            missing_argument=(
                "Replace proxy labels with formal reductions or assumption mismatches."
            ),
            activation_condition="A composition-frontier proof table in the manuscript.",
            artifact_refs=(str(composition_artifact),),
            permitted_wording="The comparison is an encoded audit until manually proved.",
        ),
        ProofLedgerEntry(
            entry_id="LB-B01",
            pillar="lower_bound",
            status="local_fact",
            statement="Boundary localization has a local angular discrimination barrier.",
            evidence=(
                "lower_bound_program artifact reports "
                f"{lower_bound_summary.get('total_local_facts')} local facts."
            ),
            missing_argument=(
                "Lift the local two-state fact to the full unknown-boundary relation."
            ),
            activation_condition="Use as one ingredient in a global adversary proof.",
            artifact_refs=(str(lower_bound_artifact),),
            permitted_wording="This is a local lower-bound ingredient.",
        ),
        ProofLedgerEntry(
            entry_id="LB-B02",
            pillar="lower_bound",
            status="proof_obligation",
            statement="Active-history recovery has a direct-sum lower-bound component.",
            evidence="The lower-bound program records the target proxy only.",
            missing_argument=(
                "Adversary matrix or polynomial relation forcing coherent history "
                "discovery across charged levels."
            ),
            activation_condition="Formal all-algorithms lower-bound proof.",
            artifact_refs=(str(lower_bound_artifact),),
            permitted_wording="This lower-bound block is open.",
        ),
        ProofLedgerEntry(
            entry_id="LB-B03",
            pillar="lower_bound",
            status="proof_obligation",
            statement="Direct multi-output extraction has an output-sensitive barrier.",
            evidence="The lower-bound program records the target proxy only.",
            missing_argument=(
                "Proof that arbitrary adaptive quantum algorithms must pay the "
                "multi-output relation cost, not only estimate-then-sort algorithms."
            ),
            activation_condition="Formal all-algorithms lower-bound proof.",
            artifact_refs=(str(lower_bound_artifact),),
            permitted_wording="This lower-bound block is open.",
        ),
        ProofLedgerEntry(
            entry_id="LB-B04",
            pillar="lower_bound",
            status="proof_obligation",
            statement="Composition-frontier exclusion is necessary for optimality.",
            evidence=(
                "The lower-bound program records "
                f"{lower_bound_summary.get('total_proof_obligations')} proof "
                "obligation rows across configured records."
            ),
            missing_argument=(
                "Show that known same-interface algorithms cannot beat the claimed "
                "lower-target relation, or revise the candidate."
            ),
            activation_condition="Manual composition frontier plus lower-bound theorem.",
            artifact_refs=(str(composition_artifact), str(lower_bound_artifact)),
            permitted_wording="Optimality is not proved.",
        ),
    )
    status_counts = Counter(entry.status for entry in entries)
    return ProofLedger(
        claim_status=CLAIM_STATUS,
        readiness=READINESS,
        theorem_claimable=False,
        ccf_a_claimable=False,
        entry_count=len(entries),
        status_counts=dict(sorted(status_counts.items())),
        artifact_summaries={
            "stopping_unitary_theorem": stopping_summary,
            "composition_frontier": composition_summary,
            "lower_bound_program": lower_bound_summary,
        },
        entries=entries,
        next_required_manual_proofs=(
            "Write ST-P01 compiled stopping-unitary construction and cleanup proof.",
            "Write ST-P02 bounded-error confidence-composition lemma.",
            "Solve or replace ST-P03 coherent boundary localization.",
            "Instantiate CF-T01/CF-T02 against published composition theorems.",
            "Prove LB-B02 through LB-B04 with adversary or polynomial method.",
        ),
    )


def proof_ledger_markdown(ledger: ProofLedger) -> str:
    """Render a proof-ledger Markdown report."""

    lines = [
        "# Q-GapSelect proof ledger",
        "",
        f"Claim status: `{ledger.claim_status}`",
        "",
        f"Readiness: `{ledger.readiness}`",
        "",
        f"Theorem claimable: `{ledger.theorem_claimable}`",
        "",
        f"CCF-A claimable: `{ledger.ccf_a_claimable}`",
        "",
        "## Status counts",
        "",
    ]
    for status, count in ledger.status_counts.items():
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(["", "## Entries", ""])
    for entry in ledger.entries:
        lines.extend(
            [
                f"### {entry.entry_id}: {entry.statement}",
                "",
                f"- pillar: `{entry.pillar}`",
                f"- status: `{entry.status}`",
                f"- evidence: {entry.evidence}",
                f"- missing argument: {entry.missing_argument}",
                f"- activation condition: {entry.activation_condition}",
                f"- permitted wording: {entry.permitted_wording}",
                "",
            ]
        )
    lines.extend(["## Next required manual proofs", ""])
    for item in ledger.next_required_manual_proofs:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)
