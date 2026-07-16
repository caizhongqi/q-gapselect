"""Stopping-time unitary lemma scaffold.

The stopping transducer gives an executable relation.  This module turns that
relation into a paper-facing theorem scaffold: which parts are already checked
by exact-state execution, which resource identities are audited, and which
parts remain mathematical proof obligations.

Nothing here is a substitute for the final variable-time theorem.  The scaffold
exists to keep the manuscript honest and to make the next proof step concrete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .stopping_time_transducer import StoppingProfile, VariableTimeStoppingTransducer

CLAIM_STATUS = "stopping_time_unitary_lemma_scaffold_no_proof"
READINESS = "execution_checks_passed_proof_obligations_remain"
CLEANUP_TOLERANCE = 1e-10

CheckStatus = Literal["passed", "failed", "proof_obligation"]


@dataclass(frozen=True, slots=True)
class LemmaCheck:
    """One executable or proof-only obligation in the theorem scaffold."""

    check_id: str
    status: CheckStatus
    statement: str
    evidence: str
    missing_for_theorem: str


@dataclass(frozen=True, slots=True)
class StoppingUnitaryLemmaScaffold:
    """Auditable theorem scaffold for the variable-time stopping relation."""

    theorem_name: str
    claim_status: str
    readiness: str
    phase_on: str
    n_arms: int
    level_count: int
    index_dimension: int
    stop_dimension: int
    statevector_dimension: int
    compute_involution_residual_l2: float
    phase_equivalence_residual_l2: float
    work_garbage_probability_after_phase: float
    serial_full_history_cost_per_branch: int
    coherent_branch_rms_cost: float
    all_branch_rms_search_proxy: float
    branch_rms_over_serial: float
    all_branch_rms_formula_residual: float
    output_count: int
    active_count: int
    unresolved_count: int
    passed_count: int
    failed_count: int
    proof_obligation_count: int
    checks: tuple[LemmaCheck, ...]


def _predicate(profile: StoppingProfile, phase_on: str) -> bool:
    ever_active = bool(profile.ever_active)
    output = bool(profile.output)
    if phase_on == "active":
        return ever_active
    if phase_on == "output":
        return output
    if phase_on == "active_output":
        return ever_active and output
    raise ValueError("phase_on must be active, output, or active_output")


def _expected_phase_state(
    transducer: VariableTimeStoppingTransducer,
    state: np.ndarray,
    *,
    phase_on: str,
) -> np.ndarray:
    expected = state.reshape(transducer.shape).copy()
    for profile in transducer.profiles():
        if _predicate(profile, phase_on):
            expected[profile.index, 0, 0, 0] *= -1.0
    return expected.reshape(-1)


def build_stopping_unitary_lemma_scaffold(
    transducer: VariableTimeStoppingTransducer,
    *,
    phase_on: str = "output",
    tolerance: float = CLEANUP_TOLERANCE,
) -> StoppingUnitaryLemmaScaffold:
    """Build the executable theorem scaffold for one stopping transducer."""

    if not isinstance(transducer, VariableTimeStoppingTransducer):
        raise TypeError("transducer must be a VariableTimeStoppingTransducer")
    if phase_on not in {"active", "output", "active_output"}:
        raise ValueError("phase_on must be active, output, or active_output")
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")

    state = transducer.uniform_index_state()
    restored = transducer.apply_compute(transducer.apply_compute(state))
    compute_residual = float(np.linalg.norm(restored - state))
    phase_result = transducer.apply_phase(state, phase_on=phase_on)
    expected = _expected_phase_state(transducer, state, phase_on=phase_on)
    phase_residual = float(np.linalg.norm(phase_result.state - expected))
    view = phase_result.state.reshape(transducer.shape)
    garbage = float(
        np.sum(np.abs(view[:, 1:, :, :]) ** 2)
        + np.sum(np.abs(view[:, :, 1, :]) ** 2)
        + np.sum(np.abs(view[:, :, :, 1]) ** 2)
    )
    summary = transducer.summary()
    formula_residual = abs(
        summary.all_branch_rms_search_proxy
        - summary.coherent_branch_rms_cost * math.sqrt(summary.n_arms)
    )

    execution_checks = (
        LemmaCheck(
            check_id="ST-U01",
            status="passed" if compute_residual <= tolerance else "failed",
            statement="The stopping compute relation is an involution on the tested state.",
            evidence=f"L2 residual {compute_residual:.3e}.",
            missing_for_theorem=(
                "Generalize from exact-state fixture to a symbolic unitary over "
                "all valid basis states and padded registers."
            ),
        ),
        LemmaCheck(
            check_id="ST-U02",
            status="passed" if phase_residual <= tolerance else "failed",
            statement="Compute--phase--uncompute implements the intended phase predicate.",
            evidence=f"L2 residual {phase_residual:.3e} for phase_on={phase_on}.",
            missing_for_theorem=(
                "Prove the predicate equivalence for all superpositions and "
                "for the eventual unknown-boundary predicate."
            ),
        ),
        LemmaCheck(
            check_id="ST-U03",
            status="passed" if garbage <= tolerance else "failed",
            statement="The stop, active, and output work registers are cleaned after phase.",
            evidence=f"Work-garbage probability {garbage:.3e}.",
            missing_for_theorem=(
                "Lift the cleanup identity to the final circuit with boundary, "
                "QPE, verification, and confidence workspaces."
            ),
        ),
        LemmaCheck(
            check_id="ST-R01",
            status=(
                "passed"
                if summary.coherent_branch_rms_cost
                <= summary.serial_full_history_cost_per_branch + tolerance
                else "failed"
            ),
            statement="The branch-RMS stopping ledger is no larger than serial full history.",
            evidence=(
                "branch_rms="
                f"{summary.coherent_branch_rms_cost:.6g}, serial="
                f"{summary.serial_full_history_cost_per_branch}."
            ),
            missing_for_theorem=(
                "Derive the same inequality from charged stopping times in the "
                "formal variable-time algorithm."
            ),
        ),
        LemmaCheck(
            check_id="ST-R02",
            status="passed" if formula_residual <= tolerance else "failed",
            statement="The all-branch RMS proxy equals sqrt(sum_i T_i^2).",
            evidence=f"Formula residual {formula_residual:.3e}.",
            missing_for_theorem=(
                "Connect this proxy to the chosen variable-time search or "
                "amplitude-amplification theorem."
            ),
        ),
    )
    proof_only_checks = (
        LemmaCheck(
            check_id="ST-P01",
            status="proof_obligation",
            statement="There is a compiled stopping-time unitary with charged QPE calls.",
            evidence="Current code supplies a relation skeleton and exact-state traces.",
            missing_for_theorem=(
                "Write the circuit construction using controlled QPE, stopping "
                "flags, inverse calls, and explicit workspace dimensions."
            ),
        ),
        LemmaCheck(
            check_id="ST-P02",
            status="proof_obligation",
            statement="The bounded-error predicate composes across levels and outputs.",
            evidence="The scaffold uses deterministic finite-phase fixtures.",
            missing_for_theorem=(
                "Add confidence allocation and prove no unresolved/duplicate "
                "output is accepted."
            ),
        ),
        LemmaCheck(
            check_id="ST-P03",
            status="proof_obligation",
            statement="The boundary is localized coherently rather than supplied.",
            evidence="The current transducer receives a numeric boundary phase.",
            missing_for_theorem=(
                "Integrate the unknown-boundary localization unitary and prove "
                "that it does not leak membership."
            ),
        ),
        LemmaCheck(
            check_id="ST-P04",
            status="proof_obligation",
            statement="Known loop/composition theorems do not already imply the same bound.",
            evidence="Composition frontier is still a separate P1 audit.",
            missing_for_theorem=(
                "Instantiate strongest loop composition, k-minimum, QBAI, and "
                "variable-time baselines under identical assumptions."
            ),
        ),
    )
    checks = execution_checks + proof_only_checks
    passed = sum(check.status == "passed" for check in checks)
    failed = sum(check.status == "failed" for check in checks)
    proof_obligations = sum(check.status == "proof_obligation" for check in checks)
    readiness = (
        READINESS
        if failed == 0
        else "execution_check_failed_theorem_scaffold_blocked"
    )
    return StoppingUnitaryLemmaScaffold(
        theorem_name="Variable-time stopping-unitary lemma scaffold",
        claim_status=CLAIM_STATUS,
        readiness=readiness,
        phase_on=phase_on,
        n_arms=summary.n_arms,
        level_count=summary.level_count,
        index_dimension=transducer.index_dimension,
        stop_dimension=transducer.stop_dimension,
        statevector_dimension=transducer.statevector_dimension,
        compute_involution_residual_l2=compute_residual,
        phase_equivalence_residual_l2=phase_residual,
        work_garbage_probability_after_phase=garbage,
        serial_full_history_cost_per_branch=summary.serial_full_history_cost_per_branch,
        coherent_branch_rms_cost=summary.coherent_branch_rms_cost,
        all_branch_rms_search_proxy=summary.all_branch_rms_search_proxy,
        branch_rms_over_serial=summary.variable_over_serial_per_branch,
        all_branch_rms_formula_residual=formula_residual,
        output_count=summary.output_count,
        active_count=summary.active_count,
        unresolved_count=summary.unresolved_count,
        passed_count=passed,
        failed_count=failed,
        proof_obligation_count=proof_obligations,
        checks=checks,
    )


def stopping_unitary_lemma_markdown(
    scaffold: StoppingUnitaryLemmaScaffold,
) -> str:
    """Render a compact theorem-scaffold note."""

    lines = [
        "# Variable-time stopping-unitary lemma scaffold",
        "",
        f"Claim status: `{scaffold.claim_status}`",
        "",
        f"Readiness: `{scaffold.readiness}`",
        "",
        "## Executable resource summary",
        "",
        f"- arms: `{scaffold.n_arms}`",
        f"- levels: `{scaffold.level_count}`",
        f"- phase predicate: `{scaffold.phase_on}`",
        f"- serial full-history cost per branch: `{scaffold.serial_full_history_cost_per_branch}`",
        f"- coherent branch-RMS cost: `{scaffold.coherent_branch_rms_cost}`",
        f"- branch-RMS / serial: `{scaffold.branch_rms_over_serial}`",
        f"- compute involution residual: `{scaffold.compute_involution_residual_l2}`",
        f"- phase equivalence residual: `{scaffold.phase_equivalence_residual_l2}`",
        f"- work garbage after phase: `{scaffold.work_garbage_probability_after_phase}`",
        "",
        "## Checks",
        "",
    ]
    for check in scaffold.checks:
        lines.extend(
            [
                f"### {check.check_id}: {check.statement}",
                "",
                f"- status: `{check.status}`",
                f"- evidence: {check.evidence}",
                f"- missing for theorem: {check.missing_for_theorem}",
                "",
            ]
        )
    return "\n".join(lines)
