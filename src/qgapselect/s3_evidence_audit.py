"""Fail-closed integration audit for the S3 unknown-boundary evidence stack.

The audit consumes four already generated artifacts.  It verifies their
machine-readable evidence boundaries and reports which paper gates are closed.
It is a claim guard, not a theorem checker: a source artifact cannot turn an
unproved mathematical statement into a proved one merely by setting a flag.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

ADAPTIVE_ARTIFACT = "q_gapselect_adaptive_unknown_boundary_topk_s3_panel"
COHERENT_ARTIFACT = "q_gapselect_tiny_true_coherent_stopping_history_s3_panel"
FRONTIER_ARTIFACT = "q_gapselect_frontier_lower_bound_witness_s3"
COMPOSITION_ARTIFACT = "q_gapselect_same_interface_strong_composition_s3"

CLAIM_SCOPE = (
    "s3_integration_claim_guard_only_no_independent_theorem_or_advantage_proof"
)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a mapping with string keys")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _sequence(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return value


def _artifact(
    document: Mapping[str, object], *, expected_type: str, name: str
) -> Mapping[str, object]:
    resolved = _mapping(document, name)
    if resolved.get("artifact_type") != expected_type:
        raise ValueError(f"{name} has the wrong artifact_type")
    return resolved


def _coherent_record_audit(document: Mapping[str, object]) -> bool:
    """Recompute the tiny-circuit evidence gate from individual records.

    This checks artifact structure and executed ledgers.  It is not a formal
    verification of the Python source or a theorem certificate.
    """

    if document.get("schema_version") != 1:
        return False
    records = _sequence(document.get("records"), "coherent.records")
    required_roles = {
        "exact_grid_first_stop",
        "exact_grid_second_stop",
        "exact_grid_arm1_winner",
        "exact_grid_tie",
        "off_grid_fail_closed",
    }
    if len(records) != len(required_roles):
        return False
    by_role: dict[str, Mapping[str, object]] = {}
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"coherent.records[{index}]")
        role = record.get("role")
        if not isinstance(role, str) or role in by_role:
            return False
        by_role[role] = record
        result = _mapping(record.get("result"), f"coherent result {role}")
        resources = _mapping(result.get("resources"), f"coherent resources {role}")
        ledger = _mapping(resources.get("query_ledger"), f"coherent ledger {role}")
        counts = _mapping(ledger.get("query_counts"), f"coherent counts {role}")
        expected = _mapping(
            ledger.get("expected_query_counts"),
            f"coherent expected counts {role}",
        )
        charged = sum(
            _integer(counts.get(field), f"coherent {role} {field}")
            for field in (
                "forward",
                "inverse",
                "controlled_forward",
                "controlled_inverse",
            )
        )
        if not (
            counts == expected
            and charged == 176
            and _integer(counts.get("coherent_total"), "coherent_total") == 176
            and _integer(counts.get("classical_total"), "classical_total") == 0
            and _integer(counts.get("total"), "total") == 176
            and _integer(counts.get("qram_queries"), "qram_queries") == 0
            and _boolean(ledger.get("reconciled"), "coherent reconciled")
            and not _boolean(ledger.get("qram_assumed"), "coherent qram_assumed")
            and _boolean(
                result.get("fixed_expected_query_ledger_respected"),
                "fixed ledger respected",
            )
            and _boolean(result.get("budget_valid"), "coherent budget_valid")
        ):
            return False
        levels = _sequence(
            ledger.get("per_level_runtime_records"),
            f"coherent levels {role}",
        )
        if len(levels) != 2:
            return False
        for level, expected_one_way in zip(levels, (28, 60), strict=True):
            level_record = _mapping(level, f"coherent level {role}")
            one_way = _mapping(
                level_record.get("runtime_derived_one_way_counts"),
                f"coherent one-way level {role}",
            )
            if not (
                _integer(one_way.get("coherent_total"), "one-way coherent_total")
                == expected_one_way
                and _boolean(
                    level_record.get("full_replay_reconciled"),
                    "full replay reconciled",
                )
                and _boolean(
                    level_record.get("one_way_reconciled"),
                    "one-way reconciled",
                )
            ):
                return False
        history = _mapping(result.get("history"), f"coherent history {role}")
        durable = _mapping(
            result.get("durable_output"),
            f"coherent durable output {role}",
        )
        certificate = _mapping(
            result.get("certificate"),
            f"coherent certificate {role}",
        )
        if not (
            _boolean(
                history.get("single_statevector_history_register"),
                "single statevector history",
            )
            and _boolean(
                history.get("later_level_oracles_controlled_by_active_flag"),
                "later-level active control",
            )
            and _boolean(
                durable.get("scratch_to_durable_copy_executed"),
                "durable copy",
            )
            and _boolean(
                durable.get("full_history_replay_executed"),
                "full replay",
            )
            and not _boolean(certificate.get("issued"), "certificate issued")
            and not _boolean(
                result.get("quantum_advantage_claimable"),
                "record quantum advantage",
            )
        ):
            return False

    if set(by_role) != required_roles:
        return False
    winner_roles = (
        "exact_grid_first_stop",
        "exact_grid_second_stop",
        "exact_grid_arm1_winner",
    )
    if not all(
        _mapping(by_role[role]["result"], f"winner result {role}").get(
            "output_status"
        )
        == "MASK"
        and _boolean(
            _mapping(
                _mapping(by_role[role]["result"], f"winner result {role}")[
                    "resources"
                ],
                f"winner resources {role}",
            )["cleanup"]["passed"],  # type: ignore[index]
            f"winner cleanup {role}",
        )
        and _mapping(
            by_role[role].get("trusted_fixture_and_scoring"),
            f"trusted scoring {role}",
        ).get("diagnostic_mask_exact")
        is True
        for role in winner_roles
    ):
        return False
    tie_result = _mapping(by_role["exact_grid_tie"]["result"], "tie result")
    tie_cleanup = _mapping(
        _mapping(tie_result["resources"], "tie resources")["cleanup"],
        "tie cleanup",
    )
    off_grid_result = _mapping(
        by_role["off_grid_fail_closed"]["result"],
        "off-grid result",
    )
    off_grid_cleanup = _mapping(
        _mapping(off_grid_result["resources"], "off-grid resources")["cleanup"],
        "off-grid cleanup",
    )
    inactive = _mapping(
        document.get("inactive_level_clean_dirty_subspace_audit"),
        "inactive subspace audit",
    )
    provenance = _mapping(document.get("provenance"), "coherent provenance")
    config_hash = document.get("config_hash")
    return bool(
        tie_result.get("output_status") == "INCONCLUSIVE"
        and _boolean(tie_cleanup.get("passed"), "tie cleanup passed")
        and off_grid_result.get("output_status") == "INCONCLUSIVE"
        and not _boolean(off_grid_cleanup.get("passed"), "off-grid cleanup passed")
        and _boolean(
            inactive.get("clean_identity_witness_passed"),
            "clean identity witness",
        )
        and _boolean(
            inactive.get("dirty_negative_control_activated"),
            "dirty negative control",
        )
        and isinstance(config_hash, str)
        and len(config_hash) == 64
        and provenance.get("config_sha256") == config_hash
        and not _boolean(
            provenance.get("source_tree_dirty_at_execution"),
            "coherent dirty provenance",
        )
    )


@dataclass(frozen=True, slots=True)
class S3EvidenceGate:
    """One conjunctive paper gate and its current evidence boundary."""

    gate_id: str
    claim_gate: bool
    status: str
    satisfied: bool
    evidence: str
    blocker: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "gate_id": self.gate_id,
            "claim_gate": self.claim_gate,
            "status": self.status,
            "satisfied": self.satisfied,
            "evidence": self.evidence,
            "blocker": self.blocker,
        }


@dataclass(frozen=True, slots=True)
class S3EvidenceReport:
    """Integrated, fail-closed status of the four S3 evidence artifacts."""

    stage: str
    gates: tuple[S3EvidenceGate, ...]
    adaptive_case_count: int
    coherent_case_count: int
    coherent_exact_grid_complete_count: int
    coherent_fail_closed_count: int
    coherent_executed_queries_per_case: int
    composition_attempt_count: int
    composition_certified_exact_count: int
    composition_inconclusive_count: int
    composition_incorrect_certificate_count: int
    composition_budget_violation_count: int
    pair_hybrid_witness_count: int
    johnson_witness_count: int
    finite_composition_kill_count: int
    finite_comparator_dominance_count: int
    uncovered_required_baseline_count: int
    upstream_claims_remain_false: bool
    upstream_positive_claim_flags_detected: bool
    execution_audits_passed: bool
    coherent_record_audit_passed: bool
    independent_theorem_verifier_available: bool
    theorem_claim_activation_locked: bool
    quantum_advantage_claimable: bool
    ccf_a_claimable: bool
    claim_scope: str = CLAIM_SCOPE

    @property
    def satisfied_gate_count(self) -> int:
        return sum(gate.satisfied for gate in self.gates)

    @property
    def open_gate_count(self) -> int:
        return len(self.gates) - self.satisfied_gate_count

    @property
    def claim_gate_count(self) -> int:
        return sum(gate.claim_gate for gate in self.gates)

    @property
    def satisfied_claim_gate_count(self) -> int:
        return sum(gate.claim_gate and gate.satisfied for gate in self.gates)

    @property
    def open_claim_gate_count(self) -> int:
        return self.claim_gate_count - self.satisfied_claim_gate_count

    @property
    def checkpoint_count(self) -> int:
        return len(self.gates) - self.claim_gate_count

    @property
    def satisfied_checkpoint_count(self) -> int:
        return sum(not gate.claim_gate and gate.satisfied for gate in self.gates)

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "gates": [gate.as_dict() for gate in self.gates],
            "satisfied_gate_count": self.satisfied_gate_count,
            "open_gate_count": self.open_gate_count,
            "claim_gate_count": self.claim_gate_count,
            "satisfied_claim_gate_count": self.satisfied_claim_gate_count,
            "open_claim_gate_count": self.open_claim_gate_count,
            "checkpoint_count": self.checkpoint_count,
            "satisfied_checkpoint_count": self.satisfied_checkpoint_count,
            "adaptive_case_count": self.adaptive_case_count,
            "coherent_case_count": self.coherent_case_count,
            "coherent_exact_grid_complete_count": (
                self.coherent_exact_grid_complete_count
            ),
            "coherent_fail_closed_count": self.coherent_fail_closed_count,
            "coherent_executed_queries_per_case": (
                self.coherent_executed_queries_per_case
            ),
            "composition_attempt_count": self.composition_attempt_count,
            "composition_certified_exact_count": (
                self.composition_certified_exact_count
            ),
            "composition_inconclusive_count": self.composition_inconclusive_count,
            "composition_incorrect_certificate_count": (
                self.composition_incorrect_certificate_count
            ),
            "composition_budget_violation_count": (
                self.composition_budget_violation_count
            ),
            "pair_hybrid_witness_count": self.pair_hybrid_witness_count,
            "johnson_witness_count": self.johnson_witness_count,
            "finite_composition_kill_count": self.finite_composition_kill_count,
            "finite_comparator_dominance_count": (
                self.finite_comparator_dominance_count
            ),
            "uncovered_required_baseline_count": (
                self.uncovered_required_baseline_count
            ),
            "upstream_claims_remain_false": self.upstream_claims_remain_false,
            "upstream_positive_claim_flags_detected": (
                self.upstream_positive_claim_flags_detected
            ),
            "execution_audits_passed": self.execution_audits_passed,
            "coherent_record_audit_passed": self.coherent_record_audit_passed,
            "independent_theorem_verifier_available": (
                self.independent_theorem_verifier_available
            ),
            "theorem_claim_activation_locked": self.theorem_claim_activation_locked,
            "quantum_advantage_claimable": self.quantum_advantage_claimable,
            "ccf_a_claimable": self.ccf_a_claimable,
            "claim_scope": self.claim_scope,
        }


def audit_s3_evidence(
    adaptive_document: Mapping[str, object],
    coherent_document: Mapping[str, object],
    frontier_document: Mapping[str, object],
    composition_document: Mapping[str, object],
) -> S3EvidenceReport:
    """Audit the complete S3 stack without trusting prose or external statuses."""

    adaptive = _artifact(
        adaptive_document, expected_type=ADAPTIVE_ARTIFACT, name="adaptive"
    )
    coherent = _artifact(
        coherent_document, expected_type=COHERENT_ARTIFACT, name="coherent"
    )
    frontier = _artifact(
        frontier_document, expected_type=FRONTIER_ARTIFACT, name="frontier"
    )
    composition = _artifact(
        composition_document,
        expected_type=COMPOSITION_ARTIFACT,
        name="composition",
    )

    adaptive_summary = _mapping(adaptive.get("summary"), "adaptive.summary")
    coherent_summary = _mapping(coherent.get("summary"), "coherent.summary")
    frontier_summary = _mapping(frontier.get("summary"), "frontier.summary")
    composition_audit = _mapping(
        composition.get("aggregate_audit"), "composition.aggregate_audit"
    )
    composition_boundary = _mapping(
        composition.get("claim_boundary"), "composition.claim_boundary"
    )
    adaptive_assertions = _mapping(
        adaptive.get("aggregate_assertions"), "adaptive.aggregate_assertions"
    )
    coherent_assertions = _mapping(
        coherent.get("aggregate_assertions"), "coherent.aggregate_assertions"
    )

    execution_audits = all(
        (
            _boolean(
                adaptive_assertions.get("all_assertions_passed"),
                "adaptive assertions",
            ),
            _boolean(
                coherent_assertions.get("all_assertions_passed"),
                "coherent assertions",
            ),
            _boolean(
                composition_audit.get("all_checks_passed"),
                "composition checks",
            ),
            _boolean(
                frontier_summary.get("pair_hybrid_all_verified"),
                "pair witnesses",
            ),
            _boolean(
                frontier_summary.get("johnson_all_verified"),
                "Johnson witnesses",
            ),
        )
    )
    coherent_record_audit_passed = _coherent_record_audit(coherent)

    true_coherent = _boolean(
        coherent_summary.get("true_coherent_stopping_history_unitary_implemented"),
        "true coherent history flag",
    )
    active_control = _boolean(
        coherent_summary.get("later_level_active_control_implemented"),
        "later level control flag",
    )
    replay = _boolean(
        coherent_summary.get("durable_copy_and_full_replay_implemented"),
        "durable replay flag",
    )
    generic_off_grid = _boolean(
        coherent_summary.get("generic_off_grid_cleanup_proved"),
        "generic off-grid flag",
    )
    variable_time_bound = _boolean(
        coherent_summary.get("variable_time_query_speedup_proved"),
        "variable-time speedup flag",
    )
    upper_bound = _boolean(
        coherent_summary.get("new_query_upper_bound_proved"),
        "new upper-bound flag",
    )
    strongest_coverage = _boolean(
        frontier_summary.get("registered_strongest_composition_coverage_complete"),
        "strongest coverage flag",
    )
    matching_lower = _boolean(
        frontier_summary.get("matching_lower_bound_claimable"),
        "matching lower-bound flag",
    )
    official_reproduction = _boolean(
        composition_boundary.get("official_literature_reproduction"),
        "official reproduction flag",
    )
    candidate_in_cer_panel = _boolean(
        composition_boundary.get("candidate_included_in_cer_panel"),
        "candidate CER inclusion flag",
    )
    empirical_superiority = _boolean(
        composition_boundary.get("paired_candidate_cer_superiority_verified"),
        "paired CER superiority flag",
    )
    claim_bearing_sample_size = _boolean(
        composition_boundary.get("claim_bearing_sample_size_met"),
        "claim-bearing sample-size flag",
    )

    positive_claim_flags = any(
        (
            generic_off_grid,
            variable_time_bound,
            upper_bound,
            strongest_coverage,
            matching_lower,
            official_reproduction,
            candidate_in_cer_panel,
            empirical_superiority,
            claim_bearing_sample_size,
            _boolean(
                frontier_summary.get("strongest_composition_claimable"),
                "composition claim flag",
            ),
            *(
                _boolean(summary.get(field), f"upstream {field}")
                for summary in (adaptive_summary, coherent_summary, frontier_summary)
                for field in ("quantum_advantage_claimable", "ccf_a_claimable")
            ),
            *(
                _boolean(composition_boundary.get(field), f"composition {field}")
                for field in ("quantum_advantage_claimed", "ccf_a_claimable")
            ),
        )
    )
    upstream_claims_false = not positive_claim_flags

    # This integration layer validates schemas and finite execution ledgers. It
    # has no proof-assistant checker or source-faithful theorem-certificate
    # verifier. Consequently an upstream JSON boolean can never activate a
    # theorem, advantage, or venue-readiness gate here.
    independent_theorem_verifier_available = False
    theorem_claim_activation_locked = True

    circuit_satisfied = (
        coherent_record_audit_passed and true_coherent and active_control and replay
    )
    finite_dominance_count = _integer(
        frontier_summary.get("finite_fixture_query_dominance_count"),
        "finite fixture query dominance count",
    )
    gates = (
        S3EvidenceGate(
            gate_id="S3-C0-TINY-CIRCUIT-CHECKPOINT",
            claim_gate=False,
            status="VERIFIED_TINY_PROMISE_DOMAIN" if circuit_satisfied else "BLOCKED",
            satisfied=circuit_satisfied,
            evidence=(
                "One-statevector stopping history, active-controlled later level, "
                "durable copy, replay, and runtime query reconciliation."
            ),
            blocker=None if circuit_satisfied else "S3 execution assertions failed.",
        ),
        S3EvidenceGate(
            gate_id="S3-G1-CIRCUIT-GENERIC",
            claim_gate=True,
            status=("OPEN_UNVERIFIED_UPSTREAM_FLAG" if generic_off_grid else "OPEN"),
            satisfied=False,
            evidence="The current off-grid fixture exposes cleanup entanglement.",
            blocker=(
                "No independently verified generic confidence/correctness and "
                "cleanup theorem certificate."
            ),
        ),
        S3EvidenceGate(
            gate_id="S3-G2-EMPIRICAL-CER",
            claim_gate=True,
            status=(
                "OPEN_UNVERIFIED_UPSTREAM_FLAG"
                if candidate_in_cer_panel
                or empirical_superiority
                or claim_bearing_sample_size
                else "OPEN_CANDIDATE_NOT_IN_CER_PANEL"
            ),
            satisfied=False,
            evidence=(
                "The S3 control panel evaluates three proxy baselines only; the "
                "true-coherent candidate is not in a matched CER@Q campaign."
            ),
            blocker=(
                "No preregistered candidate-versus-baseline fixture campaign, "
                "paired effect-size test, bootstrap interval, or Holm correction."
            ),
        ),
        S3EvidenceGate(
            gate_id="S3-G3-VARIABLE-TIME-UPPER",
            claim_gate=True,
            status=(
                "OPEN_UNVERIFIED_UPSTREAM_FLAG"
                if variable_time_bound or upper_bound
                else "OPEN"
            ),
            satisfied=False,
            evidence="Executed worst-case cost is retained; branch RMS is a target only.",
            blocker=(
                "No independently verified scalable stopping/cleanup resource "
                "lemma or new upper-bound certificate."
            ),
        ),
        S3EvidenceGate(
            gate_id="S3-G4-STRONG-COMPOSITION-FIDELITY",
            claim_gate=True,
            status=(
                "OPEN_UNVERIFIED_UPSTREAM_FLAG"
                if strongest_coverage or official_reproduction
                else "OPEN"
            ),
            satisfied=False,
            evidence="Executable same-interface controls retain fail-closed fidelity labels.",
            blocker=(
                "No independently verified, source-faithful certificate covers "
                "the registered strongest literature reductions."
            ),
        ),
        S3EvidenceGate(
            gate_id="S3-G5-COMPOSITION-SEPARATION",
            claim_gate=True,
            status=(
                "OPEN_UNVERIFIED_UPSTREAM_FLAG"
                if _boolean(
                    frontier_summary.get("strongest_composition_claimable"),
                    "composition claim flag",
                )
                else (
                    "OPEN_S3_FINITE_DIAGNOSTIC_DOMINATED"
                    if finite_dominance_count > 0
                    else "OPEN"
                )
            ),
            satisfied=False,
            evidence=(
                "Independent exact-grid executions show the all-arm comparator "
                "uses fewer queries than the current S3 candidate."
            ),
            blocker="No all-published-composition separation for the S3 candidate.",
        ),
        S3EvidenceGate(
            gate_id="S3-G6-MATCHING-LOWER-BOUND",
            claim_gate=True,
            status=(
                "OPEN_UNVERIFIED_UPSTREAM_FLAG"
                if matching_lower
                else "OPEN_LOCAL_WITNESSES_ONLY"
            ),
            satisfied=False,
            evidence="Pair-hybrid and finite Johnson witnesses verify only local statements.",
            blocker=(
                "No independently verified continuous-angle Johnson composition "
                "or direct-sum theorem certificate."
            ),
        ),
    )

    advantage = False
    uncovered = frontier_summary.get("uncovered_required_baseline_ids")
    if not isinstance(uncovered, list) or any(not isinstance(item, str) for item in uncovered):
        raise TypeError("uncovered_required_baseline_ids must be a list of strings")
    return S3EvidenceReport(
        stage="s3_tiny_true_coherent_kernel_theorem_stack_open",
        gates=gates,
        adaptive_case_count=_integer(
            adaptive_summary.get("case_count"), "adaptive case_count"
        ),
        coherent_case_count=_integer(
            coherent_summary.get("case_count"), "coherent case_count"
        ),
        coherent_exact_grid_complete_count=_integer(
            coherent_summary.get("exact_grid_complete_count"),
            "coherent exact_grid_complete_count",
        ),
        coherent_fail_closed_count=_integer(
            coherent_summary.get("fail_closed_count"), "coherent fail_closed_count"
        ),
        coherent_executed_queries_per_case=_integer(
            coherent_summary.get("executed_queries_per_case"),
            "coherent executed_queries_per_case",
        ),
        composition_attempt_count=_integer(
            composition_audit.get("attempt_count"), "composition attempt_count"
        ),
        composition_certified_exact_count=_integer(
            composition_audit.get("certified_exact_count"),
            "composition certified_exact_count",
        ),
        composition_inconclusive_count=_integer(
            composition_audit.get("inconclusive_count"),
            "composition inconclusive_count",
        ),
        composition_incorrect_certificate_count=_integer(
            composition_audit.get("incorrect_certificate_count"),
            "composition incorrect_certificate_count",
        ),
        composition_budget_violation_count=_integer(
            composition_audit.get("budget_violation_count"),
            "composition budget_violation_count",
        ),
        pair_hybrid_witness_count=_integer(
            frontier_summary.get("pair_hybrid_witness_count"),
            "pair_hybrid_witness_count",
        ),
        johnson_witness_count=_integer(
            frontier_summary.get("johnson_witness_count"), "johnson_witness_count"
        ),
        finite_composition_kill_count=_integer(
            frontier_summary.get("composition_kill_count"),
            "composition_kill_count",
        ),
        finite_comparator_dominance_count=finite_dominance_count,
        uncovered_required_baseline_count=len(uncovered),
        upstream_claims_remain_false=upstream_claims_false,
        upstream_positive_claim_flags_detected=positive_claim_flags,
        execution_audits_passed=execution_audits,
        coherent_record_audit_passed=coherent_record_audit_passed,
        independent_theorem_verifier_available=independent_theorem_verifier_available,
        theorem_claim_activation_locked=theorem_claim_activation_locked,
        quantum_advantage_claimable=advantage,
        ccf_a_claimable=advantage,
    )


__all__ = [
    "ADAPTIVE_ARTIFACT",
    "CLAIM_SCOPE",
    "COHERENT_ARTIFACT",
    "COMPOSITION_ARTIFACT",
    "FRONTIER_ARTIFACT",
    "S3EvidenceGate",
    "S3EvidenceReport",
    "audit_s3_evidence",
]
