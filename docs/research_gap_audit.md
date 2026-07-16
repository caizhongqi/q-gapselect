# Q-GapSelect research gap audit

Current stage: `active_unknown_boundary_history_candidate_no_theorem`

Readiness: `pre_theorem_not_ccf_a_ready`

## Evidence now

### quantum_benchmark_diagnostic

- path: `artifacts/quantum_benchmark_diagnostic.json`
- status: `finite_exact_state_and_analytic_diagnostics_no_advantage_theorem`
- records: `778`
- interpretation: Executed finite exact-state and analytic diagnostics.  Supports implementation sanity, the rejected orientation witness, and an open unknown-boundary candidate gate; does not prove an advantage.

Key metrics:

- `experiment_name`: `q-gapselect-quantum-core-audit-v4-unknown-boundary-history`
- `suite_count`: `13`
- `claim_status`: `finite_exact_state_and_analytic_diagnostics_no_advantage_theorem`
- `unknown_boundary_gates`: `{'open_no_encoded_baseline_match_requires_unitary_and_lower_bound': 6}`
- `unknown_boundary_last_baseline_over_candidate`: `5.365213073666337`
- `orientation_composition_gate`: `failed_explicit_family`
- `orientation_failure_count`: `6`

### unknown_boundary_grid

- path: `artifacts/unknown_boundary_grid.json`
- status: `analytic_unknown_boundary_history_grid_no_theorem`
- records: `30`
- interpretation: Analytic parameter sweep for the no-free-QRAM candidate.  Open families are proof targets; failed families are rejected witnesses.

Key metrics:

- `experiment_name`: `q-gapselect-unknown-boundary-history-grid-v1`
- `claim_status`: `analytic_unknown_boundary_history_grid_no_theorem`
- `case_count`: `6`
- `total_records`: `30`
- `novelty_gate_counts`: `{'failed_encoded_baseline_match': 1, 'open_no_encoded_baseline_match_requires_unitary_and_lower_bound': 29}`
- `strongest_encoded_valid_baseline_counts`: `{'variable_time_rebuild_rms': 30}`
- `open_cases`: `['default_v4', 'larger_inactive_sea', 'growing_activity', 'decaying_activity', 'costly_activity_predicate']`
- `failed_cases`: `['loose_gate_negative_control']`

### charged_activity_history

- path: `artifacts/charged_activity_history.json`
- status: `charged_phase_history_prototype_no_upper_bound_theorem`
- records: `7`
- interpretation: Finite-phase predicate-generation prototype.  It removes supplied active/output rows from the toy relation and records charged compute/uncompute traces, but it is not a variable-time theorem.

Key metrics:

- `experiment_name`: `charged_activity_history_phase_window_audit_v1`
- `claim_status`: `charged_phase_history_prototype_no_upper_bound_theorem`
- `case_count`: `2`
- `total_records`: `7`
- `all_no_supplied_predicate_rows`: `True`
- `all_output_subset_active`: `True`
- `exact_state_trace_count`: `7`

### variable_time_charged_history

- path: `artifacts/variable_time_charged_history.json`
- status: `variable_time_charged_history_alignment_no_theorem`
- records: `16`
- interpretation: Mainline alignment audit: finite-QPE charged predicate costs are inserted into the unknown-boundary history target and compared with same-interface proxy baselines.  Open gates are proof targets.

Key metrics:

- `experiment_name`: `variable_time_charged_history_alignment_v1`
- `claim_status`: `variable_time_charged_history_alignment_no_theorem`
- `case_count`: `3`
- `total_records`: `16`
- `novelty_gate_counts`: `{'failed_charged_baseline_match': 4, 'open_charged_variable_time_gap_requires_upper_and_lower_bound': 12}`
- `strongest_valid_baseline_counts`: `{'variable_time_rebuild_rms': 16}`
- `open_case_count`: `2`
- `open_cases`: `['default_mainline_charged', 'decaying_activity_charged']`
- `failed_cases`: `['loose_gate_negative_control']`

### stopping_time_transducer

- path: `artifacts/stopping_time_transducer.json`
- status: `variable_time_stopping_transducer_skeleton_no_theorem`
- records: `7`
- interpretation: Executable variable-time stopping relation skeleton.  It checks stop-register compute/phase/uncompute traces and serial versus branch-RMS ledgers, but it is not a circuit theorem.

Key metrics:

- `experiment_name`: `stopping_time_transducer_skeleton_v1`
- `claim_status`: `variable_time_stopping_transducer_skeleton_no_theorem`
- `case_count`: `2`
- `total_records`: `7`
- `exact_state_trace_count`: `7`
- `all_variable_over_serial_at_most_one`: `True`

### stopping_unitary_theorem

- path: `artifacts/stopping_unitary_theorem.json`
- status: `stopping_time_unitary_lemma_scaffold_no_proof`
- records: `5`
- interpretation: Paper-facing stopping-unitary lemma scaffold.  It separates execution-checked identities from proof obligations; it is not a completed upper-bound theorem.

Key metrics:

- `experiment_name`: `stopping_unitary_theorem_scaffold_v1`
- `claim_status`: `stopping_time_unitary_lemma_scaffold_no_proof`
- `case_count`: `2`
- `total_records`: `5`
- `check_status_counts`: `{'passed': 25, 'proof_obligation': 20}`
- `all_execution_checks_passed`: `True`

### composition_frontier

- path: `artifacts/composition_frontier.json`
- status: `composition_frontier_audit_no_novelty_theorem`
- records: `13`
- interpretation: Known-composition frontier audit for loop composition, generated predicate extraction, QBAI, independent QPE, and forbidden free-history QRAM baselines.  It is an encoded novelty screen, not a theorem against all prior work.

Key metrics:

- `experiment_name`: `composition_frontier_mainline_2026_07_15`
- `claim_status`: `composition_frontier_audit_no_novelty_theorem`
- `case_count`: `3`
- `total_records`: `13`
- `novelty_gate_counts`: `{'failed_encoded_composition_match': 4, 'open_no_encoded_composition_match': 9}`
- `strongest_valid_baseline_counts`: `{'loop_variable_time_rebuild': 13}`
- `open_case_count`: `2`
- `open_cases`: `['default_mainline_frontier', 'decaying_activity_frontier']`
- `failed_cases`: `['loose_gate_negative_control']`

### lower_bound_program

- path: `artifacts/lower_bound_program.json`
- status: `lower_bound_program_scaffold_no_adversary_proof`
- records: `12`
- interpretation: Symbolic lower-bound proof program.  It enumerates hard-family blocks and missing proof arguments; it does not prove L-07.

Key metrics:

- `experiment_name`: `lower_bound_program_mainline_2026_07_15`
- `claim_status`: `lower_bound_program_scaffold_no_adversary_proof`
- `case_count`: `2`
- `total_records`: `12`
- `strongest_block_counts`: `{'LB-B04': 12}`
- `total_proof_obligations`: `36`
- `total_local_facts`: `12`

### proof_ledger

- path: `artifacts/proof_ledger.json`
- status: `proof_ledger_started_no_quantum_advantage_theorem`
- records: `10`
- interpretation: Machine-readable proof ledger tying stopping-unitary, composition, and lower-bound obligations together.  It explicitly marks the theorem stack as not CCF-A-claimable yet.

Key metrics:

- `claim_status`: `proof_ledger_started_no_quantum_advantage_theorem`
- `readiness`: `proof_program_structured_not_ccf_a_claimable`
- `entry_count`: `10`
- `status_counts`: `{'execution_checked': 1, 'local_fact': 1, 'manual_instantiation_required': 2, 'proof_obligation': 4, 'proof_outline_started': 2}`
- `theorem_claimable`: `False`
- `ccf_a_claimable`: `False`

## Completed evidence

- Old orientation witness is rejected by explicit composition audit.
- Unknown-boundary history candidate is encoded as a no-free-QRAM audit target.
- Parameter grid contains both open families and a failing negative control.
- Finite exact-state runner and analytic grid produce reproducible artifacts.
- Charged finite-phase prototype removes supplied predicate rows from the activity-history relation fixture.
- Variable-time charged alignment inserts finite-QPE predicate costs back into the main unknown-boundary candidate comparison.
- Stopping-time relation skeleton adds explicit stop-register compute/phase/uncompute traces and branch-RMS ledgers.
- Stopping-unitary theorem scaffold separates executable checks from manual proof obligations.
- Composition-frontier audit encodes same-interface known-composition baselines and a forbidden free-history QRAM collapse.
- Lower-bound proof program enumerates local facts and adversary-proof obligations for L-07.
- Proof ledger connects upper-bound, composition-frontier, and lower-bound obligations without activating theorem claims.

## Rejected claims

- H_orient by itself is not a new quantum core.
- Boundary-only membership certificates are not discovery evidence.
- Simulator runtime or finite slopes are not asymptotic quantum advantage.

## Highest-priority gaps

### P0-U08: Construct the no-free-QRAM activity-history transducer

- priority: `P0`
- current status: `stopping_unitary_lemma_scaffold_started`
- evidence now: unknown_boundary_history.py defines the cost target and gate; charged_activity_history.py derives finite-phase predicates; variable_time_charged_history.py aligns charged finite-QPE costs; stopping_time_transducer.py gives an executable stopping-register skeleton; stopping_time_theorem.py now separates executable lemma checks from proof obligations.
- missing evidence: Manual theorem proof for ST-P01 through ST-P04: compiled stopping unitary, bounded-error confidence composition, coherent boundary localization, and strongest-composition separation.
- next artifact: `Write the manual proof for the stopping-time unitary lemma and connect it to the composition frontier.`
- activation target: `Claim U-08 / N-03 upper-bound side`
- failure mode: Any hidden scan over inactive arms or free QRAM lookup collapses the candidate into a known rebuilt-history baseline.

### P0-L07: Prove a matching all-algorithms lower bound

- priority: `P0`
- current status: `proof_ledger_started`
- evidence now: proof_ledger.py connects LB-B02 through LB-B04 to the upper-bound and composition-frontier obligations; it still marks them as open proof obligations.
- missing evidence: Adversary or polynomial-method proof under the same unknown-boundary/no-free-QRAM oracle interface.
- next artifact: `Replace LB-B02 through LB-B04 proof_obligation labels with a formal adversary matrix or polynomial-method proof.`
- activation target: `Claim L-07 / N-03 lower-bound side`
- failure mode: A proof that only covers estimate-then-sort algorithms will not support a CCF-A algorithmic claim.

### P1-COMP: Extend the strongest-composition audit

- priority: `P1`
- current status: `proof_ledger_started`
- evidence now: proof_ledger.py records CF-T01 and CF-T02 as manual theorem instantiation requirements; composition_frontier.py supplies the encoded screen but not the final proof.
- missing evidence: Manual theorem-by-theorem instantiation showing no published composition theorem matches the same interface, or a valid published theorem that kills this candidate.
- next artifact: `docs/composition_frontier_proof.md with explicit theorem instantiations and assumption mismatches.`
- activation target: `Prior-work novelty boundary for N-03`
- failure mode: If a valid known composition matches the candidate, the new core must be rejected or reframed.

### P1-VERIFY: Build verifier and confidence accounting for the new relation

- priority: `P1`
- current status: `not_started_for_new_core`
- evidence now: Existing direct Top-k and BBHT paths have verifier/query-ledger tests, but they still rely on measured calibration.
- missing evidence: A verifier for unknown-boundary history outputs with duplicate, unresolved, and non-separating failure semantics.
- next artifact: `tests/test_activity_history_transducer.py and verifier traces in a new diagnostic artifact.`
- activation target: `Constructive algorithm correctness`
- failure mode: If verification consumes membership-equivalent information, it cannot support discovery claims.

### P2-PAPER: Convert the code audit into paper-ready theorem statements

- priority: `P2`
- current status: `documentation_partial`
- evidence now: claim_matrix.md and unknown_boundary_history_spec.md define the active problem and blocked claims.
- missing evidence: Precise theorem statements, assumptions, proof sketches, and a table separating proved facts from finite artifacts.
- next artifact: `docs/paper_readiness_plan.md generated from this audit plus manual proof sections in paper/main.tex.`
- activation target: `Submission narrative discipline`
- failure mode: A manuscript that states open gates as proved advantages will be rejected by informed reviewers.

## Recommended next steps

- Convert proof-ledger entries ST-P01, ST-P02, CF-T01/CF-T02, and LB-B02 through LB-B04 into manual proof sections.
- Manually instantiate the composition frontier against published composition theorems.
- Convert LB-B02 through LB-B04 into a formal adversary or polynomial-method proof.
- Keep every open novelty gate labelled as a proof obligation in paper text.
