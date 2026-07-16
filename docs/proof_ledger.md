# Q-GapSelect proof ledger

Claim status: `proof_ledger_started_no_quantum_advantage_theorem`

Readiness: `proof_program_structured_not_ccf_a_claimable`

Theorem claimable: `False`

CCF-A claimable: `False`

## Status counts

- `execution_checked`: `1`
- `local_fact`: `1`
- `manual_instantiation_required`: `2`
- `proof_obligation`: `4`
- `proof_outline_started`: `2`

## Entries

### ST-U-CLEANUP: The tested stopping relation has compute involution, phase equivalence, cleanup, and RMS ledger identities.

- pillar: `upper_bound`
- status: `execution_checked`
- evidence: stopping_unitary_theorem artifact reports {'passed': 25, 'proof_obligation': 20} with all_execution_checks_passed=True.
- missing argument: Lift exact-state fixture checks to a symbolic unitary over all valid basis states and workspaces.
- activation condition: A written lemma proves the same identities for the compiled controlled-QPE stopping circuit.
- permitted wording: Execution checks pass for the scaffold; no upper-bound theorem is claimed.

### ST-P01: Compile a stopping-time unitary with charged QPE calls.

- pillar: `upper_bound`
- status: `proof_outline_started`
- evidence: The stopping scaffold identifies the required registers and calls.
- missing argument: Formal circuit map, inverse-call accounting, padding convention, and exact cleanup proof.
- activation condition: Manual proof of the stopping-unitary lemma.
- permitted wording: This is a proof outline, not a theorem.

### ST-P02: Compose bounded-error finite-QPE predicates across levels.

- pillar: `upper_bound`
- status: `proof_outline_started`
- evidence: The scaffold isolates deterministic predicates from proof obligations.
- missing argument: Confidence split, correlated-error accounting, and rejection of unresolved or duplicate outputs.
- activation condition: A bounded-error lemma with explicit failure budget and verifier semantics.
- permitted wording: Confidence composition remains open.

### ST-P03: Localize the Top-k boundary coherently rather than supplying it.

- pillar: `upper_bound`
- status: `proof_obligation`
- evidence: Current stopping fixtures still receive a numeric boundary.
- missing argument: A boundary-localization unitary that does not leak selected/rejected membership and composes with the stopping relation.
- activation condition: Coherent boundary lemma plus cleanup proof.
- permitted wording: The boundary-localization step is not solved.

### CF-T01: Known loop and variable-time composition theorems do not match the same no-free-QRAM interface.

- pillar: `composition_frontier`
- status: `manual_instantiation_required`
- evidence: composition_frontier artifact reports gate counts {'failed_encoded_composition_match': 4, 'open_no_encoded_composition_match': 9} and strongest baselines {'loop_variable_time_rebuild': 13}.
- missing argument: Manual theorem-by-theorem instantiation with exact oracle, output, memory, and stopping assumptions.
- activation condition: Every applicable published theorem is either instantiated as weaker or shown to use a strictly stronger interface.
- permitted wording: The encoded frontier is a screen, not an exhaustive proof.

### CF-T02: Known k-minimum, marked extraction, and QBAI baselines do not dominate the active candidate under identical assumptions.

- pillar: `composition_frontier`
- status: `manual_instantiation_required`
- evidence: The frontier artifact includes generated-predicate extraction, coarse+QBAI, independent QPE, and serial rebuild rows.
- missing argument: Replace proxy labels with formal reductions or assumption mismatches.
- activation condition: A composition-frontier proof table in the manuscript.
- permitted wording: The comparison is an encoded audit until manually proved.

### LB-B01: Boundary localization has a local angular discrimination barrier.

- pillar: `lower_bound`
- status: `local_fact`
- evidence: lower_bound_program artifact reports 12 local facts.
- missing argument: Lift the local two-state fact to the full unknown-boundary relation.
- activation condition: Use as one ingredient in a global adversary proof.
- permitted wording: This is a local lower-bound ingredient.

### LB-B02: Active-history recovery has a direct-sum lower-bound component.

- pillar: `lower_bound`
- status: `proof_obligation`
- evidence: The lower-bound program records the target proxy only.
- missing argument: Adversary matrix or polynomial relation forcing coherent history discovery across charged levels.
- activation condition: Formal all-algorithms lower-bound proof.
- permitted wording: This lower-bound block is open.

### LB-B03: Direct multi-output extraction has an output-sensitive barrier.

- pillar: `lower_bound`
- status: `proof_obligation`
- evidence: The lower-bound program records the target proxy only.
- missing argument: Proof that arbitrary adaptive quantum algorithms must pay the multi-output relation cost, not only estimate-then-sort algorithms.
- activation condition: Formal all-algorithms lower-bound proof.
- permitted wording: This lower-bound block is open.

### LB-B04: Composition-frontier exclusion is necessary for optimality.

- pillar: `lower_bound`
- status: `proof_obligation`
- evidence: The lower-bound program records 36 proof obligation rows across configured records.
- missing argument: Show that known same-interface algorithms cannot beat the claimed lower-target relation, or revise the candidate.
- activation condition: Manual composition frontier plus lower-bound theorem.
- permitted wording: Optimality is not proved.

## Next required manual proofs

- Write ST-P01 compiled stopping-unitary construction and cleanup proof.
- Write ST-P02 bounded-error confidence-composition lemma.
- Solve or replace ST-P03 coherent boundary localization.
- Instantiate CF-T01/CF-T02 against published composition theorems.
- Prove LB-B02 through LB-B04 with adversary or polynomial method.
