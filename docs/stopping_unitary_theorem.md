# Variable-time stopping-unitary theorem scaffold

Claim status: `stopping_time_unitary_lemma_scaffold_no_proof`

The JSON artifact contains the full per-record check table.  The first record is summarized here as the canonical proof template.

- readiness: `execution_checks_passed_proof_obligations_remain`
- passed checks: `5`
- failed checks: `0`
- proof obligations: `4`

## Checks

### ST-U01: The stopping compute relation is an involution on the tested state.

- status: `passed`
- evidence: L2 residual 0.000e+00.
- missing for theorem: Generalize from exact-state fixture to a symbolic unitary over all valid basis states and padded registers.

### ST-U02: Compute--phase--uncompute implements the intended phase predicate.

- status: `passed`
- evidence: L2 residual 0.000e+00 for phase_on=output.
- missing for theorem: Prove the predicate equivalence for all superpositions and for the eventual unknown-boundary predicate.

### ST-U03: The stop, active, and output work registers are cleaned after phase.

- status: `passed`
- evidence: Work-garbage probability 0.000e+00.
- missing for theorem: Lift the cleanup identity to the final circuit with boundary, QPE, verification, and confidence workspaces.

### ST-R01: The branch-RMS stopping ledger is no larger than serial full history.

- status: `passed`
- evidence: branch_rms=41.0023, serial=44.
- missing for theorem: Derive the same inequality from charged stopping times in the formal variable-time algorithm.

### ST-R02: The all-branch RMS proxy equals sqrt(sum_i T_i^2).

- status: `passed`
- evidence: Formula residual 0.000e+00.
- missing for theorem: Connect this proxy to the chosen variable-time search or amplitude-amplification theorem.

### ST-P01: There is a compiled stopping-time unitary with charged QPE calls.

- status: `proof_obligation`
- evidence: Current code supplies a relation skeleton and exact-state traces.
- missing for theorem: Write the circuit construction using controlled QPE, stopping flags, inverse calls, and explicit workspace dimensions.

### ST-P02: The bounded-error predicate composes across levels and outputs.

- status: `proof_obligation`
- evidence: The scaffold uses deterministic finite-phase fixtures.
- missing for theorem: Add confidence allocation and prove no unresolved/duplicate output is accepted.

### ST-P03: The boundary is localized coherently rather than supplied.

- status: `proof_obligation`
- evidence: The current transducer receives a numeric boundary phase.
- missing for theorem: Integrate the unknown-boundary localization unitary and prove that it does not leak membership.

### ST-P04: Known loop/composition theorems do not already imply the same bound.

- status: `proof_obligation`
- evidence: Composition frontier is still a separate P1 audit.
- missing for theorem: Instantiate strongest loop composition, k-minimum, QBAI, and variable-time baselines under identical assumptions.
