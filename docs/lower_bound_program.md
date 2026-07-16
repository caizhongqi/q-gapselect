# Q-GapSelect lower-bound proof program

Claim status: `lower_bound_program_scaffold_no_adversary_proof`

This document is a proof scaffold. It records target blocks and missing arguments; it is not an adversary lower-bound proof.

## Summary

- cases: `2`
- records: `12`
- proof obligations: `36`
- local facts: `12`

## Proof blocks

### LB-B01

- description: Boundary localization by angular discrimination near the Top-k cut.
- role: `local angular boundary barrier`
- status: `local_fact_available`
- missing argument: Lift local two-state discrimination to the unknown-boundary multi-output relation.

### LB-B02

- description: Direct active-history recovery barrier over charged levels.
- role: `history direct-sum component`
- status: `proof_obligation`
- missing argument: Construct adversary matrix or polynomial relation forcing coherent active-history discovery.

### LB-B03

- description: Direct multi-output extraction barrier from active history.
- role: `output-sensitive extraction component`
- status: `proof_obligation`
- missing argument: Show all adaptive quantum algorithms must pay the charged multi-output relation cost, not only estimate-then-sort.

### LB-B04

- description: Composition-frontier exclusion barrier.
- role: `prior-composition separation requirement`
- status: `proof_obligation`
- missing argument: Prove known loop, k-minimum, QBAI, and variable-time theorems do not imply a matching same-interface upper bound.

## Activation condition

The L-07 claim can only be activated after the proof-obligation blocks are replaced by a formal adversary or polynomial-method argument under the same no-free-QRAM oracle interface.
