# Stopping-unitary proof skeleton

Status: proof skeleton, not an upper-bound theorem.

This document is the manual proof target for `ST-P01` and `ST-P02`.  The
executable scaffold in `artifacts/stopping_unitary_theorem.json` checks finite
fixtures; the theorem below still needs a symbolic circuit proof.

## Target lemma

Let `O_theta` be canonical coherent block access to arm angles.  For precision
levels `r = 0, ..., R-1`, let `QPE_r` be the finite-precision phase estimator
with charged query cost `c_r`.  Let `S(i)` be the first level at which arm `i`
becomes stopping-resolved by the unknown-boundary predicate.

The target stopping unitary should implement, up to bounded error,

```text
U_stop |i>|0_stop>|0_work>
  = |i>|S(i)>|clean_work_i>
```

followed by a predicate phase and exact uncomputation:

```text
U_stop^dagger P U_stop |psi>|0>
  = sum_i alpha_i (-1)^{b_i} |i>|0>.
```

The intended query ledger is controlled by

```text
sqrt(sum_i T_i^2)
```

where `T_i = sum_{r <= S(i)} c_r` is the charged stopping time for arm `i`.

## ST-P01: compiled stopping-time unitary

Required proof components:

1. Register layout for index, level, QPE phase, stopping code, active flag,
   output flag, verifier flag, and scratch space.
2. Controlled finite-QPE construction for each precision level.
3. Reversible update rule that writes the first stopping level without measuring.
4. Exact inverse call sequence that removes QPE and predicate work garbage.
5. Query ledger counting controlled and inverse oracle calls.

Current code evidence:

- `stopping_time_transducer.py` verifies the stop-register compute/phase/uncompute
  relation on finite fixtures.
- `stopping_time_theorem.py` records passed execution checks for involution,
  phase equivalence, cleanup, and RMS accounting.

Missing theorem step:

- generalize from deterministic fixture predicates to charged finite-QPE
  predicates over all valid basis states.

## ST-P02: bounded-error confidence composition

Required proof components:

1. Assign per-level and per-output failure budgets.
2. Prove false-active, false-inactive, duplicate-output, and unresolved-output
   events are rejected or bounded.
3. Show that coherent reuse of QPE predicates does not silently assume classical
   membership tables.
4. Connect verifier calls to the same oracle model and query ledger.

Current status:

- no bounded-error lemma is proved yet;
- deterministic fixtures are code sanity only;
- confidence accounting remains a theorem obligation.

## Cleanup and resource invariant

The final proof must show:

```text
Pr[work != 0 after U_stop^dagger P U_stop] = 0
```

for exact predicate circuits, and bounded by the declared confidence budget for
finite-QPE predicates.

It must also prove that no hidden scan over all inactive arms or hidden QRAM
lookup is used.  If cleanup requires materialized active lists, the candidate
collapses into a rebuilt-history baseline.

## Activation rule

The upper-bound side of `N-03` can be strengthened only after this skeleton is
replaced by a formal lemma with:

- symbolic circuit maps;
- query counts;
- workspace dimensions;
- cleanup identities;
- confidence composition;
- connection to the composition-frontier proof.
