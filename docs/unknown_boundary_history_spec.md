# Unknown-boundary coherent activity-history core

Status: **active candidate problem, no theorem yet**.

This document is the replacement theory target after the orientation-family
composition audit.  It is deliberately application-neutral and does not use an
application-specific reward oracle or QNN.

## 1. Problem interface

The algorithm receives coherent canonical block access to unknown angles
`theta_1, ..., theta_n`, knows `n`, `k`, and `delta`, and must output the exact
Top-k set with probability at least `1-delta`.

The algorithm is not given:

- the sorted order;
- the Top-k boundary;
- selected/rejected membership;
- active arms at any precision level; or
- a QRAM table of unresolved-history lists.

A legal implementation must generate any activity predicate coherently from
charged oracle access and must uncompute its workspace.

## 2. Candidate family

The configured audit family is:

- `n = m^3` arms;
- `m` precision epochs;
- `m` hidden active arms per epoch;
- one newly required output per epoch;
- minimum angular gap `gamma = m^-2`;
- mildly heterogeneous precision `epsilon_r = gamma (r+1)^0.25`.

The inactive sea is intentionally large.  A rebuilt-history baseline that
scans all arms at every precision level should pay for that sea.  The candidate
may avoid this only if we build a reversible activity-history transducer.

## 3. Candidate cost target

The audit records the proxy

```text
C_candidate
  = C_boundary
  + C_history
  + sqrt(sum_r A_r (M_r + 1) / epsilon_r^2).
```

Here `A_r` is the active count at level `r`, and `M_r` is the number of newly
required outputs at that level.  This is a cost target, not a proved upper
bound.

The encoded legal baselines are:

- variable-time rebuilt-history RMS;
- Grover activity reconstruction per level;
- independent all-arm scan at the finest precision;
- coarse partition plus BAI.

The suite also reports a known-boundary/free-history layered proxy, but marks
it invalid under the no-free-QRAM interface.

The extended grid runner is:

```bash
python scripts/run_unknown_boundary_grid.py \
  --config configs/unknown_boundary_grid.json \
  --output artifacts/unknown_boundary_grid.json
```

It sweeps nearby families and includes a loose-tolerance negative control to
verify that the novelty gate can fail.

## 4. Current finite audit result

With the default sweep `m in {8,16,32,64,128,256}`, the open candidate has:

- novelty gate: open for all six points;
- strongest encoded valid baseline: variable-time rebuilt-history RMS;
- last-point baseline/candidate ratio: about `5.36`;
- candidate finite-family slope: about `3.42`;
- lower-target proxy slope: about `3.50`.

These numbers only show that the candidate has not been killed by the encoded
baselines.  They are not a complexity theorem.

The grid artifact should be read the same way: a family that stays open is a
candidate for proof work; a failed family is a rejected witness.

## 5. Required algorithm design

Current code status has five layers:

1. `src/qgapselect/activity_history_transducer.py` contains a toy relation-level
   transducer.  It verifies reversible compute--phase--uncompute semantics and
   query-ledger accounting when the active/output predicates are supplied by the
   benchmark harness.
2. `src/qgapselect/charged_activity_history.py` removes that supplied-predicate
   shortcut for a finite prototype.  It derives active/output flags from
   normalized phase values, finite-QPE precision levels, and a boundary window;
   `artifacts/charged_activity_history.json` records exact-state traces and QPE
   query-unit ledgers.
3. `src/qgapselect/variable_time_charged_history.py` moves the charged prototype
   back onto the main unknown-boundary line.  It sets finite-QPE level costs from
   the required `epsilon_r`, computes charged active-history and direct-output
   quadrature proxies, and compares them with serial rebuild, variable-time
   rebuild, Grover-activity, independent scan, and coarse+BAI baselines.
4. `src/qgapselect/stopping_time_transducer.py` makes the next circuit-shape
   component executable.  It computes a first-output stopping code for each arm,
   XORs the stop register and active/output flags, performs
   compute--phase--uncompute traces, and records serial full-history versus
   branch-RMS stopping ledgers.
5. `src/qgapselect/stopping_time_theorem.py` converts that skeleton into a
   theorem scaffold.  It records executable checks for involution, phase
   equivalence, cleanup, and RMS identities, while marking circuit synthesis,
   confidence composition, coherent boundary localization, and composition
   dominance as explicit proof obligations.
6. `src/qgapselect/composition_frontier.py` encodes the current known-composition
   frontier under the same proxy interface.  Its artifact records open families
   and failed negative controls; it is not a proof that no published theorem
   applies.
7. `src/qgapselect/lower_bound_program.py` records the L-07 proof program:
   local boundary localization is a local fact, while active-history recovery,
   direct multi-output extraction, and composition-frontier exclusion remain
   proof obligations.
8. `src/qgapselect/proof_ledger.py` ties the upper-bound, composition-frontier,
   and lower-bound obligations into one theorem-stack ledger.  It marks the
   current state as not theorem-claimable and not CCF-A-claimable.

The second through eighth layers are stronger code-sanity evidence, but still
not the final unknown-boundary algorithm.  The boundary is not yet localized by
a coherent circuit, and the theorem scaffold is a proof checklist rather than a
proved variable-time transducer theorem.

The next constructive proof must supply:

1. a reversible unknown-boundary representation that does not reveal full
   membership;
2. a coherent unresolved-history predicate for each precision epoch;
3. a variable-time search schedule over heterogeneous finite-QPE precision costs;
4. direct multi-output extraction from the coherent history relation;
5. cleanup of boundary, activity, QPE, and verification garbage;
6. exact query accounting for controlled, inverse, and verification calls; and
7. a verifier that rejects unresolved, duplicate, or non-separating outputs.

If any step uses a measured active-list table or a free QRAM lookup, the
candidate fails its own interface.

## 6. Required lower bound

The lower bound must cover all adaptive quantum algorithms under the same
oracle interface.  It cannot be restricted to estimate-then-sort algorithms.

The target is an adversary or polynomial-method theorem showing that the
candidate family requires the recorded lower-target proxy up to declared
logarithms.  A valid proof must also rule out loop/transducer composition
matching the candidate under the same no-free-QRAM assumptions.

The current scaffold is:

```bash
python scripts/run_lower_bound_program.py \
  --config configs/lower_bound_program.json \
  --output artifacts/lower_bound_program.json \
  --markdown docs/lower_bound_program.md
```

It records the proof program but does not activate L-07.
