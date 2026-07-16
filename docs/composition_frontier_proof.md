# Composition-frontier proof scaffold

Status: proof scaffold, not a novelty theorem.

This document is the manual proof target for `P1-COMP`.  The encoded artifact
`artifacts/composition_frontier.json` is only a screen.  The manuscript can
activate a novelty-boundary claim only after every row below is replaced by a
formal theorem instantiation or a precise assumption mismatch.

## Interface to preserve

The active candidate uses the no-free-QRAM unknown-boundary interface:

- no supplied Top-k boundary;
- no supplied active-history list;
- no QRAM table of unresolved arms;
- finite-QPE predicate generation is charged;
- cleanup must erase stopping, activity, output, QPE, and verification garbage;
- output relation is exact Top-k, not one marked element or one best arm.

Any baseline that receives a stronger interface must be marked invalid for
same-interface dominance, even if its proxy expression is smaller.

## Required theorem-instantiation rows

| Row | Baseline family | Current artifact row | Same-interface question | Manual proof obligation |
|---|---|---|---|---|
| CF-T01 | Loop / variable-time composition | `loop_variable_time_rebuild` | Can loop composition generate the same active history without rebuilding the inactive sea? | Instantiate the loop theorem with charged finite-QPE stopping times and show whether it pays `variable_time_rebuild_rms_proxy` or less. |
| CF-T02 | All-marked extraction | `all_marked_extraction_generated_predicate` | Does marked extraction receive a ready predicate, or must it generate it coherently at every level? | Charge predicate generation and uncomputation before comparing to the candidate. |
| CF-T03 | Quantum best-arm identification / coarse partition | `coarse_partition_plus_qbai` | Does coarse partition expose a valid same-output relation for exact Top-k? | Prove or reject the reduction from unknown-boundary exact Top-k to the BAI oracle assumptions. |
| CF-T04 | Independent QPE / estimate-then-sort | `independent_all_arm_qpe` | Does estimating all arms dominate the active-history candidate? | State the exact precision schedule and show it pays all-arm finest-level cost. |
| CF-T05 | Serial rebuild scan | `serial_rebuild_scan` | Does any naive same-interface algorithm match the candidate on the configured family? | Keep as a sanity upper baseline, not a novelty threat unless it matches. |
| CF-T06 | Free-history QRAM / known active lists | `free_history_qram_layered` | Does the theorem assume already materialized active histories? | Mark invalid under the no-free-QRAM interface unless a charged construction is supplied. |

## Current artifact summary

The current composition-frontier run records:

- `13` total records;
- `9` open encoded gates;
- `4` failed negative-control gates;
- strongest encoded valid baseline: `loop_variable_time_rebuild`;
- theorem status: not claimable.

These numbers are useful only for locating the manual proof target.  They are
not a substitute for reading and instantiating the prior theorems.

## Activation rule

The composition-frontier part of `N-03` can be strengthened only if the final
paper contains:

1. one row per applicable published theorem;
2. the exact oracle and output relation assumed by that theorem;
3. the cost paid to generate and uncompute activity predicates;
4. whether the theorem receives a stronger interface than the candidate;
5. the resulting asymptotic comparison under the configured hard family.

If any valid same-interface theorem matches the candidate, the current novelty
claim must be rejected or reframed.
