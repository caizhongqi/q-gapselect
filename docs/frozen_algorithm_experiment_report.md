# Frozen algorithm experiment report

## Scope

This stage evaluates only the previously designed synthetic algorithm
experiments. It executes no commercial model, LLM, attack target, or external
API. The two resource regimes remain separate:

- classical selector samples and realized source costs;
- analytic Layer-C coherent `A`/`A_dagger` calls.

Those units are not converted into each other. Simulator wall time is not a
quantum runtime.

## Strict closure checkpoint (2026-07-15)

The newer experiment stack closes several earlier *protocol* gaps without
closing the quantum-advantage theorem:

- `coherent_activity_history_core.py` now executes adaptive finite-state
  history, heterogeneous stopping, cleanup traces, direct births, fresh
  verification, and exact logical-query ledgers. Its amplitude-estimation
  measurements remain analytic and per-arm, so it is the scalable matched
  reference rather than a coherent-index implementation.
- `coherent_activity_history_statevector.py` separately executes an actual
  coherent index register with controlled QPE and explicit
  active/history/stop/output/work registers. In the committed 24-trial audit,
  all eight on-grid runs complete the measured boundary and clean up their
  executed layers, all eight tied-boundary controls fail closed, and all eight
  off-grid cleanup stresses expose the intended negative path. No run completes
  global direct multi-output extraction or receives a Top-k certificate.
- The fixed-cap primary panel now compares the scalable reference with k-only
  independent adaptive estimation, coarse-partition plus BAI composition,
  repeated single-output selection, and an unknown-time variable-time
  reference under identical oracle information and hard query caps. A
  known-stopping-time method is retained only as a stronger-information
  control.
- `theorem_closure_audit.json` uses one hard family and exact oracle/output
  contract for the upper, composition, and lower-bound audit. It records a
  decisive negative result: with a public static partition, legal
  coarse-partition composition matches or beats the candidate proxy. With a
  hidden partition that composition is invalid, but the candidate discovery
  and cleanup upper bound is unproved. The weighted matching lower bound is
  also open.

The public-data external-validity layer freezes classifier prediction columns
as Bernoulli arms, with reward `R(i,z)=1[h_i(x_z)=y_z]`. Classifier configs,
feature selection, prediction-hash deduplication, and label-blind shards are
fixed without held-out correctness outcomes. Strict local loaders are present
for UCI Letter, Optdigits, and Covertype; the current environment could execute
only scikit-learn's bundled 1,797-row Digits copy, which is explicitly marked
non-official.

The Digits diagnostic contains 24 arms, `k=8`, five frozen shards, 20 seeds,
four query caps, and five matched methods (800 executed records on the two
unique-boundary shards). Three shards have an exact k/(k+1) count tie and are
rejected without jitter. At cap 524,288, certified-exact recovery is 0.825 for
the activity-history reference, 1.000 for k-only independent adaptive and the
unknown-time reference, 0.975 for coarse+BAI, and 0.000 for repeated
single-output selection under the strict certificate contract. Thus the
candidate is Pareto-dominated on this diagnostic. The result is useful negative
evidence; it is not a public-data advantage result.

## Executed campaigns

### Classical frozen-selector matrix

`artifacts/frozen_selector_benchmark_diagnostic.json` contains:

- 8 reward/cost/graph landscapes;
- 5 query budgets from 64 to 1024;
- 64 independently frozen fixtures per cell;
- 6 selectors: random, uniform, successive halving, cost-aware racing,
  internal CLUCB-style Top-k, and GCGS-style graph search;
- 15,360 total selector runs.

Every panel reopens a fresh blind oracle over the same immutable tensor. The
selector never receives configured means, frozen means, reward streams, or the
trusted evaluator. Selector and oracle query/cost ledgers are cross-checked.

Strict exact Top-k recovery at the largest budget is:

| landscape | random | uniform | succ. halving | cost racing | CLUCB-style | GCGS-style |
|---|---:|---:|---:|---:|---:|---:|
| gap 0.25 ring | 0.000 | 0.859 | 0.969 | 0.859 | 1.000 | 0.812 |
| gap 0.08 ring | 0.000 | 0.734 | 0.922 | 0.734 | 0.859 | 0.656 |
| gap 0.04 ring | 0.000 | 0.469 | 0.641 | 0.469 | 0.609 | 0.375 |
| gap 0.02 ring | 0.000 | 0.469 | 0.625 | 0.469 | 0.656 | 0.406 |
| independent costs | 0.000 | 0.531 | 0.500 | 0.016 | 0.625 | 0.422 |
| boundary expensive | 0.000 | 0.438 | 0.391 | 0.188 | 0.609 | 0.375 |
| smooth grid | 0.000 | 0.125 | 0.203 | 0.125 | 0.156 | 0.156 |
| disconnected peaks | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.984 |

These results expose negative findings rather than selecting a favorable cell:

- no selector dominates every landscape and budget;
- the current cost-aware racing rule is poor when candidate costs are
  heterogeneous, so it is not yet a sufficiently strong cost-aware paper
  baseline;
- the current GCGS-style implementation does not consistently exploit the
  smooth graph and must not be described as an official XOXO-GCGS reproduction;
- shrinking the boundary gap substantially lowers exact recovery even at 1024
  samples.

### Frozen empirical Layer-C reference matrix

`artifacts/frozen_quantum_reference_diagnostic.json` contains:

- 10 `(family,n,k,gap)` cells;
- 500 independently generated exact-count frozen tables per cell;
- 5,000 total problem instances and 15,000 method runs;
- a non-isomorphic signed angular-gap generator varying the boundary gap,
  non-boundary gaps, active counts, and heterogeneity;
- 500/500 unique permutation-invariant difficulty fingerprints in every cell;
- Q-GapSelect's all-active analytic-IAE reference;
- gap-aided independent IAE Top-k with strict simultaneous interval separation;
- known-threshold IAE as a visibly stronger-information control;
- exact paired McNemar tests, 10,000-repetition fixture-pair bootstrap intervals,
  unconditional all-pair query differences, and Holm FWER correction.

The information regimes are intentionally not pooled. Q-GapSelect receives
only `k`; independent IAE additionally receives the public gap floor; the
threshold control additionally receives the public threshold. Therefore the
repository has no fully information-matched primary quantum baseline yet.

Only certified exact recovery counts as success. Q-GapSelect empirical
completion after `MAX_ROUNDS` and independent-IAE heuristic rankings are
failures even when their post-run set happens to be correct.

The central stress results are:

| cell | Q-GapSelect certified exact | independent IAE | known-threshold control | mean coherent queries: QG / independent / control |
|---|---:|---:|---:|---:|
| all cells through `n=32,k=8,gap≈pi/64` | 500/500 | 500/500 | 500/500 | cell dependent |
| `n=32,k=16,gap≈pi/128` | 447/500 | 401/500 | 411/500 | 86,371 / 53,248 / 51,993 |
| `n=32,k=16,gap≈pi/256` | 0/500 | 0/500 | 0/500 | 187,296 / 53,248 / 53,248 |

At `pi/128`, the Q-GapSelect versus gap-aided independent-IAE paired table is
383 both-success, 64 Q-GapSelect-only success, 18 independent-only success,
and 35 neither. The risk difference is 9.2 percentage points, with a 95%
fixture-pair bootstrap interval of 5.8 to 12.6 points. The exact McNemar
`p=3.32e-7` remains `3.32e-6` after Holm correction across ten cells. This is a
finite-distribution accuracy difference, not an advantage result: Q-GapSelect
uses about 33,123 more logical calls on average in that cell, the methods are
not query-matched, and the reference receives extra gap information.

Unconditional all-pair resource differences change sign by cell. Q-GapSelect
uses fewer calls in several `n=16` and `n=32,k=8` cells, more in the easy
`n=8` cells, substantially more at `pi/128`, and about 3.5 times as many at the
all-failure `pi/256` cell. There is no uniform finite-query dominance.

## What the results support

The artifacts support the following limited statements:

- the blind frozen-oracle and dual-ledger harness is executable and
  reproducible;
- every paper-scale cell passes the preregistered non-isomorphic-instance gate;
- certificate and heuristic output semantics are separated correctly;
- the all-active Q-GapSelect reference can resolve some smaller-gap cells that
  a fixed independent-IAE schedule leaves unresolved;
- the `pi/128` mixture-average accuracy difference is statistically stable
  under the declared paired analysis;
- adaptive behavior can also cost substantially more and still fail on a
  harder cell;
- heterogeneous cost and graph structure materially change classical selector
  behavior.

They do not support:

- a new coherent activity-history algorithm;
- a uniform fixed-instance or worst-case fixed-confidence guarantee;
- an information- and query-matched superiority result;
- a quantum speedup, hardware advantage, or CCF-A theorem claim;
- any LLM attack, transfer, ASR, or commercial-model claim.

The easy cells have 500/500 successes and a two-sided 95% Wilson lower bound of
about 0.9924 for the configured instance mixture. That does not prove a
per-instance guarantee because each fixture has only one algorithm seed. The
`pi/128` Q-GapSelect rate is 0.894 with a Wilson interval of approximately
`[0.864,0.918]`, and the `pi/256` rate is zero. Thus the campaign directly
falsifies any uniform 0.95-success interpretation of the current schedule.

## Remaining paper blockers

1. The scalable fixed-cap candidate executes an adaptive finite-state circuit
   IR but still obtains its phase evidence from analytic per-arm IAE. The
   separate coherent-index statevector kernel is exact only at small size and
   does not complete global direct multi-output Top-k extraction.
2. Scalable runs account for logical calls and IR operations, not a compiled
   state-preparation/QROM/reward/cleanup circuit with end-to-end gate, depth,
   qubit, and memory bounds. Exact-state resource checks do not establish
   scalable feasibility.
3. The unified same-family audit is now complete enough to falsify the public
   partition candidate: legal composition matches or improves on its proxy.
   Hiding the partition removes that baseline but also removes the current
   candidate upper bound. A revised interface/functional is required.
4. The weighted canonical matching lower bound remains a proof obligation; no
   adversary or polynomial-method proof currently matches the desired hidden
   activity-history cost.
5. Classical sample access and coherent logical queries remain different
   resource models. Neither simulator wall time nor an arbitrary conversion
   factor may be used to claim a speedup.
6. Fixed-fixture multiple-seed and information/query-cap matched panels now
   exist, but their statistical evidence gate is fail-closed and cannot repair
   the theorem or implementation gaps.
7. The first real-data diagnostic is small and negative: three of five Digits
   shards tie, and the candidate is dominated by matched k-only references.
   Official Letter, Optdigits, and Covertype confirmatory artifacts remain
   unexecuted because their source files are not available in this environment.

## Next algorithm-only campaign

The next work should remain independent of commercial models:

1. replace the public-partition cost functional or formulate a hidden-structure
   interface for which a constructive discovery/cleanup upper bound is actually
   provable;
2. implement a coherent union/output mechanism that converts branch-local
   history writes into one exact Top-k output without measurement-dependent
   classical reconstruction;
3. prove the variable-time stopping/cleanup/resource lemma for that revised
   mechanism, then manually instantiate every applicable loop, transducer,
   k-minimum, marked-extraction, and QBAI composition under the identical
   interface;
4. prove a weighted canonical lower bound on the same hidden hard family, or
   terminate the advantage claim if a legal composition matches the new upper;
5. compile the finite-table Layer-P purification with explicit lookup,
   workspace, uncomputation, gate, depth, qubit, and peak-memory accounting;
6. retain the complete fixed-cap hard-family results and run a separately
   locked 100-seed confirmatory panel only after the algorithm is frozen;
7. obtain the official Letter, Optdigits, and Covertype files, verify hashes,
   and execute the immutable public-data manifests without changing arms,
   shards, `k`, caps, or exclusions after observing outcomes.

Until those conditions hold, the correct research status remains:

`strong reproducible algorithm-diagnostic infrastructure; new quantum advantage theorem still open`.
