# Q-GapSelect

Q-GapSelect is a research codebase for **heterogeneous-discrimination-gap, multi-output quantum
pure exploration**. Its immediate objective is to determine whether Top-$k$ and
matroid best-basis identification admit a relation-aware quantum query complexity
that is both algorithmically attainable and matched by a lower bound.

The active research program has two explicitly separated layers. The quantum
core is an unknown-boundary, coherent-activity-history, no-free-QRAM multi-output
selection problem. The application layer, Q-GapAttack, uses that selector to
choose transferable attacks against code LLMs from an authorized local surrogate.
The previous orientation-aware candidate has been falsified as a novelty core.

The project starts from a strict claim boundary: standard amplitude estimation,
amplitude amplification, quantum best-arm identification, and classical
combinatorial pure exploration are prior primitives. The paper contribution is not
claimed until the proposed direct multi-output algorithm and its lower bound are
proved. Executable simulations and conjectural query accounting are labelled
separately throughout the repository.

## Research questions

1. Can a direct quantum Top-$k$ identification algorithm exploit heterogeneous
   angular boundary gaps without estimating every arm independently?
2. What output-sensitive direct-sum lower bound interpolates correctly between
   quantum best-arm identification ($k=1$) and the linear output barrier
   ($k=\Theta(n)$)?
3. Which parts extend from uniform Top-$k$ constraints to partition and general
   matroids?
4. Can the unknown-boundary activity-history interface support a new upper bound
   and matching all-algorithms lower bound that known loop/variable-time
   compositions do not already match?
5. On a frozen semantic-transformation graph, can the selector find a stronger
   transferable attack portfolio than XOXO-GCGS, INSEC, classical racing, and
   independent quantum estimation under matched budgets?

## What is implemented

- A complete small-scale NumPy statevector oracle for the canonical phase-fixed
  Bernoulli $R_y$ model. It supports arbitrary index superpositions, true
  controlled branches, forward/inverse calls, reward reflections, and an explicit
  query ledger without a public mean/amplitude accessor.
- A natural-purification statevector oracle with explicit work garbage and a
  deterministic full-unitary Householder completion. The completion is a simulator
  convention, not a unit-cost reversible domain compiler.
- A direct, charged amplitude-threshold reflection for the canonical oracle. It
  constructs QPE from embedded controlled reward-oracle calls, compares phase bins,
  applies phase kickback, and uncomputes on the complete phase/index/reward space.
  It receives a numeric threshold and never receives a hidden marked-index set.
- Full-workspace BBHT threshold search with a rank-one diffusion about the complete
  initial state, joint accept/index decoding, output exclusion, and independently
  rerun fixed-confidence QPE verification. Failed search budgets remain explicitly
  uncertified; they are not interpreted as proof that no qualifying arm exists.
- A calibrated direct Top-$k$ controller that passes only one numeric boundary and
  simultaneous confidence intervals into fair above/below searches. Completion
  requires a final strict interval certificate for every selected and rejected arm;
  the calibration still samples every arm and therefore is not an advantage result.
- Executable `QBoundary`, reversible compute--phase--uncompute `QGapFlag`, known-count
  Grover and BBHT `QBatchExtract`, and a resumable selected/complement dovetail
  controller. Every reward-oracle experiment and measured flag verification is
  charged. This earlier controller still compiles its flag from the resulting
  certificate and is retained as a clearly labelled baseline.
- An analytic iterative-amplitude-estimation simulator and multiscale Q-GapSelect
  research selector with mean/angular confidence diagnostics and separate
  simulated versus conjectural cost fields.
- Classical Hoeffding and Bernoulli-information comparators, independent exact-state
  and analytic quantum-estimation baselines, and orientation-optimized candidate
  layer-complexity accounting.
- Deterministic analytic, reference, and exact-state coherent experiment runners.
  Raw records keep heuristic output agreement, certificate-gated success, executed
  query/gate/depth/qubit resources, and conjectural theory fields separate.
- A unified executed quantum-core audit runner covering random complex-state
  unitarity, exact and off-grid QPE resolution, measured verifier calibration,
  four seeded angular instance families, full-workspace BBHT scheduling, both
  threshold orientations, multi-output exclusion, direct/independent/classical
  fixed-parameter and same-logical-budget controls, an analytic iterative-AE
  comparator, Top-k boundary-only negative control, invalid index-only diffusion,
  and explicit failure semantics. Dense NumPy QFT and comparator peak allocations
  are reported separately from logical query/gate/depth counts.
- A composition-falsification audit showing that the original `n=3m`
  orientation witness is matched by coarse grouping plus strong-oracle BAI.
- A new unknown-boundary history audit harness that tracks candidate cost,
  legal encoded baselines, invalid free-history baselines, lower-target proxies,
  and novelty gates under a no-free-QRAM interface.
- A charged finite-phase activity-history prototype. Unlike the first toy
  transducer, it derives active/output flags from finite-QPE phase windows and
  records exact compute--phase--uncompute traces plus charged classifier query
  units. It is code-sanity for the no-free-QRAM construction, not an upper-bound
  theorem.
- A variable-time charged history alignment audit that moves the charged
  predicate costs back onto the main unknown-boundary candidate. It compares the
  charged candidate proxy with serial rebuild, variable-time rebuild,
  Grover-activity, independent scan, and coarse+BAI proxies under the same
  finite-QPE level costs.
- A variable-time stopping transducer skeleton. It adds an explicit stop-level
  register, XOR compute/uncompute semantics, output phase kickback, and serial
  versus branch-RMS stopping-cost ledgers for the charged finite-QPE relation.
- A stopping-unitary theorem scaffold that separates execution-checked
  identities from remaining proof obligations. It records passed checks for
  involution, phase equivalence, cleanup, and RMS accounting, while keeping
  circuit synthesis, confidence composition, coherent boundary localization,
  and composition dominance open.
- A composition-frontier audit for the active no-free-QRAM candidate. It encodes
  loop variable-time rebuild, generated-predicate extraction, coarse+QBAI,
  independent QPE, serial rebuild, and forbidden free-history QRAM comparators
  under the same proxy interface.
- A lower-bound proof program for the active candidate. It records the boundary
  localization local fact plus open active-history, direct multi-output, and
  composition-exclusion proof blocks. This is a scaffold for L-07, not an
  adversary or polynomial-method proof.
- A proof ledger that ties the upper-bound scaffold, composition-frontier audit,
  and lower-bound program into one machine-readable theorem-stack checkpoint.
  It marks local facts, execution checks, proof outlines, manual instantiation
  requirements, and open proof obligations separately.
- An authorization-scoped offline attack replay layer with source/victim
  separation, paired clean/attack seeds, ASR@Q, FV-ASR@Q,
  paired-counterfactual-ASR@Q, and a certificate-checked Q-GapSelect result
  adapter. Its bundled fixture is pipeline testing only, not attack evidence.
- A machine-readable Q-GapAttack experiment matrix covering 38 quantum,
  selector, attack, control, and diagnostic baselines across 11 preregistered
  panels. The design audit is valid but explicitly records
  `empirical_ready=false` and `ccf_a_claimable=false`.
- A blind frozen-source oracle and six budget-accounted classical selector
  baselines: random, uniform, successive halving, cost-aware racing, internal
  CLUCB-style Top-k, and GCGS-style graph search. The committed synthetic
  campaign contains 15,360 runs and no LLM execution.
- An exact-count frozen empirical Layer-C benchmark for the all-active
  Q-GapSelect reference, gap-aided independent IAE Top-k, and a known-threshold
  stronger-information control. Its powered panel contains 5,000 non-isomorphic
  instances and 15,000 method runs, with paired inference and strict
  certificate semantics. It records logical coherent query ledgers, not
  hardware resources.
- A tiny end-to-end coherent unknown-boundary Top-k reference unitary. It
  receives only the charged canonical oracle, `k`, and public precision/resource
  limits; runs coherent QPE, an exhaustive reversible rank relation, complete
  membership-mask copy, and inverse-QPE cleanup; and records the exact
  rank-copy identity `P_garbage = 1 - sum_y p_y^2`. Exact-grid cases are circuit
  sanity checks, while generic off-grid entanglement fails closed.
- A deterministic hidden-frontier fixture generator with seven registered
  positive and negative-control families. It separates method-visible blind or
  public-partition interfaces from trusted truth, hides the numeric gap and
  activity history, supports exact replay, and deduplicates permutation orbits.
- A primary-source identity and version-locator inventory of the strongest estimation, QBAI,
  approximate-k-minimum, variable-time, loop-composition, tunable-VTAA, and
  all-marked baselines. Declarative rows and self-reported test JSON never
  count as trusted runtime evidence; uncovered reductions keep the advantage
  gate false.
- Tests, continuous integration, a research protocol, claim matrix, and manuscript
  scaffold.

The orientation-optimized angular layer complexity reported by the current experiments is a
**research hypothesis**, not a proved runtime theorem. The manuscript states the
proof obligations and falsification criteria explicitly.

The direct threshold path fixes the earlier semantic flaw in which the coherent
flag merely encoded an already known answer. It is nevertheless a careful
composition of amplitude estimation and unknown-success amplitude amplification,
not yet the proposed new heterogeneous-gap quantum algorithm. Only a strictly
stronger unknown-boundary procedure with a matching lower bound could activate
that claim. Likewise, the repository does not compile any application-domain
generator into a reversible circuit; it supplies the natural oracle contract and
finite exact-state reference implementations.

## Current measured status

### Strict matched-evidence checkpoint (2026-07-15)

The repository now separates three experiment tiers that must not be pooled:

1. `artifacts/coherent_statevector_history.json` executes a genuine
   coherent-index exact-state activity-history kernel on 24 small trials. It
   records controlled forward/inverse queries and explicit work/history/stop
   registers, but obtains zero complete direct multi-output Top-k certificates.
2. `artifacts/ccfa_matched_benchmark_diagnostic.json.gz` is the scalable
   fixed-fixture x multi-seed, fixed-logical-cap panel. Its candidate is still
   the analytic finite-state IR reference, so the artifact's implementation
   gate remains blocked even if a finite cell were favorable.
3. `artifacts/uci_classifier_benchmark_diagnostic.json.gz` is public-data
   external-validity evidence. The committed offline Digits run has 24 arms,
   `k=8`, five label-blind shards, 20 seeds, four caps, and five matched
   methods. Three shards tie exactly at the Top-k boundary and are retained as
   fail-closed exclusions. On the two executable shards at cap 524,288, the
   candidate reaches 0.825 certified-exact recovery, below both k-only
   adaptive references at 1.000 and coarse+BAI at 0.975.

The unified audit in `artifacts/theorem_closure_audit.json` also records that a
legal public-partition composition matches or beats the current candidate
proxy. Hiding the partition invalidates that comparator, but the corresponding
candidate discovery/cleanup upper bound and weighted matching lower bound are
unproved. Therefore the machine-readable CCF-A quantum-advantage gate is false.

`docs/coherent_unknown_boundary_experiment_protocol.md` freezes the next
algorithm-only campaign: a charged unknown-boundary constructor, complete
durable Top-k output, replay cleanup, strongest same-interface composition
baselines, fixture-cluster statistics, and explicit empirical/theorem
falsification gates. Supplied schedules remain regression controls only.

### S2 unknown-boundary checkpoint (2026-07-16)

The executable S2 checkpoint closes circuit semantics, fixture isolation, and
fail-closed baseline inventory—not the new upper/lower-bound theorem:

- Four exact-state cases reconcile all canonical-oracle queries. Both exact-grid
  strict controls write the complete Top-k mask and clean; the generic off-grid
  control leaves `0.034769713197` transient probability, matches
  `1 - sum_y p_y^2`, and rejects; the exact tie cleans and rejects.
- The backend reports exact queries, executed NumPy kernel macros, undecomposed
  logical macros, and rank-structure proxies separately. It deliberately
  reports no elementary-gate ledger, transpiled depth, or compiled ancilla count.
- The frozen manifest contains 21 fixtures across seven families and three
  seeds. It has 18 raw geometric orbits but 21 algorithmic
  `(orbit_hash, interface_id)` keys, so hidden and public-partition controls are
  never collapsed.
- The version-locator inventory contains ten strong baselines; nine require
  trusted implementations and remain uncovered. Even nine structurally valid
  self-reports cannot activate coverage because a trusted attestation pipeline
  is not implemented.

Accordingly, the S2 artifact records `quantum_advantage_claimable=false` and
`ccf_a_claimable=false`. Its positive result is a stricter executable problem
definition and a concrete off-grid cleanup obstruction for the next algorithm.

CCF publishes a venue directory, not a mandatory dataset list. For that
reason, `docs/ccfa_algorithm_experiment_protocol.md` defines a top-conference
dataset protocol rather than claiming a nonexistent "CCF-A dataset." Strict
local loaders and source-hash manifests are implemented for UCI Letter,
Optdigits, and Covertype. Their official confirmatory campaigns remain
unexecuted until the official files are present; scikit-learn Digits is clearly
labelled as a non-official diagnostic fallback.

The versioned quantum-core diagnostic
`artifacts/quantum_benchmark_diagnostic.json` is a v4 audit with 13 configured
suites and 778 raw records. It adds the `unknown_boundary_history` suite while
retaining the negative composition audit for the rejected orientation witness.

Key v4 results:

- the old orientation-family novelty gate fails: 6/6 composition-audit records
  are `failed_explicit_family`;
- the new unknown-boundary history suite remains open on 6/6 default sweep
  points under the encoded legal baselines;
- the strongest encoded legal baseline for that suite is
  `variable_time_rebuild_rms`;
- at the largest default point, baseline/candidate is about `5.36`;
- the default candidate finite-family slope is about `3.42`, and the recorded
  lower-target proxy slope is about `3.50`;
- the artifact explicitly marks the result as finite exact-state and analytic
  diagnostics only, not an asymptotic theorem.

The additional grid artifact `artifacts/unknown_boundary_grid.json` sweeps
multiple unknown-boundary-history families. It is designed to find candidate
families that remain open, and to expose parameter choices that collapse under
encoded baselines. Open gates are proof obligations, not theorem claims.

The algorithm-only frozen selector campaign reports that no one classical
baseline dominates all gap, graph, and heterogeneous-cost cells. In the
Layer-C reference campaign, every one of the ten cells contains 500
non-isomorphic exact-count fixtures and passes a 100%-unique,
permutation-invariant difficulty-fingerprint audit. All three methods certify
500/500 fixtures in the easy cells. At the `n=32, k=16, gap≈pi/128` stress
cell, Q-GapSelect certifies 447/500, gap-aided independent IAE 401/500, and the
stronger-information threshold control 411/500. The paired Q-GapSelect versus
independent-IAE risk difference is 9.2 percentage points (fixture-pair
bootstrap 95% CI 5.8--12.6; Holm-adjusted `p≈3.32e-6`). Q-GapSelect also uses
about 33,123 more logical calls on average in that cell. At `gap≈pi/256`, all
three certify 0/500 and Q-GapSelect spends about 3.5 times the independent-IAE
calls. The methods are neither information- nor query-matched, so these are
finite algorithm diagnostics, not fixed-confidence or quantum-advantage
claims.

The charged predicate-generation artifact
`artifacts/charged_activity_history.json` is the next P0-U08 implementation
checkpoint. It removes supplied active/output predicate rows from the toy
relation fixture: predicates are generated from normalized phase values,
finite-QPE precision levels, and a boundary window. This strengthens the
constructive evidence chain, but it still does not localize the unknown Top-k
boundary or prove a variable-time coherent-search bound.

The mainline alignment artifact
`artifacts/variable_time_charged_history.json` inserts those finite-QPE charged
costs into the unknown-boundary history proxy. It contains 16 records: two
configured charged families stay open against the encoded proxy baselines, while
the loose-gate negative control fails as intended. The strongest encoded valid
baseline in this audit is `variable_time_rebuild_rms`. This is stronger
evidence that the current direction is coherent, but it is still a proof target,
not a theorem.

The stopping skeleton artifact `artifacts/stopping_time_transducer.json`
contains 7 exact-state records. It verifies stop-register
compute--phase--uncompute traces and records branch-RMS stopping costs no larger
than serial full-history costs on the configured fixtures. This is the closest
current artifact to a circuit-shape construction, but it still uses a supplied
phase boundary and is not a variable-time theorem.

The theorem scaffold artifact `artifacts/stopping_unitary_theorem.json` contains
5 records. All executable checks pass (`25` passed checks total), but it also
records `20` proof-obligation entries. The accompanying
`docs/stopping_unitary_theorem.md` is the current proof checklist for the
stopping-unitary lemma.

The composition-frontier artifact `artifacts/composition_frontier.json` contains
13 records. Two configured families remain open under the encoded
same-interface baselines, while the loose-gate negative control fails. The
strongest encoded valid baseline is `loop_variable_time_rebuild`. This screens
the current candidate against the baselines we encoded; it is not a theorem
against all published compositions.

The lower-bound program artifact `artifacts/lower_bound_program.json` contains
12 records. It records `12` local boundary facts and `36` open proof-obligation
blocks. The accompanying `docs/lower_bound_program.md` states the current L-07
proof scaffold. It does not prove an all-algorithms lower bound.

The proof-ledger artifact `artifacts/proof_ledger.json` contains 10 theorem-stack
entries: 1 execution-checked scaffold item, 2 proof outlines, 2 manual
composition-instantiation requirements, 1 local lower-bound fact, and 4 open
proof obligations. Its Markdown rendering is `docs/proof_ledger.md`. It
explicitly records `ccf_a_claimable=false`.

The planning artifact `artifacts/research_gap_audit.json` and its Markdown
rendering `docs/research_gap_audit.md` summarize the current paper-readiness
state. The current stage is `active_unknown_boundary_history_candidate_no_theorem`;
the readiness label is `pre_theorem_not_ccf_a_ready`.

These results validate implementation invariants, record negative novelty
evidence for the old candidate, and start the new no-free-QRAM candidate audit.
They do **not** demonstrate a quantum query advantage, a new quantum algorithm,
hardware feasibility, or an application-domain result.

## Repository layout

```text
src/qgapselect/       oracles, coherent primitives, selectors, audit proxies
scripts/              analytic, exact-state, and unknown-boundary audit entry points
tests/                unit, invariant, and special-case regression tests
configs/              checked-in experiment configurations
paper/                LaTeX manuscript and bibliography
docs/                 research protocol and claim-level evidence matrix
.github/workflows/    automated linting and tests
```

## Reproduce

```bash
python -m venv --copies .venv
. .venv/bin/activate
python -m pip install -e '.[dev,plots,datasets]'
pytest
make quantum-core
```

The same setup is encoded in the Makefile:

```bash
make install
make test-quantum
make unknown-boundary-grid
make charged-history
make variable-time-charged
make stopping-transducer
make theorem-scaffold
make composition-frontier
make lower-bound
make proof-ledger
make research-gap
make attack-design
make frozen-selector-benchmark
make frozen-quantum-reference
make coherent-statevector-history
make replay-coherent-frontier
make coherent-rank-baseline
make coherent-unknown-boundary-topk
make hidden-frontier-fixtures
make strong-composition-registry
make theorem-closure-audit
make ccfa-matched-benchmark
make install-datasets
make uci-classifier-benchmark
make quantum-history
make quantum-core
```

`make install-uv` uses the checked-in lockfile when `uv` works in the host
environment. If a sandbox has broken interpreter symlinks, prefer the `venv
--copies` command above.

Longer optional diagnostics:

```bash
python scripts/run_scaling.py --output artifacts/scaling.json
python scripts/run_reference.py --output artifacts/reference_results.json
python scripts/run_coherent.py --output artifacts/coherent_results.json
python scripts/run_direct_search.py --output artifacts/direct_search_results.json
python scripts/run_quantum_benchmarks.py \
  --config configs/quantum_benchmarks.json \
  --output artifacts/quantum_benchmark_diagnostic.json
python scripts/run_unknown_boundary_grid.py \
  --config configs/unknown_boundary_grid.json \
  --output artifacts/unknown_boundary_grid.json
python scripts/run_charged_activity_history.py \
  --config configs/charged_activity_history.json \
  --output artifacts/charged_activity_history.json
python scripts/run_variable_time_charged_history.py \
  --config configs/variable_time_charged_history.json \
  --output artifacts/variable_time_charged_history.json
python scripts/run_stopping_time_transducer.py \
  --config configs/stopping_time_transducer.json \
  --output artifacts/stopping_time_transducer.json
python scripts/run_stopping_unitary_theorem.py \
  --config configs/stopping_unitary_theorem.json \
  --output artifacts/stopping_unitary_theorem.json \
  --markdown docs/stopping_unitary_theorem.md
python scripts/run_coherent_unknown_boundary_topk.py \
  --config configs/coherent_unknown_boundary_topk.json \
  --output artifacts/coherent_unknown_boundary_topk.json
python scripts/run_hidden_frontier_fixture_manifest.py \
  --config configs/hidden_frontier_fixture_manifest.json \
  --output artifacts/hidden_frontier_fixture_manifest.json
python scripts/run_strong_composition_registry.py \
  --config configs/strong_composition_registry.json \
  --output artifacts/strong_composition_registry.json
python scripts/run_composition_frontier.py \
  --config configs/composition_frontier.json \
  --output artifacts/composition_frontier.json
python scripts/run_lower_bound_program.py \
  --config configs/lower_bound_program.json \
  --output artifacts/lower_bound_program.json \
  --markdown docs/lower_bound_program.md
python scripts/run_proof_ledger.py \
  --stopping-artifact artifacts/stopping_unitary_theorem.json \
  --composition-artifact artifacts/composition_frontier.json \
  --lower-bound-artifact artifacts/lower_bound_program.json \
  --output artifacts/proof_ledger.json \
  --markdown docs/proof_ledger.md
python scripts/run_qgapattack_experiment_design.py \
  --config configs/qgapattack_experiments.json \
  --output artifacts/qgapattack_experiment_design.json \
  --markdown docs/qgapattack_experiment_design_audit.md \
  --strict-design
python scripts/run_frozen_selector_benchmarks.py \
  --config configs/frozen_selector_benchmarks.json \
  --output artifacts/frozen_selector_benchmark_diagnostic.json
python scripts/run_frozen_quantum_reference_benchmarks.py \
  --config configs/frozen_quantum_reference_benchmarks.json \
  --output artifacts/frozen_quantum_reference_diagnostic.json
python scripts/run_ccfa_matched_benchmarks.py \
  --config configs/ccfa_matched_benchmarks.json \
  --mixture-artifact artifacts/frozen_quantum_reference_diagnostic.json \
  --theory-artifact artifacts/theorem_closure_audit.json \
  --output artifacts/ccfa_matched_benchmark_diagnostic.json.gz
python scripts/run_uci_classifier_benchmarks.py \
  --config configs/uci_classifier_benchmarks.json \
  --data-root data/uci \
  --theory-artifact artifacts/theorem_closure_audit.json \
  --output artifacts/uci_classifier_benchmark_diagnostic.json.gz
```

Use each script's `--help` option for experiment controls. The quantum-core
runner supports `--suite`, `--trials`, and `--seed`; its default configuration
records exact-state and analytic-measurement evidence in separate resource
classes. The reference
configuration schedules 500 repetitions per scenario; an explicit `--trials`
override is recorded in the resolved configuration and hash. Generated data are
written beneath `artifacts/`. The repository versions the analytic scaling table,
the 4-trial-per-scenario reference run, the direct-search run, the full
quantum-core run, and the unknown-boundary parameter grid. Other outputs stay
ignored until a recorded run is intentionally promoted. The charged
activity-history, variable-time alignment, stopping-transducer, and
stopping-unitary theorem-scaffold artifacts are also versioned because they are
P0-U08 construction checkpoints. The composition-frontier and lower-bound
program artifacts are versioned because they define the current prior-work
frontier and L-07 proof obligations. The proof-ledger artifact is versioned
because it is now the canonical theorem-stack checklist. The coherent
unknown-boundary circuit, hidden-frontier manifest, and strong-composition
registry artifacts are versioned as the S2 experiment-interface checkpoint;
they are not an upper-bound or lower-bound theorem.

## Paper-grade completion criteria

The central result is ready to be described as a new quantum algorithm only after
all of the following hold:

- a fully specified polynomial-time circuit/query algorithm, including noisy
  thresholding, batch extraction, failure-budget allocation, and stopping rules;
- an upper bound that recovers known best-arm behavior at $k=1$ and the
  $\Theta(\sqrt{k(n-k)}/\gamma)$ equal-angular-gap limit (with
  $\gamma=\Theta(\Delta)$ only for means bounded away from 0 and 1);
- a matching adversary or polynomial-method lower bound, including the
  $\Omega(n/\gamma)$ angular-gap dense-output target (hence $\Omega(n)$ at
  constant $\gamma$) and additive partition direct sums;
- ablations against independent amplitude estimation and unstructured search;
- a reversible unknown-boundary activity-history transducer without free QRAM;
- a manual composition-frontier proof showing that known loop/k-minimum/QBAI
  compositions do not match the same interface, or a candidate revision if they do;
- release of seeds, raw outputs, configs, and negative results.

No repository structure, benchmark table, or simulated curve by itself establishes
a CCF-A-level contribution or guarantees acceptance.

## Responsible-use boundary

The repository currently runs quantum-core audits and a non-executing attack
experiment-design audit. Any real application-domain experiment must be
separately authorized, isolated, and reviewed before execution. See
[SECURITY.md](SECURITY.md).
