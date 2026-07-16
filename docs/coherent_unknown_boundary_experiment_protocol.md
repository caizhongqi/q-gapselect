# Replay-preserving coherent unknown-boundary experiment protocol

Status: **prospective protocol; no quantum-advantage or CCF-A claim**

Version: `v1`, 2026-07-16

## 1. Purpose and claim boundary

The mainline object is exact Top-k selection under coherent canonical
Bernoulli-oracle access when the Top-k boundary, arm ordering, active history,
and stopping schedule are unknown.  The candidate must discover its frontier
coherently, copy a complete durable Top-k output, and replay its computation to
erase every transient register.

This protocol is deliberately application-neutral.  It contains no LLM,
security, commercial API, QNN, or hardware experiment.  Its purpose is to test
and attempt to falsify a quantum-algorithm candidate before any downstream
application is attached.

The current repository establishes only the following starting facts:

- `artifacts/replay_coherent_frontier.json` verifies compute--copy--uncompute
  semantics for a **supplied public schedule** on 11 tiny exact-state records;
  its maximum cleanup residual is zero and its largest state uses 14 qubits,
  but it does not discover a boundary or return a complete multi-output set;
- `artifacts/ccfa_history_certificate_ablation.json.gz` shows that replayable
  history certification repairs the previous certification collapse, but the
  largest-cap candidate result (`0.9887` certified exact, `297,543` mean
  queries) remains below the strongest k-only reference (`0.9973`, `270,387`);
- `artifacts/composition_frontier.json` is an encoded proxy screen, not a
  theorem against known composition methods;
- `artifacts/lower_bound_program.json` contains 36 proof obligations and no
  adversary or polynomial-method lower bound; and
- the public-partition family in `docs/unified_theorem_closure_audit.md` is
  already matched or beaten by a legal partition-plus-selection composition.

Accordingly, experimental success may support circuit correctness and
finite-family efficiency.  It cannot by itself support a new quantum-advantage
theorem, guarantee acceptance at any venue, or convert an unproved proxy into
a complexity result.

## 2. Research questions

The confirmatory study answers five questions in this order.

| ID | Research question | Evidence that can answer it |
|---|---|---|
| RQ1 | Can an executable unitary derive an unknown Top-k boundary and nested active frontier from charged oracle calls, without a supplied schedule or active-list QRAM? | Exact-state circuit semantics, blinded oracle execution, and resource traces. |
| RQ2 | Can it retain a complete durable Top-k output while replay-uncomputing boundary, QPE, history, stop, counter, verifier, and extraction work? | Basis-state and superposition tests with exact cleanup identities. |
| RQ3 | At matched information and hard query caps, does it improve the certified-exact/query frontier over every strongest legal baseline? | Fixed-fixture, repeated-seed, paired experiments. |
| RQ4 | Does the measured stopping profile agree with the cost expression used in the proposed upper bound, including predicate generation and inverse calls? | Instrumented logical-query and circuit-resource ledgers. |
| RQ5 | Is the proposed bound genuinely outside the same-interface composition frontier and matched by a lower bound for all adaptive quantum algorithms? | Formal theorem instantiation and proof; experiments and log--log fits are insufficient. |

RQ1 and RQ2 are implementation gates.  RQ3 and RQ4 are empirical gates.  RQ5
is a theory gate.  All three kinds of gate must remain separate in every
artifact and paper table.

## 3. Frozen oracle and output interface

### 3.1 Canonical oracle

For arm angles `theta_i in [0, pi/2]`, define `mu_i = sin(theta_i)^2` and the
canonical block rotation

```text
B_theta |i>|0> = |i>(cos(theta_i)|0> + sin(theta_i)|1>),
B_theta |i>|1> = |i>(-sin(theta_i)|0> + cos(theta_i)|1>).
```

Computational-basis indices padding `n` to a power of two are fixed by the
identity and may never be returned.  `B_theta`, `B_theta^dagger`, controlled
`B_theta`, and controlled `B_theta^dagger` are the only hidden-instance
access.  A call to any one base unitary costs one logical coherent query.
Applying a power through `q` base calls costs `q`, not one.  QPE, amplitude
amplification, verification, and replay must expose their complete forward,
inverse, and controlled call counts.

The primary query model does not charge known Clifford gates as oracle calls,
but it reports them separately.  Classical Bernoulli samples use a separate
sample-access model and are never converted to coherent queries by wall-clock
time.

### 3.2 Strict Top-k relation

Let `theta_(1) >= ... >= theta_(n)` denote sorted angles.  Every positive
fixture satisfies

```text
Delta = theta_(k) - theta_(k+1) > 0.
```

The required output is the unique set of the `k` largest valid arm indices,
encoded canonically as either a sorted k-tuple or an n-bit indicator string.
The final executable map must contain one complete durable output register
`Y`, not one branch-membership bit:

```text
U |0> -> sqrt(p_ok) |Y_top-k>|0_transient>|ok>
       + sqrt(1-p_ok) |0_Y>|0_transient>|fail>.
```

The failure branch may contain an explicit timeout or reject flag.  A result
is accepted only if it contains exactly `k` distinct valid indices, passes the
algorithm's oracle-derived certificate, stays under the hard cap, and leaves
all declared transient registers clean.  The trusted harness may inspect the
hidden angles only after return to score exactness; it may not repair or
certify an algorithm output.

### 3.3 Information supplied to every primary method

Every primary method receives exactly:

```text
(fresh blind oracle handle, n, k, delta, hard query cap,
 public structural-promise parameters if the theorem requires them).
```

Every such public promise parameter is serialized into the interface ID and
given to every baseline.  The method is not given the numeric boundary, true
means or angles, fixture family label, fixture seed, hidden permutation,
minimum realized gap, arm ranking, selected/rejected membership, active counts,
active indices, partition table, stopping times, or a QRAM/QROM table derived
from the instance.

If the candidate requires a public count profile such as `(A_r, M_r)`, that
profile is also given to all baselines and its storage and use are charged.
If it requires the instance-specific identities realizing that profile, the
primary interface has been violated.

### 3.4 Candidate execution contract

The candidate implementation must expose the following stages rather than
only their analytic proxies:

1. `U_boundary`: construct a bounded-error coherent representation of the
   unknown Top-k boundary from canonical-oracle calls;
2. `U_frontier(r)`: at precision epoch `r`, generate an unresolved/stopping
   predicate without reading a supplied active list;
3. `U_stop`: write the first stopping epoch and decision flags while preserving
   branches that have already stopped;
4. `U_extract`: produce a canonical complete Top-k output and reject duplicate,
   incomplete, invalid-index, or non-separating results;
5. `COPY_Y`: copy the durable output into a clean output register;
6. replay `U_extract^dagger U_stop^dagger U_frontier^dagger U_boundary^dagger`
   to clean all transient work; and
7. run a charged certificate/verifier whose total failure probability is at
   most the declared `delta`.

Measurements used by the final output protocol must be explicit.  Measuring
an active list between epochs and then treating it as free classical control is
not a coherent frontier implementation.  Repeating a one-output routine `k`
times is a legal baseline, but it is not evidence for direct multi-output
extraction.

## 4. Instance families

### 4.1 Fixture definition

A fixture is one immutable angle vector together with `n`, `k`, its hidden
permutation, public promise parameters, and generator provenance.  Algorithm
seeds repeat measurement randomness **within** a fixture; they do not create
new independent instances.  Isomorphic permutations are deduplicated by a
canonical multiset-and-orbit hash before anchor selection.

For all primary families, a hidden center `beta` is selected from a frozen
grid inside `[pi/6, pi/3]`.  Signed offsets from `beta` determine the ordering,
and a frozen permutation hides the identities.  All families at the same
`(n,k)` use the same declared minimum design gap and the same cap grid, so a
cap does not reveal the family.

### 4.2 Mandatory families

| ID | Construction and purpose | Expected role |
|---|---|---|
| `F-EQ` | `k` positive offsets and `n-k` negative offsets separated by one common boundary scale. | Homogeneous-stopping calibration; it must not be used as the novelty witness. |
| `F-DYADIC` | Offsets occupy geometrically spaced shells around hidden `beta`; shell identities are randomly permuted.  Far arms should stop early and a small boundary shell should stop late. | Heterogeneous stopping and scaling. |
| `F-CLUSTER` | Many arms lie within a constant multiple of the minimum boundary gap, with symmetric deterministic jitter and no ties. | Hard certification, timeout, and bounded-error stress test. |
| `F-HIDDEN-FRONTIER` | A frozen nested count profile `(A_r, M_r)` is realized only through angle shells and a hidden permutation.  Neither shell membership nor the numeric center is supplied. | Primary candidate theorem witness.  Its structural promise must be written mathematically before execution. |
| `F-PUBLIC-PARTITION` | The same angle multisets as `F-HIDDEN-FRONTIER`, but the static shell/block partition is public to every method. | Mandatory falsification control: partition plus QBAI/all-marked composition is expected to match or win.  It is a different interface and is never pooled with the hidden-family result. |
| `F-UNKNOWN-TIME-NC` | Heavy-tailed stopping profiles covering the generic unknown-time regime, with no special hidden-frontier promise. | Negative control against silently claiming a universal `sqrt(sum_i T_i^2)` unknown-time bound. |
| `F-TIE-NC` | `theta_(k) = theta_(k+1)` with an explicit non-unique-output label. | Fail-closed negative control; the exact strict Top-k certificate must reject. |

For `F-HIDDEN-FRONTIER`, let decreasing radii
`w_0 > w_1 > ... > w_R = gamma/2` define the nested relation

```text
H_r(theta, beta) = {i : |theta_i - beta| <= w_r}.
```

The fixture generator freezes the counts `|H_r| = A_r` and the number of new
selected outputs at each epoch, but randomizes their identities.  Arms in
shell `H_r \ H_(r+1)` receive deterministic tie-free offsets whose signs fix
their true selected/rejected status.  The candidate receives only any public
promise parameters explicitly used by its theorem, never `beta`, `H_r`, or
the permutation.  If the coherent algorithm cannot construct an equivalent
frontier from oracle access, the family falsifies rather than supports the
candidate upper bound.

### 4.3 Scale grid and anchors

The confirmatory hard-family grid is frozen before outcomes:

- `n in {32, 64, 128, 256}` for scalable blinded execution;
- `k in {ceil(sqrt(n)), floor(n/2)}`;
- four shared design-gap scales per `(n,k)`, selected geometrically;
- at least 30 non-isomorphic fixtures per family/scale cell; and
- 100 common random-number seeds per fixed fixture, per method, and per cap.

The exact-state tier uses `n in {4, 6, 8}`, every valid `k`, at least three
hidden centers, all non-isomorphic small shell profiles, and exhaustive
permutations when tractable.  `n > 8` may be added only when the full output
register and workspace remain under the preregistered statevector limit.

The old 30-fixture/50-seed campaign remains a diagnostic regression panel.  It
is not silently pooled with this new confirmatory campaign.

## 5. Baselines and interface legality

### 5.1 Mandatory primary baselines

Every baseline below must output the same complete certified Top-k relation,
receive the same information, use the same `delta`, and obey the same atomic
query cap.

| Baseline | Required implementation/fidelity check |
|---|---|
| Independent adaptive QPE/IAE plus sort | Blind per-arm estimation with a summable confidence schedule; all verification calls charged. |
| MIQAE per-arm estimate-then-sort | Implement modified iterative QAE with its proved confidence/query schedule, allocate failure across all arms, and charge every Grover iterate before sorting and certification. |
| Unknown-time variable-time search/computation-tree reference | Published theorem assumptions instantiated against the exact interface; unknown stopping times and logarithmic overhead retained. |
| Loop/transducer composition | Activity-predicate generation, every loop invocation, and inverse cleanup charged; no materialized hidden frontier. |
| Generated-predicate all-marked extraction | The marking predicate is built from oracle calls on every use; full k-output extraction and duplicate control included. |
| Coarse partition plus quantum BAI | Primary-valid only when its partition is generated legally from the blind oracle.  A supplied partition belongs to the stronger-information control panel. |
| Repeated single-output selection | Remove/deflate already returned indices coherently and charge all `k` repetitions and verification. |
| Current strongest k-only adaptive reference | Retained for continuity with the 37,500-attempt campaign and revalidated on every new family. |

Baseline names such as “QBAI” or “variable time” are not fidelity evidence.
Before preregistration, each strongest baseline must have a theorem-to-code
mapping that records the source theorem, oracle conversion, confidence terms,
output conversion, hidden constants used in code, and tests against published
special cases.  A paper-informed stand-in is labelled `proxy` and cannot close
the strongest-baseline gate.

The minimum prior-method registry contains the following primary-source rows.
Each row is a mandatory composition threat even when it is not ultimately a
legal executable baseline.  Exclusion requires a written assumption mismatch,
not the absence of repository code.

| Registry row | Required audit on this interface |
|---|---|
| Gao--Ji--Wang, [Quantum Approximate k-Minimum Finding](https://arxiv.org/abs/2412.16586) (ESA 2025) | Instantiate its approximate-value oracle and complete k-output guarantee.  Either charge the reduction from `B_theta` and show when approximation resolves the strict boundary, or prove why its approximation/output promise does not give exact Top-k here. |
| Vihrovs, [Quantum Search on Computation Trees](https://arxiv.org/abs/2505.22405) (2025) | Instantiate the optimal unknown-time VTS corollary, including the `sqrt(T log min(n,t_max))` scale.  A generic `sqrt(T)` candidate bound is not admissible unless the hidden-frontier promise is proved to exclude the relevant unknown-time family. |
| Low--Su, [Quantum linear system algorithm with optimal queries to initial state preparation](https://arxiv.org/abs/2410.18178) (Quantum 2026) | Audit the paper's Tunable VTAA primitive against the nested frontier: thresholds, state-preparation calls, success amplitudes, predicate construction, and inverse cleanup must all be instantiated before claiming that the candidate is a new nested-amplification primitive. |
| Fukuzawa--Ho--Irani--Zion, [Modified Iterative Quantum Amplitude Estimation is Asymptotically Optimal](https://arxiv.org/abs/2208.14612) | Use MIQAE as the mandatory strong per-arm estimate-then-sort baseline; retain its confidence and query constants rather than substituting the existing analytic IAE by name. |
| Rall, [Faster Coherent Quantum Algorithms for Phase, Energy, and Amplitude Estimation](https://arxiv.org/abs/2103.09717) (Quantum 2021) | Coherent estimation must audit the rounding promise.  The hidden-angle generator includes values near every finite precision-cell boundary; the candidate must either prove the promise, charge a valid shifting/robustification construction, or mark Rall-style coherent rounding as inapplicable.  It may not silently assume a unique rounded estimate. |

The Rall stress rows are generated before execution by placing non-boundary
arms on both sides of every used QPE grid edge while preserving the strict
Top-k gap.  They are implementation tests for coherent rounding and cleanup,
not a modification of the hidden scoring relation.

### 5.2 Stronger-information controls

The following methods are useful ablations but are never primary competitors:

- supplied numeric boundary;
- supplied gap floor or true arm order;
- known per-arm stopping time;
- supplied active-history rows or static partition;
- free QRAM/QROM lookup of an instance-derived table; and
- exhaustive access to hidden angles.

Their records carry a different `information_regime` and `interface_id` and
are shown in a separate table.

### 5.3 Automatically invalid comparisons

A run is invalid for primary evidence if it does any of the following:

- reconstructs a schedule in the trusted harness and injects it into the
  candidate;
- charges `B_theta^q`, controlled powers, or compute--uncompute pairs as one
  query;
- omits inverse, verification, or failed-attempt calls;
- reads true means, rankings, family IDs, active counts not declared public, or
  the scoring oracle;
- returns one branch-membership bit instead of the same complete k-set;
- treats a measured active list as free coherent memory;
- conditions query means on successful runs;
- drops timeout, unresolved, wrong-certificate, cleanup-failure, or cap-failure
  attempts from the denominator;
- chooses fixtures, caps, seeds, methods, or a favorable asymptotic range after
  seeing outcomes; or
- compares simulator wall time with quantum query complexity.

An invalid run remains available as a labelled diagnostic; it cannot be
reclassified post hoc.

## 6. Budget and information matching

### 6.1 Hard cap

The oracle wrapper owns the hard cap.  Before any atomic logical call that
would exceed it, execution stops and records a timeout.  Partial sets and
uncertified sets are failures.  Candidate and baseline receive fresh oracle
objects so no cached state, ledger, or hidden data crosses method boundaries.

All methods use the same failure budget `delta = 0.05`.  A method may allocate
that budget across levels and outputs, but the allocation is frozen in its
configuration and its union/sequence bound must be auditable.

### 6.2 Cap grid

Within a confirmatory `(n,k,design-gap)` block, define an outcome-independent
reference scale from the family design, not from any method's observed cost:

```text
Q_ref = 2^ceil(log2(sqrt(n*k) / gamma_design)).
Q_caps = Q_ref * {1/8, 1/4, 1/2, 1, 2, 4, 8},
```

rounded upward to integers.  `gamma_design` is used only by the harness to
construct a shared cap panel; it is not passed as an algorithm input.  Every
family at that `(n,k,gamma_design)` uses the same caps.  If pilot executions
show that all methods are identically zero or one across the entire grid, the
grid may be changed only before preregistration, and all pilot fixtures are
permanently excluded from confirmatory analysis.

In addition to fixed-cap results, the report includes the complete
success-versus-cap Pareto frontier.  A cap is never selected after observing
which method wins.

### 6.3 Resource equivalence

The primary query table reports, for every attempt:

```text
forward, inverse, controlled-forward, controlled-inverse,
boundary, predicate/QPE, amplification, extraction,
verification, cleanup, total coherent queries.
```

The sum of the four base-call directions is the authoritative total.  Tagged
subtotals must reconcile exactly to it.  Gate count, T/Toffoli count when a
decomposition is available, circuit depth, maximum live qubits, output qubits,
ancilla qubits, classical bits, state-preparation cost, schedule-storage bits,
and maximum statevector dimension are reported separately.

For the hidden interface, `instance_derived_schedule_storage_bits` must be
zero before coherent execution.  Public promise constants may be compiled,
but their bit count and SELECT cost are included.  A claimed asymptotic query
improvement that requires exponentially larger unreported gates, depth, or
workspace fails the resource gate.

## 7. Outcomes and metrics

### 7.1 Primary outcome

For every attempted run,

```text
certified_exact =
    complete canonical k-output
    AND exact hidden Top-k agreement
    AND algorithm-produced certificate accepted
    AND no timeout or cap violation
    AND replay/cleanup checks passed.
```

All preregistered attempts remain in the denominator.  This is evaluated at
each fixed cap, not only at the largest cap.

### 7.2 Secondary correctness outcomes

- exact k-set without certificate;
- false-certificate rate;
- incomplete, duplicate, invalid-index, and unresolved-output rates;
- timeout and cap-violation rates;
- per-family and per-shell stopping error;
- boundary-localization error, reported only by the trusted scorer;
- certificate coverage and calibration against the declared `delta`; and
- number of outputs obtained per extraction invocation.

### 7.3 Coherence and cleanup outcomes

Exact-state records report:

- compute involution residual `||U_compute^2|psi> - |psi>||_2`;
- forward/inverse residual `||U^dagger U|psi> - |psi>||_2`;
- durable-output residual against the specified ideal relation;
- probability of nonzero boundary, history, QPE, stop, counter, verifier, or
  extraction work after replay;
- norm error and transcript-replay equality;
- behavior on every basis index, random complex superpositions, invalid padded
  indices, dirty-output inputs, and deliberately tampered transcripts; and
- whether coherent boundary discovery and complete direct multi-output
  extraction actually executed.

Default numerical tolerance is `1e-12` for double-precision exact-state tests.
Any relaxed tolerance must be frozen and justified before execution.

### 7.4 Efficiency outcomes

- unconditional mean, median, and restricted mean coherent queries, counting
  a timeout at its full cap;
- empirical distribution of stopping costs `T_i` and active counts by epoch;
- candidate/baseline query ratio at fixed certified-exact targets;
- the seven-point success/query Pareto frontier;
- qubits, gates, depth, and schedule loading/storage; and
- descriptive log--log slopes with simultaneous uncertainty intervals.

Simulator wall time and memory are reproducibility diagnostics only.  A fitted
slope is labelled finite-range descriptive evidence and is never written as an
asymptotic complexity theorem.

## 8. Statistical plan

The fixture, not the algorithm seed, is the independent experimental unit.
Common deterministic block seeds are used across methods and caps to enable
paired comparisons, while every run uses a fresh oracle object.

1. Report attempts, successes, and exact Clopper--Pearson intervals for every
   family/size/cap/method cell.
2. Compute within-fixture paired risk differences for candidate versus each
   primary baseline and exact McNemar tests on paired binary outcomes.
3. Form confidence intervals by cluster bootstrap over fixtures, retaining all
   seeds of one fixture as one cluster.  Use at least 10,000 bootstrap draws.
4. Apply Holm family-wise correction across the preregistered
   family-by-size-by-cap-by-baseline primary comparisons.
5. Report simultaneous confidence bands for the complete success/cap frontier.
6. Define the conservative certified-95% cap of a method as the smallest cap
   whose simultaneous lower confidence bound reaches `0.95`.  For a baseline,
   the smallest cap whose simultaneous **upper** bound can reach `0.95` is used
   as the optimistic comparator.  This prevents uncertainty from favoring the
   candidate.
7. Estimate slopes only on the full preregistered size grid.  No endpoint may
   be removed because it weakens the apparent separation.

The minimum of 30 fixtures per family/scale and 100 seeds per fixture is a
floor, not a retrospective power claim.  Before the manifest is locked, a
cluster-aware power calculation based only on excluded pilot fixtures must
record the detectable paired risk difference and may increase, but never
decrease, the fixture count.

## 9. Preregistration and artifact contract

The confirmatory manifest is committed before the first confirmatory outcome.
It contains:

- immutable family-generator source and mathematical promise;
- all fixture hashes, hidden-permutation hashes, and anchor-selection rule;
- oracle/interface/output-relation IDs;
- candidate and baseline source commits and baseline fidelity records;
- method configurations, confidence allocations, caps, fixture counts, seeds,
  exclusions, numerical tolerances, and statevector limits;
- all primary/secondary metrics, statistical tests, multiplicity family, and
  effect-size gates;
- theorem statements, composition rows, lower-bound target, and their initial
  status (`open`, never prefilled as `proved`);
- environment and dependency lock hashes; and
- a canonical SHA-256 digest of the complete manifest.

Development and pilot fixtures carry a separate namespace and can never be
promoted.  Confirmatory execution fails closed if the worktree is dirty, the
manifest hash differs, a method/config is missing, a run is missing, an
unexpected exclusion occurs, or provenance cannot be reproduced.

Attempt-level artifacts are append-only and contain the full resource ledger,
outcome flags, seed derivation, configuration hash, source tree, and failure
reason.  Aggregates are regenerated from raw attempts.  A second clean run
from the frozen commit must reproduce the schema, record count, fixture set,
and deterministic summaries before results are promoted.

## 10. Success and falsification gates

### 10.1 Circuit/implementation gate

This gate passes only if all of the following hold:

1. the numeric boundary and frontier schedule are absent from the
   algorithm-facing input;
2. coherent boundary discovery, stopping, and complete k-output extraction are
   executed rather than reconstructed by the harness;
3. exact-state involution, output, and cleanup residuals are at most `1e-12`
   on every preregistered small fixture and negative control;
4. all base-oracle directions and powered calls reconcile exactly with the
   query ledger;
5. no instance-derived QRAM/QROM or measured active-list shortcut is used; and
6. complete gate/depth/qubit/storage accounting is verified independently of
   the candidate's own counters.

One failing exact-state fixture blocks a coherent-unitary claim until the bug
is fixed and a new preregistration version is created.

### 10.2 Empirical gate

The empirical gate requires all preregistered data plus both conditions below:

- on `F-HIDDEN-FRONTIER`, the candidate's conservative certified-95% cap is at
  most `0.8` times the optimistic certified-95% cap of **every** strongest legal
  baseline at the two largest sizes for every preregistered primary gap scale;
  and
- the simultaneous lower confidence bound for the paired certified-exact risk
  difference is at least `+0.03` at two adjacent interior caps, with no
  corrected significant deficit at a larger cap.

These thresholds are intentionally fixed before outcomes.  Passing them is
finite-family evidence only.  Failure against one legal strongest baseline,
failure confined to one selected cap, or an advantage that vanishes after
full query charging falsifies the empirical superiority claim.

`F-PUBLIC-PARTITION` is expected to fail the novelty direction.  If the
candidate appears to beat partition composition there, the first conclusion
is a baseline-fidelity or accounting defect, not a new result.

### 10.3 Upper-bound theorem gate

Before a new-algorithm complexity claim is activated, a symbolic proof must
construct the complete unknown-boundary unitary and give an instance-dependent
bound of the following **template** form (not currently a theorem):

```text
Q_A(theta, delta)
  <= C * polylog(n, R, 1/delta)
       * [C_boundary(theta)
          + C_history({T_i})
          + C_multi({A_r, M_r, c_r})].
```

`T_i`, `A_r`, and `M_r` must arise from the algorithm's charged coherent
predicates, not a ground-truth history table.  The proof must state constants,
all logarithms, confidence composition, full output semantics, and cleanup.
Any logarithmic penalty required for unknown-time search remains in the bound
unless a mathematically explicit promise is proved to exclude the relevant
lower-bound family.

An analytic proxy, RMS identity, successful finite statevector, or fitted slope
does not pass this gate.

### 10.4 Composition-frontier gate

For each applicable published loop/transducer composition, Vihrovs
unknown-time computation-tree/VTS, Low--Su Tunable VTAA, Gao--Ji--Wang
approximate k-minimum, MIQAE/Rall coherent estimation, quantum BAI, and
all-marked extraction theorem, the final ledger must record:

1. its exact oracle and output relation;
2. boundary, partition, stopping-time, and QRAM information it receives;
3. the cost of constructing and erasing every predicate under this interface;
4. its instantiated bound on the exact `F-HIDDEN-FRONTIER` promise; and
5. either the resulting asymptotic comparison or a proved assumption mismatch.

If any legal same-interface construction matches the candidate up to the
claimed factors, the novelty gate fails.  Merely omitting that baseline from
the executable panel or assigning it an unfavorable proxy cannot reopen the
gate.

### 10.5 Lower-bound gate

The lower bound must apply to all adaptive quantum algorithms using the same
canonical controlled/inverse oracle and returning the same exact Top-k
relation with error at most `delta`.  It must use the same hidden-frontier
promise and public information as the upper bound.

A valid proof must address boundary localization, coherent history discovery,
direct multi-output extraction, queries across shells in superposition, and
the angular cost of small gaps.  It may use a general adversary, polynomial,
or hybrid/direct-product construction, but a maximum single-shell fact or a
bound limited to estimate-then-sort algorithms is insufficient.  The target
must match the new upper bound up to explicitly declared logarithmic factors.

A counteralgorithm below the target, an adversary that assumes public blocks,
or a reduction that loses the claimed gap dependence falsifies the target.

### 10.6 Claim activation

The repository evidence gate remains false unless every row below passes:

| Gate | Required status |
|---|---|
| Preregistration completeness | `LOCKED_BEFORE_RUN` with valid manifest hash |
| Unknown-boundary coherent implementation | `VERIFIED` |
| Complete direct multi-output and cleanup | `VERIFIED` |
| Resource accounting | `VERIFIED` by independent reconciliation |
| Strongest-baseline fidelity | `VERIFIED` for every claimed strongest baseline |
| Empirical matched superiority | `PASSED` under the frozen simultaneous tests |
| New same-interface upper bound | `PROVED` |
| Composition frontier | `PROVED_NO_MATCH` |
| Matching lower bound | `PROVED` |

Passing this internal stack would make a top-tier submission technically
plausible; it still would not guarantee a CCF-A acceptance.  If a theory gate
fails, the admissible result is a circuit/benchmark or a falsification report,
not a quantum-advantage claim.

## 11. Experiment matrix

| Tier | Panel | Grid | Repetitions | Required output |
|---|---|---|---:|---|
| `T0` | Contract and fail-closed tests | invalid indices, ties, dirty work, tampered transcript, missing inverse charge, cap edge | Exhaustive deterministic | Every invalid shortcut rejected. |
| `S1` | Supplied-schedule regression | Existing 3 schedule fixtures and every prefix | Deterministic exact state | Preserve current replay semantics; labelled stronger information. |
| `S2` | Unknown-boundary exact-state core | `n={4,6,8}`, all valid k, hidden centers/shells/permutations | Exhaustive or frozen complete orbit | Boundary discovery, full output, inverse, cleanup, and independent resource reconciliation. |
| `S3` | Bounded-error exact-state calibration | Same S2 fixtures across QPE precisions and confidence allocations | 1,000 frozen seeds per fixture where measurement is present | False-certificate and cleanup calibration. |
| `H1` | Matched homogeneous/heterogeneous panel | `F-EQ`, `F-DYADIC`, `F-CLUSTER`; `n={32,64,128,256}`, two k values, four gaps, seven caps | 30 fixtures x 100 seeds | Full certified-exact/query frontier. |
| `H2` | Primary hidden-frontier panel | `F-HIDDEN-FRONTIER`, same scale grid | 30 fixtures x 100 seeds | Main empirical and stopping-profile tests. |
| `N1` | Mandatory negative controls | `F-PUBLIC-PARTITION`, `F-UNKNOWN-TIME-NC`, `F-TIE-NC` | Same paired seeds where defined | Expected composition match, generic-log barrier, and fail-closed tie rejection. |
| `A1` | Information ablations | known boundary, known schedule, free-history control, fixed-time, repeated single-output, no-replay fault injection | Same H2 fixtures | Quantify the price of each information/resource assumption; never pooled. |
| `R1` | Resource scaling | Every S2/H1/H2 attempt | No extra runs | Base calls, gates, depth, qubits, storage, stopping profiles. |
| `C1` | Composition theorem table | Every relevant theorem x exact interface | Symbolic/manual verification | Assumption map and instantiated asymptotic bound. |
| `L1` | Lower-bound program | Exact H2 promise and canonical oracle | Formal proof, not simulation | Weighted all-algorithms lower bound or explicit falsification. |

## 12. Execution order

The order is designed to stop expensive experiments when the core claim is
already falsified.

1. Freeze the canonical oracle, exact output relation, hidden-frontier promise,
   and interface hash.
2. Complete the composition-theorem registry.  If a legal construction already
   matches the proposed bound, revise or stop before running a large campaign.
3. State the candidate upper-bound lemma and lower-bound target symbolically,
   including all unknown-time logarithms and discovery/cleanup costs.
4. Replace supplied schedules with executable `U_boundary` and `U_frontier`
   circuits, then implement complete durable k-output extraction.
5. Pass `T0`, `S1`, and `S2`; independently reconcile the oracle ledger and
   circuit resources.
6. Implement and fidelity-test every primary baseline under the identical
   interface.
7. Run excluded pilot fixtures only to locate a nondegenerate cap range and
   perform cluster-aware power analysis.
8. Commit and hash the complete confirmatory manifest.
9. Run `S3`, then `H1`, `H2`, and `N1` without changing code or configuration.
10. Regenerate all aggregates from attempt-level records, run the frozen
    statistical analysis once, and evaluate every gate mechanically.
11. Repeat the campaign from a clean environment and compare provenance,
    schema, record completeness, and deterministic summaries.
12. Only after the implementation, empirical, composition, upper-bound, and
    lower-bound gates all pass may the claim language be reconsidered.

The immediate next implementation milestone is therefore not another
supplied-schedule statevector sweep.  It is a charged, replayable
unknown-boundary constructor plus a complete k-output relation on the `S2`
fixtures.  If that constructor needs an instance-derived classical frontier,
the present mainline is falsified under its own interface.
