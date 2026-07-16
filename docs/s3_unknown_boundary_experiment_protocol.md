# S3 unknown-boundary Top-k experiment protocol

## Claim under test

The S3 candidate is evaluated only on the blind canonical Bernoulli-rotation
oracle.  Its public input is

\[
  (n,k,\delta,Q_{\max},B_\theta,B_\theta^\dagger),
\]

and does not contain the answer, a boundary value, a gap, a fixture family, a
precision schedule selected from private truth, an active-arm list, or a
stopping-time profile.  A run may return either the complete strict Top-k set
or `INCONCLUSIVE`.  On a tie/nonpromise instance it must fail closed.  At a
fixed cap, timeouts, uncertified outputs, wrong sets, and budget violations are
all failures and remain in the denominator.

This protocol distinguishes four evidence levels that must not be merged:

1. an exact-state circuit-semantics result;
2. a measured or analytic algorithm experiment;
3. a source-faithful composition-bound instantiation;
4. a proved asymptotic upper or lower bound.

Evidence from an earlier level cannot activate a claim at a later level.

## Candidate circuit contract

The adaptive candidate must expose a public sequence of precision caps, build
its certificate from coherent phase information, record recognizable stopping
flags, copy one durable `n`-bit membership mask, and replay every transient
workspace.  It may stop at precision level \(r\) only when every value in the
declared phase uncertainty cells induces the same strict Top-k mask.  Otherwise
it advances to the next public precision or returns `INCONCLUSIVE` at the hard
cap.

The resource ledger charges every call to `B`, `B_dagger`, `controlled_B`, and
`controlled_B_dagger`.  A NumPy kernel invocation, an undecomposed logical
macro, a truth-table row, and a canonical oracle query are separate resources.
No elementary-gate count, circuit depth, compiled ancilla count, QRAM claim, or
hardware runtime is reported without an actual decomposition or execution.

## Preregistered panels

### P1: exact-state circuit semantics

- `n` in `{2, 3}` and every legal `k`;
- on-grid, half-grid, generic off-grid, near-boundary, and exact-tie angles;
- public phase precisions from 1 through the largest exact-state-feasible value;
- forward unitary, durable copy, inverse replay, output reduced purity, and
  garbage probability checked independently;
- on-grid success is not extrapolated to generic inputs.

Primary outputs are the returned membership mask or `INCONCLUSIVE`, dominant
output probability, canonical query count, stopping history, output purity,
executed cleanup residual, and a proved-or-unproved error-bound label.

### P2: fixed fixture by repeated algorithm seed calibration

- every hidden-frontier family, including public-partition and tie controls;
- at least 30 frozen fixture seeds per family for a claim-bearing run;
- at least 20 algorithm repetitions within each stochastic fixture/method cell;
- `n` in `{16, 32, 64}` and `k` in `{1, 2, 4, 8}` where legal;
- at least four preregistered hard query caps;
- identical fixture, repetition, and cap cells for the candidate and every
  executable baseline.

The fixture is the resampling unit.  Algorithm repetitions within one fixture
are not counted as independent instances.

### P3: heterogeneous variable-time stress

- matched angle multisets with different hidden permutations;
- matched hidden-frontier/public-partition controls;
- equal-gap, dyadic, clustered, and unknown-time negative-control profiles;
- fixed `n` with increasing stopping-time heterogeneity;
- fixed heterogeneity with increasing `n` and `k`.

This panel tests whether an observed gain comes from a legal variable-time
mechanism or from leaked partitions/schedules.  A gain confined to the public
partition control cannot support the blind-interface claim.

### P4: direct multi-output scaling

- hold the minimum boundary gap distribution fixed while varying `k`;
- hold `k/n` fixed while varying `n`;
- report complete-set recovery, duplicate outputs, missing outputs, and total
  extraction queries;
- compare direct output with repeated single-output removal under the same
  exclusion-memory and verification charges.

Partial recall, best-arm success, and per-arm classification accuracy are
secondary diagnostics and cannot replace complete exact Top-k recovery.

### P5: composition and lower-bound falsification

For every registered published threat, record the exact source version,
theorem locator, oracle reduction, output-relation conversion, error
allocation, inverse/cleanup charges, implementation fidelity, and resulting
bound.  An unimplemented theorem is a blocker, not a weak numerical baseline.

Small-instance lower-bound witnesses may verify local overlap, hybrid,
polynomial, or adversary constraints.  Numerical feasibility or a finite-size
slope is labelled a witness only; it is not an all-algorithms lower-bound
theorem.

## Required executable baselines

The minimum claim-bearing set is:

- fixed-precision coherent estimate/rank/copy/replay;
- independent adaptive amplitude estimation followed by certified sorting;
- repeated quantum best-arm identification with charged winner exclusion;
- approximate k-minimum with a charged approximate-value oracle;
- optimal unknown-time variable-time search plus complete-output extraction;
- straight-line and loop subroutine composition instantiated on the same
  Top-k relation;
- a classical fixed-confidence Top-k comparator, reported separately from the
  quantum-composition frontier.

Proxy-only implementations remain visible but cannot satisfy the strongest
baseline fidelity gate.

## Metrics and statistics

The primary empirical curve is certified exact recovery at query cap,
`CER@Q`.  It uses all attempts.  The companion curves are wrong-set rate,
`INCONCLUSIVE@Q`, timeout rate, budget-violation rate, mean and quantiles of
canonical queries, cleanup error, and output purity.  Resource comparisons are
unconditional and do not condition on both methods succeeding.

Candidate-versus-baseline comparisons use paired fixture/repetition outcomes,
two-sided McNemar tests, whole-fixture paired bootstrap confidence intervals,
and Holm correction across preregistered family/cap hypotheses.  The minimum
risk difference and all caps are fixed before the claim-bearing run.

## Gates for a CCF-A-level quantum-advantage claim

All gates are conjunctive:

1. **Circuit gate:** coherent index execution, durable complete output, bounded
   cleanup, and reconciled oracle queries are verified.
2. **Empirical gate:** the preregistered `CER@Q` frontier beats every executable
   strongest baseline by the fixed effect size after multiplicity correction.
3. **Fidelity gate:** every mandatory published threat is implemented or
   instantiated faithfully under the same information and output relation.
4. **Upper-bound gate:** a new query upper bound is proved for the actual
   algorithm, including confidence and cleanup.
5. **Composition gate:** published subroutine, loop, k-minimum, QBAI, and
   variable-time compositions do not imply the same asymptotic bound.
6. **Lower-bound gate:** a matching lower bound holds for all algorithms in the
   declared canonical oracle model.

If any gate is missing, the artifact must set both
`quantum_advantage_claimable` and `ccf_a_claimable` to `false`.

## Current S3 checkpoint versus this protocol

The tiny `n=2,k=1` coherent stopping-history execution is tracked as a
non-claim circuit checkpoint, not as satisfaction of the generic circuit gate.
The current 486-attempt control panel excludes the stronger public-partition
interface and correctly retains all failures, but it contains only proxy
controls: the true-coherent candidate is not in its CER@Q comparison. It also
uses three fixture seeds, three measurement seeds, and three caps, below the
claim-bearing P2 design above. Therefore the integrated audit records one
checkpoint passed and zero of six paper gates passed.
