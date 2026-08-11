# Q-COLLIDE Fixed-Subset Capacity Estimator v3 — Result Audit

## Frozen execution

- Workflow: `qcollide-fixed-subset-capacity-v3`
- Run: `31502830093`
- Grid: `N = 256,512,1024,2048,4096,8192,16384`
- Capacity scales: `N/64, N/32, N/16`
- True-capacity multipliers within each registered factor-two window: `1.0,1.5,2.0`
- Total cells: 63
- Monte Carlo trials per cell: 8192
- Registered relative-error target: `epsilon = 0.25`
- Merged artifact ID: `9105814969`
- Digest: `sha256:c72ffe1b43a7df51a8b86c39ba6b352d96eb9dbab02fadaf63d3bfc58d5312f8`

All seven component jobs and the merge job completed successfully.

## Theorem-validation gates

All registered fail-closed gates pass:

1. **Exact finite-N inversion:** 63/63 exact survival probabilities invert to the exact integer capacity.
2. **Bonferroni envelope:** 63/63 exact survival probabilities lie between the registered first/second-order bounds.
3. **Stable inversion:** perturbing the survival probability by the theorem-prescribed additive precision `eta = 3 epsilon / 128` yields relative capacity error at most `epsilon` in every registered cell.

The maximum observed worst-case relative capacity error under the registered probability perturbation is `0.125`, below the registered target `0.25`.

## Monte Carlo validation

Monte Carlo is used only to validate the finite-N survival probability and inversion numerically. It is **not** treated as quantum execution or query-speed evidence.

Across all 63 cells:

- mean absolute survival-probability error: `0.0031947422`;
- maximum absolute survival-probability error: `0.0090493268`;
- mean relative capacity error after exact-curve inversion: `0.0197844329`;
- maximum relative capacity error: `0.125`;
- fraction of cells satisfying the registered 25% relative-capacity target: `1.0`.

## Query-complexity interpretation

The fixed-subset capacity construction separates two layers.

### New/derived capacity layer

The repository proves for the packed/disjoint promise:

- an exact fixed-size-subset survival probability;
- monotonicity and a discrete survival-curve slope bound;
- stable inversion from additive survival-probability error to relative capacity error;
- a registered constant-factor capacity-scale window requiring only `Theta(epsilon)` survival-probability precision.

### Prior-art inner pair-detection layer

The inner yes/no question on a selected `r x r` endpoint domain — whether at least one marked pair is present — uses known quantum pair/claw-finding machinery. The structural `r^(2/3)` query exponent is therefore **prior art**, not claimed as Q-COLLIDE novelty.

At the registered stable scale,

```text
r = Theta(N / sqrt(K)),
```

so combining the proved capacity inversion with the known inner detector gives the candidate packed-capacity record-query contour

```text
~ O(epsilon^-1 * (N^2/K)^(1/3))
```

for a constant-factor-correct capacity scale, before gate/resource overheads and unknown-K scale search.

The workflow reports this only as an **analytic query contour**. The v3 experiment does not generate fake quantum runtime data or fit a complexity exponent from those proxy values.

## What this experiment establishes

The packed-capacity estimator has now passed three distinct checks:

1. the finite-N combinatorial survival formula is exact on brute-force small instances;
2. the theorem-specified inversion stability holds throughout the full registered large-N matrix;
3. finite-sample Monte Carlo survival estimates recover capacity well inside the registered 25% target on all 63 cells.

This is stronger than the older Query-Budget v2 quantum proxy because v3 validates a **capacity-estimation reduction**, not only a pair-detection scaling expression.

## What remains open

- a same-endpoint-record-oracle lower bound for estimating capacity;
- optimal unknown-K scale selection without a naive logarithmic scan;
- full reversible subset-unranking / data-structure / collision-predicate gate accounting;
- bounded-degree overlap capacity estimation;
- arbitrary-overlap capacity estimation;
- fault-tolerant or hardware runtime crossover.

## Claim boundary

A defensible current statement is:

> Under the packed/disjoint collision promise and a constant-factor capacity scale, Q-COLLIDE has an exact fixed-subset survival estimator whose capacity inversion is provably stable, and whose candidate quantum record-query contour combines this estimator with existing quantum pair-detection primitives.

It is not yet defensible to call this an optimal arbitrary-overlap quantum maximum-matching estimator or a demonstrated hardware speedup.
