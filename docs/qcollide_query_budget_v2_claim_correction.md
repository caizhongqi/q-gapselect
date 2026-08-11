# Q-COLLIDE Query-Budget v2 — Quantum-Proxy Claim Correction

## Why this correction is required

Query-Budget Scaling v2 measured real **classical endpoint-record query counts** and compared them with a field named `qccs_packed_query_proxy`, whose structural factor was

```text
(N^2 / K)^(1/3).
```

Subsequent literature audit establishes that this cube-root factor is not a novel Q-COLLIDE pair-detection result. Quantum pair finding with `K` disjoint marked pairs already has a corresponding cube-root structural dependence in prior work (e.g. Allcock et al., ESA 2022 / arXiv:2111.07059, Theorem 2.8), and optimal quantum claw finding is established prior art.

Therefore the v2 field must be interpreted as a **prior-art pair-detection reference contour**, not as a proved Q-COLLIDE capacity-estimation complexity.

## What remains valid from v2

The classical side of Query-Budget Scaling v2 remains a valid empirical result under its explicitly implemented endpoint-record accounting:

- endpoint queries are measured, not generated from a formula;
- the matching-promise relative-error query exponent is approximately `0.5484` over the registered grid;
- the additive-error exponent is approximately `0.450`;
- hub / overlap / biclique negative controls demonstrate that edge-count scaling does not estimate maximum-matching capacity;
- full induced-graph reconstruction plus exact matching recovers the controlled ground-truth capacity.

These results remain useful as empirical classical baselines and overlap falsification evidence.

## What is retracted/reclassified

The following interpretation is no longer admissible:

> `qccs_packed_query_proxy` is a proved Q-COLLIDE quantum capacity-estimation algorithm with complexity `~O(epsilon^-1 (N^2/K)^(1/3))`.

That statement mixed two logically separate layers:

1. a known quantum **pair-detection** structural exponent;
2. an unproved assumption that this detector automatically yields a stable estimator of `K`.

The numerical v2 ratios between measured classical queries and the analytic proxy — including the previously reported approximate range `6.35x–40.32x` and mean `~21.15x` — are therefore retained only as **historical classical-versus-pair-detection-reference ratios**. They are not evidence of a proved capacity-estimation advantage and must not be used as a headline quantum result.

## Corrected capacity-estimation result

The replacement quantum-capacity result is **Fixed-Subset Capacity Estimator v3**.

It explicitly separates:

### Capacity-estimation layer — Q-COLLIDE contribution candidate

For the packed/disjoint promise:

- choose uniformly random left/right subsets of fixed size `r`;
- compute the exact probability that at least one of `K` disjoint marked pairs survives;
- prove monotonicity and a discrete slope bound;
- invert the finite-N survival curve;
- prove that additive survival-probability precision `Theta(epsilon)` gives relative capacity precision `epsilon` in a registered constant-factor scale window.

### Inner pair-detection layer — prior art

The yes/no test on the selected `r x r` subproblem uses known quantum pair/claw-finding primitives. The `r^(2/3)` structural exponent is cited as prior art and is not claimed as a new quantum search primitive.

The resulting packed-capacity contour

```text
~O(epsilon^-1 * (N^2/K)^(1/3))
```

is now derived as **proved capacity inversion + prior-art inner detector**, subject to still-open coherent data-structure/predicate resource accounting and unknown-K scale selection.

Fixed-Subset v3 run `31502830093` completed all 63 registered validation cells and all theorem-validation gates passed. See `docs/qcollide_fixed_subset_capacity_v3_result_audit.md`.

## Paper-writing rule

Any future paper draft, table, plot or PR description must distinguish these three categories:

1. **measured classical endpoint-query count** — executable empirical quantity;
2. **prior-art pair-detection query contour** — analytic baseline/reference;
3. **Q-COLLIDE capacity-estimation contour** — only admissible when the capacity inversion/reduction and its oracle assumptions are stated explicitly.

No numerical ratio between category 1 and category 2 may be labeled a demonstrated quantum capacity-estimation speedup.
