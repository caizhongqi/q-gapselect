# Overlap-aware functional collision capacity: benchmark verdict

## Scope

This benchmark tests whether the exact packed-matching survival inversion can be
extended to collision graphs with shared endpoints. It compares four bipartite
families at `N=128` with 12,000 Monte Carlo trials:

1. disjoint matching;
2. sparse bounded-degree overlap;
3. a high-degree hub/star negative control;
4. a dense random bipartite graph.

The exact target is always the maximum bipartite matching size `K_C = nu(G)`.
Raw edge count is never treated as capacity.

## Empirical results

| family | `|E|` | `Delta` | true `K_C` | uniform packed inversion | abs. error | degree-aware packed inversion | abs. error |
|---|---:|---:|---:|---:|---:|---:|---:|
| matching | 16 | 1 | 16 | 15.913 | 0.087 | 15.884 | 0.116 |
| sparse overlap | 48 | 2 | 38 | 47.812 | 9.812 | 39.482 | 1.482 |
| hub | 64 | 64 | 1 | 1.749 | 0.749 | 0.134 | 0.866 |
| dense | 1336 | 19 | 128 | 777.341 | 649.341 | 98.362 | 29.638 |

The benchmark artifact was produced by GitHub Actions run `31449858481`,
artifact `9085920720`, digest
`sha256:fc69d0daa60b58ed7736067be1cde4fa0bdba84ec0d037b6d6e19beaa6830f8b`.

## What is falsified

The packed survival identity

\[
S_K(q)=1-(1-q^2)^K
\]

is exact when the `K` witnesses are vertex-disjoint. The hub and dense controls
show that applying its inverse to an arbitrary overlapping graph is not valid.
In particular, the dense graph has 1,336 edges but matching capacity only 128;
the naive packed inversion estimates about 777.

Degree-aware isolation is therefore not presented as a universal exact
correction. It sharply reduces error on the sparse-overlap instance and on the
dense packed-inversion diagnostic, but it underestimates the hub. The current
implementation remains an empirical estimator outside the disjoint case.

## A theorem that survives arbitrary overlap

For any finite bipartite collision graph with edge count `m=|E|` and maximum
degree `Delta`, the edge set can be partitioned into `Delta` matchings. Hence at
least one matching contains `ceil(m/Delta)` edges. Together with the trivial
upper bounds,

\[
\boxed{
\left\lceil\frac{|E|}{\Delta}\right\rceil
\leq
K_C
\leq
\min\{|A|,|B|,|E|\}.
}
\]

For the frozen benchmark this gives:

| family | provable capacity envelope | true `K_C` |
|---|---:|---:|
| matching | `[16,16]` | 16 |
| sparse overlap | `[24,48]` | 38 |
| hub | `[1,64]` | 1 |
| dense | `[71,128]` | 128 |

The lower endpoint is exact on the star negative control and both endpoints are
exact on a disjoint matching.

## Consequence for the Q-COLLIDE theorem program

The general-overlap result should be separated into two layers:

1. **arbitrary bipartite overlap:** deterministic capacity envelope from
   `(|E|, Delta)` with no independent-witness assumption;
2. **bounded sparse overlap:** degree-aware isolation followed by survival
   sketching, for which a tighter approximation theorem remains to be proved.

The paper must not claim a general exact maximum-matching estimator from a
single survival probability. The remaining theorem target is to quantify the
approximation factor and query complexity under a bounded-overlap or bounded
local-congestion promise, then prove a matching lower bound under the same
promise and endpoint-oracle interface.

## Claim boundary

Supported now:

- exact packed-matching inversion;
- explicit hub/dense counterexamples to naive generalization;
- exact maximum-matching ground truth in every benchmark;
- arbitrary-overlap deterministic capacity envelope;
- empirical evidence that degree-aware isolation is useful in sparse overlap.

Not yet supported:

- an unbiased estimator for arbitrary overlap;
- a tight `Delta`-dependent quantum capacity-estimation theorem;
- the full multiplicative `1/epsilon` lower bound in the raw endpoint oracle;
- an end-to-end fault-tolerant runtime advantage.
