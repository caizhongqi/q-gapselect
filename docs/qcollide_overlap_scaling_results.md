# Q-COLLIDE overlap/quantum scaling main experiment

## Frozen execution

GitHub Actions run `31460807922` completed all nine scaling components and the merge job successfully. The merged artifact is `9089771579` with digest

```text
sha256:fbe0dfd346f69f70c9d3c7777606b70bcc153a8f5854d587131298512c08338c
```

The grid spans `N in {64,128,256}`, requested matching densities `{1/16,1/8,1/4}`, and four graph families: disjoint matching, bounded sparse overlap, hub/star, and dense bipartite overlap. Each component uses 1,024 Monte Carlo trials and retains exact maximum bipartite matching as ground truth.

## Capacity-certificate result

Across all 36 graph cells:

- every deterministic envelope contained the exact matching capacity;
- the finite-sample survival-capacity certificate covered the exact capacity in `36/36` cells;
- disjoint-matching packed inversion had mean relative error `0.0327`.

The certificate is deliberately conservative under heavy overlap. Its mean relative width is approximately `0.599` for matching, `1.379` for sparse overlap, `2.087` for hub graphs, and `0.954` for dense graphs. The widening is evidence of the explicit overlap penalty rather than hidden estimator bias.

## Degree-aware isolation result

For sparse bounded overlap, the mean absolute error of naive packed inversion is `17.49`, while the degree-aware packed estimate has mean absolute error `2.59`, an error reduction of approximately `85.2%`.

This improvement does **not** generalize into a universal unbiased estimator. On hub graphs the degree-aware estimate is no better on average, and dense graphs retain substantial bias. These negative controls remain part of the main result.

## Analytic query-scaling layer

For positive capacity the paper reports the endpoint-query scaling proxies

```text
Q_classical ~ sqrt(N^2 / K)
Q_quantum   ~ (N^2 / K)^(1/3)
```

as theoretical quantities, not measured runtime.

At fixed matching density `K/N=1/16`, the classical-to-quantum proxy ratio increases from approximately `3.175` at `N=64`, to `3.564` at `N=128`, to `4.000` at `N=256`. For densities `1/8` and `1/4`, the corresponding ratios are `2.828 -> 3.175 -> 3.564` and `2.520 -> 2.828 -> 3.175`.

This numerical table is a consistency visualization of the analytic square-root versus cube-root query laws. It is **not** experimental evidence of wall-clock quantum speedup.

## Defensible conclusion

The scaling experiment supports three distinct claims:

1. packed survival inversion remains accurate on disjoint matching fixtures;
2. degree-aware isolation is a strong empirical correction for sparse bounded overlap but fails as a universal overlap estimator;
3. arbitrary-overlap capacity can still be reported with valid deterministic and finite-sample survival certificates, with the loss of precision exposed through degree/overlap.

The experiment does not close the optimal `Delta`-dependent quantum theorem, the full multiplicative `1/epsilon` lower bound, coherent execution, or fault-tolerant runtime advantage.
