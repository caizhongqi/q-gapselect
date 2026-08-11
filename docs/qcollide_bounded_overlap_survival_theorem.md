# Bounded-overlap survival theorem for functional collision capacity

## Setting

Let `G=(A,B,E)` be a finite bipartite functional-collision graph. Write

- `K = nu(G)` for its maximum matching capacity;
- `Delta` for its maximum degree;
- `G_q` for the induced graph obtained by retaining every left and right vertex
  independently with the same probability `q`;
- `S_G(q) = Pr[E(G_q) != empty]` for the collision-survival probability.

No forest, acyclicity, independent-component, or unique-witness assumption is
made.

## Theorem 1: survival envelope under arbitrary overlap

For every `q in [0,1]`,

\[
\boxed{
1-(1-q^2)^K
\leq
S_G(q)
\leq
\min\{1,K\Delta q^2\}.
}
\]

### Proof of the lower side

Fix any maximum matching `M` with `|M|=K`. Its edges are vertex-disjoint. An
edge of `M` survives exactly when both endpoints are retained, with probability
`q^2`. Because distinct matching edges use disjoint endpoints, their survival
events are independent. Thus the probability that at least one matching edge
survives is

\[
1-(1-q^2)^K.
\]

Whenever a matching edge survives, `G_q` is nonempty, proving the lower bound.

### Proof of the upper side

By Konig's theorem, a bipartite graph with maximum matching size `K` has a
vertex cover `C` of size `K`. Every edge touches at least one vertex in `C`, and
each cover vertex is incident to at most `Delta` edges. Therefore

\[
|E|\leq K\Delta.
\]

Each edge survives with probability `q^2`. A union bound gives

\[
S_G(q)\leq |E|q^2\leq K\Delta q^2,
\]

and probability is at most one.

## Corollary 1: population capacity interval

For `0<q<1` and `S=S_G(q)<1`, monotonic inversion gives

\[
\boxed{
\frac{S}{\Delta q^2}
\leq
K
\leq
\frac{\log(1-S)}{\log(1-q^2)}.
}
\]

The right endpoint is the familiar packed-matching inverse. Under overlap it is
not a point estimator: it is a valid upper bound. The left endpoint is the
explicit overlap correction.

## Corollary 2: finite-sample high-probability certificate

Suppose `T` independent retention experiments produce empirical survival
`S_hat`. For failure probability `delta`, Hoeffding gives

\[
r_T=\sqrt{\frac{\log(2/\delta)}{2T}},
\qquad
S_-=(S_{hat}-r_T)_+,
\qquad
S_+=\min\{1,S_{hat}+r_T\}.
\]

With probability at least `1-delta`,

\[
\boxed{
\frac{S_-}{\Delta q^2}
\leq K \leq
\frac{\log(1-S_+)}{\log(1-q^2)}
}
\]

with the trivial domain-size upper bound applied when `S_+=1`.

This is implemented by `survival_capacity_confidence_interval` in
`overlap_bounds.py`.

## Corollary 3: bounded-overlap capacity sketch at the critical scale

Take `q=c/sqrt(K)` for a fixed constant `c>0`, ignoring the harmless `q<=1`
clipping at very small `K`. Then

\[
1-(1-c^2/K)^K = \Theta(1),
\]

so collision survival is bounded below by a positive constant independent of
`N`. The population interval becomes

\[
\Omega(K/\Delta)\leq K\leq O(K).
\]

Thus a single critical-scale survival statistic provides an `O(Delta)`-factor
capacity bracket. If `K` is unknown, a dyadic schedule of retention scales can
locate the transition with logarithmic overhead.

This is the first general-overlap statement in Q-COLLIDE that preserves the
maximum-matching interpretation without pretending that edge count equals
capacity.

## Quantum query implication under the endpoint-local claw interface

At the critical retention scale, each side contains `Theta(qN)` retained
endpoints in expectation. Applying the same endpoint-local claw detector used
in the packed analysis to this retained domain has the query-scale proxy

\[
\widetilde O((qN)^{2/3})
=
\widetilde O((N^2/K)^{1/3}).
\]

Constant-accuracy estimation of the survival probability adds only constant
outer repetition/amplitude-estimation overhead. A dyadic unknown-`K` search
adds logarithmic factors. Consequently the bounded-overlap theorem supports

\[
\boxed{
\widetilde O((N^2/K)^{1/3})
}
\]

as a quantum **`O(Delta)`-factor capacity-bracketing upper-bound scale** under
the same endpoint-local oracle assumptions as the packed-claw analysis.

This statement is intentionally weaker than a `(1+epsilon)` maximum-matching
estimator. The latter remains open for overlapping collision graphs.

## Lower-bound boundary

The disjoint packed-matching family (`Delta=1`) is a subfamily of every class
with a maximum-degree promise `Delta>=1`. Therefore any algorithm claimed to
solve the general bounded-overlap problem must at least respect the existing
packed-claw hard instances. What is not yet proved is a tight lower bound with
an explicit optimal dependence on both `Delta` and the estimation accuracy
`epsilon` for the full overlapping family.

## Empirical falsification and theorem alignment

The frozen overlap benchmark confirms why the interval, rather than a universal
point inverse, is necessary:

- matching: packed inverse recovers `K=16` to error about `0.09`;
- sparse overlap (`Delta=2`): degree-aware isolation reduces packed-inverse
  error from about `9.81` to `1.48`;
- hub (`Delta=64`, `K=1`): high edge count does not imply high capacity;
- dense graph (`Delta=19`, `K=128`): naive packed inversion overestimates by
  more than `600`.

The deterministic arbitrary-overlap envelope

\[
\left\lceil\frac{|E|}{\Delta}\right\rceil
\leq K\leq\min\{|A|,|B|,|E|\}
\]

and the survival envelope above both contain the correct quantity `K=nu(G)`.

## Claim boundary

Supported:

1. exact packed survival inversion for disjoint matching witnesses;
2. arbitrary-overlap deterministic capacity envelope;
3. arbitrary-overlap survival-probability envelope;
4. finite-sample high-probability capacity interval;
5. an `O(Delta)` critical-scale capacity bracket;
6. the cubic-root quantum upper-bound scale for that bracket under the existing
   endpoint-local claw interface.

Not yet supported:

1. an unbiased point estimator for arbitrary overlap;
2. a `(1+epsilon)` quantum estimator of `nu(G)` for arbitrary collision graphs;
3. a tight optimal `Delta` dependence;
4. the full multiplicative `1/epsilon` adversary lower bound;
5. fault-tolerant end-to-end runtime advantage.
