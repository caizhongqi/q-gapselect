# Q-COLLIDE Fixed-Subset Collision Capacity Estimator

## 1. Motivation and prior-art correction

For `K` disjoint marked pairs in two indexed sets of size `N`, quantum pair finding with complexity of order

\[
\widetilde O((N^2/K)^{1/3})
\]

is prior art, not a Q-COLLIDE contribution. In particular, Allcock et al. (ESA 2022, arXiv:2111.07059, Theorem 2.8) give a quantum pair-finding algorithm for `K` disjoint marked pairs. Standard claw finding is also prior art.

The Q-COLLIDE algorithmic question is different:

> **Can we estimate the number of independent collision pairs / packed collision capacity, rather than merely find one pair, without reconstructing the full collision graph?**

This note defines a candidate estimator that does not require Bernoulli-thinned endpoints to be compacted into a free indexed list.

---

## 2. Packed matching model

Let the left and right domains both have size `N`. Exactly `K` disjoint endpoint pairs are marked by the collision predicate.

Independently choose:

- a uniformly random left subset `L` of fixed size `r`;
- a uniformly random right subset `R` of fixed size `r`.

Define the survival event

\[
Z=1
\]

iff at least one of the `K` marked pairs has its left endpoint in `L` and its right endpoint in `R`.

Let

\[
S_{N,K}(r)=\Pr[Z=1].
\]

The subsets are already compact indexed objects of size `r`; no Bernoulli compaction assumption is required.

---

## 3. Exact survival probability

Let `T` be the number of matched left endpoints selected into `L`. Then

\[
\Pr[T=t]
=
\frac{\binom Kt\binom{N-K}{r-t}}{\binom Nr}.
\]

Conditioned on `T=t`, the right subset avoids all `t` corresponding matched partners with probability

\[
\frac{\binom{N-t}{r}}{\binom Nr}.
\]

Therefore

\[
\boxed{
S_{N,K}(r)
=
1-
\sum_t
\frac{\binom Kt\binom{N-K}{r-t}}{\binom Nr}
\frac{\binom{N-t}{r}}{\binom Nr}
}
\]

where

\[
\max(0,r-(N-K))\le t\le\min(K,r).
\]

This is an exact finite-`N` formula.

---

## 4. Simple survival bounds

For each marked pair `i`, let `E_i` be the event that both endpoints are selected.

A single pair survives with probability

\[
p=\left(\frac rN\right)^2.
\]

Two distinct disjoint pairs survive jointly with probability

\[
p_2
=
\left(
\frac{r(r-1)}{N(N-1)}
\right)^2
\le p^2.
\]

By the union bound and the first Bonferroni inequality,

\[
Kp-\binom K2p_2
\le
S_{N,K}(r)
\le
Kp.
\]

These bounds are intentionally elementary and remain useful for theorem diagnostics.

---

## 5. Monotonicity and discrete inverse stability

### Lemma 1 — monotonicity

For fixed `N,r`, `S_{N,K}(r)` is nondecreasing in `K`.

#### Proof

Couple the instance with `K+1` marked disjoint pairs to the instance with the first `K` pairs. Every subset pair that contains a marked pair in the `K`-pair instance also contains one in the `K+1` instance. QED.

### Lemma 2 — one-pair increment lower bound

For `K` existing disjoint pairs,

\[
S_{N,K+1}(r)-S_{N,K}(r)
\ge
p-Kp_2
\ge
p(1-Kp).
\]

#### Proof

The increment equals the probability that the new pair survives while none of the previous `K` pairs survives:

\[
\Pr[E_{K+1}\cap E_1^c\cap\cdots\cap E_K^c].
\]

By subtracting a union bound on intersections with previous marked-pair events,

\[
\Pr[E_{K+1}]-\sum_{i=1}^K\Pr[E_{K+1}\cap E_i]
\ge p-Kp_2.
\]

Since `p_2 <= p^2`, the second inequality follows. QED.

---

## 6. Stable capacity-scale window

Suppose a scale `kappa` satisfies

\[
\kappa\le K\le2\kappa.
\]

Choose a fixed subset size `r` for which

\[
\frac{1}{16\kappa}
\le
p=\left(\frac rN\right)^2
\le
\frac{1}{8\kappa}.
\]

Then for every intermediate capacity up to `2 kappa`,

\[
Kp\le\frac14,
\]

and Lemma 2 gives

\[
S_{N,K+1}(r)-S_{N,K}(r)
\ge
\frac34p.
\]

Thus the survival curve has a uniformly positive discrete slope throughout the registered scale window.

---

## 7. Capacity inversion theorem

### Theorem 3 — additive survival precision gives relative capacity precision

Assume:

\[
\kappa\le K\le2\kappa
\]

and

\[
\frac{1}{16\kappa}\le p\le\frac{1}{8\kappa}.
\]

Let an estimator produce `S_hat` satisfying

\[
|\widehat S-S_{N,K}(r)|\le\eta.
\]

Define `K_hat` as the integer in `[kappa,2 kappa]` whose exact survival probability is closest to `S_hat`, with a fixed deterministic tie-break.

Then

\[
|\widehat K-K|
\le
\frac{8\eta}{3p}.
\]

Consequently,

\[
\frac{|\widehat K-K|}{K}
\le
\frac{128}{3}\eta.
\]

Therefore choosing

\[
\boxed{
\eta=\frac{3\epsilon}{128}
}
\]

is sufficient for relative capacity error at most `epsilon`.

#### Proof

For any integer difference `d=|K_hat-K|`, repeated application of the increment lower bound gives

\[
|S_{N,\widehat K}(r)-S_{N,K}(r)|
\ge d\frac34p.
\]

Because `K_hat` minimizes distance to `S_hat`,

\[
|S_{N,\widehat K}(r)-\widehat S|
\le
|S_{N,K}(r)-\widehat S|
\le\eta.
\]

Hence by the triangle inequality,

\[
d\frac34p
\le2\eta,
\]

so

\[
d\le\frac{8\eta}{3p}.
\]

Using `K>=kappa` and `p>=1/(16 kappa)` yields

\[
\frac dK
\le
\frac{128}{3}\eta.
\]

Set `eta=3 epsilon / 128`. QED.

---

## 8. Quantum implementation contour

The estimator can be written as amplitude/probability estimation over a coherent experiment:

1. prepare uniform superpositions over rank-`r` subsets of `[N]` on both sides;
2. reversibly unrank each subset seed into an indexed `r`-element subset;
3. query endpoint records only for the inner pair-detection procedure;
4. mark whether any collision pair exists between the two selected subsets;
5. estimate the marked probability to additive precision `eta = Theta(epsilon)`;
6. invert the exact survival curve.

For the inner `r x r` pair-detection task, the standard quantum-walk pair/claw machinery supplies an **existing baseline** with record-query scale

\[
\widetilde O(r^{2/3})
\]

for the one-or-more-pair decision/search regime in the usual query-cost abstraction.

Because the stable scale has

\[
r=\Theta(N/\sqrt K),
\]

the resulting candidate capacity-estimation contour is

\[
\boxed{
\widetilde O\left(
\frac{1}{\epsilon}
\left(\frac{N^2}{K}\right)^{1/3}
\right)
}
\]

for a constant-factor-correct capacity scale.

### Important distinction from the old proxy

The exponent is **not novel pair-finding complexity**. Its use here is different: it is the cost of the inner yes/no pair predicate inside an outer probability-estimation procedure whose output is an estimate of `K`.

---

## 9. No free compacted list

The fixed-size construction is chosen specifically to eliminate an unresolved assumption in Bernoulli thinning.

A rank-`r` subset can be represented by an integer in

\[
[\binom Nr]
\]

and reversibly unranked using standard combinatorial-number-system arithmetic. This may have significant gate cost, but it requires **no endpoint-record query** merely to define the subset.

The formal resource theorem must still bound:

- reversible subset unranking;
- storage of `r` indices or on-demand indexed access;
- endpoint-record loading;
- pair-predicate evaluation;
- uncomputation;
- quantum-walk data structure and checking gates.

Therefore the current result is a **record-query estimator contour plus a concrete gate-resource obligation**, not an end-to-end fault-tolerant runtime theorem.

---

## 10. Unknown capacity

For unknown `K`, use geometric scale hypotheses

\[
\kappa_s=2^s.
\]

For the true `K`, one scale satisfies

\[
\kappa_s\le K<2\kappa_s.
\]

A naive scan incurs `O(log N)` scale overhead. The registered future algorithmic task is to reduce this with coarse-to-fine or variable-time scale selection while retaining the same endpoint-record interface.

No optimal unknown-`K` overhead is claimed yet.

---

## 11. Relation to approximate counting

Standard quantum approximate counting estimates the number of marked elements of a directly queryable Boolean list. Treating all `N^2` endpoint pairs as independent marked elements would give a Grover-type dependence on the total marked-pair count, not the independent matching capacity, and it loses the structured pair-finding exponent.

The fixed-subset estimator instead uses the **probability that a random compact pair of endpoint subsets contains an independent witness**, and then inverts that probability under the packed/disjoint promise.

A complete novelty audit must still search for prior work on quantum estimation of the number of disjoint marked pairs. Until that audit closes, the paper should describe this as the **QCCS candidate packed-capacity estimator**, not claim priority.

---

## 12. Bounded-overlap status

The exact formula and Theorem 3 assume `K` disjoint marked pairs and no additional marked edges that change the survival event.

For overlap graphs:

- edge count cannot replace `K`;
- the fixed-subset survival probability depends on overlap topology, not only maximum matching size;
- bounded-degree isolation/certification remains required.

Thus no arbitrary-overlap capacity theorem follows from this note.

---

## 13. Current closure status

**PROVED in this note:**

- exact fixed-subset survival formula;
- Bonferroni survival bounds;
- monotonicity;
- discrete slope lower bound in the stable scale window;
- stable inversion from additive survival error to relative capacity error.

**Uses prior-art primitive:**

- quantum pair/claw detection on the compact `r x r` selected domains.

**OPEN:**

- full coherent fixed-subset preparation gate bound;
- arbitrary-predicate checking time/gate cost;
- unknown-`K` optimal scale selection;
- same-record-oracle lower bound for capacity estimation;
- bounded-degree overlap extension;
- fault-tolerant resource crossover.
