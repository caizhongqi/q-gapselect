# Q-COLLIDE Overlap-Aware Capacity Extension

## 0. Scope correction: what is and is not new

The target quantity remains the maximum matching capacity of a functional collision graph,

\[
K_C(G)=\nu(G), \qquad G=(A\sqcup B,E_C).
\]

Q-COLLIDE must **not** claim to introduce the first quantum algorithm for bipartite maximum matching. In the standard quantum **edge-query** model, exact maximum-cardinality bipartite matching is already known with near-tight \(\widetilde O(n^{3/2})\) query complexity (Blikstad, van den Brand, Efron, Mukhopadhyay, Nanongkai, FOCS 2022, arXiv:2208.02526). Optimal quantum claw-finding algorithms are also classical literature (Tani, TCS 2009, arXiv:0708.2584), and multiple-solution pair-finding has been studied beyond the single-claw setting.

The admissible Q-COLLIDE novelty target is therefore narrower and structurally different:

> **Estimate collision capacity under a neural endpoint-record / collision-predicate oracle, without reconstructing the full collision graph, with complexity that adapts to capacity and overlap structure.**

The theorem must explicitly distinguish this oracle from an adjacency-matrix edge query.

---

## 1. Oracle contract

Let \(|A|=|B|=N\). An endpoint record contains the information needed by the frozen functional-collision predicate, for example control coordinates, non-control payload coordinates or a reversible handle to them, and behavior coordinates.

### 1.1 Endpoint-record access

The intended coherent record oracles are

\[
O_A:|i\rangle|0\rangle\mapsto |i\rangle|r^A_i\rangle,
\qquad
O_B:|j\rangle|0\rangle\mapsto |j\rangle|r^B_j\rangle.
\]

A classical endpoint query reveals one record. The executable Query-Budget Scaling v2 benchmark queries \(m\) records from each side and charges

\[
Q_{\rm endpoint}=2m.
\]

### 1.2 Collision-predicate access

A coherent algorithm additionally needs a reversible predicate

\[
O_C:|r^A_i,r^B_j,z\rangle
\mapsto
|r^A_i,r^B_j,z\oplus C(i,j)\rangle,
\]

where \(C(i,j)=1\) iff the frozen control-close, payload-separated and behavior-divergent conditions are all satisfied.

**Current executable caveat.** Query-Budget Scaling v2 measures endpoint-record queries only; after retaining endpoints, induced-subgraph matching and edge-count calculations are classical post-processing and pair-comparison cost is not separately charged. A formal quantum theorem must therefore use one of two resource contracts:

1. **record-query theorem:** charge only calls to \(O_A,O_B\), while explicitly bounding reversible evaluation of \(C\) as arithmetic/resource overhead; or
2. **hybrid theorem:** charge \(Q_{\rm rec}+\lambda Q_C\), with a declared relative cost \(\lambda\).

Until this is frozen, an endpoint-record complexity must not be compared numerically to the standard edge-query matching bound as if the queries were identical.

---

## 2. Estimation problem

### Problem: QCCS-REC

Input:

- coherent endpoint-record access to \(A,B\);
- the frozen collision predicate \(C\);
- confidence \(1-\delta\);
- either an additive or relative accuracy target.

Output an estimate \(\widehat K\) of

\[
K=\nu(G_C).
\]

Two admissible guarantees are kept separate:

### Additive form

\[
|\widehat K-K|\le \epsilon N.
\]

This remains meaningful when \(K\) is small.

### Relative form

\[
|\widehat K-K|\le \epsilon K,
\]

under an explicit promise \(K\ge K_{\min}>0\).

The experiment and theorem must never silently switch between these two targets.

---

## 3. Packed/disjoint promise: established internal reference point

For \(K\) vertex-disjoint collision witnesses, retaining each endpoint independently with probability \(q\) yields exact witness survival probability

\[
S_K(q)=1-(1-q^2)^K.
\]

Hence

\[
K=\frac{\log(1-S_K(q))}{\log(1-q^2)}.
\]

Choosing \(q=\Theta(K^{-1/2})\) leaves an effective retained domain of order \(N/\sqrt K\). The current QCCS packed contour

\[
\widetilde O\!\left(\epsilon^{-1}(N^2/K)^{1/3}\right)
\]

is retained as an **analytic promise-model target**, not as an arbitrary-overlap theorem and not as a hardware runtime measurement.

---

## 4. Why overlap changes the problem

When collision edges share endpoints,

\[
|E_C|\not\asymp K.
\]

The extreme star family has many edges and \(K=1\). Dense bicliques similarly decouple edge count from independent witness capacity. Therefore an overlap-aware estimator must preserve information about independent witnesses, not merely collision mass.

Let

\[
\Delta(G)=\max_{v\in A\cup B} d(v).
\]

The first defensible extension regime is

\[
\Delta(G)\le D,
\]

with \(D\) known or independently upper-bounded.

---

## 5. Bounded-degree isolation theorem target

Fix a maximum matching \(M\) of size \(K\). Independently retain vertices with a degree-aware or uniform thinning rule. The proof objective is not merely to preserve edges; it is to produce a controlled number of **isolated matching witnesses** whose survival statistics can be inverted.

### Lemma target A — one-witness survival

For each \(e=(u,v)\in M\), derive an explicit lower and upper bound

\[
a_D(q)\le
\Pr[e\text{ survives and is isolated}]
\le b_D(q).
\]

A crude candidate scale for uniform thinning is

\[
q^2(1-q)^{O(D)},
\]

but the exponent and neighborhood event must be proved from the actual isolation definition. This expression is **not yet a lemma**.

### Lemma target B — aggregate isolated-witness concentration

If \(X\) is the number of isolated witnesses obtained from a fixed maximum matching, prove a concentration or second-moment statement sufficient to distinguish different values of \(K\):

\[
\mathbb E[X]=\Theta(K a_D(q)),
\]

plus a variance/dependency bound that remains useful for the registered \(D\) regime.

Linearity of expectation alone is insufficient because witness-isolation events can be dependent through non-matching edges.

### Lemma target C — observable reduction

Prove that the statistic used by the quantum routine can be implemented from \(O_A,O_B,O_C\) without reconstructing all \(N^2\) candidate pairs.

This is the most important oracle-level obligation. A graph-theoretic isolation lemma that assumes an explicit adjacency list does not by itself imply a sublinear endpoint-record algorithm.

---

## 6. QCCS bounded-overlap algorithm skeleton

1. **Scale selection.** Use a geometric grid of retention scales \(q_0,q_1,\ldots\) rather than assuming \(K\) is known.
2. **Optional degree stratification.** Use only a degree sketch whose own endpoint/predicate cost is explicitly charged. No free degree oracle is assumed.
3. **Coherent thinning.** Prepare the retained endpoint domain at a chosen scale.
4. **Isolated-witness predicate.** Reversibly certify the registered isolation event.
5. **Survival-probability estimation.** Estimate the probability or count associated with isolated collision witnesses.
6. **Capacity inversion.** Convert the multi-scale survival profile into \(\widehat K\), with confidence amplification.
7. **Fail closed on heavy overlap.** If the degree/variance certificate exceeds the theorem regime, return an out-of-regime certificate rather than an unsupported estimate.

---

## 7. Upper-bound theorem target

A publishable bounded-degree theorem would have the form:

> For every bipartite functional collision graph with \(|A|=|B|=N\), maximum degree at most \(D\), and matching capacity \(K\), QCCS-REC returns \(\widehat K\) satisfying the registered accuracy guarantee with probability at least \(1-\delta\), using
> \[
> \widetilde O\!\left(
> F(D,\epsilon,\delta)
> (N^2/K)^{1/3}
> \right)
> \]
> endpoint/predicate resources under the frozen oracle contract.

At present:

- the function \(F\) is **OPEN**;
- no exponent \(D^\alpha\) is proved;
- the \(1/\epsilon\) dependence outside the packed survival model is **OPEN**;
- the theorem is not valid for arbitrary unbounded-degree graphs.

---

## 8. Required baselines under matched information

Every claimed query advantage must compare under the same oracle contract and accuracy target:

1. uniform endpoint thinning;
2. adaptive/degree-aware classical endpoint thinning;
3. exact reconstruction of all returned endpoint records plus exact matching;
4. edge-count proxy as a negative control, not a competitive capacity estimator;
5. known quantum claw/pair-finding baselines in promise regimes where their input model matches;
6. standard quantum edge-query BMM only as an **oracle-model reference**, unless an explicit reduction makes the interfaces comparable.

---

## 9. Synthetic theorem-test families

The required matrix is:

- disjoint matching (\(D=1\));
- bounded-degree cyclic/random overlap;
- multi-hub/star;
- biclique/dense overlap;
- degree-preserving randomized controls.

For each family vary \(N,K,D\) independently where possible and report:

- additive/relative error;
- failure probability;
- endpoint queries;
- collision-predicate calls if charged;
- retained endpoints;
- isolated witness count;
- exact \(K\);
- edge count.

Hub/star is mandatory because it certifies that edge mass cannot substitute for maximum matching capacity.

---

## 10. Claim boundary

Current admissible statement:

> Q-COLLIDE provides an executable packed-promise capacity sketch and a bounded-overlap theorem program under an endpoint-record collision oracle. The arbitrary-overlap maximum-matching problem remains outside the proved QCCS regime.

Current inadmissible statements:

- first quantum bipartite matching algorithm;
- first quantum claw/pair-finding algorithm;
- proved arbitrary-overlap \((N^2/K)^{1/3}\) estimator;
- proved \(D^\alpha\) dependence;
- proved \(\epsilon^{-1}\) lower bound under the endpoint-record oracle;
- end-to-end quantum runtime advantage with free neural inference or free QRAM.
