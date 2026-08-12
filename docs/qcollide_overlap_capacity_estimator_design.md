# QCCS: Overlap-Aware Collision Capacity Estimator Design

## 1. Target problem

Given a bipartite functional collision graph

\[
G_C=(A,B,E_C), \qquad |A|=|B|=N,
\]

estimate its independent collision capacity

\[
K_C=\nu(G_C)
\]

without reconstructing the full \(N\times N\) collision matrix.

The algorithmic target is **capacity estimation under endpoint-record access**, not exact arbitrary bipartite matching in the standard edge-query model. Standard quantum edge-query BMM and quantum claw finding are prior baselines, not QCCS novelty.

---

## 2. Frozen access model

QCCS assumes coherent access to endpoint records

\[
O_A|i,0\rangle=|i,r^A_i\rangle,
\qquad
O_B|j,0\rangle=|j,r^B_j\rangle,
\]

plus a reversible collision predicate \(C(i,j)\) derived from the frozen functional-collision thresholds.

The current Query-Budget Scaling v2 classical benchmark charges only the number of endpoint records selected:

\[
Q_{\rm endpoint}=2m
\]

when \(m\) endpoints are queried on each side. Its induced-subgraph matching computation is classical post-processing. A theorem-level implementation must additionally account for coherent collision-predicate evaluation and record cleanup.

---

## 3. Why the estimator targets matching rather than edge count

Real collision graphs can contain:

- repeated collisions around one anchor;
- repeated collisions around one candidate;
- hubs;
- dense collision basins;
- nearly disjoint witness families.

Consequently two graphs with equal \(|E_C|\) may have radically different \(K_C\). A star can have arbitrarily many collision edges while its matching number remains one. Therefore \(|E_C|\) is retained only as a negative control.

---

## 4. Registered regimes

### Regime P — packed/disjoint

Maximum degree \(D=1\). This is the exact survival-sketch reference model.

### Regime B — bounded overlap

\[
\Delta(G_C)\le D
\]

for a registered finite degree cap. This is the first theorem-extension target.

### Regime H — heavy overlap / hubs

The estimator must detect that its bounded-overlap certificate is violated. It is acceptable to return `OUT_OF_REGIME`; it is not acceptable to silently output the packed inversion formula.

### Regime U — arbitrary overlap

Experimental stress-test only until a theorem is proved.

---

## 5. Estimator v1 design

### Phase A — multiscale endpoint thinning

Use a geometric scale grid

\[
q_s=2^{-s/2},
\]

clipped to the finite domain, or an equivalent preregistered grid. The purpose is to cover unknown \(K\) without assuming it as input.

At scale \(q_s\), coherently retain endpoint indices with probability \(q_s\). The expected retained endpoint count is \(2Nq_s\).

### Phase B — local overlap certificate

For retained collision witnesses, collect enough information to test whether the theorem regime is plausible. Possible statistics include:

- collision multiplicity around sampled endpoints;
- an upper confidence bound on local degree;
- isolated-witness fraction;
- repeated-endpoint collision rate.

**No free degree oracle is assumed.** Every statistic that requires checking potential neighbors must pay endpoint/predicate resources under the frozen oracle model.

### Phase C — isolated-witness observable

Define an event \(I(i,j)\) that is true only when:

1. \(C(i,j)=1\);
2. both endpoints survive the scale-\(q_s\) thinning;
3. the witness satisfies the registered isolation rule inside the retained domain.

The isolation rule must be chosen so that its survival probability is analyzable as a function of \(K,D,q_s\).

### Phase D — quantum observable estimation

Estimate an isolation/survival statistic

\[
\widehat S(q_s)
\]

for each registered scale. Candidate implementations may combine quantum search/walk primitives and amplitude estimation, but the paper's novelty is the **capacity-specific reduction and oracle accounting**, not these primitives individually.

### Phase E — capacity inversion

Infer \(\widehat K\) from the full multi-scale vector

\[
(\widehat S(q_0),\widehat S(q_1),\ldots).
\]

For \(D=1\), this must reduce to the exact packed formula

\[
S_K(q)=1-(1-q^2)^K.
\]

For \(D>1\), inversion must use a proved bounded-overlap survival envelope rather than the packed equation.

### Phase F — certificate / abstention

Return one of:

- `ESTIMATE(\widehat K, interval)` if the theorem/certificate conditions hold;
- `OUT_OF_REGIME` if overlap statistics exceed the registered domain;
- `INSUFFICIENT_BUDGET` if the requested confidence/accuracy cannot be certified.

Fail-closed abstention is part of the algorithm design, not an implementation failure.

---

## 6. Accuracy contracts

### Additive-capacity mode

\[
|\widehat K-K|\le \epsilon N
\]

with failure probability at most \(\delta\).

This is the preferred general neural-graph contract because it remains defined near \(K=0\).

### Relative-capacity mode

\[
|\widehat K-K|\le \epsilon K
\]

with a declared lower-capacity promise \(K\ge K_{\min}\).

The existing Query-Budget Scaling v2 reports both additive and relative classical success criteria; theorem statements must keep them separate.

---

## 7. Bounded-degree theorem obligations

A valid estimator theorem needs all of the following.

### O1 — isolation expectation

For a fixed maximum matching \(M\), establish

\[
\mathbb E[X_q]=K\,s_D(q)
\]

or a two-sided envelope with explicit \(s_D\).

### O2 — dependency control

Bound variance or provide a concentration inequality for \(X_q\). Shared non-matching edges can correlate isolation events even though the matching edges themselves are vertex-disjoint.

### O3 — inversion stability

Prove that estimation error in \(S(q)\) yields the registered error in \(K\). The condition number of the inversion must appear in the resource bound.

### O4 — unknown-K scale selection

Bound the cost of searching the geometric \(q\) grid. The theorem may hide logarithmic factors but may not assume the optimal \(q\) is given for free.

### O5 — coherent cleanup

Bound ancilla, record-loading, predicate, uncomputation and confidence-amplification resources. A query theorem can separate gate cost, but it must name what is excluded.

---

## 8. Candidate complexity form — OPEN

The intended structural form is

\[
\widetilde O\left(
F(D,\epsilon,\delta)
\left(\frac{N^2}{K}\right)^{1/3}
\right),
\]

where \(F\) is derived from the bounded-degree isolation analysis.

No current proof establishes:

- \(F=D^\alpha/\epsilon\) for any \(\alpha\);
- the same exponent for arbitrary overlap;
- a relative-error guarantee for \(K\) near zero;
- a matching lower bound with identical \(D,\epsilon\) dependence.

All such terms remain **OPEN** until theorem proofs are committed.

---

## 9. Classical same-interface baselines

The estimator benchmark must include:

1. uniform endpoint thinning + exact matching on the induced graph;
2. degree-aware classical thinning whose degree-estimation queries are charged;
3. adaptive endpoint allocation across sides/scales;
4. full endpoint acquisition/reconstruction reference;
5. edge-count scaling as a negative control;
6. known edge-query BMM only as an oracle-model reference unless a reduction equates the interfaces.

The primary plot is **minimum charged queries to hit the same \((\epsilon,\delta)\) target**, not wall-clock runtime from incomparable implementations.

---

## 10. Synthetic validation matrix

For each \(N\), sweep matching density \(K/N\) and overlap degree.

Registered graph families:

- disjoint matching;
- bounded-overlap cyclic/random graph;
- multi-hub/star;
- biclique;
- degree-preserving randomized variants.

Report at least:

- exact \(K\);
- \(|E|\);
- \(D\);
- estimator bias;
- additive and relative error;
- 95% success/failure rate;
- endpoint query count;
- predicate-query count if separately charged;
- isolated-witness count;
- abstention rate.

The star/hub family is mandatory because it makes the distinction between collision mass and independent collision capacity explicit.

---

## 11. Neural-graph validation

After the synthetic theorem regime closes, apply QCCS to saved real neural collision graphs without changing thresholds post hoc. For each neural graph record:

- \(N\);
- exact \(K_C\) from the offline reference graph;
- maximum/quantile degree;
- topology/basin statistics;
- QCCS estimate and interval;
- charged queries;
- whether the bounded-overlap certificate holds.

Graphs outside the theorem regime remain useful empirical stress tests but cannot support a proved quantum-advantage claim.

---

## 12. Claim boundary

QCCS is currently an **algorithm design plus packed-promise executable reference**, with bounded-degree overlap as the next theorem target. The design does not yet constitute a proved general maximum-matching estimator.
