# Q-COLLIDE Adversary / Composition Theorem Program

## 1. Theorem objective

The lower-bound target is the endpoint-record collision-capacity estimation problem, not standard adjacency-matrix matching.

Let

\[
F_{\epsilon}(R_A,R_B)
\]

denote the registered capacity-estimation task on two endpoint-record arrays, with collision graph induced by one fixed reversible predicate \(C\). The output is an estimate of

\[
K_C=\nu(G_C).
\]

We seek a lower bound on the number of calls to record oracles \(O_A,O_B\), and—if the resource contract charges it separately—the collision-predicate oracle \(O_C\).

---

## 2. Prior-art boundary

Three facts constrain the theorem design.

1. Quantum claw finding is established prior art; QCCS cannot treat the claw primitive itself as novel.
2. Quantum algorithms for pair finding with multiple solutions also exist; “there are many collisions” is not by itself a new quantum primitive.
3. Exact maximum-cardinality bipartite matching already has near-tight \(\widetilde O(n^{3/2})\) quantum **edge-query** complexity.

Therefore the theorem contribution must arise from the **record-oracle reduction, capacity-estimation target, structural promises and matching lower bound**, not from renaming known search primitives.

---

## 3. Oracle-normalized decision versions

Before proving an estimation lower bound, define binary promise problems.

### CAP-GAP-PACKED

Given endpoint records inducing a packed collision graph, distinguish

\[
K\le K_0
\qquad\text{from}\qquad
K\ge K_1.
\]

### CAP-GAP-DEGREE-\(D\)

Same task under

\[
\Delta(G_C)\le D.
\]

### CAP-GAP-RELATIVE

Given a scale \(K_0\), distinguish capacities separated by a multiplicative factor, e.g.

\[
K\le K_0
\qquad\text{from}\qquad
K\ge (1+c\epsilon)K_0,
\]

for a fixed constant \(c\) chosen by the reduction.

An \(\epsilon\)-accurate estimator solves an appropriate CAP-GAP problem. This reduction must be written explicitly for additive and relative error separately.

---

## 4. Inner record gadget

The intended composition has two conceptual layers:

\[
\text{hidden hard symbol}
\xrightarrow{g}
\text{endpoint records}
\xrightarrow{C}
\text{collision graph}
\xrightarrow{f}
\text{capacity gap}.
\]

The inner gadget \(g\) must be concrete. For example a hidden symbol can determine a small tuple of control/payload/behavior values whose pairwise predicate creates one collision witness or one bounded-degree gadget.

Required properties:

- records have fixed length/precision;
- one fixed \(C\) is used for every instance;
- single-record marginals do not trivially reveal the output class;
- gadget composition preserves the promised degree;
- total matching capacity is analytically known.

---

## 5. Adversary formulation on record coordinates

Let \(X_0\) and \(X_1\) be valid record-array instances on opposite sides of the capacity-gap promise. Choose an adversary matrix \(\Gamma\) with nonzero entries only between opposite outputs.

For endpoint coordinate \(t\), define

\[
(\Delta_t)_{x,y}=1
\]

iff querying that endpoint returns different records on instances \(x,y\).

Then the negative-weight adversary bound is evaluated on these **record coordinates**:

\[
\frac{\|\Gamma\|}
{\max_t\|\Gamma\circ\Delta_t\|}.
\]

If \(O_C\) is charged independently, the input-variable model must be extended so the proof also bounds predicate access or proves that \(O_C\) is implementable from already loaded records without additional query access.

---

## 6. Composition theorem: no informal multiplication

The previous scaffold informally suggested multiplying “isolation hardness” and “claw hardness”. This is not sufficient.

A valid use of adversary composition requires an explicit partial-function factorization and a theorem whose hypotheses match that factorization. The proof must identify:

1. source variables;
2. inner gadget function \(g\);
3. outer promise function \(f\);
4. whether inputs to different gadgets are disjoint;
5. how degree-overlap constraints couple gadgets;
6. query simulation in both directions.

Only after these are fixed may a composition identity/inequality be invoked.

For bounded-overlap gadgets, the central difficulty is that non-matching edges can couple otherwise independent witness blocks. If those couplings violate block composition, the proof needs a direct adversary construction rather than a product theorem.

---

## 7. Packed theorem milestone

The first complete theorem should be deliberately narrow.

### Theorem P target

For a declared packed endpoint-record promise with \(K\) hidden independent witnesses, any bounded-error algorithm solving the registered CAP-GAP-PACKED problem uses

\[
\Omega\!\left(G(N,K)\right)
\]

record queries, where the goal is to establish

\[
G(N,K)=\left(\frac{N^2}{K}\right)^{1/3}
\]

up to constant/polylog factors under the exact chosen multiple-witness promise.

This scale is a **proof target**, not considered closed merely because the single-claw literature has a related exponent.

Proof sequence:

- [ ] specify record encoding;
- [ ] reduce a known multiple-solution search/collision promise or construct direct adversary;
- [ ] prove capacity is exactly \(K\);
- [ ] prove query preservation;
- [ ] state constant-error lower bound.

---

## 8. Estimation-precision milestone

After Theorem P, derive an estimation lower bound through a capacity gap.

### Additive target

Choose \(K_0,K_1\) such that

\[
K_1-K_0>2\epsilon N.
\]

### Relative target

Choose

\[
K_1-K_0>2\epsilon K_0
\]

with \(K_0\) bounded away from zero.

The hard-instance indistinguishability must strengthen as the gap shrinks if an explicit \(1/\epsilon\)-type factor is to follow. No such factor is currently proved.

Amplitude-estimation upper-bound behavior is not evidence for a lower bound.

---

## 9. Bounded-degree composition milestone

The next theorem introduces \(D\) while preserving a hard packed core.

### Construction target

Replace or decorate each hidden witness with a gadget satisfying:

\[
\Delta(G)\le D,
\qquad
\nu(G)=K,
\]

while ensuring decoy overlap does not reveal the hidden witness locations too cheaply.

### Theorem D target

Prove

\[
Q_{\rm REC}(N,K,D)
\ge
L(N,K,D)
\]

for an explicit \(L\), and compare it to the QCCS bounded-degree upper bound using identical oracle and promise definitions.

No \(D^\alpha\) exponent is frozen in advance.

---

## 10. Edge-query BMM as a reference, not a compositional shortcut

Blikstad et al. show exact BMM can be solved with \(O(n^{3/2}\log^2 n)\) quantum edge queries and give a near-matching \(\Omega(n^{3/2})\) lower-bound regime for the standard problem.

This result must appear in related work and baselines because it prevents novelty overclaiming. However:

- an edge query returns one adjacency bit for a chosen pair;
- a QCCS record query returns an endpoint record potentially relevant to many pairs;
- coherent evaluation of the neural collision predicate has a separate representation/gate cost.

Therefore a same-oracle theorem requires an explicit reduction. We cannot cite the edge-query \(n^{3/2}\) lower bound as if it directly proves an endpoint-record lower bound.

---

## 11. Empirical theorem diagnostics

Synthetic experiments are used to falsify proof assumptions, not to prove the theorem.

Required families:

- packed matching;
- bounded overlap;
- multi-hub/star;
- biclique;
- randomized degree-preserving controls.

Required diagnostics:

- exact \(K\);
- exact/max degree;
- estimator bias/variance;
- isolated-witness survival;
- endpoint query count;
- predicate count if applicable;
- failure/abstention probability.

If an estimator behaves well on an unproved overlap family, report it as empirical only.

---

## 12. Formal theorem-closure gate

The quantum theory section is considered closed only when the repository contains all of:

1. **Oracle Definition.** Formal \(O_A,O_B,O_C\) and cost model.
2. **Packed Upper Bound.** Full QCCS packed estimator guarantee with unknown-\(K\) scale selection and confidence.
3. **Packed Lower Bound.** Same-record-oracle CAP-GAP lower bound.
4. **Bounded-Degree Isolation Lemma.** Explicit survival and dependency bound.
5. **Bounded-Degree Upper Bound.** Capacity-estimation theorem with explicit \(D,\epsilon,\delta\) dependence.
6. **Matching Lower Bound or Gap Statement.** Either a same-parameter lower bound, or an explicit theorem gap retained as open.
7. **Cleanup/Resource Lemma.** Ancilla, uncomputation, predicate-evaluation and record-loading resources.
8. **Same-Interface Experiments.** Classical and analytic quantum comparisons under the frozen access model.

Until all relevant items close, the paper must not call the arbitrary-overlap QCCS estimator optimal.

---

## 13. Current status

**Established / executable:**

- exact neural collision capacity via offline bipartite matching;
- packed survival-sketch identity;
- same-endpoint-budget classical synthetic scaling experiment;
- analytic packed QCCS query contour;
- overlap stress families showing edge-count proxy failure.

**OPEN:**

- formal endpoint-record packed lower bound with the desired multiple-witness scaling;
- epsilon-dependent lower bound;
- bounded-degree isolation concentration;
- bounded-degree QCCS theorem;
- matched degree-dependent lower bound;
- coherent neural-record construction cost and end-to-end fault-tolerant resources.
