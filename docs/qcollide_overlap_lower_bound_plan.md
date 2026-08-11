# Q-COLLIDE Overlap Collision-Capacity Lower-Bound Program

## 1. Scope

We seek lower bounds for estimating

\[
K_C=\nu(G_C)
\]

under the **same endpoint-record / collision-predicate oracle** used by QCCS.

This is not the standard adjacency-matrix edge-query problem. Exact bipartite maximum matching in the quantum edge-query model already has a near-tight \(\widetilde\Theta(n^{3/2})\) characterization. That result is a required baseline but cannot be copied as a lower bound for endpoint-record QCCS without an explicit oracle reduction.

---

## 2. Oracle models must not be conflated

### Edge-query BMM

One query selects a pair \((i,j)\) and returns whether \((i,j)\in E\).

### QCCS endpoint-record model

A query loads an endpoint record:

\[
O_A|i,0\rangle=|i,r_i^A\rangle,
\qquad
O_B|j,0\rangle=|j,r_j^B\rangle.
\]

The collision predicate is then evaluated reversibly from two records, or charged as a separate \(O_C\) call under a hybrid accounting model.

An endpoint record can encode information relevant to many potential pair relations. Conversely, evaluating all pair relations among loaded records may have nontrivial gate/predicate cost. Therefore neither oracle dominates the other for free.

**Lower-bound rule:** every reduction must state exactly how one source-oracle query simulates the target oracle and with what cost.

---

## 3. Lower-bound layers

The proof program is split into claims that can close independently.

### LB-P — packed/disjoint structural lower bound

Restrict to instances consisting of \(K\) disjoint collision witnesses hidden among \(N\) endpoints per side. Reduce from an appropriate claw/pair-finding or collision-detection promise under a record oracle.

Target structural scale:

\[
\Omega\!\left((N^2/K)^{1/3}\right)
\]

for constant success probability, **only after the multiple-witness reduction is written under the exact record oracle**.

The existence of optimal claw-finding algorithms in the literature does not by itself prove this exact multiple-witness lower bound for our record encoding.

### LB-E — estimation lower bound

Construct two capacity distributions

\[
\mathcal D_0:K=K_0,
\qquad
\mathcal D_1:K=K_1,
\]

where \(|K_1-K_0|\) equals the registered additive or relative estimation gap. Show that endpoint-record transcripts remain hard to distinguish below \(Q\) queries.

This layer is where any \(\epsilon\) dependence must be proved.

### LB-D — bounded-degree overlap lower bound

Embed the packed hard family into graphs with maximum degree at most \(D\), while preserving the capacity gap and preventing a record query from revealing many hard bits at once.

A theorem may yield a \(D\)-dependent lower bound, but the sign and exponent of that dependence must follow from the reduction; it must not be guessed in advance.

---

## 4. Hard-instance record construction

A valid hard family must specify actual endpoint records, not merely adjacency matrices.

For each endpoint record define fields sufficient for the frozen predicate, for example

\[
r=(c,p,g,\text{aux}),
\]

such that

\[
C(r_i^A,r_j^B)=1
\]

produces the desired collision graph.

The construction must satisfy four properties:

1. **realizability:** the records generate the intended graph under one fixed collision predicate;
2. **balanced marginals:** individual endpoint records do not trivially reveal whether the instance came from \(\mathcal D_0\) or \(\mathcal D_1\);
3. **capacity gap:** \(\nu(G_0)\) and \(\nu(G_1)\) differ by the registered amount;
4. **bounded information per query:** one endpoint query cannot decode the complete hidden matching structure.

Without a concrete record construction, an adjacency-graph adversary matrix is not yet a lower bound for QCCS-REC.

---

## 5. Candidate hard families

### H1 — hidden packed matching

Embed a random matching of size \(K\) in otherwise non-colliding endpoints. This is the cleanest bridge to claw/pair-finding lower bounds.

### H2 — packed matching with dummy overlap

Add bounded-degree decoy collision edges that do not change the maximum matching. This tests whether overlap can leak the hidden packing.

### H3 — local replacement gadget

Replace each packed witness by a constant- or \(D\)-size bounded-degree gadget with one unit of matching capacity. The gadget should preserve indistinguishability while controlling \(\Delta(G)\).

### H4 — hub negative control

Hub/star graphs are essential experimentally but may be poor lower-bound instances because one queried hub record can expose atypical structure. They should not be assumed hard without proof.

---

## 6. Adversary-method route

Let \(X\) and \(Y\) be two sets of record-encoded instances with different capacity values. Construct a negative-weight adversary matrix \(\Gamma\) indexed by \(X\cup Y\).

For each **record-query coordinate** \(t\), define \(\Delta_t\) to indicate pairs of instances whose returned record differs at coordinate \(t\). The standard adversary objective has the form

\[
\mathrm{Adv}^{\pm}(F)
=
\max_{\Gamma}
\frac{\|\Gamma\|}{\max_t\|\Gamma\circ\Delta_t\|},
\]

subject to the promise/output constraints.

The key obligation is that \(t\) indexes the actual QCCS record oracle. An adversary matrix over adjacency bits would prove an edge-query bound, not automatically a record-query bound.

---

## 7. Composition: what must actually be proved

It is tempting to write

\[
\mathrm{Adv}^{\pm}(f\circ g)
\approx
\mathrm{Adv}^{\pm}(f)\mathrm{Adv}^{\pm}(g),
\]

but this cannot be used until the QCCS problem has been factored into explicit partial functions with compatible oracle variables.

A valid composition proof must provide:

1. the inner record-encoding function \(g\);
2. the outer capacity-gap decision/estimation function \(f\);
3. a promise set closed under the reduction;
4. a query-preserving simulation;
5. the exact adversary composition theorem applicable to those partial functions.

Until then, “isolation hardness × claw hardness × estimation hardness” is a proof plan, not a theorem.

---

## 8. Epsilon dependence — OPEN

The desired estimation theorem may target either

\[
|\widehat K-K|\le \epsilon N
\]

or, under \(K\ge K_{\min}\),

\[
|\widehat K-K|\le \epsilon K.
\]

A factor \(\Omega(1/\epsilon)\) must **not** be inserted solely because amplitude estimation has such a familiar dependence in some regimes. The lower bound must come from an explicit family whose capacity values separated by the required epsilon gap remain query-indistinguishable.

Possible routes to test:

- reduction from approximate counting;
- polynomial/approximate-degree lower bound for a packed-capacity promise;
- adversary direct-sum/composition across independent witness blocks.

All are currently **OPEN** under the QCCS record oracle.

---

## 9. Degree dependence — OPEN

Do not preregister a formula

\[
D^{-\alpha}(N^2/K)^{1/3}
\]

without a proof. Increasing degree can make an instance either more redundant or easier to detect, depending on record encoding. The bounded-degree lower-bound theorem should instead be stated abstractly as

\[
Q_{\rm REC}(N,K,D,\epsilon,\delta)
\ge L(N,K,D,\epsilon,\delta),
\]

with \(L\) derived from the hard gadget family.

The upper and lower bounds are considered matched only when they use the same \(N,K,D,\epsilon,\delta\) promises and exactly the same oracle contract.

---

## 10. Relation to known baselines

Required literature anchors:

- Tani, *Claw Finding Algorithms Using Quantum Walk*, TCS 2009 / arXiv:0708.2584 — optimal quantum claw-finding baseline.
- Blikstad et al., *Nearly Optimal Communication and Query Complexity of Bipartite Matching*, FOCS 2022 / arXiv:2208.02526 — near-tight exact BMM complexity in the quantum edge-query model.
- Multiple-solution pair-finding constructions, including Allcock et al., arXiv:2111.07059, must be treated as prior art for quantum searches with many solutions.

These results constrain novelty but do not replace a same-record-oracle lower bound.

---

## 11. Proof-completion checklist

A lower-bound claim is publishable only after all checked items are true:

- [ ] explicit endpoint record schema;
- [ ] one fixed collision predicate for both hard distributions;
- [ ] exact capacity values proved;
- [ ] oracle simulation cost proved;
- [ ] constant-error structural lower bound proved;
- [ ] epsilon dependence proved separately if claimed;
- [ ] bounded-degree extension proved if claimed;
- [ ] matching upper bound uses same promises/oracle;
- [ ] resource model states whether collision-predicate gates are charged;
- [ ] synthetic experiments instantiate the same hard regimes.

---

## 12. Current claim boundary

Currently supported:

> the packed/disjoint family provides an executable analytic QCCS contour and a concrete lower-bound reduction target.

Currently **not** proved:

- arbitrary-overlap lower bound;
- bounded-degree matching lower bound;
- \(\Omega(\epsilon^{-1}(N^2/K)^{1/3})\) under endpoint-record access;
- optimal \(D\) dependence;
- equivalence between endpoint-record and adjacency edge-query complexity.
