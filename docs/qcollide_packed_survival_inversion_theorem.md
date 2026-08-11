# Q-COLLIDE Packed Collision Survival Inversion Theorem

## 1. Purpose

This note closes one narrow but exact piece of the QCCS theory: **how accurately the number of disjoint collision witnesses can be recovered from a survival-probability estimate at a constant-factor correct thinning scale.**

It does **not** by itself close the end-to-end quantum algorithm. In particular, coherent construction/indexing of the retained endpoint domain, collision-predicate resources, unknown-scale search and the bounded-overlap extension remain separate obligations.

---

## 2. Packed promise

Assume a bipartite collision instance contains exactly `K >= 1` vertex-disjoint functional collision witnesses and no additional collision edges relevant to the promise instance.

Retain each endpoint independently with probability `q`. A witness survives iff both of its endpoints survive, hence independently across the `K` disjoint witnesses with probability `q^2`.

Therefore the probability of observing at least one surviving witness is exactly

\[
S_K(q)=1-(1-q^2)^K.
\]

For `0 < q < 1`, the inverse map is

\[
K=\Phi_q(S)
:=
\frac{\log(1-S)}{\log(1-q^2)}.
\]

This identity is exact for the packed promise.

---

## 3. Constant-factor scale theorem

### Theorem 1 — survival probability stays away from 0 and 1

Let `kappa >= 1` and suppose

\[
\kappa \le K \le 2\kappa.
\]

Choose

\[
q^2=\frac{1}{2\kappa}.
\]

Then

\[
1-(1-1/(2\kappa))^{\kappa}
\le S_K(q)
\le
1-(1-1/(2\kappa))^{2\kappa}.
\]

In particular, for every `kappa >= 1`,

\[
1-e^{-1/2}
\le S_K(q)
\le \frac34.
\]

Thus

\[
0.3934\ldots \le S_K(q)\le 0.75.
\]

#### Proof

`S_K(q)` is increasing in `K`, so the first two-sided inequality follows from the promise `kappa <= K <= 2 kappa`.

For the lower bound, use `1-x <= e^{-x}`:

\[
(1-1/(2\kappa))^{\kappa}\le e^{-1/2}.
\]

For the upper bound on `S`, it suffices to lower-bound the no-survival probability. The sequence

\[
(1-1/(2\kappa))^{2\kappa}
\]

is at least its `kappa=1` value `1/4`, hence `S <= 3/4`. QED.

---

## 4. Stable inversion theorem

### Theorem 2 — additive survival error gives relative capacity error

Under the conditions of Theorem 1, let `S=S_K(q)` and suppose an estimator returns `S_hat` with

\[
|\widehat S-S|\le \eta,
\qquad
0<\eta\le 0.05.
\]

Define

\[
\widehat K
=
\frac{\log(1-\widehat S)}{\log(1-q^2)}.
\]

Then

\[
\frac{|\widehat K-K|}{K}
\le 10\eta.
\]

Consequently, for any `0 < epsilon <= 0.4`, estimating survival probability to

\[
\eta=\epsilon/16
\]

is sufficient to guarantee

\[
|\widehat K-K|\le \epsilon K.
\]

#### Proof

From Theorem 1,

\[
1-S\ge 1/4.
\]

Because `eta <= 0.05`, every point between `S` and `S_hat` satisfies

\[
1-x\ge 0.20.
\]

The derivative of `log(1-x)` has magnitude `1/(1-x)`, so by the mean-value theorem,

\[
|\log(1-\widehat S)-\log(1-S)|
\le 5\eta.
\]

Also, for `0<x<1`, `-log(1-x) >= x`. With `q^2=1/(2 kappa)`,

\[
|\log(1-q^2)|
\ge \frac{1}{2\kappa}.
\]

Therefore

\[
|\widehat K-K|
\le 10\kappa\eta.
\]

Since `K >= kappa`,

\[
\frac{|\widehat K-K|}{K}\le 10\eta.
\]

For `eta=epsilon/16`, the relative error is at most `(10/16) epsilon < epsilon`; the condition `epsilon <= 0.4` ensures `eta <= 0.025 < 0.05`. QED.

---

## 5. Probability-estimation corollary

### Corollary 3 — outer estimation cost

Suppose there exists a coherent bounded-error procedure `D_q` that marks whether the retained packed instance contains at least one surviving collision witness. If a standard quantum probability/amplitude-estimation routine estimates its success probability to additive error `eta` and failure probability at most `delta` using

\[
O\!\left(\eta^{-1}\log(1/\delta)\right)
\]

controlled calls to `D_q` and `D_q^{-1}` (up to the chosen estimator's standard polylog/constant factors), then under the constant-factor capacity promise

\[
\kappa\le K\le2\kappa
\]

QCCS obtains relative capacity error `epsilon` using

\[
O\!\left(\epsilon^{-1}\log(1/\delta)\right)
\]

calls to the retained-witness detector.

This corollary concerns the **outer survival-probability estimation layer only**. It does not state the cost of one `D_q` call in the endpoint-record oracle.

---

## 6. Conditional claw-detection contour

At the scale of Theorem 1,

\[
q=\Theta(\kappa^{-1/2}).
\]

If the retained left/right endpoint sets can be exposed through a compact indexed domain of size

\[
M=\Theta(Nq)=\Theta(N/\sqrt\kappa)
\]

with no larger asymptotic record-loading overhead, then a standard equal-domain quantum claw detector has structural query scale

\[
\widetilde O(M^{2/3})
=
\widetilde O\!\left((N^2/\kappa)^{1/3}\right).
\]

Combining this detector scale with Corollary 3 gives the familiar **conditional packed contour**

\[
\widetilde O\!\left(
\epsilon^{-1}
(N^2/K)^{1/3}
\log(1/\delta)
\right)
\]

when `kappa` is a constant-factor estimate of `K`.

### Critical claim boundary

The word **conditional** is essential. A complete endpoint-record theorem still has to prove that:

1. Bernoulli-thinned endpoints can be coherently indexed/prepared at the claimed cost;
2. record loading and uncomputation are charged consistently;
3. the functional collision predicate is implemented reversibly with declared resource cost;
4. the detector used inside amplitude estimation is a valid coherent subroutine under that representation;
5. no QRAM or retained-list construction is silently treated as free.

Until those are proved, the expression above is an analytic query contour, not an end-to-end theorem.

---

## 7. Unknown-K scale search

Theorems 1–2 assume a constant-factor scale `kappa` is available. For unknown `K`, the registered plan is a geometric family

\[
\kappa_s=2^s,
\qquad s=0,1,\ldots,\lceil\log_2 N\rceil.
\]

At least one scale satisfies

\[
\kappa_s\le K<2\kappa_s.
\]

A complete algorithm must provide a bounded-error rule for finding/validating such a scale without spending the full high-precision estimation budget at every level. A naive scan contributes at most an additional `O(log N)` detector-scale factor, but a sharper variable-time or coarse-to-fine schedule is **OPEN** until formally analyzed.

Therefore the current theorem closure is:

- exact packed survival identity: **PROVED**;
- constant-factor scale survival window: **PROVED**;
- stable inversion with relative error controlled by additive survival error: **PROVED**;
- `O(1/epsilon)` outer probability-estimation call dependence: **conditional on the chosen coherent estimator, standard**;
- compact thinned-domain detector cost: **OPEN as an endpoint-record resource theorem**;
- unknown-K optimal scale-selection overhead: **OPEN**;
- bounded-overlap extension: **OPEN**.

---

## 8. Relation to prior art

The theorem does not claim novelty for amplitude estimation or claw finding. Those are standard primitives. The candidate Q-COLLIDE contribution is the collision-capacity survival reduction under the neural endpoint-record oracle, its stable capacity inversion, the scale-adaptive construction, and the bounded-overlap extension if the latter is proved.

Relevant baselines that must remain in related work include:

- Tani, *Claw Finding Algorithms Using Quantum Walk*, TCS 2009 / arXiv:0708.2584;
- multiple-solution pair-finding work such as Allcock et al., arXiv:2111.07059;
- Blikstad et al., *Nearly Optimal Communication and Query Complexity of Bipartite Matching*, FOCS 2022 / arXiv:2208.02526 for standard edge-query BMM.

---

## 9. Next proof obligation

The next theorem work should **not** be another survival-inversion calculation. It should prove the missing endpoint-record implementation lemma:

> Given coherent record access and Bernoulli thinning at rate `q`, implement the retained-witness detector with an explicit record-query, predicate-query, ancilla and cleanup bound that recovers the `M^{2/3}` structural contour without assuming a free compact retained list.

That lemma is the bridge between this now-closed statistical inversion result and a genuine QCCS endpoint-record upper bound.
