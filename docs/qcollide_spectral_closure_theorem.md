# Spectral closure and adaptive tunnel rotation

## Scope

This note formalizes the theorem that is supported by the Q-COLLIDE closure-spectrum implementation. The fixed-candidate spectral statement is exact. The adaptive statement is a deterministic perturbation bound. Neither statement is a new quantum lower bound, and the singular-value minimization step is an application of the classical Eckart--Young theorem.

## Setup

Let the current hidden control row space be `row(P)` and let

\[
D=[d_1,\ldots,d_s]\in\mathbb R^{h\times s}
\]

contain hidden displacements of `s` independently indexed collision-tunnel candidates. Each displacement is first projected into the current control null space, so

\[
P d_i=0.
\]

A rank-`k` control augmentation is represented by a matrix

\[
U\in\mathbb R^{h\times k},\qquad U^\top U=I,\qquad P U=0.
\]

After adding `U` to the control system, the residual non-control displacement is

\[
\bar d_i(U)=(I-UU^\top)d_i,
\]

and the total residual dangerous energy is

\[
\mathcal E(U;D)=\|(I-UU^\top)D\|_F^2.
\]

## Proposition 1: optimal fixed-candidate spectral closure

Let

\[
D=L\Sigma R^\top,
\qquad
\sigma_1\ge\sigma_2\ge\cdots\ge0
\]

be a singular-value decomposition. Among all admissible rank-`k` augmentations, the choice

\[
U_k=L_{[:,1:k]}
\]

minimizes the residual dangerous energy, and

\[
\min_U\mathcal E(U;D)
=
\sum_{j>k}\sigma_j^2.
\]

### Proof

Because all columns of `D` lie in `ker(P)`, the left singular vectors with non-zero singular values also lie in `ker(P)` and are admissible control directions. Minimizing

\[
\|(I-UU^\top)D\|_F^2
\]

is equivalent to finding the best rank-`k` orthogonal projection of `D`. The Eckart--Young theorem gives the leading left singular vectors and the tail-energy expression. `QED`.

## Corollary 1: fixed-candidate packing bound

Assume that a candidate can remain an effective collision tunnel only when its residual non-control displacement satisfies

\[
\|\bar d_i(U)\|\ge\Delta.
\]

Let `m(U)` be the number of candidate attack endpoints satisfying that necessary condition, and let `nu(U)` be the maximum matching of the resulting collision graph. Then

\[
\nu(U)\le m(U)
\le
\frac{\mathcal E(U;D)}{\Delta^2}.
\]

Consequently, the leading-singular-vector augmentation minimizes this deterministic upper bound among equal-rank linear augmentations:

\[
\nu(U_k)
\le
\frac{\sum_{j>k}\sigma_j^2}{\Delta^2}.
\]

### Proof

For every surviving endpoint,

\[
\|\bar d_i(U)\|^2\ge\Delta^2.
\]

Summing over the surviving endpoints gives

\[
m(U)\Delta^2
\le
\sum_i\|\bar d_i(U)\|^2
=
\mathcal E(U;D).
\]

A bipartite matching cannot use more distinct attack endpoints than the number of surviving attack endpoints, so `nu(U) <= m(U)`. `QED`.

This bound is necessary-condition based and may be loose because a valid edge must additionally satisfy control distance, behaviour divergence, input validity, and anchor compatibility.

## Proposition 2: adaptive rotation penalty

The previous result controls fixed candidates. Let

\[
\widetilde D(U)
\]

be the displacement matrix produced after the attacker re-optimizes against the augmented control system. Define the adaptive rotation magnitude

\[
\rho(U)=\|\widetilde D(U)-D\|_F.
\]

Then

\[
\|(I-UU^\top)\widetilde D(U)\|_F
\le
\sqrt{\mathcal E(U;D)}+\rho(U),
\]

and the same necessary residual threshold yields

\[
\nu_{\mathrm{adaptive}}(U)
\le
\frac{
\left(
\sqrt{\mathcal E(U;D)}+\rho(U)
\right)^2
}{\Delta^2}.
\]

### Proof

Orthogonal projection is non-expansive. Therefore,

\[
\begin{aligned}
\|(I-UU^\top)\widetilde D(U)\|_F
&\le
\|(I-UU^\top)D\|_F\\
&\quad+
\|(I-UU^\top)(\widetilde D(U)-D)\|_F\\
&\le
\sqrt{\mathcal E(U;D)}+\rho(U).
\end{aligned}
\]

Apply the endpoint-counting argument from Corollary 1 to the adaptive displacement matrix. `QED`.

## New measurable quantities

The normalized fixed-candidate residual spectrum is

\[
\mathcal R_k
=
\frac{\sum_{j>k}\sigma_j^2}{\sum_j\sigma_j^2}.
\]

The static 95-percent dangerous rank is

\[
r_{95}^{\mathrm{static}}
=
\min\left\{
 k:
 \frac{\sum_{j\le k}\sigma_j^2}{\sum_j\sigma_j^2}
 \ge0.95
\right\}.
\]

For a fixed packing threshold `tau`, define the adaptive closure rank

\[
k_\tau^{\mathrm{adaptive}}
=
\min\{k:\nu_{\mathrm{adaptive}}(U_k)/s\le\tau\}.
\]

The adaptive spectral inflation is

\[
\boxed{
I_\tau
=
k_\tau^{\mathrm{adaptive}}
-r_{95}^{\mathrm{static}}
}.
\]

A positive value means that a static spectrum underestimates the number of control directions required after attack re-optimization.

## Executed synthetic result

Across ten trained dual-head models, the baseline dangerous-displacement spectrum had

\[
r_{95}^{\mathrm{static}}=1
\]

at every non-full visible control rank. However, at packing threshold `tau=0.05`, the mean adaptive closure ranks were:

| visible control rank | mean adaptive closure rank | mean spectral inflation |
|---:|---:|---:|
| 1 | 3.2 | 2.2 |
| 2 | 1.7 | 0.7 |
| 4 | 1.0 | 0.0 |
| 6 | 1.0 | 0.0 |

The targeted augmentation minimized baseline residual dangerous energy in every equal-rank comparison, as required by Proposition 1. Nevertheless, baseline residual energy correlated with adaptive packing only at `0.596`, whereas post-intervention openness correlated at `0.899`. The result is consistent with adaptive tunnel rotation: after the dominant static mode is controlled, new attack directions appear inside the remaining control null space.

## Connection to Q-COLLIDE search complexity

Under the previously frozen random-packed, endpoint-local query model, the analytic search proxies are

\[
R_C(k)
=\Theta\left(\frac{N}{\sqrt{\nu_k}}\right),
\qquad
Q_Q(k)
=\widetilde O\left(\left(\frac{N^2}{\nu_k}\right)^{1/3}\right).
\]

The closure spectrum therefore changes attack-search hardness through its effect on `nu_k`. These expressions remain query-complexity proxies. The repository does not infer a hardware runtime advantage from them, and the matching lower bound for weighted prefix-conditional claw finding remains open.

## Next theorem boundary

The remaining mathematical target is not another use of Eckart--Young. It is a stability or adversarial-rotation theorem that controls `rho(U)` from properties of the neural control map, for example Jacobian Lipschitz constants, spectral gaps, and the geometry of the re-optimized control level set. Without such a bound, static spectral closure is provably optimal only for fixed candidates, not for a fully adaptive attacker.
