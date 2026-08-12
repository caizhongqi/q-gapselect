# Constant-precision lower bound for bounded-overlap collision capacity

## Problem class

Fix `Delta >= 1`. Let `BDelta(N)` be the family of endpoint-local bipartite
functional-collision instances with `N` endpoints on each side and maximum
collision degree at most `Delta`.

Consider the promise problem

```text
NO:  K_C = 0,
YES: K_C >= k,
```

where `K_C = nu(G_C)` is maximum-matching collision capacity. Any capacity
sketch that returns a valid nonzero multiplicative bracket for every `K_C>0`
and returns zero on the empty instance solves this promise problem as a
subroutine.

## Theorem: packed restriction lower bound

For every `Delta >= 1`, the class `BDelta(N)` contains the degree-one packed
matching family. Restrict an arbitrary bounded-overlap algorithm to instances
consisting of exactly `k` vertex-disjoint endpoint-local collision witnesses and
no other collision edges. On this restriction, the capacity is exactly `k` and
the problem is the packed multi-solution claw-detection problem used in the
Q-COLLIDE calibration layer.

Therefore every bounded-overlap capacity algorithm that solves the promise
above with constant bounded error inherits the packed-claw lower bound

\[
\boxed{
Q_{\Delta}(N,k)
=\Omega\!\left((N^2/k)^{1/3}\right)
}
\]

up to the same polylogarithmic/interface qualifications as the packed endpoint-
local claw model.

The reduction is identity-on-instances: no graph transformation, QRAM, or free
preprocessing is introduced.

## Matching upper exponent for the survival bracket

The bounded-overlap survival theorem shows that at the critical retention scale
`q = Theta(k^{-1/2})`, collision survival is constant whenever `K_C >= k`.
The retained domains contain `Theta(qN)` endpoints per side. Applying the same
endpoint-local claw detector to the retained instance costs

\[
\widetilde O((qN)^{2/3})
=
\widetilde O((N^2/k)^{1/3}).
\]

A dyadic schedule removes prior knowledge of `k` with logarithmic overhead.
Combining the observed survival probability with the overlap envelope yields an
`O(Delta)`-factor capacity bracket.

Consequently, for constant `Delta` and constant accuracy/confidence, the
`N,k` exponent of the bounded-overlap capacity-sketch problem is closed:

\[
\boxed{
\widetilde\Theta((N^2/k)^{1/3})
}
\]

for the detection/bracketing formulation under the frozen endpoint-local claw
interface.

## What this theorem does not prove

The restriction lower bound deliberately does **not** establish that the
optimal dependence on overlap is `Delta`, `sqrt(Delta)`, or any other function.
The hard restriction has degree one, so it can only certify the base `N,k`
exponent.

It also does not establish the full multiplicative dependence on an arbitrary
estimation accuracy `epsilon`. A separate composition/direct-product argument
would be required to prove

\[
\Omega\!\left(\epsilon^{-1}(N^2/k)^{1/3}\right)
\]

in the raw endpoint oracle rather than merely an outer amplitude-estimation
model.

Finally, the result is for an `O(Delta)`-factor bracket/detection task, not a
`(1+epsilon)` approximation of maximum matching on arbitrary overlap graphs.

## Paper-safe statement

A defensible theorem statement is:

> Under the endpoint-local conditional-collision interface, bounded-overlap
> functional collision capacity admits a survival-based `O(Delta)` bracket with
> quantum query scale `O-tilde((N^2/K_C)^(1/3))`. For constant maximum degree and
> constant precision, this `N,K_C` exponent is optimal by restriction to the
> degree-one packed-claw family.

Do not strengthen this to an optimal `Delta` dependence or a fully tight
`epsilon`-dependent maximum-matching estimator without an additional proof.
