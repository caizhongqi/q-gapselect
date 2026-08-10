# Weighted prefix-conditional claw: audited lower-bound layer

This note separates four statements that must not be conflated:

1. a rigorous adversary-composition transfer theorem;
2. a rigorous homogeneous single-claw instantiation;
3. a rigorous distributional random-range packed-claw theorem;
4. a parameterized worst-case multi-solution profile that remains conditional
   on an outer packed-claw adversary bound for the stated promise family.

The fully heterogeneous variable-time matching theorem remains open.

## 1. Composed relation model

Let `R` be an efficiently verifiable outer relation on endpoint-record bits and
let `G_C` be a Boolean inner oracle problem satisfying

\[
\operatorname{ADV}^{\pm}(G_C)=\Theta(C).
\]

Each endpoint bit is supplied by an independent copy of `G_C`; it is not
available at unit cost. The outer algorithm may query the inner copies
coherently.

## 2. Rigorous composition transfer

**Theorem 1 (relation-composition transfer).** If

\[
\operatorname{ADV}^{\pm}_{\mathrm{rel}}(R)=A,
\]

then the composed endpoint problem obeys

\[
Q\!\left(R\circ G_C^n\right)=\Theta(AC)
\]

under the hypotheses of the relation-composition theorem, including efficient
verification of `R`.

The negative-weight relation adversary bound composes multiplicatively with a
Boolean inner function. For efficiently verifiable relations it characterizes
bounded-error quantum query complexity up to constant factors. Therefore the
endpoint-generation cost multiplies the outer-relation adversary value; it
cannot be treated as a free database read.

## 3. Homogeneous single-claw instantiation

For balanced domains of size `N` and the ordinary single-claw promise,

\[
Q(\mathrm{Claw}_N)=\Theta(N^{2/3}).
\]

Consequently, when every endpoint record costs `C` inner queries, Theorem 1
gives

\[
\boxed{Q=\Theta(CN^{2/3})}
\]

in the explicit composed-oracle model. This matches the homogeneous
product-Johnson upper bound up to implementation logarithms and constants.

## 4. Random-range packed claws: rigorous average-case theorem

Let

\[
f:[N_A]\to[R],\qquad g:[N_B]\to[R]
\]

be independent uniform random functions. Their disjoint union is a uniform
random function on a domain of size `N_A+N_B`. Any algorithm that outputs a
cross-domain claw

\[
f(i)=g(j)
\]

with constant probability is therefore also a random-function collision finder.
The tight random-oracle collision lower bound implies

\[
\boxed{Q=\Omega(R^{1/3})}
\]

in the unit-cost model whenever the domain is large enough to support that
query regime. Under uniform endpoint-generation cost `C`, adversary composition
gives

\[
\boxed{Q=\Omega(CR^{1/3})}.
\]

The corresponding classical random-query lower scale is

\[
\boxed{R_{\mathrm C}=\Omega(C\sqrt R)}.
\]

### 4.1 Cross-claw non-vacuity certificate

The collision lower bound alone does not ensure that a random collision is
cross-domain. Define `X` as the number of output labels with exactly one
preimage under `f` and exactly one preimage under `g`. Every such label is an
isolated cross claw, and all isolated cross claws form a vertex-disjoint
matching.

Its first moment is

\[
\mathbb E X
=
\frac{N_A N_B}{R}
\left(1-\frac1R\right)^{N_A+N_B-2}.
\]

For two distinct output labels, the exact joint probability gives

\[
\begin{aligned}
\mathbb E X^2
={}&\mathbb E X\\
&+R(R-1)
\frac{N_A(N_A-1)N_B(N_B-1)}{R^4}
\left(1-\frac2R\right)^{N_A+N_B-4}.
\end{aligned}
\]

Therefore Paley--Zygmund yields

\[
\Pr[X>0]
\ge
\frac{(\mathbb E X)^2}{\mathbb E X^2}.
\]

The implementation exposes all three quantities and certifies the theorem only
when:

- `N_A+N_B >= sqrt(R)`;
- the expected isolated-claw mass exceeds a preregistered constant;
- the explicit Paley--Zygmund lower bound exceeds a preregistered constant.

No Poisson approximation is required.

### 4.2 Packing parameterization

In the balanced sparse regime, choose

\[
R=\Theta\!\left(\frac{N_A N_B}{\nu}\right).
\]

Then the expected cross-claw and isolated-claw counts are `Theta(nu)`, while the
query lower scales become

\[
Q
=
\Omega\!\left[
C\left(\frac{N_A N_B}{\nu}\right)^{1/3}
\right],
\]

and

\[
R_{\mathrm C}
=
\Omega\!\left[
C\left(\frac{N_A N_B}{\nu}\right)^{1/2}
\right].
\]

This closes the multi-solution exponent for the explicit independent
random-range distribution. It does **not** establish a universal worst-case
bound for every graph with maximum matching `nu`.

## 5. Random-range prefix-stage restriction theorem

A prefix verifier has stages `1,...,L`, incremental costs `c_l`, and cumulative
costs

\[
C_l=\sum_{j\le l}c_j.
\]

At stage `l`, fix every endpoint outside public survivor sets `A_l,B_l` to an
early reject and let the survivors be independent random functions into a range
of size `R_l`. Applying Section 4 to each restricted subfamily gives

\[
\boxed{
Q
\ge
\Omega\!\left[
\max_{l\in\mathcal C}
C_l R_l^{1/3}
\right],
}
\]

where `C` contains only stages passing the explicit domain and second-moment
cross-claw certificates.

When

\[
R_l=\Theta\!\left(
\frac{N_{A,l}N_{B,l}}{\nu_l}
\right),
\]

this becomes the rigorous distributional profile

\[
Q
\ge
\Omega\!\left[
\max_{l\in\mathcal C}
C_l
\left(
\frac{N_{A,l}N_{B,l}}{\nu_l}
\right)^{1/3}
\right].
\]

This stage maximum is a valid lower bound for the complete prefix promise family
because every restricted random-range family is a subfamily of it. It still
does not couple multiple stopping branches into one tight variable-time
adversary matrix.

## 6. General worst-case packed-claw profile: conditional status

For an arbitrary promise family with domains `N_A,N_B` and `nu`
vertex-disjoint claws, the repository also reports

\[
C\left(\frac{N_A N_B}{\nu}\right)^{1/3}.
\]

Outside the random-range theorem above, this expression is a valid composed
lower bound only after the corresponding unit-cost outer relation has been
shown to satisfy

\[
\operatorname{ADV}^{\pm}_{\mathrm{rel}}(R_{N_A,N_B,\nu})
=
\Omega\!\left[
\left(\frac{N_A N_B}{\nu}\right)^{1/3}
\right].
\]

The repository has the matching product-Johnson-style upper profile and exact
unit-cost calibration on planted fixtures, but no universal worst-case relation
adversary theorem parameterized solely by maximum matching. The generic
numerical formula therefore remains labelled a **conditional profile**.

## 7. Remaining theorem

The current variable-time upper bound is controlled by second moments of setup,
update, and checking costs. A stage-by-stage maximum can be strictly smaller
because it never couples all stopping branches in one adversary matrix. The
missing result must jointly represent:

- two endpoint domains;
- adversarially planted or correlated packed claws;
- prefix-dependent heterogeneous stopping times;
- coherent early stopping;
- solution endpoints correlated with an expensive survivor tail.

The target is a costed relation adversary matching the variable-time
product-Johnson upper bound. Until that matrix is constructed, the paper may
claim:

- relation-composition transfer;
- the homogeneous single-claw matching bound;
- the random-range multi-solution average-case theorem;
- certified random-range prefix-stage restrictions;
- conditional general worst-case profiles.

It may not claim a fully heterogeneous or universal worst-case multi-solution
matching lower bound.
