# Weighted prefix-conditional claw: audited lower-bound layer

This note separates three statements that must not be conflated:

1. a rigorous adversary-composition transfer theorem;
2. a rigorous homogeneous single-claw instantiation;
3. a parameterized multi-solution profile that remains conditional on an outer
   packed-claw lower bound for the stated promise family.

The fully heterogeneous variable-time theorem remains open.

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

up to the conditions of the relation-composition theorem, including efficient
verification of `R`.

**Justification.** The negative-weight relation adversary bound composes
multiplicatively with a Boolean inner function. For efficiently verifiable
relations it characterizes bounded-error quantum query complexity up to
constant factors. Therefore the endpoint-generation cost multiplies the
outer-relation adversary value; it cannot be treated as a free database read.

This is the part of the lower-bound package that is fully closed.

## 3. Homogeneous single-claw instantiation

For balanced domains of size `N` and the ordinary single-claw promise,

\[
Q(\mathrm{Claw}_N)=\Theta(N^{2/3}).
\]

Consequently, when every endpoint record costs `C` inner queries, Theorem 1
gives

\[
\boxed{
Q=\Theta(CN^{2/3})
}
\]

in the explicit composed-oracle model. This matches the homogeneous
product-Johnson upper bound up to implementation logarithms and constants.

## 4. Multi-solution packed-claw profile: conditional status

For a promise family with domains `N_A,N_B` and `nu` vertex-disjoint claws,
the repository reports the analytic profile

\[
C\left(\frac{N_A N_B}{\nu}\right)^{1/3}.
\]

This expression is a valid composed lower bound **only after** the corresponding
unit-cost outer relation has independently been shown to satisfy

\[
\operatorname{ADV}^{\pm}_{\mathrm{rel}}(R_{N_A,N_B,\nu})
=
\Omega\!\left[
\left(\frac{N_A N_B}{\nu}\right)^{1/3}
\right].
\]

The current repository has:

- the matching product-Johnson-style upper profile;
- exact unit-cost calibration on planted packed fixtures;
- an average-case random-range lower-bound route from the collision literature;
- no universal worst-case relation-adversary proof parameterized solely by
  maximum matching `nu`.

Therefore the multi-solution formula must be labelled a **conditional
lower-bound profile**, not a completed general theorem.

## 5. Prefix-stage restriction profile

A prefix verifier has stages `1,...,L`, incremental costs `c_l`, and cumulative
costs

\[
C_l=\sum_{j\le l}c_j.
\]

Let `A_l,B_l` be endpoint subsets surviving through stage `l`, and let `R_l` be
the outer relation obtained by restricting all other endpoints to public early
rejects. The rigorous restriction statement is

\[
\boxed{
Q\ge
\Omega\!\left[
\max_l C_l\,
\operatorname{ADV}^{\pm}_{\mathrm{rel}}(R_l)
\right].
}
\]

If a particular promise family additionally establishes

\[
\operatorname{ADV}^{\pm}_{\mathrm{rel}}(R_l)
=
\Omega\!\left[
\left(
\frac{N_{A,l}N_{B,l}}{\nu_l}
\right)^{1/3}
\right],
\]

then the familiar numeric profile follows:

\[
Q\ge
\Omega\!\left[
\max_{l:\nu_l>0}
C_l
\left(
\frac{N_{A,l}N_{B,l}}{\nu_l}
\right)^{1/3}
\right].
\]

The Python implementation computes this numeric **conditional profile** for
fixture design and falsification. It does not certify the missing outer
adversary premise.

## 6. Remaining theorem

The current variable-time upper bound is controlled by second moments of
setup, update, and checking costs. A stage-by-stage maximum can be strictly
smaller because it never couples all stopping branches in one adversary
matrix. The missing result must jointly represent:

- two endpoint domains;
- randomly located or adversarially planted packed claws;
- prefix-dependent stopping times;
- coherent early stopping;
- solution endpoints correlated with an expensive survivor tail.

The target is a costed relation adversary matching the variable-time
product-Johnson upper bound. Until that matrix is constructed, the paper may
claim the composition transfer, the homogeneous single-claw matching bound,
and conditional stage profiles—but not a fully heterogeneous or universal
multi-solution matching lower bound.
