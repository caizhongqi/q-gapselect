# Disjoint-stage selector lower bound

This note closes one rigorously defined part of the missing heterogeneous
prefix lower bound. It does **not** claim the complete shared-prefix,
variable-time matching theorem.

## Explicit promise family

Partition candidate endpoints into public, pairwise-disjoint stage blocks
`B_1,...,B_L`. Endpoints in block `l` reveal their complete record only after
cumulative cost

\[
C_l=\sum_{j\le l}c_j.
\]

Block `l` contains two balanced domains of size `n_l` and obeys the ordinary
no-claw versus single-claw decision promise. The global promise states that
exactly one block is a yes-instance. An algorithm must identify that block and
return its claw witness.

## Theorem

The unit-cost adversary value of ordinary balanced single-claw detection is

\[
\Theta(n_l^{2/3}).
\]

Costed endpoint composition gives per-stage hardness

\[
H_l=\Theta(C_l n_l^{2/3}).
\]

The negative-weight adversary bound for the exact-one selector composes with
unequal stage costs in Euclidean norm. Therefore the explicit disjoint-stage
subfamily satisfies

\[
\boxed{
Q
=\Omega\!\left[
\left(
\sum_{l=1}^L C_l^2 n_l^{4/3}
\right)^{1/2}
\right].
}
\]

Witness finding is at least as hard as deciding which stage is the unique
yes-instance, so the same lower profile applies to the search relation.

## What this improves

A stage-restriction argument alone yields only

\[
\max_l C_l n_l^{2/3}.
\]

The selector construction couples multiple stopping-cost levels and can be up
to a factor `sqrt(L)` stronger. It is the first repository theorem that
produces an explicit L2 aggregation across several endpoint-cost stages.

## Remaining gap

The construction uses public disjoint stage blocks. The full Q-COLLIDE verifier
has shared prefixes, input-dependent stopping histories, and solution endpoints
whose stopping level may correlate with hidden collision structure. The current
theorem does not yet establish

\[
\widetilde\Theta\!\left[
T_{\rm rms}
\left(\frac{N_A N_B}{\nu}\right)^{1/3}
\right]
\]

for that general model. The next theoretical step is to replace the public
disjoint selector by one adversary matrix with off-diagonal couplings between
shared-prefix cost levels.
