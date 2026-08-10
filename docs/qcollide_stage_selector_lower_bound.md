# Tight disjoint-stage selector complexity

This note closes one rigorously defined heterogeneous claw problem. It does
**not** claim the complete shared-prefix, input-dependent variable-time theorem.

## Explicit promise family

Partition candidate endpoints into public, pairwise-disjoint stage blocks
`B_1,...,B_L`. Endpoints in block `l` reveal their complete record only after
cumulative cost

\[
C_l=\sum_{j\le l}c_j.
\]

Block `l` contains two balanced domains of size `n_l` and obeys the ordinary
no-claw versus single-claw decision promise. Exactly one block is a yes-instance.
An algorithm must identify that block and return its claw witness.

## Lower bound

The unit-cost adversary value of ordinary balanced single-claw detection is

\[
\Theta(n_l^{2/3}).
\]

Costed endpoint composition gives per-stage hardness

\[
H_l=\Theta(C_l n_l^{2/3}).
\]

The exact-one selector with unequal stage costs has Euclidean adversary value.
Therefore

\[
Q
=\Omega\!\left[
\left(
\sum_{l=1}^L C_l^2 n_l^{4/3}
\right)^{1/2}
\right].
\]

Witness finding is at least as hard as deciding which stage is the unique
yes-instance.

## Matching upper bound

For every stage, use a coherent optimal claw-finding subroutine with endpoint
cost included. Its stage runtime is

\[
T_l=\widetilde O(C_l n_l^{2/3}).
\]

Compose these unequal-runtime stage subroutines with variable-time exact-one
search. The resulting cost is

\[
\widetilde O\!\left[
\left(
\sum_{l=1}^L T_l^2
\right)^{1/2}
\right]
=
\widetilde O\!\left[
\left(
\sum_{l=1}^L C_l^2 n_l^{4/3}
\right)^{1/2}
\right].
\]

Combining both directions gives the tight profile

\[
\boxed{
Q
=
\widetilde\Theta\!\left[
\left(
\sum_{l=1}^L C_l^2 n_l^{4/3}
\right)^{1/2}
\right].
}
\]

The tilde records coherent subroutine and error-reduction overheads in the upper
bound; the lower bound itself does not require those logarithmic factors.

## Improvement over stage restriction

A single-stage restriction proves only

\[
\max_l C_l n_l^{2/3}.
\]

The coupled result can be a factor `sqrt(L)` larger. It is therefore a genuine
multi-level heterogeneous theorem rather than a restatement of the hardest
stage.

## Remaining gap

The explicit construction uses public disjoint stage blocks. The full
Q-COLLIDE verifier has shared prefixes, input-dependent stopping histories, and
solution endpoints correlated with the stopping level. The next theorem must
embed off-diagonal adversary couplings between shared-prefix histories. Until
that construction is complete, this result must not be advertised as a
universal variable-time packed-claw theorem.
