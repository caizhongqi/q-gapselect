# Partitioned heterogeneous claw complexity law

This note closes one rigorously defined heterogeneous claw problem and derives
the cost norm that actually controls it. It does **not** claim the complete
shared-prefix, input-dependent variable-time theorem.

## Explicit promise family

Partition candidate endpoints into public, pairwise-disjoint stage blocks
`B_1,...,B_L`. Endpoints in block `l` reveal their complete record only after
cumulative cost

\[
C_l=\sum_{j\le l}c_j.
\]

Block `l` contains two balanced domains of size `n_l` and obeys the ordinary
no-claw versus single-claw decision promise. Exactly one block is a yes-instance.

## Tight L2 theorem

The stage hardness is

\[
H_l=\Theta(C_l n_l^{2/3}).
\]

Costed exact-one adversary composition gives

\[
Q
=\Omega\!\left[
\left(
\sum_l C_l^2 n_l^{4/3}
\right)^{1/2}
\right].
\]

Coherent optimal claw subroutines combined by variable-time exact-one search
give the matching upper bound

\[
Q
=\widetilde O\!\left[
\left(
\sum_l C_l^2 n_l^{4/3}
\right)^{1/2}
\right].
\]

Therefore

\[
\boxed{
Q
=\widetilde\Theta\!\left[
\left(
\sum_l C_l^2 n_l^{4/3}
\right)^{1/2}
\right].
}
\]

## Collision-weighted stopping norm

Let

\[
N=\sum_l n_l,
\qquad
p_l=\frac{n_l}{N}.
\]

Then the same theorem can be written as

\[
\boxed{
Q
=\widetilde\Theta\!\left[
N^{2/3}\,
\mathcal T_{\rm claw}
\right],
\qquad
\mathcal T_{\rm claw}
=
\left(
\sum_l C_l^2p_l^{4/3}
\right)^{1/2}.
}
\]

The ordinary endpoint RMS cost is

\[
T_{\rm rms}
=
\left(
\sum_l C_l^2p_l
\right)^{1/2}.
\]

Because `p_l^(4/3) <= p_l`,

\[
\mathcal T_{\rm claw}\le T_{\rm rms}.
\]

For `L` equal-mass blocks with equal endpoint costs,

\[
\frac{T_{\rm rms}}{\mathcal T_{\rm claw}}=L^{1/6}.
\]

Thus a universal lower bound of the form `N^(2/3) T_rms` cannot be inferred
from this family. The correct cost quantity couples endpoint cost with the
collision exponent. This is a substantive correction to the earlier target
that treated ordinary stopping-time RMS as the only heterogeneous factor.

## Improvement over stage restriction

A single-stage restriction proves only

\[
\max_l C_l n_l^{2/3}.
\]

The coupled theorem can be a factor `sqrt(L)` larger. It is therefore a genuine
multi-level result rather than a restatement of the hardest stage.

## Remaining gap

The explicit construction uses public disjoint stage blocks. The full
Q-COLLIDE verifier has shared prefixes, input-dependent stopping histories, and
solution endpoints correlated with stopping level. The next theorem should seek
a shared-prefix analogue of the mixed norm, rather than forcing an ordinary
`T_rms` factor that this partitioned family already shows is not universally
tight.
