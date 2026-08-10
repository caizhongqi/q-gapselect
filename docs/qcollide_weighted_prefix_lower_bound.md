# Weighted prefix-conditional claw: proven lower-bound layer

This note separates the lower bound that is now closed from the fully
heterogeneous variable-time theorem that remains open.

## 1. Composed packed-claw model

Let `PCF(N_A,N_B,nu)` be the relation that receives endpoint records on two
domains and must return one of at least `nu` vertex-disjoint claws. In the
balanced random-packing regime its unit-cost quantum query complexity is

\[
\Theta\!\left(\left(\frac{N_A N_B}{\nu}\right)^{1/3}\right).
\]

Each endpoint record is not supplied for free. It is produced by an inner
oracle problem `G_C` whose bounded-error quantum query complexity and general
adversary value are both `Theta(C)`. The outer algorithm can query the inner
oracles coherently, but cannot access any bit of an endpoint record without
solving the corresponding inner problem.

## 2. Homogeneous composition theorem

**Theorem 1 (homogeneous endpoint-cost lower bound).** In the composed model

\[
\operatorname{PCF}(N_A,N_B,\nu)\circ G_C^{N_A+N_B},
\]

any bounded-error quantum algorithm requires

\[
\Omega\!\left[
C\left(\frac{N_A N_B}{\nu}\right)^{1/3}
\right]
\]

inner-oracle queries.

**Proof.** The negative-weight general adversary bound characterizes bounded-
error quantum query complexity up to constants and composes multiplicatively
for a relation whose input variables are independently produced by copies of
an inner function. The outer packed-claw relation contributes
`Omega(((N_A N_B)/nu)^(1/3))`; every endpoint variable contributes adversary
value `Theta(C)`. Multiplication gives the stated bound. The product-Johnson
upper bound with a homogeneous endpoint generator matches it up to logarithmic
and constant factors. `square`

This closes the matching lower bound for the **uniform endpoint-cost** version
of Q-COLLIDE. It does not yet close the nonuniform stopping-time model.

## 3. Prefix-stage restriction theorem

A prefix verifier has stages `1,...,L`, incremental costs `c_l`, and cumulative
costs

\[
C_l=\sum_{j\le l}c_j.
\]

Let `A_l` and `B_l` be endpoint subsets whose records remain indistinguishable
until stage `l`, with sizes `N_{A,l}` and `N_{B,l}`. Suppose the restriction to
those subsets contains `nu_l` vertex-disjoint valid claws.

**Theorem 2 (stage-restriction profile).** Every bounded-error algorithm in the
composed prefix model satisfies

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

**Proof.** Fix a stage `l`. Set every endpoint outside `A_l union B_l` to a
public record rejected before stage `l`. Inside the survivor subsets, fix all
earlier prefixes to the same public value and encode the first distinguishing
record bit through an inner oracle of query complexity `Theta(C_l)`. The
restricted problem is exactly the homogeneous composed packed-claw instance of
Theorem 1. A lower bound for a restriction is a lower bound for the original
problem. Maximizing over stages proves the claim. `square`

The repository implementation calls the resulting quantity the **prefix
restriction hardness profile**.

## 4. What remains open

The current variable-time upper bound is controlled by second moments of
setup, update, and checking costs. The stage-restriction theorem can be
strictly smaller because it considers one homogeneous restriction at a time.
The missing theorem must couple, in one adversary matrix,

- two endpoint domains;
- randomly located packed claws;
- prefix-dependent stopping times;
- coherent early stopping;
- solution endpoints correlated with the expensive survivor tail.

The target is a costed relation adversary matching the variable-time
product-Johnson upper bound, not another restatement of the unit-cost claw
bound. Until this matrix is constructed, the paper may claim a homogeneous
matching lower bound and the multi-stage restriction profile, but not a fully
heterogeneous matching theorem.
