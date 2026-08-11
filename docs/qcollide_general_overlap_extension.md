# Q-COLLIDE Overlap-Aware Capacity Extension (Theorem Scaffold)

## Goal

Extend the packed-matching Collision-Basin Survival Sketch from disjoint witnesses to general collision graphs.

The target object is the maximum matching capacity

\[
K_C(G)=\nu(G)
\]

of the conditional collision graph

\[
G=(A\cup B,E_C).
\]

## Step 1: witness isolation instead of direct edge sampling

For arbitrary graphs, independent endpoint thinning no longer gives

\[
S_K(q)=1-(1-q^2)^K
\]

because edges are correlated through shared vertices.

The replacement is a random isolation operator:

\[
\mathcal I_q(G)\rightarrow G_q
\]

such that surviving isolated witnesses form a packing. The analysis target is:

\[
\Pr[\nu(G_q)\ge 1]\approx f(q,\nu(G),\Delta(G)).
\]

where \(\Delta(G)\) is maximum degree.

## Step 2: bounded-degree extension

For collision graphs satisfying

\[
\Delta(G)\le d,
\]

a conservative decomposition is:

\[
\nu(G)\le |M_1|+\cdots+|M_{O(d)}|
\]

where each \(M_i\) is a matching extracted by edge coloring.

This converts the arbitrary graph problem into a logarithmic/degree number of packed subproblems.

## Step 3: adversary target

The missing theorem is:

> Given endpoint oracle access to a collision graph with matching number \(K\), estimating \(K\) to relative error \(\epsilon\) requires
> \[
> \Omega(Q(N,K,\epsilon))
> \]
> queries.

The current completed proof only covers the packed claw subclass.

## Required next proof obligations

1. Construct an adversary matrix over overlap classes.
2. Show composition between isolation and claw detection.
3. Prove matching lower bound for the same oracle interface.
4. Validate against synthetic overlap families:
   - star collision graphs;
   - random regular collision graphs;
   - expander collision graphs;
   - union-of-basin graphs.

## Claim boundary

Until these four obligations are completed, the main claim remains:

**quantum collision capacity sketch for packed or bounded-overlap collision witnesses**, not a universal maximum matching estimator.
