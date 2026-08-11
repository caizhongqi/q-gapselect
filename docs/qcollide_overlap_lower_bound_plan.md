# Overlap Collision Capacity Lower Bound Plan

## Target

Establish lower bounds for estimating functional collision capacity

\[
K_C=\nu(G_C)
\]

under endpoint oracle access.

## Hard instance family

Construct two distributions:

- \(\mathcal G_0\): low matching capacity;
- \(\mathcal G_1\): hidden collision packing of size \(K_C\).

The oracle transcript should remain indistinguishable unless the algorithm performs enough collision queries.

## Proof route

### Step 1: Isolation reduction

Reduce arbitrary bounded-degree overlap graphs to isolated witness families.

### Step 2: Adversary construction

Construct an adversary matrix \(\Gamma\) over collision graph instances.

Bound:

\[
\frac{\|\Gamma\circ\Delta_i\|}{\|\Gamma\|}
\]

for every oracle coordinate.

### Step 3: Composition

Combine:

- isolation hardness;
- claw detection hardness;
- capacity estimation precision.

## Open points

1. General unbounded-degree graphs.
2. Exact epsilon dependence.
3. Matching lower bound for all overlap families.

## Final theorem target

For bounded overlap degree \(D\):

\[
Q=\widetilde\Omega\left(D^{-\alpha}(N^2/K_C)^{1/3}\right).
\]

The exponent \(\alpha\) is determined after the isolation analysis.
