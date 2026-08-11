# Q-COLLIDE Adversary Composition Theorem Plan

## Statement target

Let \(O_G\) be the endpoint collision oracle. We want a lower bound for estimating

\[
K_C=\nu(G)
\]

under the same oracle interface as the algorithm.

## Current gap

The packed witness proof gives a claw-finding reduction, but does not yet cover:

- shared endpoints;
- arbitrary degree distributions;
- adaptive isolation procedures.

## Proposed proof decomposition

### Lemma 1: isolation hardness

Construct a family \(G_k\) where random isolation preserves the distinction

\[
\nu(G_k)=k_0 \quad vs \quad \nu(G_k)=k_1.
\]

Bound the adversary progress under one oracle query.

### Lemma 2: composition

For an algorithm using an isolation stage followed by quantum search,
compose adversary matrices:

\[
\mathrm{Adv}^{\pm}(f\circ g)
\geq
\mathrm{Adv}^{\pm}(f)\mathrm{Adv}^{\pm}(g).
\]

The required step is proving the collision-isolation primitive fits the same oracle model.

### Lemma 3: epsilon dependence

Amplitude estimation contributes the estimation factor:

\[
\Omega(1/\epsilon)
\]

only if the output probability is encoded by a valid bounded-error oracle promise.

## Experimental validation before theorem claim

Add synthetic families:

1. disjoint matching baseline;
2. bounded-degree overlap;
3. heavy-star collision graph;
4. random bipartite expander.

Measure:

- estimator bias;
- query count;
- variance under isolation;
- failure probability.

## Final theorem threshold

A 9.5+ theoretical claim requires:

\[
\text{upper bound}
+
\text{same-interface lower bound}
+
\text{resource accounting}
\]

all closed under the same collision oracle.
