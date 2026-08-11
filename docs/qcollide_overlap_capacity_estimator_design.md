# Overlap-aware Collision Capacity Estimator

## Objective

Extend packed collision capacity estimation from disjoint claw witnesses to general functional collision graphs.

Given a bipartite collision graph

\[
G_C=(A,B,E),
\]

we estimate

\[
K_C=\nu(G_C),
\]

where \(\nu\) is the maximum matching size.

## Problem

The packed model assumes independent witness edges. Real neural collision graphs contain overlap:

- shared attack candidates;
- shared anchors;
- hub collision regions;
- bounded-degree collision basins.

Therefore \(|E|\) cannot be used as a proxy for capacity.

## Proposed pipeline

1. Degree sketching

Estimate endpoint collision degree distribution.

2. Degree-aware isolation

Apply adaptive sampling:

\[
p_v=f(d(v))
\]

with lower retention probability for high-degree vertices.

3. Witness extraction

Construct approximately isolated claw components.

4. Quantum capacity sketch

Estimate isolated matching survival probability and invert the survival curve.

## Benchmark families

### Forest

\(\Delta=1\), recovers packed claw model.

### Sparse overlap

Bounded degree collision graph.

### Hub graph

Large edge count but small matching number.

### Expander collision graph

High connectivity stress test.

## Success criteria

- Capacity estimator error scales with epsilon.
- Edge count is not used as capacity surrogate.
- Forest case matches existing collision sketch bound.
- Bounded overlap produces explicit degree-dependent degradation.

## Claim boundary

This document defines the algorithmic extension. It does not claim a universal tight bound for arbitrary collision graphs until adversary lower bounds are completed.
