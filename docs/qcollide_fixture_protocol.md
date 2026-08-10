# Q-COLLIDE F0--F4 calibration protocol

This package freezes the first falsifiable implementation layer for the
Q-COLLIDE research line. It does **not** claim that an analytic query-cost
formula is a quantum circuit or a runtime result.

## Fixtures

- **F0 pair-oracle negative control:** endpoint records are identical and the
  valid relation is hidden in an arbitrary pair oracle. Structured claw costs
  are rejected; only pair-Grover scaling is admissible.
- **F1 exact packed claw:** exactly `nu` vertex-disjoint, endpoint-local claws.
- **F2 random range:** independent random endpoint signatures, used to measure
  edge count, maximum matching, degree concentration, and the difference
  between raw collisions and independent tunnel packing.
- **F3 weighted prefix claw:** progressively revealed signatures with explicit
  incremental costs and exact prefix survival rates.
- **F4 geometry-to-packing:** controlled tangent dimension, control rank,
  openness, curvature, and local certificate margin.

## Frozen output semantics

Every fixture uses one immutable endpoint-record interface. The report stores
exact graph packing statistics and analytic classical/quantum endpoint-query
proxies. It does not grant the measured maximum matching to any search method;
that quantity is an evaluator-only diagnostic.

## Current claim boundary

The implementation can validate protocol invariants and expected cost scaling.
It cannot establish a new weighted-prefix claw lower bound, coherent
variable-time quantum execution, compiled resource advantage, or empirical
neural-network causality. Those remain separate gates.
