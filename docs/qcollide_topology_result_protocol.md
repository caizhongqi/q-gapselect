# Collision-topology result protocol

The graph-instrumented visual pipeline emits one topology profile for every
model, visible control rank, intervention, and added rank. The primary baseline
analysis aggregates only `intervention=baseline, closure_rank=0` rows.

## Primary descriptors

- maximum-matching capacity fraction `K_C`;
- active component count `beta_0`;
- collision-cycle density `beta_1/|E|`;
- normalized component edge-mass entropy `H_C`;
- effective component count `exp(H_C)`;
- largest-component edge fraction;
- matching-to-edge independence ratio;
- non-control displacement entropy rank and stable rank.

The protocol reports associations with capacity, monotonicity across visible
control rank, and matched-rank architecture contrasts. These are descriptive
statistics. They do not establish persistent homology, a training phase
transition, a dataset-invariant law, or quantum topology-reconstruction
advantage.

## Interpretation tests

The metrics separate different mechanisms behind the same capacity:

- high `beta_0` and high entropy indicate many distributed collision basins;
- high `beta_1/|E|` indicates redundant cycles inside basins;
- a high largest-component fraction indicates concentration into a giant
  collision basin;
- high displacement entropy rank indicates geometrically diverse non-control
  payload directions;
- low independence ratio means that many raw collision edges reuse the same
  endpoints and therefore overstate independently exploitable capacity.
