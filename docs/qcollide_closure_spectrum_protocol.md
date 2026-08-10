# Q-COLLIDE dangerous-direction closure-spectrum protocol

The adaptive dual-head audit showed that one targeted control direction is not
universally sufficient. This campaign measures the complete closure-rank
dose-response instead of selecting one favourable control rank.

For each trained model and visible control rank, baseline collision tunnels are
constructed on calibration anchors. Their hidden, control-null displacements
form a dangerous-direction matrix `D`. The campaign records its singular-value
spectrum and compares equal-rank augmentations:

- `targeted`: leading left singular vectors of `D`;
- `random_null`: random directions in the same hidden control null space;
- `variance_null`: maximum-variance directions in that null space;
- `row_sham`: redundant directions from the existing control row space.

Every projection is recalibrated to the same benign-acceptance quantile. Attacks
are regenerated after every intervention, payload separation is measured in the
post-intervention non-control hidden representation, and packing is the exact
cross-anchor maximum matching.

The key preregistered objects are the residual dangerous energy, the minimum
closure rank required to reduce packing below fixed thresholds, and whether the
targeted subspace minimizes residual energy and adaptive packing at equal rank.

All quantum quantities remain analytic endpoint-query proxies. This experiment
is synthetic trained-network evidence, not a pretrained-model, hardware, or new
quantum-lower-bound claim.
