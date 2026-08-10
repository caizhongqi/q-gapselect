# Trainable dual-head Geometry-to-Packing protocol

This experiment is the first trained-network causal gate for Q-COLLIDE. It uses
a deterministic one-hidden-layer tanh model with a scalar behaviour head and an
anisotropically trained multi-output control head.

## Separation of estimation and evaluation

For every model seed, the experiment uses disjoint calibration and evaluation
anchor sets. Attack directions and the targeted closure direction are learned
only from calibration anchors. Packing is measured only on evaluation anchors.
A separate support sample estimates representation variance.

## Interventions

Every intervention adds at most one control direction and is recalibrated to
the same benign-acceptance quantile:

- `targeted`: leading control-null displacement learned from calibration attacks;
- `random_null`: random direction in the same control null space;
- `variance_null`: maximum-variance direction restricted to that null space;
- `row_sham`: a redundant direction from the existing control row space.

## Frozen claim boundary

The experiment provides causal evidence on a trained synthetic neural network.
It is not evidence on pretrained production models, a real-world attack, a
coherent quantum circuit, a hardware speedup, or a new quantum lower bound.
Quantum costs remain analytic endpoint-query proxies.

The full diagnostic is identified by SHA-256
`70697ea4155dadc7ffbaa70260d496a199b8b5d1463cfe9de0d01f6cd515d675`.
