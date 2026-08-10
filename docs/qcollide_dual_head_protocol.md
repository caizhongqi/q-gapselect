# Trainable dual-head Geometry-to-Packing protocol

This experiment is the first trained-network causal gate for Q-COLLIDE. It uses
a deterministic one-hidden-layer tanh model with a scalar behaviour head and an
anisotropically trained multi-output control head.

## Separation of estimation and evaluation

For every model seed, the experiment uses disjoint calibration and evaluation
anchor sets. Baseline attack directions and the targeted closure direction are
learned only from calibration anchors. Packing is measured only on evaluation
anchors. A separate support sample estimates representation variance.

## Transfer and adaptive evaluation

Every intervention is evaluated twice:

- `transfer`: reuse attacks generated against the original control projection;
- `adaptive`: recalibrate the intervened projection and regenerate attacks from
  its post-intervention control-null geometry.

The collision graph no longer encodes anchor identity. All attack candidates may
connect to all anchors through the same continuous control-distance predicate,
and the reported packing number is the exact maximum bipartite matching.
Payload separation is measured in the orthogonal complement of the current
hidden control row space rather than in raw latent coordinates.

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
Quantum costs remain analytic endpoint-query proxies. A new v2 summary must be
frozen only after the adaptive campaign is executed; the existing v1 summary is
retained as a transfer-only historical diagnostic.
