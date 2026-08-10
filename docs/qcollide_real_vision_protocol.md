# Q-COLLIDE real-image control-bottleneck protocol

This experiment is the first external-validity layer beyond synthetic latent
fixtures and the trainable synthetic dual-head model.

## Dataset and architectures

The experiment uses the scikit-learn handwritten Digits images. The model
training, closure-direction estimation, and final attack evaluation partitions
are stratified and disjoint. Two independently implemented encoders are used:

- a convolutional network;
- a tiny patch-token vision Transformer.

Both encoders have a 10-class main head and an eight-output auxiliary control
head trained on label-agnostic low-frequency image attributes.

## Control bottleneck

For a visible rank `r`, the control projection is the leading right-singular
subspace of the trained auxiliary head. Every intervention is orthonormalized
and recalibrated against exactly the same benign perturbation panel. The
control threshold is the preregistered benign-distance quantile. Payload
separation is measured in the complement of the current hidden control row
space and is also calibrated from benign perturbations.

## Attack and packing

For each correctly classified anchor, the target is the highest-logit incorrect
class. The input-space target-margin gradient is projected into the local
control-Jacobian nullspace, followed by a bounded line search. A candidate is
valid only when it reaches the target margin, remains inside the calibrated
control threshold, and exceeds the non-control payload threshold.

Candidates may match any held-out anchor with the same source class. The
reported collision quantity is the exact maximum bipartite matching, not the
raw edge count.

## Interventions

The baseline is compared with equal-rank additions:

- `targeted`: leading control-null hidden target-gradient modes estimated only
  from calibration anchors;
- `random_null`: random directions in the same control nullspace;
- `variance_null`: maximum-variance directions restricted to the nullspace;
- `row_sham`: redundant directions from the existing control row space.

All attacks are regenerated after each intervention.

## Claim boundary

This is a real image dataset and includes CNN and Transformer architectures,
but it is not production-scale vision, ImageNet, a physical attack, a coherent
quantum execution, or a hardware runtime result. Quantum costs remain analytic
endpoint-query proxies. The workflow deliberately executes each
architecture/seed component in a fresh process to avoid cross-model autograd
state and then merges raw rows before computing aggregate statistics.
