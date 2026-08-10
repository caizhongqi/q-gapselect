# CIFAR functional-collision topology pilot

This protocol is the first CIFAR-scale external-validity gate for the
Q-COLLIDE collision-topology mainline. It reuses the frozen conditional edge
predicate and persistent-filtration code; it does not introduce a new topology
definition for CIFAR.

## Model comparison

The pilot compares three approximately parameter-matched models with a common
128-dimensional hidden representation and 48-dimensional label-agnostic control
head:

- CIFAR-adapted ResNet-18: approximately 0.72M parameters;
- six-layer Tiny-ViT: approximately 0.82M parameters;
- ten-layer MLP-Mixer: approximately 0.72M parameters.

All models use the same CIFAR partition, optimizer family, epoch budget,
augmentation, control target, anchor count, visible-control ranks, and
threshold-filtration grid.

## Control target

The auxiliary control head predicts a standardized 4x4 RGB low-frequency image
descriptor. Visible control projections are leading singular subspaces of the
learned control head at each checkpoint. The experiment scans ranks
`1,2,4,8,16,32`.

## Collision graph

The left domain contains correctly classified evaluation images; the right
domain contains disjoint correctly classified calibration images. A pair is
eligible only when the true and predicted classes differ. It becomes an edge
when all of the following hold:

1. control distance is below the current filtration threshold;
2. non-control hidden displacement exceeds a benign 95% quantile;
3. logit-space behavioural divergence exceeds a benign 95% quantile.

The control threshold is scanned from 0.25 to 4 times the benign 95% radius.
Every cell reports the complete filtration, truncated H0 basin persistence,
Betti numbers, component entropy, hidden-displacement spectral ranks, and exact
maximum matching capacity.

## Training dynamics

The pilot freezes checkpoints at epochs `0,1,3,6,12`. Ranks `1,8,32` are
measured at every checkpoint; all configured ranks are measured at the final
checkpoint. These curves are change-point evidence only. They do not establish
a thermodynamic phase transition.

## Claim boundary

The pilot studies naturally occurring cross-class functional collisions. It
is not a post-intervention adaptive attack experiment. It does not claim
CIFAR-100/ImageNet generality, pretrained production-model vulnerability,
coherent quantum execution, free QRAM, or end-to-end runtime advantage.
