# Q-COLLIDE Fashion-MNIST external-validity protocol

This campaign extends the collision-tunnel audit from 8×8 handwritten digits
to 28×28 Fashion-MNIST images while preserving the same endpoint interface,
adaptive attack regeneration, covariance-whitened control metric, non-control
hidden payload criterion, exact maximum matching, and equal-rank intervention
controls.

## Frozen data partitions

The official Fashion-MNIST training split is divided into disjoint model-training
and closure-estimation partitions. The official test split supplies the attack
evaluation partition. The preregistered campaign uses:

- 12,000 stratified training images;
- 2,000 stratified calibration images;
- 2,000 stratified evaluation images;
- three independent model seeds for each architecture.

Dataset download uses the official `torchvision.datasets.FashionMNIST` interface.
No evaluation image is used to train the model or estimate closure directions.

## Architectures and control bottleneck

The campaign trains a convolutional encoder and a patch-token Tiny Vision
Transformer. Each has a 10-class main head and an eight-output auxiliary control
head trained on label-agnostic low-frequency image attributes. Visible control
ranks are 1, 2, 4, and 8. Closure ranks 1 through 6 are compared for targeted,
random-null, variance-null, and redundant row-space interventions.

## Attack and evaluation

For every correctly classified held-out anchor, the target is the highest-logit
incorrect class. The target-margin input gradient is projected into the local
control-Jacobian nullspace, followed by bounded line search. Attacks are
regenerated after every intervention. Candidates may collide with every held-out
anchor sharing the source class, and the reported packing statistic is the exact
maximum bipartite matching.

Every projection is recalibrated to the same 0.95 benign-control acceptance
quantile. Payload separation is measured in the complement of the current hidden
control row space. The row-space sham must remain exactly inert under the
basis-invariant covariance-whitened control metric.

## Claim boundary

Fashion-MNIST is a larger real image dataset than the existing Digits fixture,
but it is not ImageNet-scale evidence, a physical-world attack, or a deployed
system. Quantum numbers remain analytic endpoint-query proxies. The workflow
does not claim a coherent neural oracle, fault-tolerant runtime advantage, free
QRAM, or a completed heterogeneous weighted-prefix lower bound.
