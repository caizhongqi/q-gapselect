# CIFAR checkpoint performance-matching track

The three-seed CIFAR topology experiment revealed a substantial task-accuracy
imbalance between ResNet18, Tiny-ViT, and MLP-Mixer. Final-checkpoint topology
therefore cannot by itself identify a causal architecture effect.

This workflow deterministically replays the same architecture-by-seed training
runs and records full calibration/test classification accuracy at every frozen
checkpoint (`0,1,3,6,12`). The training seed, data partition, optimizer,
augmentation, model, and checkpoint schedule are identical to the topology
workflow.

The merged performance artifact is joined to the previously frozen topology
rows by `(architecture, model_seed, checkpoint_epoch)`. The analysis must report
both:

1. budget-matched trajectories at equal epoch;
2. performance-matched comparisons in overlapping accuracy regions.

A nearest-checkpoint comparison is admissible only with the actual accuracy
mismatch reported. Interpolation is descriptive and must not be represented as
an observed checkpoint.

This track does not erase all architecture confounding, prove a causal
architecture law, or replace longer training. It is a direct audit of whether
the observed topology ordering survives task-performance matching.
