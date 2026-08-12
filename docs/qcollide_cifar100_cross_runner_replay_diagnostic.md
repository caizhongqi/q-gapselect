# CIFAR-100 Cross-Runner Checkpoint Replay Diagnostic

## Why this diagnostic exists

The CIFAR-100 performance-matching protocol originally used a two-stage process:

1. train/replay a dense checkpoint trajectory and choose an epoch near the fixed calibration target;
2. retrain the same architecture/seed in a later workflow and measure topology at that selected epoch.

This implicitly assumed that an intermediate checkpoint could be reproduced closely enough across independent GitHub-hosted CPU runners from the same seed and source code. Selected-v3 falsified that assumption for at least one registered model instance.

## Frozen target

The performance target and tolerance were not changed:

```text
target calibration accuracy = 0.30
tolerance = ±0.05
```

Dense replay run `31482286279`, head `2dfe0fb408b5644b052ab6090abb3bbc450b580a`, selected for ResNet-18 seed 2:

```text
epoch 8
calibration accuracy = 0.3078
absolute mismatch = 0.0078
```

The later selected-checkpoint v3 run `31494581548`, head `40b3a5557f0d21665e798160b25af0d80ab349dd`, retrained the same registered model seed and attempted to measure epoch 8 topology. The component failed before topology measurement with:

```text
RuntimeError: selected checkpoint misses fixed performance gate:
0.065250 > 0.050000
```

## Code/config audit

The dense and selected-v3 configurations agree on the quantities that define the training problem:

- master seed `20260811`;
- dataset and dataset seed;
- 30,000 training examples;
- 5,000 calibration examples;
- 5,000 evaluation examples;
- hidden dimension 128;
- 30 total epochs;
- batch size 128;
- learning rate 0.001;
- control-loss weight 0.15;
- control grid size 4;
- device `auto`.

Both derive the training seed using:

```text
derived_seed(master_seed, architecture, model_seed, "training")
```

A commit comparison from the dense-run head to the selected-v3 head shows no modification to `src/qgapselect/qcollide/cifar_models.py`; selected-v3 adds its wrapper/config/results code but does not alter the underlying training implementation.

The major configuration difference is checkpoint snapshot density. Dense replay stores every epoch; selected-v3 stores only the subset needed by the frozen selected epochs. Snapshotting should not intentionally change optimization, but the observed result establishes that cross-job intermediate checkpoint equality cannot be treated as a scientific invariant on the hosted CPU execution environment.

## Methodological decision

The failed seed is **not retried until it passes**, and the tolerance is **not relaxed**.

Selected-v3 is superseded as the confirmatory main-table method by **single-run selection v4**:

```text
train once
  -> score every preregistered checkpoint in that same process
  -> select min |calibration accuracy - 0.30|
     (earlier epoch breaks an exact tie)
  -> require mismatch <= 0.05
  -> immediately measure topology from the already existing selected state
```

Thus performance matching and topology measurement share one training trajectory and no longer assume cross-run intermediate-checkpoint reproducibility.

## Scientific consequence

This diagnostic does not imply that final model conclusions are non-reproducible. It establishes a narrower point:

> **A seed plus source-code identity is insufficient evidence that a specific intermediate checkpoint on a stochastic/finite-precision neural training trajectory is interchangeable across hosted CPU jobs.**

For performance-matched topology experiments, checkpoint selection and topology measurement must therefore occur on the same realized trajectory, or the exact model state must be serialized and reused.

## Claim boundary

- v3 remains useful as a reproducibility diagnostic, not as the final CIFAR-100 main table;
- the fixed target and tolerance remain unchanged;
- no failed seed is dropped from v4;
- no post-hoc checkpoint interpolation is introduced;
- v4 fails closed if any architecture × seed has no observed epoch inside the original tolerance.
