# Q-COLLIDE Prospective CIFAR Collision-Repair Utility Protocol

## Purpose

This experiment tests a practical causal question:

> After discovering functional collision topology, can a calibration-derived representation intervention reduce **held-out independent collision capacity** more than an equal-rank, approximately energy-matched null intervention, without materially degrading task accuracy?

This is a representation-level intervention experiment. It does not retrain model weights and does not claim a production defense.

## Frozen dataset/model panel

Dataset: CIFAR-10.

Architectures:

- ResNet-18 control model;
- Tiny-ViT control model;
- MLP-Mixer control model.

Model seeds: `0,1,2`.

Training protocol is frozen to the existing CIFAR main configuration:

- 20,000 training samples;
- 4,000 calibration samples;
- 4,000 evaluation samples;
- hidden dimension 128;
- 24 epochs;
- AdamW, learning rate 0.001;
- auxiliary low-frequency control loss weight 0.15.

The raw CIFAR-10 bytes are prepared once from the public `uoft-cs/cifar10` mirror and cached as a snapshot. The standard Q-COLLIDE stratified partition logic is then used unchanged.

## Prospective design/evaluation separation

For every trained model, correctly classified examples are split independently by class.

### Repair-design fixture

From the calibration partition, select for each class:

- 6 collision anchors;
- 6 disjoint collision candidates.

This fixture is the **only** source used to derive the targeted repair basis.

### Held-out repair-evaluation fixture

From the evaluation partition, independently select for each class:

- 6 collision anchors;
- 6 disjoint collision candidates.

No collision edge from this held-out fixture may be used when constructing the targeted repair basis.

The full 4,000-example evaluation partition is separately used to measure classification accuracy after intervention.

## Frozen collision definition

Visible control rank is fixed to `r=1` because all three registered architectures exhibit nonzero baseline capacity in the established CIFAR-10 experiment at this rank.

From the unmodified calibration design fixture, freeze:

- the visible control projection;
- whitening/standardization from a 2,000-example training support set;
- nominal benign control radius `epsilon` at the 95th percentile;
- non-control payload threshold `Delta` at the 95th percentile;
- behavior-divergence threshold `Gamma` at the 95th percentile;
- the filtration `0.25x ... 4x` the nominal control radius.

**These thresholds are never recalibrated after intervention.** Recalibration would change the measurement scale and can conceal the effect of a repair.

## Targeted intervention

Construct the nominal calibration collision graph. For every calibration collision edge, form the residual hidden displacement after projecting out the visible control subspace.

Stack these displacements and compute their SVD. The leading right-singular directions define the targeted collision subspace.

Registered closure ranks:

`1,2,4`.

Primary closure rank:

`2`.

For a basis `U_r` and support mean `mu`, intervene on a hidden state by

```text
h' = h - U_r U_r^T (h - mu).
```

The classifier head is then evaluated directly on `h'`.

## Matched controls

Every closure rank is compared with two controls.

### Energy-matched random null

Generate 64 random bases in the visible-control nullspace and choose the candidate whose removed support variance is closest to the targeted basis at the same rank.

This is the **primary control**. It prevents a targeted advantage from being explained only by deleting more hidden-state energy.

### Variance-null

Use the top support-covariance directions restricted to the visible-control nullspace.

This tests whether generic high-variance removal is sufficient.

Because all intervention bases lie in the frozen control nullspace, visible control output should remain nearly invariant. RMS control drift is reported rather than assumed zero.

## Primary endpoint

For model instance `m`, let

```text
DeltaK_targeted(m) = baseline held-out K_C fraction - targeted held-out K_C fraction
DeltaK_random(m)   = baseline held-out K_C fraction - matched-random held-out K_C fraction.
```

The single primary confirmatory comparison is

```text
DeltaK_targeted > DeltaK_random
```

at closure rank 2.

The nine architecture x seed instances are paired. The registered statistical test is a one-sided exact paired sign-flip randomization test over the nine model instances.

The test is not replaced post hoc by pooling closure ranks, thresholds, collision edges, or filtration points as independent replicates.

## Utility/quality gate

The preregistered full-evaluation classification-accuracy loss budget is

```text
<= 0.03 absolute accuracy.
```

The primary targeted repair is considered operationally useful only if the result table reports both:

1. positive held-out capacity reduction relative to the energy-matched random control; and
2. the fraction of targeted model instances satisfying the 3-point accuracy-loss budget.

No accuracy-budget failures are hidden from the table.

## Additional reported endpoints

For all three interventions and all ranks report:

- held-out nominal capacity fraction;
- capacity reduction;
- filtration capacity-AUC reduction;
- basin density;
- full evaluation accuracy and accuracy loss;
- visible-control RMS drift;
- removed support energy;
- calibration dangerous-spectrum rank 90/95;
- calibration collision edge count.

## Fail-closed conditions

A component fails rather than silently changing protocol if:

- a class lacks 12 correctly classified samples in a registered partition;
- held-out baseline capacity fraction is below 0.05;
- the calibration collision-displacement spectrum has rank below the largest registered closure rank;
- prepared dataset snapshot files are missing or malformed.

The experiment does **not** respond to a failed component by lowering thresholds, reducing the class requirement, changing visible rank, or dropping an architecture after seeing results.

## Claim boundary

A positive result supports:

> calibration-derived functional-collision directions can guide a representation-level intervention that reduces independent collision capacity on a disjoint held-out fixture more effectively than a rank- and energy-matched null intervention under the same frozen collision thresholds.

It does not by itself support:

- permanent model-weight repair;
- cross-dataset repair generalization;
- production robustness;
- adversarial-training superiority;
- a causal theorem for all neural representations;
- any quantum runtime claim.
