# Q-COLLIDE Frozen Pretrained Vision Atlas — Phase-2 Result Audit

## Frozen execution

- Workflow: `qcollide-pretrained-vision-atlas`
- Run: `31496900229`
- Dataset backend: Hugging Face mirror `uoft-cs/cifar10`
- Encoders: ImageNet-pretrained ResNet-50, ConvNeXt-Tiny, Swin-T, ViT-B/16
- Fixture seeds: `20260811,20260812,20260813`
- Grid: 4 architectures × 3 fixtures = 12 components
- Merged artifact ID: `9103992521`
- Digest: `sha256:6e755fa19e226fa9b0febc7aaef28da5144e78d696fee1e0047362de39727f1f`

All 12 components and the merge job completed successfully.

## Frozen quality gate

The registered phase-2 comparison requires, for every fixture of an architecture:

- linear-probe evaluation accuracy >= 0.70;
- label-agnostic control calibration R^2 >= 0.10.

All four architectures satisfy the task-accuracy gate in all three fixtures. No architecture satisfies the control-R^2 gate in all three fixtures.

| Architecture | Mean eval accuracy | Min eval accuracy | Mean control R^2 | Min control R^2 | All task gates | All control gates |
|---|---:|---:|---:|---:|---|---|
| ResNet-50 | 0.84167 | 0.825 | 0.12890 | 0.05142 | pass | fail |
| ConvNeXt-Tiny | 0.86833 | 0.850 | -0.43954 | -0.50154 | pass | fail |
| Swin-T | 0.87833 | 0.860 | -0.73306 | -0.85878 | pass | fail |
| ViT-B/16 | 0.91333 | 0.890 | -0.67570 | -0.77762 | pass | fail |

The merged gate `all_architectures_pass_quality_gates` is therefore **false**.

## Topology values are descriptive only under the failed control gate

All architecture × rank cells have nonzero nominal collision capacity, but those values cannot support a cross-architecture functional-collision law because the chosen low-frequency control interface is not comparably valid across encoders.

Mean nominal capacity fractions:

| Architecture | rank 1 | rank 8 | rank 32 |
|---|---:|---:|---:|
| ResNet-50 | 0.1750 | 0.2667 | 0.3167 |
| ConvNeXt-Tiny | 0.4333 | 0.4417 | 0.7500 |
| Swin-T | 0.2833 | 0.3417 | 0.6167 |
| ViT-B/16 | 0.7250 | 0.9000 | 0.9833 |

These numbers are retained for falsification/diagnostic purposes. They must not be used to claim that ViT-B/16 intrinsically has higher functional collision capacity than the CNN encoders under a valid common control interface.

## Scientific interpretation

The experiment establishes a useful negative result:

> Strong task representations do not guarantee that an externally chosen label-agnostic control variable is linearly represented well enough to define a comparable functional-collision object.

This mirrors the Audio and Graph fail-closed findings and sharpens the paper's main scientific message: **validating the control interface is part of defining a functional collision graph, not an optional post hoc diagnostic.**

The result also prevents an overclaim that large pretrained Transformer representations universally exhibit larger collision capacity than CNN representations.

## What is supported

1. The four frozen ImageNet-pretrained encoders can be evaluated through one common CIFAR-10 probe/topology pipeline.
2. Task accuracy is uniformly strong across the registered fixtures.
3. The low-frequency control interface is not uniformly valid across these frozen pretrained representations.
4. Cross-architecture topology numbers must therefore remain descriptive under this protocol.

## What is not supported

- no pretrained ResNet/ConvNeXt/Swin/ViT collision-capacity ordering claim;
- no global CNN-versus-Transformer collision law;
- no claim that task performance validates the control bottleneck;
- no ImageNet-validation-set topology claim (the encoders are ImageNet-pretrained, but the common probe/evaluation fixture is CIFAR-10);
- no quantum advantage claim from these topology values.

## Next admissible step

A future pretrained-representation study must calibrate a control interface independently of the confirmatory fixtures. A valid continuation could compare several preregistered control definitions on separate calibration fixtures and freeze one only before evaluating the held-out architecture panel. Lowering the current R^2 gate after seeing these results is not admissible.
