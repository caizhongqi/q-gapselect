# Q-COLLIDE Speech Commands Audio Atlas — Phase-2 Result Audit

## Frozen execution

- Workflow: `qcollide-neural-atlas-audio-speechcommands`
- Run: `31492819408`
- Grid: 3 architectures × 3 independently resampled fixtures = 9 components
- Architectures: Audio-CNN, Wav2Vec2-base, AST
- Dataset: Speech Commands v0.02, frozen 12-command subset
- Merged artifact: `qcollide-audio-speechcommands-phase2-merged`
- Artifact ID: `9102828620`
- Artifact digest: `sha256:3dab7d89acbf3dbdf9a8de2d15b255be25e419ff5d44f375bbe0cfbe0d11d773`

All 9 component jobs and the merge job completed successfully. PCM WAV decoding is performed through the deterministic standard-library path introduced to avoid TorchCodec/FFmpeg/CUDA binary coupling on CPU runners.

## Frozen quality gates

The preregistered component gates are:

- evaluation accuracy >= 0.50;
- label-agnostic control-head calibration R^2 >= 0.10.

The merged artifact is fail-closed: the Audio domain is eligible for the Universal Atlas main claim only if every registered architecture × fixture cell passes both gates.

| Architecture | Fixture | Eval accuracy | Control R^2 | Accuracy gate | Control gate | Joint gate |
|---|---:|---:|---:|---|---|---|
| AST | 20260811 | 0.97917 | 0.05597 | pass | fail | fail |
| AST | 20260812 | 0.98611 | -0.78329 | pass | fail | fail |
| AST | 20260813 | 0.99653 | -1.04955 | pass | fail | fail |
| Audio-CNN | 20260811 | 0.50694 | 0.60452 | pass | pass | pass |
| Audio-CNN | 20260812 | 0.46528 | 0.61999 | fail | pass | fail |
| Audio-CNN | 20260813 | 0.51389 | 0.62495 | pass | pass | pass |
| Wav2Vec2 | 20260811 | 0.80903 | -0.56277 | pass | fail | fail |
| Wav2Vec2 | 20260812 | 0.83681 | -0.70506 | pass | fail | fail |
| Wav2Vec2 | 20260813 | 0.79167 | -1.62845 | pass | fail | fail |

Architecture-level means:

- AST: evaluation accuracy = 0.98727; control R^2 = -0.59229.
- Audio-CNN: evaluation accuracy = 0.49537; control R^2 = 0.61648.
- Wav2Vec2: evaluation accuracy = 0.81250; control R^2 = -0.96543.

The registered Universal Atlas main gate therefore **fails** for this Audio protocol. This is retained as a scientific negative result rather than repaired by lowering thresholds.

## Topology measurements and interpretation boundary

All 36 registered architecture × fixture × rank topology rows have non-zero nominal collision capacity. The merged non-zero-capacity row fraction is 1.0. However, topology comparisons for Wav2Vec2 and AST are not admitted as architecture-law evidence because their control heads fail calibration.

For reference only, the mean nominal capacity fractions by visible rank are:

| Architecture | rank 1 | rank 2 | rank 4 | rank 8 |
|---|---:|---:|---:|---:|
| AST | 0.79167 | 0.73611 | 0.84722 | 0.88889 |
| Audio-CNN | 0.40278 | 0.34722 | 0.40278 | 0.66667 |
| Wav2Vec2 | 0.27778 | 0.29167 | 0.27778 | 0.31944 |

These values must not be used to claim that AST has intrinsically larger functional collision capacity than Wav2Vec2 or Audio-CNN under the current protocol, because the representation-to-control interface is not comparably valid across architectures.

## What is supported

1. The Speech Commands execution path is operational and reproducible across all 9 registered cells.
2. Audio-CNN demonstrates that the low-frequency/spectral control descriptor can be encoded strongly enough to support collision-topology measurement in at least a subset of fixtures.
3. High task accuracy does not guarantee validity of the chosen control interface: Wav2Vec2 and AST are explicit counterexamples.
4. The negative gate result falsifies a naive cross-audio universal-topology claim under the current control descriptor.

## What is not supported

- no universal Audio architecture law;
- no claim that AST/Wav2Vec2 have more or fewer functional collisions than Audio-CNN;
- no control-interface invariance across pretrained and trained-from-scratch audio encoders;
- no quantum runtime or hardware advantage claim;
- no post-hoc lowering of the R^2 or task-quality gates.

## Next admissible Audio step

A future Audio v3 may change the *control-interface definition* using an independently calibrated descriptor design, but it must use separate calibration fixtures before confirmatory fixtures. Merely increasing a loss weight or relaxing the current R^2 threshold is not an admissible continuation of this phase-2 experiment.
