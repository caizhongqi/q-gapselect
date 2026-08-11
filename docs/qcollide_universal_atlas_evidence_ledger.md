# Q-COLLIDE Universal Neural Collision Atlas — Evidence Ledger

This ledger records the current evidence state for every major claim surface. It is intentionally fail-closed: a domain or claim is not promoted because an experiment ran successfully; it is promoted only if its preregistered quality gates and comparison requirements are satisfied.

## Status vocabulary

- **verified** — executable experiment complete and the registered quality gate is satisfied for the stated claim.
- **partial** — executable evidence exists, but only a subset of models/fixtures or only a weaker claim is admissible.
- **falsified/current-design-failed** — the registered experiment produced a negative result or failed its scientific gate; the negative result is retained.
- **pending** — protocol/code may exist, but the registered execution is not yet complete.
- **theory-open** — the statement remains conjectural or only proved under a narrower promise model.

## C1 — Functional Collision Topology is a reproducible neural representation phenomenon

**Status: verified in the established vision campaigns; cross-domain universality not yet verified.**

The existing Digits/Fashion campaigns repeatedly produce nontrivial functional collision graphs and exact maximum-matching capacity. Across the established 24 paired cells, capacity correlates strongly with basin density (~0.9496) and effective component count (~0.9464). The TinyViT-minus-CNN capacity gap is dominated by distributed basin proliferation: ~96.68% of the mean gap is attributable to the basin-density term and ~3.32% to within-basin multiplicity.

This supports **Distributed Collision-Basin Proliferation** as the strongest current empirical structural law in those campaigns. It does not establish a universal architecture law across every neural modality.

## C2 — Architecture-specific collision topology generalizes beyond the original small vision experiments

**Status: partial / pending confirmatory vision closure.**

### CIFAR-10

The existing CIFAR topology pipeline is executable with parameter-matched ResNet-18, Tiny-ViT and MLP-Mixer controls, exact matching, Betti statistics, displacement spectral ranks and persistent collision filtrations. The standard CI and pilot workflows have been successfully executed.

Training-time utility tables derived from the frozen CIFAR-10 trajectory show that early collision capacity contains information about final collision risk that is not reducible to contemporaneous accuracy alone. This is an operational result, not a universal architecture-law proof.

### CIFAR-100

Dense ResNet checkpoint replay has completed for five model seeds under the unchanged fixed target `0.30 ± 0.05`. The independently selected ResNet checkpoints are:

- seed 0 -> epoch 7, calibration accuracy 0.3032;
- seed 1 -> epoch 7, calibration accuracy 0.3052;
- seed 2 -> epoch 8, calibration accuracy 0.3078;
- seed 3 -> epoch 7, calibration accuracy 0.2890;
- seed 4 -> epoch 7, calibration accuracy 0.3002.

The final selected-checkpoint v3 experiment is **pending**. It recomputes topology for all 15 ResNet/Tiny-ViT/MLP-Mixer architecture × seed cells using checkpoint-local correct anchors/candidates at the frozen performance-matched checkpoint, with visible ranks 1/8/32. No CIFAR-100 architecture-law claim is admissible until that 15/15 merge succeeds.

### Frozen ImageNet-pretrained vision encoders

A phase-2 experiment is implemented for frozen ResNet-50, ConvNeXt-Tiny, Swin-T and ViT-B/16 representations with a common CIFAR-10 linear probe and label-agnostic low-frequency control head. Execution is **pending**. These are ImageNet-pretrained encoders evaluated through a common CIFAR-10 probe; they are not an ImageNet-validation topology experiment.

## C3 — Text functional collision effects survive matched-rank null controls

**Status: falsified as a universal effect; model/rank-specific effects remain.**

The completed matched-rank random-subspace falsification contains 16 model × rank cells. The mean learned-minus-random capacity difference is only about +0.0168 and exactly 50% of cells are positive. Therefore the statement "learned control subspaces generally create more functional collision capacity than matched random subspaces" is not supported.

Model/rank-specific effects remain visible:

- DistilBERT rank 8: mean capacity difference ~+0.186, positive in all 3 fixtures;
- RoBERTa rank 2: mean capacity difference ~+0.0639, positive in all 3 fixtures;
- DeBERTa ranks 1/2/4/8: all mean differences are negative.

Persistence does not show a universal learned-subspace advantage. Text is therefore retained as a **heterogeneous / falsification-constrained** domain rather than universal positive evidence.

## C4 — Audio functional collision topology is valid under a common control interface

**Status: current design failed the Universal Atlas main gate; partial feasibility only.**

The Speech Commands v0.02 phase-2 grid completed 9/9 architecture × fixture components plus merge.

Frozen artifact:

- workflow run: `31492819408`;
- merged artifact ID: `9102828620`;
- digest: `sha256:3dab7d89acbf3dbdf9a8de2d15b255be25e419ff5d44f375bbe0cfbe0d11d773`.

Preregistered gates are evaluation accuracy >= 0.50 and control calibration R^2 >= 0.10.

Architecture-level means:

- AST: accuracy 0.98727, control R^2 -0.59229;
- Audio-CNN: accuracy 0.49537, control R^2 0.61648;
- Wav2Vec2: accuracy 0.81250, control R^2 -0.96543.

Audio-CNN passes both gates in 2/3 fixtures. Wav2Vec2 and AST have strong task accuracy but fail the control-validity gate in every fixture. The complete Audio domain is therefore **not eligible** for a universal cross-architecture topology claim under the current descriptor. The negative result is preserved in `docs/qcollide_audio_phase2_result_audit.md`.

## C5 — Graph functional collision topology is valid under the current graph-control descriptor

**Status: current design failed.**

The controlled-v3 calibration is complete:

- architectures: GCN, GraphSAGE, GAT;
- independent calibration seeds: 100, 101;
- control-loss weights: 0.5, 1, 2, 5;
- total calibration cells: 24.

The selector requires both calibration seeds to satisfy control R^2 >= 0.20 and accuracy >= 0.70, then chooses the smallest eligible weight. No architecture has an eligible weight; all three selectors return `None`.

The Graph domain therefore does not proceed to confirmatory main seeds under this descriptor/objective. Lowering the gate or continuing to increase the same auxiliary loss weight is not an admissible fix. A future Graph experiment must change the control-variable design and recalibrate on independent fixtures.

## C6 — Time-series collision topology is valid and performance-comparable

**Status: feasibility only.**

The ETTm1 matrix completed 12/12 architecture × fixture components plus merge.

Control R^2 gate results:

- LSTM: 0/3 pass;
- TCN: 0/3 pass;
- TimesNet: 2/3 pass;
- PatchTST: 3/3 pass.

However, PatchTST forecast MSE is roughly 1.74–2.20, while LSTM/TCN/TimesNet are mostly around 0.03–0.12. The topology cannot therefore be interpreted as a clean architecture comparison because model quality is not matched. The current Time domain is retained as **feasibility + quality-gate evidence**, not an architecture law.

## C7 — Collision topology provides operational utility beyond descriptive geometry

**Status: verified for several narrow decisions; external replication still required.**

### Training-time collision risk monitoring

On the frozen CIFAR-10 training trajectory, rank-8 early collision capacity has Spearman correlations with final collision capacity of approximately:

- epoch 3: 0.928;
- epoch 6: 0.990;
- epoch 9: 0.991.

The corresponding early-accuracy correlations are approximately 0.763, 0.772 and 0.772. Partial correlations controlling architecture and early accuracy remain positive. This supports an **early structural risk-monitoring** use case; it is not a causal early-warning theorem.

### Limited audit prioritization

At a 20% audit budget, all six registered rank-8/rank-32 × epoch-3/6/9 cells capture more final collision-capacity mass when prioritized by early K_C than when prioritized by early accuracy. The mean lift is about +16.86 percentage points; the strongest registered cell is rank 32 / epoch 3: 70.29% versus 43.48%.

The advantage narrows or disappears at larger audit budgets. The operational claim is therefore specifically about **scarce audit budget**, not universal superiority.

### Guard structural residual risk

Frozen descriptive comparison:

- DistilBERT Guard: benign acceptance ~0.970, injection acceptance ~0.0113, K_C = 0;
- ProtectAI guard: benign acceptance ~0.955, injection acceptance ~0.3038, K_C = 16.0 ± 1.73.

Among accepted injections for the ProtectAI guard, about 33.8% form independent failure capacity and each independent failure corresponds to about 7.74 collision edges on average. This is a residual-risk structural description, not an incremental causal prediction result.

## C8 — Quantum collision-capacity estimation has a query-complexity advantage

**Status: verified empirical classical scaling versus analytic packed-promise quantum contour; general-overlap theorem remains open.**

In the completed Query-Budget Scaling v2 experiment:

- measured classical endpoint-query relative-error exponent is ~0.5484;
- measured classical additive-error exponent is ~0.450;
- the packed QCCS analytic contour has exponent 1/3;
- over the current N <= 16384 grid, the measured classical-query / analytic-QCCS-proxy ratio spans about 6.35x–40.32x, mean ~21.15x.

This supports the statement **same-oracle measured classical query scaling versus an analytic quantum contour under the packed/disjoint promise**. It does not support a hardware runtime advantage or a general arbitrary-overlap quantum theorem.

The overlap negative controls behave as intended: edge-count proxies fail in hub/overlap regimes where edge multiplicity does not identify maximum matching, while full reconstruction plus exact matching recovers K_C.

## C9 — General overlap QCCS theorem

**Status: theory-open.**

The current defensible theorem target is bounded-degree overlap, not arbitrary collision graphs. The desired result is an isolation/sketch theorem that estimates maximum matching capacity K_C under the same oracle used by the classical comparison. The dependence on maximum degree and the epsilon-dependent lower bound are not yet proved.

No document or executable benchmark should be interpreted as proving:

- arbitrary-overlap optimality;
- an epsilon^{-1} lower bound in the raw endpoint-record oracle;
- free coherent neural inference / free QRAM;
- end-to-end quantum wall-clock advantage.

## Current main-paper eligibility summary

| Surface | Evidence status | Main-claim eligibility |
|---|---|---|
| Digits/Fashion topology + basin proliferation | verified | yes, within stated datasets/models |
| CIFAR-10 topology / training utility | verified operational evidence | yes, for narrow registered utilities |
| CIFAR-100 performance-matched architecture table | pending selected-v3 | not yet |
| Text matched-rank universal learned-subspace effect | falsified | no universal effect |
| Audio common-control architecture law | current-design-failed | no |
| Graph common-control architecture law | current-design-failed | no |
| ETTm1 architecture law | feasibility / performance mismatch | no |
| Guard residual-risk structure | verified descriptive | yes, descriptive only |
| Query-Budget Scaling v2 | verified empirical-vs-analytic contour | yes, under promise/claim boundary |
| General-overlap QCCS theorem | theory-open | no |
| Frozen pretrained vision atlas | pending | not yet |

## Paper-level consequence

The strongest defensible paper is not "functional collision topology is universally identical across every neural architecture." The accumulated evidence instead supports a stronger and more falsifiable framing:

> Functional collision topology is a measurable representation-level object whose capacity, basin structure and discoverability can carry predictive/operational information, while the validity and geometry of the collision object depend critically on the chosen control interface. Quantum collision-capacity sketching is therefore meaningful only after the neural collision object has passed domain-specific validity gates.

This framing keeps the positive vision/utility/quantum results, incorporates the Text/Audio/Graph/Time falsifications, and prevents cross-domain overclaiming.
