# Q-COLLIDE Universal Neural Collision Atlas — frozen experiment protocol

## Scientific mainline

Q-COLLIDE studies a cross-domain neural object rather than a single visual benchmark:

```text
neural representation / control bottleneck
  -> functional collision graph G_C
  -> exact independent collision capacity K_C = nu(G_C)
  -> collision-basin topology and persistence
  -> query-limited collision discovery
  -> quantum capacity discovery / certification
  -> attack, transfer and intervention consequences
```

The atlas is designed to test whether functional-collision topology is a recurring structural property across neural-network paradigms, tasks and modalities, not whether one architecture wins on one dataset.

## Scope

The frozen registry contains eight domains:

1. image classification;
2. language understanding;
3. audio;
4. graph learning;
5. time series;
6. multimodal learning;
7. Guard + downstream LLM systems;
8. neural policy/control systems.

The registry must contain at least 40 system cells, 20 unique model names, 12 datasets/tasks and 12 architecture families before the atlas may be called cross-domain.

## Cross-domain metric contract

Every domain adapter must produce an aligned bipartite collision graph and the following common measurements whenever mathematically meaningful:

- `capacity_fraction = K_C / min(|A|,|B|)`;
- `basin_density = beta_0 / min(|A|,|B|)`;
- `within_basin_multiplicity = K_C / beta_0`;
- `cycle_density = beta_1 / |E|`;
- normalized component edge entropy;
- hidden/payload displacement entropy rank;
- capacity-filtration AUC;
- persistent basin lifetime;
- overlap maximum degree `Delta`;
- discovery capacity under fixed query fractions `K_C@Q`.

Raw distances are domain-specific and are never pooled directly across modalities. Cross-domain analyses use normalized topology/capacity quantities.

# Planned paper tables

## Table 1 — Universal Neural Functional Collision Atlas

One row per dataset × model system. This is the largest table and the first empirical claim surface.

| Domain | Task/Dataset | Model | Family | Params | Task score | K_C/N | beta0/N | K_C/beta0 | beta1/|E| | Capacity AUC | Persistence | D_C | Delta |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vision | CIFAR-10 | ResNet-18 | residual CNN | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Vision | CIFAR-10 | Tiny-ViT | global Transformer | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Vision | ImageNet subset | ConvNeXt-T | modern CNN | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Text | 20 Newsgroups | BERT-base | encoder Transformer | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Text | 20 Newsgroups | DeBERTa-v3-small | disentangled attention | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Audio | Speech Commands V2 | wav2vec2 | conv+Transformer | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Audio | Speech Commands V2 | AST | audio Transformer | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Graph | Cora | GCN | graph convolution | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Graph | Cora | GAT | graph attention | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Time series | ETTm1 | LSTM | recurrent | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Time series | ETTm1 | PatchTST | temporal Transformer | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Multimodal | Flickr30k | CLIP-RN50 | CNN+text Transformer | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Multimodal | Flickr30k | CLIP-ViT | dual Transformer | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Guard/LLM | prompt injection | ProtectAI + SmolLM2 | guard+LM | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Policy | MiniGrid | PPO-CNN | CNN policy | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |

`--` is a planned cell, never a fabricated value.

## Table 2 — Architecture-family law across domains

Aggregates normalized atlas metrics by inductive-bias family while controlling for domain, task score and model scale.

| Architecture factor | Compared systems | Delta K_C/N | Delta basin density | Delta multiplicity | Delta persistence | domain-adjusted effect | support |
|---|---:|---:|---:|---:|---:|---:|---:|
| local convolution -> global attention | -- | -- | -- | -- | -- | -- | -- |
| message passing -> graph attention | -- | -- | -- | -- | -- | -- | -- |
| recurrent -> temporal Transformer | -- | -- | -- | -- | -- | -- | -- |
| CNN image tower -> ViT image tower | -- | -- | -- | -- | -- | -- | -- |
| weak -> strong control bottleneck | -- | -- | -- | -- | -- | -- | -- |

This table is exploratory until enough independent model/task systems exist; individual seeds/ranks are not treated as independent architecture samples.

## Table 3 — Layerwise collision topology

Tests whether collision capacity appears only at the final representation or evolves across depth.

| Domain | Model | layer depth fraction | K_C/N | beta0/N | D_C | persistence | task separability |
|---|---|---:|---:|---:|---:|---:|---:|
| Vision | ResNet | 0.25 | -- | -- | -- | -- | -- |
| Vision | ResNet | 0.50 | -- | -- | -- | -- | -- |
| Vision | ResNet | 0.75 | -- | -- | -- | -- | -- |
| Vision | ViT | 0.25/0.50/0.75/1.0 | -- | -- | -- | -- | -- |
| Text | BERT | 0.25/0.50/0.75/1.0 | -- | -- | -- | -- | -- |
| Audio | wav2vec2/AST | 0.25/0.50/0.75/1.0 | -- | -- | -- | -- | -- |
| Graph | GCN/GAT | early/mid/late | -- | -- | -- | -- | -- |

## Table 4 — Model-scale law

Within architecture families, vary depth/width/model size while holding task/data protocol fixed.

| Domain | family | model size | Params | task score | K_C/N | basin density | persistence | K_C per million params |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Vision | ResNet | 18/34/50 | -- | -- | -- | -- | -- | -- |
| Vision | ViT | tiny/small/base | -- | -- | -- | -- | -- | -- |
| Text | BERT-like | Distil/base | -- | -- | -- | -- | -- | -- |
| Audio | Transformer | base variants | -- | -- | -- | -- | -- | -- |

## Table 5 — Performance-matched controlled architecture study

This retains the existing CIFAR-10/CIFAR-100 five-seed experiment as a controlled sub-study rather than the whole paper.

| Dataset | Architecture | matched accuracy | rank | K_C/N | beta0/N | multiplicity | capacity AUC | persistence |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | ResNet-18 | -- | 8 | -- | -- | -- | -- | -- |
| CIFAR-10 | Tiny-ViT | -- | 8 | -- | -- | -- | -- | -- |
| CIFAR-10 | Mixer | -- | 8 | -- | -- | -- | -- | -- |
| CIFAR-100 | ResNet-18 | -- | 8 | -- | -- | -- | -- | -- |
| CIFAR-100 | Tiny-ViT | -- | 8 | -- | -- | -- | -- | -- |
| CIFAR-100 | Mixer | -- | 8 | -- | -- | -- | -- | -- |

## Table 6 — Training/topology formation dynamics

| Dataset | model | checkpoint | task score | K_C/N | beta0/N | D_C | persistent basin count |
|---|---|---:|---:|---:|---:|---:|---:|
| -- | -- | -- | -- | -- | -- | -- | -- |

No phase-transition language is allowed without finite-size scaling.

## Table 7 — Control-rank and threshold spectrum

| Domain | model | visible rank | nominal epsilon | K_C/N | capacity AUC | robustness ratio | Delta |
|---|---|---:|---:|---:|---:|---:|---:|
| -- | -- | -- | -- | -- | -- | -- | -- |

This table tests whether the atlas conclusion survives control bottleneck strength and threshold choice.

## Table 8 — Distribution-shift robustness

| Domain | train/source | shift/test | model | task degradation | K_C shift | basin-density shift | fingerprint stability |
|---|---|---|---|---:|---:|---:|---:|
| Vision | CIFAR-10 | CIFAR-10-C subset | -- | -- | -- | -- | -- |
| Text | 20 Newsgroups | lexical/style perturbation | -- | -- | -- | -- | -- |
| Audio | Speech Commands | noise/SNR shift | -- | -- | -- | -- | -- |
| Graph | Cora | edge/drop-feature shift | -- | -- | -- | -- | -- |
| Time series | ETTm1 | seasonal/noise shift | -- | -- | -- | -- | -- |

## Table 9 — Query-matched collision discovery

The application metric is `K_C@Q`: maximum independent collisions recovered within a matched query budget.

| Domain | model | method | Q/AllPairs | K_C@Q | fraction of full K_C | functional success | collision diversity |
|---|---|---|---:|---:|---:|---:|---:|
| -- | -- | random | 0.01 | -- | -- | -- | -- |
| -- | -- | nearest-control | 0.01 | -- | -- | -- | -- |
| -- | -- | information-matched strong baseline | 0.01 | -- | -- | -- | -- |
| -- | -- | Q-COLLIDE candidate policy | 0.01 | -- | -- | -- | -- |

All baselines receive the same candidate information and query accounting.

## Table 10 — Cross-model collision transfer matrix

Rows are source models used to identify collision witnesses; columns are target models on which the same public/held-out candidates are evaluated.

| source -> target | CNN | ViT | Mixer | BERT | DeBERTa | AST | GAT | CLIP | Guard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | -- | -- | -- | N/A | N/A | N/A | N/A | N/A | N/A |
| ViT | -- | -- | -- | N/A | N/A | N/A | N/A | N/A | N/A |
| BERT | N/A | N/A | N/A | -- | -- | N/A | N/A | N/A | -- |
| Guard | N/A | N/A | N/A | -- | -- | N/A | N/A | N/A | -- |

Cross-domain cells are only defined when the input/task spaces are aligned; undefined cells remain `N/A`, not zero.

## Table 11 — Causal collision closure/intervention

| Domain | model | intervention | rank/budget | task loss | Delta K_C | Delta beta0 | Delta persistence | random-control gap |
|---|---|---|---:|---:|---:|---:|---:|---:|
| -- | -- | targeted closure | -- | -- | -- | -- | -- | -- |
| -- | -- | random null | -- | -- | -- | -- | -- | -- |
| -- | -- | variance matched | -- | -- | -- | -- | -- | -- |
| -- | -- | row/subspace sham | -- | -- | -- | -- | -- | -- |

## Table 12 — Real Guard + LLM application

| Guard | downstream LLM | fixture seeds | benign accept | injection accept | exact K_C | K_C fraction | behavior divergence | K_C@5%Q |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ProtectAI | SmolLM2 | 5 | existing | existing | existing | existing | existing | -- |
| DistilBERT guard | SmolLM2 | 5 | existing | existing | existing | existing | existing | -- |
| ProtectAI | second public LM | 5 | -- | -- | -- | -- | -- | -- |
| DistilBERT guard | second public LM | 5 | -- | -- | -- | -- | -- | -- |

Public existing prompts only; no generated jailbreak corpus is required for the structural application.

## Table 13 — Synthetic overlap / quantum scaling

| family | N | K | Delta | deterministic certificate | survival certificate | estimator error | classical analytic scale | quantum analytic scale |
|---|---:|---:|---:|---|---|---:|---:|---:|
| matching | 64-1024 | controlled | 1 | -- | -- | -- | -- | -- |
| sparse overlap | 64-1024 | controlled | bounded | -- | -- | -- | -- | -- |
| hub | -- | 1 | large | -- | -- | negative control | -- | -- |
| dense | -- | large | large | -- | -- | negative control | -- | -- |

Analytic query laws are never presented as hardware runtime.

## Table 14 — Quantum capacity sketching on real neural collision graphs

This table closes the empirical bridge between the neural atlas and the quantum problem.

| Domain | model | real N | real K_C | real Delta | certificate width | degree-aware error | classical query scale | quantum query scale |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Vision | ResNet | -- | -- | -- | -- | -- | -- | -- |
| Vision | ViT | -- | -- | -- | -- | -- | -- | -- |
| Text | BERT | -- | -- | -- | -- | -- | -- | -- |
| Audio | AST | -- | -- | -- | -- | -- | -- | -- |
| Graph | GAT | -- | -- | -- | -- | -- | -- | -- |
| Time series | PatchTST | -- | -- | -- | -- | -- | -- | -- |
| Multimodal | CLIP | -- | -- | -- | -- | -- | -- | -- |
| Guard | ProtectAI | -- | -- | -- | -- | -- | -- | -- |

## Table 15 — Oracle/resource accounting

| neural system | endpoint record cost | reversible inference assumption | memory/query model | query count | logical depth proxy | claim status |
|---|---:|---|---|---:|---:|---|
| -- | -- | -- | -- | -- | -- | -- |

This table is required before any end-to-end runtime claim.

## Table 16 — Negative controls and falsification ledger

| hypothesis/control | expected falsifier | observed outcome | interpretation |
|---|---|---|---|
| edge count equals capacity | hub/star | -- | -- |
| packed inversion is universal | dense/hub | -- | -- |
| topology is threshold artifact | threshold sweep | -- | -- |
| topology is only accuracy | performance matching | -- | -- |
| closure effect is generic | random/variance/sham closure | -- | -- |
| every guard has collisions | strong guard negative control | -- | -- |
| query proxy equals runtime | resource accounting | prohibited | claim boundary |

# Figure plan

1. Atlas heatmap: systems × normalized topology metrics.
2. UMAP/PCA only as visualization of topology fingerprints, not statistical evidence.
3. Layerwise formation trajectories.
4. Basin-density versus capacity with domain-stratified fits.
5. Query-budget discovery curves.
6. Cross-model transfer matrices.
7. Causal closure trajectories.
8. Synthetic and real-graph quantum scaling.

# Execution tiers

- **Existing:** Digits/Fashion/CIFAR/Guard/overlap results already implemented or running on the base branch.
- **Phase 1:** language-understanding atlas on 20 Newsgroups with four public pretrained encoders.
- **Phase 2:** audio core, graph core, time-series core, second Guard downstream LM, pretrained large vision.
- **Phase 3:** external datasets, multimodal CLIP, distribution-shift and transfer experiments.
- **Phase 4:** BLIP-2 / policy-transformer / heavy resource-accounting extensions.

A later phase may not silently replace a failed earlier result. Negative and null outcomes remain in the atlas.
