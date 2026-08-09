# Stage 16 — Cross-Dataset Functional Collisions and Cross-Family Impersonation

## Goal and frozen protocol

Stage 16 tests whether the MNIST result is a dataset-specific phenomenon. The public response code is **not retuned**:

- frozen F-v1 32-bit public functional code;
- fixed nondegenerate public-response window, response weight 10..22;
- strict collision predicate x != x', y(x) != y(x'), F(x)=F(x');
- same 24-bit Stage-15 cross-family secret challenge bank: rotation, Gaussian blur, contrast, brightness, gamma, and shear.

New datasets:

- FashionMNIST;
- KMNIST.

Representative victims:

- MLP;
- ResNet18-like convolutional network;
- TinyViT.

Seeds: 1,7,13. Each Stage-16 job trains the victim once and then generates both the public F-v1 and cross-family secret responses from that exact victim instance, avoiding cross-stage retraining drift.

Total: **18 dataset-victim-seed cells**.

## E64 — Strict public collision census

All **18/18 cells** contain at least one strict cross-semantic F-v1 collision.

Representative cell results:

| dataset | victim | seed | test acc. | valid samples | strict collisions | semantic pairs | collision density |
|---|---|---:|---:|---:|---:|---:|---:|
| FashionMNIST | MLP | 1 | 84.92% | 1,311 | 131 | 12 | 1.82e-4 |
| FashionMNIST | MLP | 7 | 86.24% | 1,183 | 45 | 10 | 7.52e-5 |
| FashionMNIST | MLP | 13 | 86.17% | 1,492 | 74 | 13 | 7.67e-5 |
| FashionMNIST | ResNet18 | 1 | 91.44% | 1,980 | 4,009 | 22 | 2.49e-3 |
| FashionMNIST | ResNet18 | 7 | 91.31% | 2,175 | 1,288 | 21 | 6.46e-4 |
| FashionMNIST | ResNet18 | 13 | 88.90% | 2,907 | 3,234 | 32 | 9.20e-4 |
| FashionMNIST | TinyViT | 1 | 82.90% | 2,969 | 78 | 12 | 2.10e-5 |
| FashionMNIST | TinyViT | 7 | 83.61% | 3,285 | 283 | 14 | 6.15e-5 |
| FashionMNIST | TinyViT | 13 | 83.38% | 4,421 | 1,429 | 15 | 1.70e-4 |
| KMNIST | MLP | 1 | 85.34% | 1,439 | 67 | 12 | 7.38e-5 |
| KMNIST | MLP | 7 | 85.27% | 1,274 | 29 | 6 | 4.10e-5 |
| KMNIST | MLP | 13 | 85.53% | 1,380 | 10 | 5 | 1.22e-5 |
| KMNIST | ResNet18 | 1 | 93.57% | 1,203 | 461 | 23 | 7.81e-4 |
| KMNIST | ResNet18 | 7 | 92.37% | 1,135 | 98 | 20 | 1.77e-4 |
| KMNIST | ResNet18 | 13 | 94.56% | 784 | 76 | 16 | 3.04e-4 |
| KMNIST | TinyViT | 1 | 75.97% | 3,235 | 1,672 | 36 | 3.62e-4 |
| KMNIST | TinyViT | 7 | 78.54% | 3,407 | 7,271 | 28 | 1.44e-3 |
| KMNIST | TinyViT | 13 | 76.43% | 3,312 | 3,629 | 38 | 7.54e-4 |

The TinyViT KMNIST accuracy is lower than the ResNet/MLP accuracy, so the strongest cross-dataset generality claim should not rely only on TinyViT. The high-accuracy KMNIST ResNet cells independently reproduce the collision phenomenon.

## E65 — Cross-family secret behavior versus class-pair-matched random

For every cell, each strict public collision pair is compared with random valid pairs preserving the exact semantic class-pair composition.

Across all 18 cells:

- mean collision cross-family secret agreement: **86.46%**;
- mean matched-random agreement: **76.36%**;
- mean uplift: **+10.11 percentage points**;
- positive-uplift cells: **18/18**;
- significant cells (empirical matched-resampling p<.05): **17/18**.

The only nonsignificant cell is KMNIST MLP seed 13, which has only 10 strict collisions and a small positive full-bank uplift of about +3.0 pp. It is retained as a weak cell rather than removed.

Dataset-victim aggregate:

| dataset | victim | mean accuracy | median strict collisions | mean secret agreement | mean matched random | mean uplift | positive/significant seeds |
|---|---|---:|---:|---:|---:|---:|---:|
| FashionMNIST | MLP | 85.78% | 74 | 89.22% | 74.78% | **+14.44 pp** | 3/3 / 3/3 |
| FashionMNIST | ResNet18 | 90.55% | 3,234 | 90.91% | 80.48% | **+10.43 pp** | 3/3 / 3/3 |
| FashionMNIST | TinyViT | 83.30% | 283 | 81.03% | 75.05% | **+5.98 pp** | 3/3 / 3/3 |
| KMNIST | MLP | 85.38% | 29 | 85.59% | 77.94% | **+7.65 pp** | 3/3 / 2/3 |
| KMNIST | ResNet18 | 93.50% | 98 | 84.00% | 71.78% | **+12.22 pp** | 3/3 / 3/3 |
| KMNIST | TinyViT | 76.98% | 3,629 | 88.03% | 78.11% | **+9.92 pp** | 3/3 / 3/3 |

## E66 — Secret information-content control

As in Stage 15, secret components whose marginal flip probability over all valid samples is outside [5%,95%] are removed independently for each victim. This selection uses no collision-pair outcome.

Across all 18 cells after filtering:

- mean collision agreement: **79.28%**;
- mean matched-random agreement: **66.40%**;
- mean uplift: **+12.88 pp**;
- positive-uplift cells: **18/18**;
- significant cells: **17/18**.

Mean filtered uplift by dataset-victim combination:

- Fashion MLP: +17.61 pp;
- Fashion ResNet18: +13.65 pp;
- Fashion TinyViT: +7.53 pp;
- KMNIST MLP: +11.18 pp;
- KMNIST ResNet18: +14.63 pp;
- KMNIST TinyViT: +12.65 pp.

Thus the cross-dataset effect is not explained by near-constant hidden-response bits.

## E67 — Random equal-K valid-pool control

A first attempted equal-K control selected samples whose public F-v1 Hamming weight was closest to 16/32. That rule substantially increased code clustering and artificially inflated collision density. This version was explicitly rejected and is **not** used as evidence.

The valid control instead samples uniformly from each cell's already frozen valid pool. For every cell, K=400 samples were drawn without replacement and the experiment was repeated 20 times.

Results:

- all 18 cells have positive mean collision-vs-matched-random secret uplift under the random equal-K control;
- median subset hit probability is 1.0;
- minimum subset hit probability is 0.70 (Fashion TinyViT seed1);
- the weakest cell remains KMNIST MLP seed13, with about +2.1 pp average uplift.

Representative random-K=400 mean uplifts:

- Fashion MLP seeds 1/7/13: +12.24 / +12.84 / +11.99 pp;
- Fashion ResNet seeds 1/7/13: +11.19 / +10.07 / +8.64 pp;
- Fashion TinyViT seeds 1/7/13: +8.93 / +7.43 / +6.11 pp;
- KMNIST ResNet seeds 1/7/13: +8.58 / +11.73 / +15.27 pp;
- KMNIST TinyViT seeds 1/7/13: +10.90 / +7.99 / +10.58 pp.

Therefore the Stage-16 direction is not a consequence of different valid-pool sizes. The rejected Hamming-weight-centered equal-K experiment is retained as a methodological warning against conditioning candidate pools on response-code weight.

## Main-line implication

The main empirical claim can now be strengthened from a single-dataset observation to a cross-dataset behavioral-aliasing phenomenon:

1. the **same frozen public F-v1 definition** produces strict cross-semantic functional collisions on MNIST, FashionMNIST, and KMNIST;
2. on the two new datasets, all 18 representative architecture/seed cells contain strict collisions;
3. all 18 show positive cross-family secret behavioral similarity relative to semantic-pair-matched random controls, with 17/18 significant;
4. the result holds in MLP, convolutional ResNet, and Transformer-style TinyViT victims and under random equal-size valid-pool resampling.

This does not establish universality over arbitrary data modalities or all neural architectures. It does substantially reduce the risk that the MNIST result is a digit-specific or single-architecture artifact.

The quantum VT-FCW resource statement remains separate and conditional on coherent component-oracle access; Stage 16 strengthens the prevalence and downstream-security side of the paper, not the physical-oracle assumption.
