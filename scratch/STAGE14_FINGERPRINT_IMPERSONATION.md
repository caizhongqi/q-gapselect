# Stage 14 — Secret-Challenge Functional-Fingerprint Impersonation

## Attack consequence

The functional-collision predicate remains unchanged and is constructed only on the public 32-bit F-v1 challenge bank:

\[
x\ne x',\qquad y(x)\ne y(x'),\qquad F_{\rm public}(x)=F_{\rm public}(x').
\]

Stage 14 asks a downstream security question that does **not** assume generic adversarial-perturbation transfer:

> If a defender authenticates or fingerprints an input by comparing its response to a hidden challenge bank, how often can a semantically different public functional collision impersonate the reference response?

The hidden challenge bank is the independent 28-probe Stage-9 holdout bank and is never used to construct the collision. A pair passes a secret threshold tau if the two inputs agree on at least tau of the hidden response bits. Every collision pair is compared with class-pair-matched random valid pairs, preserving the exact semantic-pair composition.

All seed-7 results use the independent per-model training jobs, not the earlier serial pilot whose DataLoader generator state was shared across model trainings.

## E57 — Secret-challenge threshold sweep

Across 15 independently trained MNIST victim/seed cells (MLP, TinyCNN, LeNet-5, ResNet-18, TinyViT; seeds 1,7,13):

| hidden agreement threshold | collision pass rate | matched-random pass rate | positive uplift cells | significant cells |
|---:|---:|---:|---:|---:|
| 60% | 94.50% | 78.67% | 14/15 | 12/15 |
| 70% | 70.09% | 49.10% | 15/15 | 11/15 |
| 80% | 39.09% | 17.55% | 13/15 | 12/15 |
| 90% | 12.53% | 1.62% | 13/15 | 13/15 |
| 100% | 0.184% | 0.048% | 7/15 | 7/15 |

At the 80% operating point, the median victim-level uplift is +18.45 percentage points.

Pair-weighted over all 58,172 evaluated collision pairs, the 80% pass rate is 47.26% for functional collisions versus 28.52% for the matched-random control, an absolute increase of +18.74 percentage points.

## E58 — sample-count robustness

Small collision cells are not required for the result. Restricting to the 10 victim/seed cells with at least 100 strict cross-semantic collisions gives:

| threshold | collision pass | matched random | median uplift | positive / significant |
|---:|---:|---:|---:|---:|
| 60% | 96.08% | 82.75% | +11.45 pp | 10/10 / 10/10 |
| 70% | 75.14% | 55.07% | +19.95 pp | 10/10 / 10/10 |
| 80% | 39.55% | 21.46% | +17.87 pp | 10/10 / 10/10 |
| 90% | 6.21% | 2.01% | +3.46 pp | 10/10 / 10/10 |

Thus the effect is not driven by the 1--3-pair LeNet/TinyCNN cells.

## E59 — number of hidden challenges

For each victim, random subsets of the 28 hidden challenges were drawn and the 80% acceptance test was repeated. On the 10 cells with at least 100 collisions:

| number of hidden challenges | collision pass | matched random | uplift | positive cells |
|---:|---:|---:|---:|---:|
| 4 | 35.94% | 26.56% | +9.38 pp | 10/10 |
| 8 | 43.73% | 29.78% | +13.95 pp | 10/10 |
| 12 | 46.11% | 30.09% | +16.03 pp | 10/10 |
| 20 | 50.48% | 31.53% | +18.95 pp | 10/10 |
| 28 | 39.55% | 21.40% | +18.15 pp | 10/10 |

The non-monotone absolute pass rate is expected because an 80% discrete threshold corresponds to different integer acceptance counts as K changes and because different hidden subsets have different challenge difficulty. The important robust quantity here is the collision-vs-matched-random gap, which remains positive for every evaluated stable victim cell at every K.

## E60 — hidden challenge family decomposition

At the 80% threshold, restricting again to the >=100-collision cells:

| hidden challenge family | collision pass | matched random | uplift | positive / significant |
|---|---:|---:|---:|---:|
| 3-pixel shifts (8) | 33.16% | 12.82% | **+20.33 pp** | 10/10 / 10/10 |
| 6x6 occlusions (12) | 82.18% | 66.16% | **+16.02 pp** | 10/10 / 10/10 |
| fixed noise eps=.15 (8) | 34.08% | 35.70% | -1.62 pp | 4/10 / 4/10 |

Therefore the secret-challenge impersonation effect is not explained by two inputs merely sharing stable zero responses under noise. The robust effect is concentrated in withheld geometric shifts and occlusions.

## Interpretation

Stage 13 showed that generic zero-target-query adversarial-direction transfer is architecture dependent and cannot be claimed universally. Stage 14 provides a more direct consequence of the actual attack object:

- the attacker finds a semantically different input with exactly the same public external functional code;
- the collision is then substantially more likely than a class-pair-matched random input to pass unseen functional challenge checks;
- this persists as the number of hidden challenges grows and is strongest on geometric and occlusion challenges, not saturated noise.

This supports the language **functional-fingerprint impersonation / behavioral aliasing attack**, not a universal transferable-adversarial-example claim.

Boundary: the present challenge banks are MNIST perturbation families generated from the same broad transformation types as the public bank, although coordinates/magnitudes/seeds are disjoint. A stronger future gate is cross-family or learned-secret challenge generation. The quantum search claim remains conditional on coherent component-oracle access and is separate from the classical downstream impersonation consequence.
