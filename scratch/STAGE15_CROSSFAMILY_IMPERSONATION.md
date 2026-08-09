# Stage 15 — Cross-Family Secret Functional-Fingerprint Impersonation

## Goal

Stage 14 used a secret bank whose coordinates, magnitudes, and noise seeds were disjoint from the public F-v1 bank, but whose broad perturbation families still overlapped. Stage 15 removes that overlap.

The public collision predicate is unchanged:

\[
x\ne x',\qquad y(x)\ne y(x'),\qquad F_{\rm public}(x)=F_{\rm public}(x'),
\]

where F-public is the frozen 32-bit F-v1 bank (shifts, occlusions, fixed-sign noise).

The secret bank contains only transformation families absent from F-v1:

- rotations: -15,-10,-5,+5,+10,+15 degrees;
- Gaussian blur: sigma .5,.8,1.1,1.4;
- contrast: .65,.82,1.18,1.35;
- brightness: .72,.86,1.14,1.28;
- gamma: .70,.85,1.15,1.30;
- shear: -12,+12 degrees.

Total: 24 secret hard-label flip bits. No secret bit participates in collision construction.

Each collision pair is compared against valid class-pair-matched random pairs preserving the exact semantic-pair composition. The current workflow retrains each victim once and computes both public and secret responses from the same victim instance, so Stage-15 internal comparisons are valid. Raw collision counts should not be equated to Stage 9 counts because independently retrained CPU/BatchNorm trajectories can differ.

## E61 — Full 24-bit cross-family secret agreement

Across five MNIST victim families and seeds 1/7/13, 13 of the 15 independently trained victim/seed cells contained at least one strict public functional collision. ResNet seed 1 and TinyCNN seed 7 contained no strict public collision in their Stage-15 victim instance and therefore have no downstream impersonation statistic.

Among the 13 evaluable cells:

- mean collision secret agreement: **92.24%**;
- mean class-pair-matched random agreement: **86.09%**;
- mean uplift: **+6.15 percentage points**;
- positive-uplift cells: **13/13**;
- permutation/resampling-significant cells at p<.05: **11/13**.

Representative stable cells:

| victim | seed | strict collisions | secret agreement | matched random | uplift | 80% secret-pass uplift |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 1 | 650 | 98.41% | 93.08% | +5.33 pp | +8.24 pp |
| MLP | 7 | 935 | 95.86% | 91.34% | +4.52 pp | +10.14 pp |
| MLP | 13 | 466 | 94.94% | 89.53% | +5.41 pp | +13.95 pp |
| ResNet18 | 7 | 3,178 | 93.11% | 87.08% | +6.03 pp | +16.02 pp |
| ResNet18 | 13 | 17,865 | 94.02% | 88.00% | +6.02 pp | +16.84 pp |
| TinyViT | 1 | 19,506 | 90.24% | 84.55% | +5.69 pp | +15.38 pp |
| TinyViT | 7 | 1,664 | 95.63% | 87.47% | +8.16 pp | +19.49 pp |
| TinyViT | 13 | 5,732 | 94.33% | 89.74% | +4.58 pp | +12.39 pp |

Small-collision LeNet/TinyCNN cells are reported but are not used to argue high-confidence universality.

## E62 — Information-content control

A high secret agreement can be artificially inflated if many secret bits are almost always zero or almost always one. To remove this explanation without selecting on collision outcome, each victim independently keeps only secret components whose marginal flip rate over **all valid samples** lies in [5%,95%]. The selection criterion uses no collision-pair information.

Among the same 13 evaluable cells:

- mean filtered collision agreement: **83.77%**;
- mean filtered matched-random agreement: **75.20%**;
- mean uplift: **+8.57 percentage points**;
- positive-uplift cells: **13/13**;
- significant cells: **10/13**;
- median number of retained secret components is about 10.

Representative filtered uplifts:

- MLP seeds 1/7/13: +12.74 / +8.14 / +9.37 pp;
- ResNet seeds 7/13: +9.48 / +11.10 pp;
- TinyViT seeds 1/7/13: +8.00 / +12.03 / +6.89 pp.

Tightening the marginal window to [10%,90%] leaves only 2--4 bits for some MLP/LeNet victims and becomes statistically unstable, so [5%,95%] is retained as the pre-specified information-content stress rather than tuning the threshold for a favorable result.

## E63 — Secret transformation-family decomposition

Mean uplift and positive-cell counts across the 13 evaluable cells:

| secret family | mean agreement uplift | positive cells | significant cells |
|---|---:|---:|---:|
| rotation | +9.07 pp | 13/13 | 9/13 |
| Gaussian blur | +8.81 pp | 11/13 | 9/13 |
| contrast | +5.21 pp | 12/13 | 10/13 |
| brightness | +4.18 pp | 13/13 | 10/13 |
| gamma | +2.29 pp | 13/13 | 9/13 |
| shear | +5.18 pp | 11/13 | 10/13 |

Thus the cross-family effect is not carried by one hidden transform family and cannot be explained as same-family coordinate interpolation from public shifts/occlusions/noise.

## Interpretation

Stage 13 showed that generic zero-target-query adversarial-direction transfer is not universal and should not be the central downstream claim. Stages 14--15 identify a consequence that is much more tightly aligned with the object actually found by the collision search:

> A semantically different input with the same public hard-label functional code is substantially more likely than a class-pair-matched random input to reproduce the reference input's behavior under withheld and even cross-family secret challenges.

The appropriate language is therefore **cross-family functional-fingerprint impersonation / behavioral aliasing**, not a universal transferable-adversarial-example attack.

## Boundary

The secret transformations are still hand-designed image transformations. They are deliberately disjoint in family from F-v1, but are not learned adaptive defender challenges and are not a cryptographic authentication protocol. The stronger next validation is cross-dataset reproduction and, later, secret challenge banks learned independently from public F-v1.
