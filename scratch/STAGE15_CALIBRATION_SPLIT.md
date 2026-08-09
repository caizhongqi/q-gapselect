# Stage 15 addendum — Calibration/evaluation split for secret challenge selection

To eliminate the possibility that the secret challenge filter was indirectly tuned on the same evaluation population, valid samples were deterministically split by sample index:

- defender calibration: index mod 5 = 0 (about 20%);
- evaluation: the remaining about 80%.

Only the calibration partition is used to estimate each of the 24 cross-family secret bit marginal rates and binary entropies. Secret challenges are ranked by binary entropy on calibration data alone. Public F-v1 collision construction, class-pair-matched random controls, and secret agreement evaluation are then performed only on the disjoint evaluation partition.

No collision-pair statistic enters challenge selection.

Three pre-specified secret-bank sizes were evaluated: top-8, top-12, and top-16 entropy-ranked challenges.

Across all 13 Stage-15 victim/seed cells that retain at least one evaluation-partition collision, the direction remains overwhelmingly positive but small-pair TinyCNN/LeNet cells are statistically noisy. Restricting to the **8 stable evaluation cells with at least 100 strict collision pairs** gives:

| entropy-selected secret bits | stable cells | positive uplift | significant p<.05 | mean agreement uplift | median uplift |
|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 8/8 | 8/8 | +9.63 pp | +9.38 pp |
| 12 | 8 | 8/8 | 8/8 | +8.69 pp | +8.45 pp |
| 16 | 8 | 8/8 | 8/8 | +7.57 pp | +7.64 pp |

For top-12 challenges, representative evaluation-only cells are:

- MLP seed1: 477 collision pairs, 97.33% vs 87.40%, +9.92 pp;
- MLP seed7: 639 pairs, 92.31% vs 85.01%, +7.30 pp;
- MLP seed13: 260 pairs, 88.40% vs 81.33%, +7.07 pp;
- ResNet18 seed7: 2,084 pairs, 86.22% vs 76.62%, +9.60 pp;
- ResNet18 seed13: 12,291 pairs, 88.75% vs 79.29%, +9.46 pp;
- TinyViT seed1: 11,833 pairs, 79.79% vs 72.35%, +7.43 pp;
- TinyViT seed7: 1,193 pairs, 91.77% vs 79.44%, +12.33 pp;
- TinyViT seed13: 3,712 pairs, 88.46% vs 82.08%, +6.38 pp.

This control strengthens the interpretation that public functional collisions predict cross-family secret behavioral similarity rather than exploiting challenge-selection leakage.
