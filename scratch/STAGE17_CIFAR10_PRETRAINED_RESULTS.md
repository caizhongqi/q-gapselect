# Stage 17 — CIFAR-10 Pretrained Victim Results

## Locked attack predicate

Stage 17 keeps the main-line functional collision attack unchanged:

\[
x \ne x',\qquad y(x)\ne y(x'),\qquad F_{\rm C10-v1}(x)=F_{\rm C10-v1}(x').
\]

`F-C10-v1` is a frozen 32-bit hard-label response code made of 8 two-pixel zero-fill shifts, 16 black 8x8 occlusions on a 4x4 grid, and 8 fixed RGB sign-noise probes at epsilon 8/255. The frozen nondegenerate response-weight window is 10..22. The independent 24-bit secret bank contains rotation, Gaussian blur, contrast, brightness, gamma, and shear; it is never used to construct a public collision.

The full CIFAR-10 test set (10,000 examples) was evaluated with three public pretrained victims.

| victim | clean acc. | valid samples | strict cross-semantic claws | semantic class pairs | collision density | secret agreement | matched random | uplift | empirical p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet20 | 92.60% | 540 | 27 | 11 | 2.120e-4 | 75.46% | 63.15% | **+12.32 pp** | 0.00498 |
| MobileNetV2 x0.5 | 93.12% | 558 | 80 | 27 | 6.047e-4 | 77.19% | 63.92% | **+13.26 pp** | 0.00498 |
| VGG11-BN | 92.79% | 547 | 1 | 1 | 7.612e-6 | 79.17% | 65.48% | **+13.69 pp** | 0.21393 |

The VGG cell has only one strict collision, so its positive secret uplift is descriptive rather than statistically stable. The two higher-count pretrained cells independently show significant hidden cross-family behavioral agreement.

## Why the earlier micro test looked weak

The fixed 100-per-class (N=1000) exact adaptive-prefix micro test produced 0 / 4 / 0 strict claws for ResNet20 / MobileNetV2 x0.5 / VGG11-BN. That does **not** contradict the full-set result. The full 10,000-example census contains 27 / 80 / 1 strict claws respectively. The sparse VGG population is especially easy to miss in a 1k subset.

The adaptive-prefix implementation remains exact: it prunes a sample only when its current prefix bucket is already single-semantic (future refinement cannot merge buckets) or its partial response weight can no longer enter 10..22. On the fixed 1k subset it reduced component calls by about 31% for all three victims while preserving the exact final 32-bit collision set.

## Main-line implication

1. The frozen 32-bit attack survives transfer from MNIST/FashionMNIST/KMNIST to natural RGB CIFAR-10 without weakening the collision predicate.
2. The effect is present in three independently pretrained convolutional/mobile families, not only custom trained victims.
3. Strict public collisions show positive cross-family secret behavioral similarity in all three full-set victims, with statistically stable evidence in ResNet20 and MobileNetV2 x0.5.
4. Collision abundance is strongly architecture dependent; therefore the next experiment must broaden public pretrained victim families before making any architecture-universality statement.

Stage 18 is therefore frozen as a broad-victim CIFAR-10 replication under exactly the same F-C10-v1 and secret-bank protocol, adding ResNet56, VGG16-BN, MobileNetV2 x1.0, ShuffleNetV2 x1.0, RepVGG-A0, and ViT-B16.
