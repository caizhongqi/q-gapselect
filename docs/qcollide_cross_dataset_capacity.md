# Cross-dataset collision-tunnel capacity audit

This audit combines the frozen scikit-learn digits and Fashion-MNIST
CNN/Tiny-ViT experiments. It evaluates the stable claim

\[
\text{control-tunnel openness}
\longrightarrow
\text{independent collision packing},
\]

without assuming a dataset-invariant scalar law.

## Frozen evidence

The meta-analysis contains 16 baseline cells: two datasets, two architectures,
and visible control ranks `1,2,4,8`.

Across all cells, the Pearson association between mean openness and maximum
matching packing fraction is

\[
r=0.8122.
\]

All four dataset--architecture strata have non-increasing openness and packing
as visible control rank increases.

A linear model with openness, dataset, and architecture effects obtains

\[
R^2=0.9543.
\]

Using Fashion-MNIST/CNN as the reference, the fitted coefficients are:

- openness: `+0.6131`;
- scikit-learn digits: `-0.1853`;
- Tiny-ViT: `+0.1640`.

Adding `log2(visible rank)` raises the descriptive fit to `R^2=0.9641`, but
leave-one-dataset-out RMSE remains about `0.20`. The result therefore supports a
shared geometric mechanism with substantial task and architecture offsets, not
one universal capacity curve.

## Fashion-MNIST external-validity result

Fashion-MNIST uses 12,000 training images, 2,000 calibration images, 2,000
evaluation images, and three independent seeds per architecture. Mean
evaluation accuracies are `0.8370` for CNN and `0.8313` for Tiny-ViT.

Baseline packing fractions at visible ranks `1,2,4,8` are:

- CNN: `0.7167, 0.5167, 0.4167, 0.2667`;
- Tiny-ViT: `0.8333, 0.7167, 0.7167, 0.5500`.

The corresponding baseline openness values are:

- CNN: `0.9169, 0.7389, 0.5314, 0.2558`;
- Tiny-ViT: `0.9408, 0.8774, 0.7321, 0.5803`.

At added closure rank six, targeted closure reduces CNN packing to `0.05` at
all four visible ranks. Tiny-ViT still retains packing between `0.15` and
`0.20`, so the current closure budget does not close its adaptive tunnels.

## Claim boundary

Supported:

- conditional collision packing persists across two real-image datasets and
  both convolutional and transformer encoders;
- openness and independent packing decline monotonically with control rank;
- task and architecture materially shift the capacity curve;
- redundant row-space interventions remain inert under the whitened control
  metric.

Not supported:

- one dataset-invariant scalar curve fully determines packing;
- six added control directions are sufficient for every architecture;
- the static dangerous-spectrum rank is universal;
- production-scale, physical, or coherent-quantum advantage claims.
