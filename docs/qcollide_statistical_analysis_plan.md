# Q-COLLIDE pre-registered statistical analysis plan

This plan is frozen before the new five-seed CIFAR-10/CIFAR-100 main-table artifacts are read.

## Experimental unit

The experimental unit for architecture comparisons is an independently trained **model seed**. Control ranks, threshold-filtration points, collision edges, and graph components are repeated measurements inside one trained model and are never counted as independent replicates.

## Performance matching gate

Architecture inference is conditional on the performance-matching Gate passing. For every architecture × seed, an observed checkpoint must lie within the pre-registered calibration-accuracy tolerance. No interpolation between checkpoints is permitted.

If this Gate fails, the corresponding dataset remains descriptive and cannot support the confirmatory architecture claim.

## Primary confirmatory contrast

The single primary architecture contrast is

```text
Tiny-ViT collision-capacity fraction > ResNet-18 collision-capacity fraction
```

at visible control rank `8`, after performance matching.

The test is a one-sided exact paired sign-flip randomization test across the five matched model seeds. With five pairs the smallest attainable one-sided exact p-value is `1/32 = 0.03125`. No ranks or filtration thresholds are pooled to manufacture a larger sample size.

CIFAR-10 is the confirmatory dataset. CIFAR-100 is an external-validity replication using the same direction, metric, rank, and test. The direction is therefore fixed before either new main-table result is inspected.

## Mechanism analysis

The pre-registered mechanism contrast at rank `8` is Tiny-ViT versus ResNet-18 basin density. It is secondary evidence and does not replace the primary capacity test.

For each matched seed, capacity is decomposed as

```text
C = B * M
```

where `B` is collision-basin density and `M=C/B` is within-basin multiplicity. The architecture gap is split by the symmetric exact decomposition

```text
Delta_C_basin = 0.5 * (B_A-B_B) * (M_A+M_B)
Delta_C_mult  = 0.5 * (M_A-M_B) * (B_A+B_B)
```

so the two contributions add exactly to `C_A-C_B`. This prevents the mechanism analysis from attributing the same gap twice.

## Robustness analyses

Ranks `1` and `32` repeat the primary capacity contrast as robustness checks. They are not additional confirmatory samples. Persistent capacity AUC, cycle density, component entropy, displacement entropy rank, and persistent basin lifetime are secondary topology-fingerprint measurements reported with seed-level observations, mean, and standard error.

MLP-Mixer is treated as a mechanistic bridge architecture. Mixer pairwise contrasts are secondary/exploratory rather than additional primary tests.

## Guard/LLM application

The five fixture seeds in the Guard application measure robustness to prompt subsampling. They are not five independently trained guard models, so they are reported as repeated-fixture evidence rather than used to claim a population of guard architectures.

## Quantum/overlap scaling

No statistical test is used to 'prove' the analytic classical square-root or quantum cube-root query exponents. Those are theoretical query-scaling expressions. Monte Carlo experiments evaluate survival-capacity certificate coverage, finite-sample width, and estimator bias under overlap.

## Fail-closed reporting

- Do not pool ranks or thresholds as independent observations.
- Do not change the primary rank or direction after seeing the main results.
- Do not claim architecture causality if performance matching fails.
- Do not interpret `K=0` analytic query-proxy cells as a quantum discovery advantage.
- Do not describe query-complexity proxies as wall-clock or hardware speedup.
