# Q-COLLIDE main experiment protocol

## Frozen mainline

The paper studies **functional collision topology in learned control bottlenecks**, the resulting independent collision capacity, and the query complexity of discovering or certifying that capacity.

The causal/measurement chain is frozen as

```text
learned control representation
  -> functional collision graph
  -> distributed collision basins and maximum-matching capacity
  -> persistent collision topology
  -> collision-discovery/query complexity
```

The real Guard/LLM experiment is an application of the same object; it is not a separate jailbreak paper. Quantum quantities are query-complexity objects unless coherent execution is explicitly implemented.

## Main Experiment A — performance-matched vision topology

### Datasets

1. CIFAR-10 confirmatory main experiment.
2. CIFAR-100 external-validity experiment.

### Architectures

- CIFAR-adapted ResNet-18 control model;
- Tiny-ViT control model;
- MLP-Mixer control model.

The hidden dimension and control interface are shared and parameter counts must remain within the recorded architecture budget.

### Replication

Each dataset uses five independent model seeds: `0,1,2,3,4`.

### Performance matching

Architecture claims are not made from final epochs alone. The experiment trains a frozen checkpoint grid and separately replays checkpoint classification performance.

For each architecture × seed, select the **observed checkpoint** whose calibration accuracy is closest to the preregistered dataset target. No interpolation is permitted.

- CIFAR-10 target: `0.58`, tolerance `0.04`.
- CIFAR-100 target: `0.30`, tolerance `0.05`.

If any model lacks a checkpoint inside tolerance, the performance-matched architecture Gate fails. Final-checkpoint results may still be reported descriptively but cannot support the architecture-ordering claim.

### Main ranks

The performance-matched main table uses visible control ranks `1,8,32`. The final checkpoint additionally reports ranks `1,2,4,8,16,32`.

### Primary measurements

For each performance-matched architecture × rank cell report:

- evaluation accuracy;
- nominal collision capacity fraction `K_C / min(|A|,|B|)`;
- collision-basin density `beta_0 / min(|A|,|B|)`;
- cycle density `beta_1 / |E|`;
- normalized component edge entropy;
- hidden displacement entropy rank;
- capacity filtration AUC;
- capacity robustness ratio;
- persistent basin lifetime.

### Confirmatory hypotheses

1. **Existence/persistence:** nonzero collision capacity persists over a nontrivial control-threshold interval in learned models.
2. **Architecture fingerprint:** performance-matched architectures exhibit reproducible differences in the vector of collision-topology measurements beyond seed variation.
3. **Distributed-basin mechanism:** when an architecture has larger capacity, decompose the difference into basin-density and within-basin multiplicity contributions rather than assuming one giant collision component.
4. **Training formation:** checkpoint trajectories are reported as topology-formation dynamics; they are not called a phase transition without finite-size scaling.

## Main Experiment B — real Guard/LLM functional-collision application

The frozen application uses public existing prompt-injection fixtures only. No jailbreak content is generated.

Current five-seed matrix:

- `protectai/deberta-v3-base-prompt-injection-v2`;
- `fmops/distilbert-prompt-injection`;
- downstream behavior model: `HuggingFaceTB/SmolLM2-135M-Instruct`;
- dataset: `deepset/prompt-injections`;
- fixture seeds: `20260811` through `20260815`.

Primary application columns are guard acceptance, collision edge count, exact maximum-matching capacity, capacity fraction, functional divergence, and discovery efficiency under matched query budgets.

Random-pair discovery and nearest-control discovery are classical baselines. Packed classical/quantum values are explicitly analytic endpoint-query proxies, not runtime measurements.

## Main Experiment C — overlap and quantum-capacity scaling

Synthetic graph families include disjoint matching, sparse bounded overlap, hub/star negative control, and dense bipartite overlap.

The benchmark must keep exact maximum matching as ground truth and retain negative controls that falsify naive edge-count or packed-inversion extensions.

For arbitrary finite bipartite overlap the currently defensible certificate is

```text
ceil(|E| / Delta) <= K_C <= min(|A|, |B|, |E|)
```

with the survival envelope

```text
1 - (1-q^2)^K_C <= S_G(q) <= min(1, K_C Delta q^2).
```

The experiment may study empirical degree-aware estimators, but it must not relabel them as universal unbiased estimators.

## Main tables

### Table 1 — Performance-matched collision topology

Rows: dataset × architecture × visible rank. Columns: accuracy, `K_C`, basin density, cycle density, persistence AUC, persistent basin lifetime, displacement rank.

### Table 2 — Guard/LLM application

Rows: guard model. Columns: five-seed acceptance, `K_C`, capacity fraction, nonzero-seed fraction, query budget to recover capacity, random-search capacity, nearest-control capacity, analytic query proxies.

### Table 3 — Quantum/overlap scaling

Rows: graph family × `N` × `K_C` × `Delta`. Columns: exact capacity, certificate width, estimator error, classical query scale, quantum query scale, and claim status.

## Fail-closed rules

- Do not interpret final-checkpoint architecture differences as causal when performance matching fails.
- Do not replace maximum matching with raw edge count.
- Do not report a quantum advantage for `K_C=0` proxy cells.
- Do not call analytic query proxies coherent execution or hardware speedup.
- Do not suppress hub/dense negative controls.
- Do not infer a training phase transition from checkpoint curves alone.
