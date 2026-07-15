# Q-GapSelect

Q-GapSelect is a research codebase for **heterogeneous-discrimination-gap, multi-output quantum
pure exploration**. Its immediate objective is to determine whether Top-$k$ and
matroid best-basis identification admit a relation-aware quantum query complexity
that is both algorithmically attainable and matched by a lower bound. The intended
security application is semantic combinatorial intervention search against code
LLMs, but the quantum identification problem is developed and tested independently
of that application.

The project starts from a strict claim boundary: standard amplitude estimation,
amplitude amplification, quantum best-arm identification, and classical
combinatorial pure exploration are prior primitives. The paper contribution is not
claimed until the proposed direct multi-output algorithm and its lower bound are
proved. Executable simulations and conjectural query accounting are labelled
separately throughout the repository.

## Research questions

1. Can a direct quantum Top-$k$ identification algorithm exploit heterogeneous
   angular boundary gaps without estimating every arm independently?
2. What output-sensitive direct-sum lower bound interpolates correctly between
   quantum best-arm identification ($k=1$) and the linear output barrier
   ($k=\Theta(n)$)?
3. Which parts extend from uniform Top-$k$ constraints to partition and general
   matroids?
4. Can the resulting selector reduce source-model oracle calls when searching for
   semantics-preserving LLM interventions that cross a functional-vulnerability
   boundary?

## What is implemented

- A complete small-scale NumPy statevector oracle for the canonical phase-fixed
  Bernoulli $R_y$ model. It supports arbitrary index superpositions, true
  controlled branches, forward/inverse calls, reward reflections, and an explicit
  query ledger without a public mean/amplitude accessor.
- A natural-purification statevector oracle with explicit work garbage and a
  deterministic full-unitary Householder completion. The completion is a simulator
  convention, not a unit-cost reversible LLM compiler.
- A direct, charged amplitude-threshold reflection for the canonical oracle. It
  constructs QPE from embedded controlled reward-oracle calls, compares phase bins,
  applies phase kickback, and uncomputes on the complete phase/index/reward space.
  It receives a numeric threshold and never receives a hidden marked-index set.
- Full-workspace BBHT threshold search with a rank-one diffusion about the complete
  initial state, joint accept/index decoding, output exclusion, and independently
  rerun fixed-confidence QPE verification. Failed search budgets remain explicitly
  uncertified; they are not interpreted as proof that no qualifying arm exists.
- A calibrated direct Top-$k$ controller that passes only one numeric boundary and
  simultaneous confidence intervals into fair above/below searches. Completion
  requires a final strict interval certificate for every selected and rejected arm;
  the calibration still samples every arm and therefore is not an advantage result.
- Executable `QBoundary`, reversible compute--phase--uncompute `QGapFlag`, known-count
  Grover and BBHT `QBatchExtract`, and a resumable selected/complement dovetail
  controller. Every reward-oracle experiment and measured flag verification is
  charged. This earlier controller still compiles its flag from the resulting
  certificate and is retained as a clearly labelled baseline.
- An analytic iterative-amplitude-estimation simulator and multiscale Q-GapSelect
  research selector with mean/angular confidence diagnostics and separate
  simulated versus conjectural cost fields.
- Classical Hoeffding and Bernoulli-information comparators, independent exact-state
  and analytic quantum-estimation baselines, and orientation-optimized candidate
  layer-complexity accounting.
- Deterministic analytic, reference, and exact-state coherent experiment runners.
  Raw records keep heuristic output agreement, certificate-gated success, executed
  query/gate/depth/qubit resources, and conjectural theory fields separate.
- A unified executed quantum-core audit runner covering random complex-state
  unitarity, exact and off-grid QPE resolution, measured verifier calibration,
  four seeded angular instance families, full-workspace BBHT scheduling, both
  threshold orientations, multi-output exclusion, direct/independent/classical
  fixed-parameter and same-logical-budget controls, an analytic iterative-AE
  comparator, Top-k boundary-only negative control, invalid index-only diffusion,
  and explicit failure semantics. Dense NumPy QFT and comparator peak allocations
  are reported separately from logical query/gate/depth counts.
- An authorization-scoped offline LLM attack-study pipeline: typed local-model and
  validator adapters, strict JSONL replay, source-only portfolio selection,
  held-out-victim isolation, paired clean/attack seeds, and task-level `ASR@Q`,
  `FV-ASR@Q`, paired-seed `Delta-FV-ASR@Q`, paired-counterfactual ASR, functionality retention,
  query-to-first-success, timeout, and indeterminate metrics.
- Tests, continuous integration, a research protocol, claim matrix, and manuscript
  scaffold.

The orientation-optimized angular layer complexity reported by the current experiments is a
**research hypothesis**, not a proved runtime theorem. The manuscript states the
proof obligations and falsification criteria explicitly.

The direct threshold path fixes the earlier semantic flaw in which the coherent
flag merely encoded an already known answer. It is nevertheless a careful
composition of amplitude estimation and unknown-success amplitude amplification,
not yet the proposed new heterogeneous-gap quantum algorithm. Only a strictly
stronger unknown-gap procedure with a matching lower bound could activate that
claim. Likewise, the repository does not compile an LLM generator into a reversible
circuit; it supplies the natural oracle contract and finite exact-state reference
implementations.

## Current measured status

The versioned quantum-core diagnostic was executed from source commit
`eeed49c2ce9d845a374967085a078b317984410d`. It contains 688 primary records and
288 additional paired same-logical-query-cap records across all ten configured
suites. The most important finite-size results are:

- all 18 random-complex-state compute/inverse and reflection-involution checks
  passed; the largest residual was `8.08e-15`;
- all 134 exact phase-grid predicate checks agreed with their expected
  classification up to a maximum absolute error of `7.55e-15`;
- the measured verifier made 2 wrong resolved decisions in 320 trials, and the
  minimum per-cell empirical interval coverage was `0.90`; these small cells are
  calibration diagnostics, not evidence of nominal-coverage certification;
- on the 96 fixed-parameter random instances per method, direct BBHT returned an
  exact complete answer in 85 cases, independent QPE scan in 84, and the classical
  scan in 58, at mean logical-query counts `9668`, `8949`, and `2464`, respectively;
- direct BBHT and independent QPE each completed the finite-QPE predicate in 86
  cases, but only 85 and 84 of those outputs matched the true-mean threshold target.
  Predicate completion is therefore never reported as true-mean correctness;
- under a per-instance query cap copied from the paired direct run, the exact
  counts were 85/96 for direct, 82/96 for independent QPE, and 79/96 for the
  classical scan. This is not an accuracy-matched advantage result;
- the analytic iterative-AE comparator completed exactly on 32/32 diagnostic
  cases with a mean of 4,642 analytic oracle calls, but its measurement-law
  backend is deliberately not mixed with exact-state gate or qubit resources;
- all 24 calibrated direct Top-k trials stopped with
  `phase_resolution_insufficient`, while the boundary-only negative control was
  exact in 24/24 trials only because its certificate already contains the full
  membership answer. It is forbidden evidence for quantum discovery.

These results validate important implementation invariants and expose current
failure modes. They do **not** demonstrate a quantum query advantage, a new quantum
algorithm, hardware feasibility, or a leading LLM attack result. No local LLM or
attack reward oracle was run in this diagnostic. The bundled attack fixture remains
synthetic state-only data used to test metric and split semantics; its rates are not
empirical LLM evidence.

## Repository layout

```text
src/qgapselect/       oracles, coherent primitives, selectors, attack metrics
scripts/              analytic, exact-state, and offline attack-study entry points
tests/                unit, invariant, and special-case regression tests
configs/              checked-in experiment configurations
paper/                LaTeX manuscript and bibliography
docs/                 research protocol and claim-level evidence matrix
.github/workflows/    automated linting and tests
```

## Reproduce

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,plots]'
pytest
python scripts/run_scaling.py --output artifacts/scaling.json
python scripts/run_reference.py --output artifacts/reference_results.json
python scripts/run_coherent.py --output artifacts/coherent_results.json
python scripts/run_direct_search.py --output artifacts/direct_search_results.json
python scripts/run_quantum_benchmarks.py \
  --config configs/quantum_benchmarks.json \
  --output artifacts/quantum_benchmark_diagnostic.json
python scripts/run_attack_study.py \
  --config configs/attack_study.json \
  --output artifacts/attack_study_results.json \
  --raw-output artifacts/attack_study_raw.jsonl
```

Use each script's `--help` option for experiment controls. The quantum-core
runner supports `--suite`, `--trials`, and `--seed`; its default configuration
runs no local LLM and records exact-state and analytic-measurement evidence in
separate resource classes. The reference
configuration schedules 500 repetitions per scenario; an explicit `--trials`
override is recorded in the resolved configuration and hash. Generated data are
written beneath `artifacts/`. The repository versions four audited diagnostics:
the analytic scaling table, the 4-trial-per-scenario reference run, the direct-search
run, and the full quantum-core run. Other outputs stay ignored until a recorded run
is intentionally promoted. The default attack command runs only the clearly
labelled, non-empirical state fixture; pass authorized offline JSONL through
`--replay` for an actual study.

## Paper-grade completion criteria

The central result is ready to be described as a new quantum algorithm only after
all of the following hold:

- a fully specified polynomial-time circuit/query algorithm, including noisy
  thresholding, batch extraction, failure-budget allocation, and stopping rules;
- an upper bound that recovers known best-arm behavior at $k=1$ and the
  $\Theta(\sqrt{k(n-k)}/\gamma)$ equal-angular-gap limit (with
  $\gamma=\Theta(\Delta)$ only for means bounded away from 0 and 1);
- a matching adversary or polynomial-method lower bound, including the
  $\Omega(n/\gamma)$ angular-gap dense-output target (hence $\Omega(n)$ at
  constant $\gamma$) and additive partition direct sums;
- ablations against independent amplitude estimation and unstructured search;
- LLM experiments that measure functional-vulnerability ASR under a fixed query
  budget, with semantic validity and transfer evaluated separately;
- release of seeds, raw outputs, validators, and negative results.

No repository structure, benchmark table, or simulated curve by itself establishes
a CCF-A-level contribution or guarantees acceptance.

## Responsible-use boundary

The LLM-security portion is for controlled evaluation of models and code generators
that the experimenter owns or is authorized to test. This repository does not ship
real-service exploitation, credential handling, persistence, malware, or automated
deployment of vulnerable code. See [SECURITY.md](SECURITY.md).
