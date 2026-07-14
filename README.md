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
- Executable `QBoundary`, reversible compute--phase--uncompute `QGapFlag`, known-count
  Grover and BBHT `QBatchExtract`, and a resumable selected/complement dovetail
  controller. Every reward-oracle experiment and measured flag verification is
  charged. The current boundary stage still samples arms independently, and the
  flag is compiled from its resulting certificate.
- An analytic iterative-amplitude-estimation simulator and multiscale Q-GapSelect
  research selector with mean/angular confidence diagnostics and separate
  simulated versus conjectural cost fields.
- Classical Hoeffding and Bernoulli-information comparators, independent exact-state
  and analytic quantum-estimation baselines, and orientation-optimized candidate
  layer-complexity accounting.
- Deterministic analytic, reference, and exact-state coherent experiment runners.
  Raw records keep heuristic output agreement, certificate-gated success, executed
  query/gate/depth/qubit resources, and conjectural theory fields separate.
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

The coherent primitives are executable reference semantics, but they do not yet
constitute the proposed new heterogeneous-gap quantum algorithm. Their ingredients
overlap substantially with prior order-statistic, variable-time search, and
all-marked enumeration algorithms. Only a strictly stronger unknown-gap procedure
with a matching lower bound could activate that claim. Likewise, the repository
does not compile an LLM generator into a reversible circuit; it supplies the natural
oracle contract and a finite exact-state reference implementation.

## Current measured status

The checked coherent diagnostic contains four fixed scenarios, eight seeds, and
three executable methods (96 raw records). All methods certify the answer on these
easy instances. The current boundary-plus-certificate-enumeration controller averages
2,100 reward oracle calls, while the independent exact-state boundary baseline averages 1,188.
This is a negative baseline result: the present code does **not** demonstrate a
quantum query advantage or a leading attack result. The bundled attack fixture is
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
python scripts/run_attack_study.py \
  --config configs/attack_study.json \
  --output artifacts/attack_study_results.json \
  --raw-output artifacts/attack_study_raw.jsonl
```

Use each script's `--help` option for experiment controls. The reference
configuration schedules 500 repetitions per scenario; an explicit `--trials`
override is recorded in the resolved configuration and hash. Generated data are
written beneath `artifacts/`. The repository versions only the audited analytic
scaling table and a 4-trial-per-scenario reference diagnostic; other outputs stay
ignored until a recorded run is intentionally promoted. The default attack command
runs only the clearly labelled, non-empirical state fixture; pass authorized offline
JSONL through `--replay` for an actual study.

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
