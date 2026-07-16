# Audited code-sanity artifacts

This directory versions machine-readable audit outputs:

- `scaling.json` evaluates declared analytic complexity proxies. It contains no
  observed quantum execution and proves no scaling theorem.
- `reference_diagnostic.json` contains four simulator trials for each of four
  fixed mean vectors (16 records total). It samples analytic Grover measurement
  laws; it is not coherent batch execution, random-instance evidence, or a
  quantum-acceleration result.
- `direct_search_diagnostic.json` contains four exact-state executions for each
  of four direct unknown-oracle threshold scenarios (16 records total). It runs
  charged QPE reflections, full-workspace BBHT, joint accept/index measurement,
  and fresh verification; it is simulation evidence, not a speedup theorem.
- `quantum_benchmark_diagnostic.json` is the full quantum-core audit generated
  from clean source commit `eeed49c2ce9d845a374967085a078b317984410d`. It contains
  688 primary records plus 288 paired same-logical-query-cap controls across ten
  suites. Its SHA-256 digest is
  `b8188265ea3a0aeb63fe54c3df213c484d6c84e7b018f8f204cff6b02108d1e2`.
  The artifact includes successful unitary and phase-grid invariants as well as
  negative verifier, query-comparison, and Top-k results; it proves no advantage
  theorem and contains no local-LLM execution.
- `frozen_selector_benchmark_diagnostic.json` contains 15,360 executions of six
  classical selectors over eight synthetic frozen reward/cost landscapes, five
  budgets, and 64 independent fixtures per cell. It performs no LLM evaluation
  and makes no quantum claim.
- `frozen_quantum_reference_diagnostic.json` compares Q-GapSelect's all-active
  analytic-IAE reference, independent IAE Top-k, and a stronger-information
  known-threshold control on exact-count frozen empirical Layer-C oracles. It
  contains 5,000 non-isomorphic instances and 15,000 method runs, exposes one
  permutation-invariant difficulty record per instance, and includes paired
  McNemar, fixture-pair bootstrap, unconditional query-difference, and Holm
  analyses. It records logical A/A-dagger calls and strict certificates, not
  circuit or hardware execution.
- `coherent_statevector_history.json` is the exact-state coherent-index
  activity-history semantics audit. It executes controlled forward/inverse
  oracle calls and explicit boundary/history/work registers on 24 small
  trials. The on-grid cases clean up their executed layers, but no trial
  completes direct multi-output Top-k extraction or receives a certificate;
  therefore it is code-sanity rather than advantage evidence.
- `theorem_closure_audit.json` places the candidate upper proxy, public/hidden
  partition interfaces, legal composition, and lower-bound targets on one
  family. The public-partition composition matches or beats the candidate,
  while the hidden-partition upper bound and weighted matching lower bound are
  open. Its CCF-A/advantage gate is false.
- `ccfa_matched_benchmark_diagnostic.json.gz` is the fixed-fixture,
  multiple-seed, fixed-cap information-matched hard-family campaign. Every
  timeout, unresolved result, and budget failure remains in the denominator;
  its evidence gate also consumes the theorem-closure status.
- `uci_classifier_benchmark_diagnostic.json.gz` contains 800 matched attempts
  on the bundled real Digits classifier-selection diagnostic. Of five frozen
  label-blind shards, three have an exact k/(k+1) boundary tie and are rejected
  fail-closed; two enter the matched panel. At the largest cap, the current
  analytic activity-history reference reaches 0.825 certified-exact recovery,
  below the 1.000 k-only adaptive references. This is deliberately retained
  negative external-validity evidence, not the official Optdigits campaign.

They are regenerated from the current source with:

```bash
python scripts/run_scaling.py \
  --config configs/scaling.json \
  --output artifacts/scaling.json
python scripts/run_reference.py \
  --config configs/reference.json \
  --trials 4 \
  --output artifacts/reference_diagnostic.json
python scripts/run_direct_search.py \
  --config configs/direct_search.json \
  --output artifacts/direct_search_diagnostic.json
python scripts/run_quantum_benchmarks.py \
  --config configs/quantum_benchmarks.json \
  --output artifacts/quantum_benchmark_diagnostic.json
python scripts/run_composition_frontier.py \
  --config configs/composition_frontier.json \
  --output artifacts/composition_frontier.json
python scripts/run_lower_bound_program.py \
  --config configs/lower_bound_program.json \
  --output artifacts/lower_bound_program.json \
  --markdown docs/lower_bound_program.md
python scripts/run_proof_ledger.py \
  --stopping-artifact artifacts/stopping_unitary_theorem.json \
  --composition-artifact artifacts/composition_frontier.json \
  --lower-bound-artifact artifacts/lower_bound_program.json \
  --output artifacts/proof_ledger.json \
  --markdown docs/proof_ledger.md
python scripts/run_qgapattack_experiment_design.py \
  --config configs/qgapattack_experiments.json \
  --output artifacts/qgapattack_experiment_design.json \
  --markdown docs/qgapattack_experiment_design_audit.md \
  --strict-design
python scripts/run_frozen_selector_benchmarks.py \
  --config configs/frozen_selector_benchmarks.json \
  --output artifacts/frozen_selector_benchmark_diagnostic.json
python scripts/run_frozen_quantum_reference_benchmarks.py \
  --config configs/frozen_quantum_reference_benchmarks.json \
  --output artifacts/frozen_quantum_reference_diagnostic.json
```

Each JSON document carries its resolved configuration or source hash and an
explicit claim boundary. The configured 500-trial reference run has not been
promoted as a completed artifact. The quantum-core file records exact-state and
analytic-measurement backends as separate resource classes; its simulator wall
time is not hardware time or speedup evidence. Its file SHA identifies the recorded
run; byte-for-byte reproduction is not promised because simulator wall-time fields
are intentionally retained. Runtime versions and the volatile-field declaration
are stored in the artifact provenance.

The composition-frontier and lower-bound program artifacts are proof-planning
evidence only. They screen known-composition baselines and enumerate L-07 proof
blocks; they do not prove a new quantum advantage theorem.

The proof-ledger artifact aggregates those obligations into a single theorem
checklist and explicitly records that the current stack is not CCF-A-claimable.

The Q-GapAttack experiment-design artifact audits the preregistered baseline,
benchmark, metric, fairness, and statistics matrix. A valid design is not an
executed campaign: the artifact records `empirical_ready=false` and
`ccf_a_claimable=false` until real frozen runs and the theorem gates are complete.

The two frozen-oracle artifacts close part of that execution gap without
changing the claim boundary. The selector artifact contains no proposed quantum
method. The Layer-C reference artifact now completes its 500-instance-per-cell
mixture panel, but Q-GapSelect receives only `k`, independent IAE receives an
additional public gap floor, and the schedules are not query matched. Its
analytic canonical oracle still does not implement or charge state preparation,
QROM, cleanup, gates, depth, or qubits. Neither artifact is evidence of an LLM
attack result or a quantum advantage theorem.
