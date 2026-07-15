# Audited code-sanity artifacts

This directory versions four machine-readable audit outputs:

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
```

Each JSON document carries its resolved configuration or source hash and an
explicit claim boundary. The configured 500-trial reference run has not been
promoted as a completed artifact. The quantum-core file records exact-state and
analytic-measurement backends as separate resource classes; its simulator wall
time is not hardware time or speedup evidence. Its file SHA identifies the recorded
run; byte-for-byte reproduction is not promised because simulator wall-time fields
are intentionally retained. Runtime versions and the volatile-field declaration
are stored in the artifact provenance.
