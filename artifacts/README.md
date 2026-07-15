# Audited code-sanity artifacts

This directory versions three small, machine-readable audit outputs:

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
```

Each JSON document carries its resolved configuration or source hash and an
explicit claim boundary. The configured 500-trial reference run has not been
promoted as a completed artifact.
