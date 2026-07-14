# Contributing

This repository treats claim discipline as part of the implementation. A change
that adds an algorithmic claim must also add its assumptions, evidence class, and a
test or proof obligation.

## Claim labels

Use one of these labels in documentation, experiment schemas, and manuscript notes:

- `implemented`: executable behavior covered by a test;
- `measured`: a value produced by a versioned command with raw output and seed;
- `derived`: an algebraic consequence whose derivation is included;
- `proved`: a theorem with a complete proof under stated oracle assumptions;
- `conjectured`: an unproved target or complexity proxy;
- `external`: a prior result with a checked primary citation.

Never turn `conjectured` accounting into a measured speedup. State whether a query
count comes from circuit execution, analytic probability sampling, or a symbolic
cost formula.

## Development workflow

1. Add or update an invariant test before changing an oracle or cost convention.
2. Run `python -m ruff check .` and `python -m pytest`.
3. Record experiment commands, seeds, environment, and raw machine-readable output.
4. Keep generated plots and tables out of Git until their provenance is recorded.
5. Update `docs/claim_matrix.md` when a change affects a paper claim.

The canonical query convention counts each application of the reward oracle or its
inverse. Controlled powers must be expanded into this unit before methods are
compared. Gate depth, qubit count, wall time, and victim-model API calls are distinct
resources and must be reported in separate columns.
