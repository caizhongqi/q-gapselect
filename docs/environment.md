# Reproducible environment

Status: development setup for the pure quantum-core branch.

## Python

Use Python 3.10 through 3.12.  The CI matrix tests 3.10 and 3.12.

The most robust local setup is:

```bash
python -m venv --copies .venv
. .venv/bin/activate
python -m pip install -e '.[dev,plots,datasets]'
```

`--copies` avoids broken interpreter symlinks in some containerized
workspaces.  On a normal workstation, `python -m venv .venv` is also fine.

## uv setup

The repository includes `uv.lock`.  On hosts where `uv` can build editable
local packages normally, run:

```bash
UV_CACHE_DIR=.uv-cache uv sync --frozen --all-extras
```

The explicit cache directory keeps `uv` from writing to a read-only home cache
inside restricted containers.

## Core checks

```bash
make test-quantum
make unknown-boundary-grid
make quantum-history
make coherent-statevector-history
make theorem-closure-audit
make ccfa-matched-benchmark
make uci-classifier-benchmark
```

For a full pure quantum-core artifact:

```bash
make quantum-core
```

This writes:

- `artifacts/quantum_benchmark_diagnostic.json`
- `artifacts/unknown_boundary_grid.json`

Both are analytic or exact-state diagnostic artifacts.  They are not hardware
runs, upper-bound theorems, or lower-bound theorems.

The bundled Digits public-data diagnostic works offline after installing the
`datasets` extra. Official Letter, Optdigits, and Covertype runs require local
UCI source files below `data/uci`; `make download-uci` fetches them only on a
host that permits access to the official UCI archive domain. Every loader
checks source structure and records SHA-256 commitments before execution.

## GitHub publishing prerequisite

Publishing a branch and opening a PR requires GitHub CLI:

```bash
gh --version
gh auth status
```

If `gh` is missing, install it and run `gh auth login` before asking the agent
to push or open a pull request.
