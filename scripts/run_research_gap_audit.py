#!/usr/bin/env python3
"""Generate the current Q-GapSelect paper-readiness gap audit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from qgapselect.research_gap_audit import (
    build_research_gap_audit,
    research_gap_markdown,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_QUANTUM = REPOSITORY / "artifacts" / "quantum_benchmark_diagnostic.json"
DEFAULT_GRID = REPOSITORY / "artifacts" / "unknown_boundary_grid.json"
DEFAULT_CHARGED = REPOSITORY / "artifacts" / "charged_activity_history.json"
DEFAULT_VARIABLE_TIME = REPOSITORY / "artifacts" / "variable_time_charged_history.json"
DEFAULT_STOPPING = REPOSITORY / "artifacts" / "stopping_time_transducer.json"
DEFAULT_THEOREM = REPOSITORY / "artifacts" / "stopping_unitary_theorem.json"
DEFAULT_COMPOSITION = REPOSITORY / "artifacts" / "composition_frontier.json"
DEFAULT_LOWER_BOUND = REPOSITORY / "artifacts" / "lower_bound_program.json"
DEFAULT_PROOF_LEDGER = REPOSITORY / "artifacts" / "proof_ledger.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "research_gap_audit.json"
DEFAULT_MARKDOWN = REPOSITORY / "docs" / "research_gap_audit.md"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantum-artifact", type=Path, default=DEFAULT_QUANTUM)
    parser.add_argument("--grid-artifact", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--charged-artifact", type=Path, default=DEFAULT_CHARGED)
    parser.add_argument("--variable-time-artifact", type=Path, default=DEFAULT_VARIABLE_TIME)
    parser.add_argument("--stopping-artifact", type=Path, default=DEFAULT_STOPPING)
    parser.add_argument("--theorem-artifact", type=Path, default=DEFAULT_THEOREM)
    parser.add_argument("--composition-artifact", type=Path, default=DEFAULT_COMPOSITION)
    parser.add_argument("--lower-bound-artifact", type=Path, default=DEFAULT_LOWER_BOUND)
    parser.add_argument("--proof-ledger-artifact", type=Path, default=DEFAULT_PROOF_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_research_gap_audit(
        args.quantum_artifact,
        args.grid_artifact,
        args.charged_artifact,
        args.variable_time_artifact,
        args.stopping_artifact,
        args.theorem_artifact,
        args.composition_artifact,
        args.lower_bound_artifact,
        args.proof_ledger_artifact,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(research_gap_markdown(audit), encoding="utf-8")
    print(f"wrote research gap audit JSON to {args.output.resolve()}")
    print(f"wrote research gap audit Markdown to {args.markdown.resolve()}")
    print(f"stage={audit.stage}")
    print(f"readiness={audit.readiness}")
    print(f"top_gap_count={len(audit.top_gaps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
