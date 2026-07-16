#!/usr/bin/env python3
"""Generate the current theorem-stack proof ledger."""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qgapselect.proof_ledger import (
    CLAIM_STATUS,
    build_proof_ledger,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_STOPPING = REPOSITORY / "artifacts" / "stopping_unitary_theorem.json"
DEFAULT_COMPOSITION = REPOSITORY / "artifacts" / "composition_frontier.json"
DEFAULT_LOWER_BOUND = REPOSITORY / "artifacts" / "lower_bound_program.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "proof_ledger.json"
DEFAULT_MARKDOWN = REPOSITORY / "docs" / "proof_ledger.md"
ARTIFACT_TYPE = "q_gapselect_proof_ledger"


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def _git_provenance() -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ("git", *args),
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return "unknown"
        return completed.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_tree": run("rev-parse", "HEAD^{tree}"),
        "source_tree_dirty_at_execution": bool(status),
        "source_status_capture": "before_proof_ledger_execution",
    }


def build_report(
    *,
    stopping_artifact: Path,
    composition_artifact: Path,
    lower_bound_artifact: Path,
) -> dict[str, object]:
    ledger = build_proof_ledger(
        stopping_artifact=stopping_artifact,
        composition_artifact=composition_artifact,
        lower_bound_artifact=lower_bound_artifact,
    )
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "claim_status": CLAIM_STATUS,
        "provenance": {
            **_git_provenance(),
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "byte_for_byte_reproduction_expected": False,
            "volatile_fields": [],
        },
        "claim_boundaries": {
            "supports": [
                "machine-readable proof obligation tracking",
                "separation of local facts, execution checks, proof outlines, "
                "and open proof obligations",
            ],
            "does_not_support": [
                "a completed upper-bound theorem",
                "a completed lower-bound theorem",
                "a CCF-A-ready quantum advantage claim",
            ],
        },
        "ledger": ledger.as_dict(),
    }


def write_report(report: Mapping[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path.resolve()


def write_markdown(report: Mapping[str, object], path: Path) -> Path:
    ledger_data = report.get("ledger")
    if not isinstance(ledger_data, Mapping):
        raise TypeError("report missing ledger object")
    # Reconstruct only the fields needed by the renderer would add unnecessary
    # coupling.  The CLI writes a compact markdown rendering directly.
    lines = [
        "# Q-GapSelect proof ledger",
        "",
        f"Claim status: `{ledger_data.get('claim_status')}`",
        "",
        f"Readiness: `{ledger_data.get('readiness')}`",
        "",
        f"Theorem claimable: `{ledger_data.get('theorem_claimable')}`",
        "",
        f"CCF-A claimable: `{ledger_data.get('ccf_a_claimable')}`",
        "",
        "## Status counts",
        "",
    ]
    status_counts = ledger_data.get("status_counts", {})
    if isinstance(status_counts, Mapping):
        for status, count in status_counts.items():
            lines.append(f"- `{status}`: `{count}`")
    lines.extend(["", "## Entries", ""])
    entries = ledger_data.get("entries", [])
    if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
        for raw_entry in entries:
            if not isinstance(raw_entry, Mapping):
                continue
            lines.extend(
                [
                    f"### {raw_entry.get('entry_id')}: {raw_entry.get('statement')}",
                    "",
                    f"- pillar: `{raw_entry.get('pillar')}`",
                    f"- status: `{raw_entry.get('status')}`",
                    f"- evidence: {raw_entry.get('evidence')}",
                    f"- missing argument: {raw_entry.get('missing_argument')}",
                    f"- activation condition: {raw_entry.get('activation_condition')}",
                    f"- permitted wording: {raw_entry.get('permitted_wording')}",
                    "",
                ]
            )
    next_steps = ledger_data.get("next_required_manual_proofs", [])
    lines.extend(["## Next required manual proofs", ""])
    if isinstance(next_steps, Sequence) and not isinstance(next_steps, (str, bytes)):
        for item in next_steps:
            lines.append(f"- {item}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stopping-artifact", type=Path, default=DEFAULT_STOPPING)
    parser.add_argument("--composition-artifact", type=Path, default=DEFAULT_COMPOSITION)
    parser.add_argument("--lower-bound-artifact", type=Path, default=DEFAULT_LOWER_BOUND)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(
            stopping_artifact=args.stopping_artifact,
            composition_artifact=args.composition_artifact,
            lower_bound_artifact=args.lower_bound_artifact,
        )
        output = write_report(report, args.output)
        markdown = write_markdown(report, args.markdown)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"proof ledger error: {error}") from error
    ledger = report["ledger"]
    if not isinstance(ledger, Mapping):
        raise SystemExit("proof ledger error: internal ledger object missing")
    print(f"wrote proof ledger JSON to {output}")
    print(f"wrote proof ledger Markdown to {markdown}")
    print(f"claim_status={report['claim_status']}")
    print(f"entry_count={ledger['entry_count']}")
    print("This is proof bookkeeping, not a completed theorem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
