#!/usr/bin/env python3
"""Materialize the fail-closed strong-composition registry audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from qgapselect.strong_composition_registry import (  # noqa: E402
    audit_strong_composition_registry,
    load_strong_composition_registry,
    machine_readable_registry_audit,
)

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "q_gapselect_strong_composition_registry_audit"
DEFAULT_CONFIG = REPOSITORY / "configs" / "strong_composition_registry.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "strong_composition_registry.json"


def _git_provenance() -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ("git", *args),
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else "unknown"

    status = run("status", "--porcelain")
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_tree": run("rev-parse", "HEAD^{tree}"),
        "source_tree_dirty_at_execution": bool(status),
        "source_status_capture": "before_registry_artifact_write",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def build_artifact(config_path: Path) -> dict[str, object]:
    """Build an inventory audit without inventing runtime-fidelity evidence."""

    config_bytes = config_path.read_bytes()
    registry = load_strong_composition_registry(config_path)
    runtime_self_reports: tuple[()] = ()
    audit = audit_strong_composition_registry(registry, runtime_self_reports)
    machine_audit = machine_readable_registry_audit(audit)
    required_rows = sum(row.coverage_required for row in registry.baselines)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "claim_status": audit.claim_status,
        "registry_inventory": asdict(registry),
        "runtime_fidelity_self_reports": list(runtime_self_reports),
        "trusted_runtime_attestations": [],
        "machine_readable_audit": machine_audit,
        "summary": {
            "registry_row_count": len(registry.baselines),
            "required_coverage_row_count": required_rows,
            "runtime_self_report_row_count": 0,
            "trusted_runtime_attestation_row_count": 0,
            "uncovered_required_row_count": len(
                audit.uncovered_required_baseline_ids
            ),
            "inventory_complete": audit.inventory_complete,
            "source_identity_pin_complete": audit.source_identity_pin_complete,
            "source_version_locator_pin_complete": (
                audit.source_version_locator_pin_complete
            ),
            "trusted_runtime_attestation_pipeline_implemented": (
                audit.trusted_runtime_attestation_pipeline_implemented
            ),
            "strongest_composition_coverage_complete": (
                audit.strongest_composition_coverage_complete
            ),
            "strongest_composition_claimable": audit.strongest_composition_claimable,
            "ccf_a_quantum_advantage_claimable": (
                audit.ccf_a_quantum_advantage_claimable
            ),
            "theorem_claimed": False,
        },
        "claim_boundaries": {
            "supports": [
                "a source-identity and explicit version-locator inventory",
                "machine-readable identification of uncovered fidelity work",
            ],
            "does_not_support": [
                "runtime fidelity for any baseline",
                "trusted runtime attestation from self-reported JSON",
                "content pinning of cited theorem PDFs",
                "a composition-separation theorem",
                "a CCF-A quantum-advantage claim",
            ],
        },
        "provenance": {
            "config_path": str(config_path.resolve()),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            **_git_provenance(),
        },
    }


def write_artifact(artifact: Mapping[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            artifact,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        artifact = build_artifact(args.config)
        output = write_artifact(artifact, args.output)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"strong-composition registry error: {error}") from error
    summary = artifact["summary"]
    if not isinstance(summary, Mapping):
        raise SystemExit("strong-composition registry error: internal summary missing")
    print(f"wrote strong-composition registry audit to {output}")
    print(
        f"rows={summary['registry_row_count']} "
        f"required={summary['required_coverage_row_count']} "
        f"uncovered={summary['uncovered_required_row_count']}"
    )
    print(
        f"strongest_composition_claimable={summary['strongest_composition_claimable']} "
        f"ccf_a_claimable={summary['ccf_a_quantum_advantage_claimable']}"
    )
    print("This is a fail-closed registry audit, not a theorem or runtime evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
