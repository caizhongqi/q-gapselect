#!/usr/bin/env python3
"""Audit the preregistered Q-GapAttack experiment and baseline matrix."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qgapselect.qgapattack_experiments import (
    SCHEMA_VERSION,
    experiment_design_markdown,
    load_and_audit_experiment_design,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs" / "qgapattack_experiments.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "qgapattack_experiment_design.json"
DEFAULT_MARKDOWN = REPOSITORY / "docs" / "qgapattack_experiment_design_audit.md"
ARTIFACT_TYPE = "qgapattack_preregistered_experiment_design"


def _portable_path(path: Path) -> str:
    """Return a repository-relative path when the input lives in this checkout."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY).as_posix()
    except ValueError:
        return str(resolved)


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
    raise TypeError(f"cannot encode {type(value)!r}")


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
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--strict-design",
        action="store_true",
        help="return a nonzero status when design coverage or fairness fails",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_bytes = args.config.read_bytes()
    design, audit = load_and_audit_experiment_design(args.config)
    report = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "claim_status": audit.claim_status,
        "design": _jsonable(design),
        "audit": audit.as_dict(),
        "provenance": {
            "config_path": _portable_path(args.config),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            **_git_provenance(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(
        experiment_design_markdown(design, audit), encoding="utf-8"
    )
    sys.stdout.write(
        f"wrote Q-GapAttack design audit to {args.output}\n"
        f"baselines={audit.baseline_count} panels={audit.panel_count} "
        f"design_valid={str(audit.design_valid).lower()} "
        f"empirical_ready={str(audit.empirical_ready).lower()} "
        f"ccf_a_claimable={str(audit.ccf_a_claimable).lower()}\n"
    )
    if args.strict_design and not audit.design_valid:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
