#!/usr/bin/env python3
"""Integrate the four S3 artifacts into one fail-closed evidence report."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from qgapselect.s3_evidence_audit import CLAIM_SCOPE, audit_s3_evidence  # noqa: E402

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "q_gapselect_s3_integrated_evidence_audit"
DEFAULT_ADAPTIVE = REPOSITORY / "artifacts" / "adaptive_unknown_boundary_topk.json"
DEFAULT_COHERENT = REPOSITORY / "artifacts" / "coherent_adaptive_stopping_history.json"
DEFAULT_FRONTIER = REPOSITORY / "artifacts" / "frontier_lower_bound_witness.json"
DEFAULT_COMPOSITION = REPOSITORY / "artifacts" / "strong_composition_s3.json.gz"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "s3_evidence_audit.json"


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object with string keys")
    return value


def _load(path: Path) -> tuple[Mapping[str, object], bytes]:
    payload = path.read_bytes()
    if path.suffix == ".gz":
        decoded = gzip.decompress(payload)
    else:
        decoded = payload
    return _mapping(json.loads(decoded), str(path)), payload


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY).as_posix()
    except ValueError:
        return str(resolved)


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
        "source_status_capture": "before_s3_evidence_audit_artifact_write",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def build_artifact(
    *,
    adaptive_path: Path,
    coherent_path: Path,
    frontier_path: Path,
    composition_path: Path,
) -> dict[str, object]:
    adaptive, adaptive_bytes = _load(adaptive_path)
    coherent, coherent_bytes = _load(coherent_path)
    frontier, frontier_bytes = _load(frontier_path)
    composition, composition_bytes = _load(composition_path)
    report = audit_s3_evidence(adaptive, coherent, frontier, composition)
    inputs = (
        ("adaptive", adaptive_path, adaptive_bytes, adaptive),
        ("coherent", coherent_path, coherent_bytes, coherent),
        ("frontier", frontier_path, frontier_bytes, frontier),
        ("composition", composition_path, composition_bytes, composition),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "claim_scope": CLAIM_SCOPE,
        "source_artifacts": {
            name: {
                "path": _portable_path(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "artifact_type": document["artifact_type"],
            }
            for name, path, payload, document in inputs
        },
        "report": report.as_dict(),
        "claim_boundary": {
            "claim_guard_only": True,
            "independently_proves_theorems": False,
            "independent_theorem_verifier_available": (
                report.independent_theorem_verifier_available
            ),
            "theorem_claim_activation_locked": report.theorem_claim_activation_locked,
            "quantum_advantage_claimable": report.quantum_advantage_claimable,
            "ccf_a_claimable": report.ccf_a_claimable,
        },
        "provenance": _git_provenance(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--adaptive", type=Path, default=DEFAULT_ADAPTIVE)
    parser.add_argument("--coherent", type=Path, default=DEFAULT_COHERENT)
    parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--composition", type=Path, default=DEFAULT_COMPOSITION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = build_artifact(
        adaptive_path=args.adaptive,
        coherent_path=args.coherent,
        frontier_path=args.frontier,
        composition_path=args.composition,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    report = artifact["report"]
    print(
        "S3 evidence audit: "
        f"satisfied={report['satisfied_gate_count']}/{len(report['gates'])} "
        f"open={report['open_gate_count']} "
        f"ccf_a_claimable={report['ccf_a_claimable']}"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
