#!/usr/bin/env python3
"""Materialize the unified upper/composition/lower theorem-closure audit."""

from __future__ import annotations

import argparse
import json
import math
import operator
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

from qgapselect.theorem_closure_audit import (  # noqa: E402
    CLAIM_STATUS,
    CLOSURE_BLOCKED,
    build_unified_theorem_closure_audit,
    machine_readable_status_map,
)

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "q_gapselect_unified_theorem_closure_audit"
DEFAULT_CONFIG = REPOSITORY / "configs" / "theorem_closure_audit.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "theorem_closure_audit.json"


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def load_config(path: Path) -> dict[str, object]:
    document = _mapping(json.loads(path.read_text(encoding="utf-8")), "config")
    if _integer(document.get("schema_version"), "schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    raw_values = document.get("m_values")
    if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
        raise TypeError("m_values must be a sequence")
    m_values = tuple(_integer(value, "m value", minimum=2) for value in raw_values)
    if not m_values or len(set(m_values)) != len(m_values):
        raise ValueError("m_values must be non-empty and unique")
    failure = _number(document.get("failure_probability"), "failure_probability")
    if not 0.0 < failure < 0.5:
        raise ValueError("failure_probability must lie in (0, 0.5)")
    tolerance = _number(
        document.get("composition_match_tolerance"),
        "composition_match_tolerance",
    )
    if tolerance < 1.0:
        raise ValueError("composition_match_tolerance must be at least one")
    raw_notes = document.get("notes")
    if isinstance(raw_notes, (str, bytes)) or not isinstance(raw_notes, Sequence):
        raise TypeError("notes must be a sequence")
    notes = tuple(_nonempty(value, "note") for value in raw_notes)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": _nonempty(
            document.get("experiment_name"), "experiment_name"
        ),
        "m_values": m_values,
        "failure_probability": failure,
        "composition_match_tolerance": tolerance,
        "notes": notes,
    }


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

    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_tree": run("rev-parse", "HEAD^{tree}"),
        "source_tree_dirty_at_execution": bool(run("status", "--porcelain")),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def run_audit(config: Mapping[str, object]) -> dict[str, object]:
    audits = tuple(
        build_unified_theorem_closure_audit(
            int(m),
            failure_probability=float(config["failure_probability"]),
            match_tolerance=float(config["composition_match_tolerance"]),
        )
        for m in config["m_values"]
    )
    all_public_matched = all(
        audit.public_instantiation.partitioned_matches_candidate
        for audit in audits
    )
    all_hidden_upper_open = all(
        not audit.hidden_instantiation.candidate_upper_proved for audit in audits
    )
    all_lower_open = all(
        not audit.hidden_instantiation.matching_lower_bound_established
        for audit in audits
    )
    status_maps = tuple(machine_readable_status_map(audit) for audit in audits)
    asymptotic_public_failure = all(
        bool(
            status["asymptotic_public_witness"][
                "partitioned_is_little_o_of_candidate_proxy"
            ]
        )
        for status in status_maps
    )
    resolved_config = {
        **dict(config),
        "m_values": list(config["m_values"]),
        "notes": list(config["notes"]),
    }
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "experiment_name": config["experiment_name"],
        "claim_status": CLAIM_STATUS,
        "resolved_config": resolved_config,
        "summary": {
            "scale_count": len(audits),
            "single_oracle_output_contract_per_row": True,
            "public_partition_composition_matches_every_scale": all_public_matched,
            "hidden_partition_upper_bound_open_every_scale": all_hidden_upper_open,
            "weighted_matching_lower_bound_open_every_scale": all_lower_open,
            "public_composition_asymptotically_beats_candidate_proxy": (
                asymptotic_public_failure
            ),
            "theorem_chain_closed": False,
            "quantum_advantage_claimable": False,
            "ccf_a_quantum_advantage_claimable": False,
            "closure_status": CLOSURE_BLOCKED,
        },
        "status_maps": list(status_maps),
        "audits": [asdict(audit) for audit in audits],
        "provenance": _git_provenance(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = run_audit(load_config(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
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
    summary = artifact["summary"]
    sys.stdout.write(
        f"wrote unified theorem audit to {args.output}\n"
        f"scales={summary['scale_count']} closure={summary['theorem_chain_closed']}\n"
        "scope=analytic closure falsification; no quantum-advantage claim\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
