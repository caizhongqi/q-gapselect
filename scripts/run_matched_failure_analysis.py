#!/usr/bin/env python3
"""Analyze where the formal fixed-cap matched candidate loses to baselines."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from qgapselect.matched_failure_analysis import analyze_matched_failure  # noqa: E402

DEFAULT_INPUT = REPOSITORY / "artifacts" / "ccfa_matched_benchmark_diagnostic.json.gz"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "ccfa_matched_failure_attribution.json"
CANDIDATE = "variable_time_coherent_activity_history"
BASELINES = (
    "k_only_independent_adaptive",
    "coarse_partition_bai_composition",
    "repeated_single_output_selection",
    "unknown_time_variable_time_reference",
)


def _load(path: Path) -> Mapping[str, object]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            document = json.load(handle)
    else:
        document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise TypeError("input artifact must be a JSON object")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = _load(args.input)
    raw_records = source.get("records")
    if not isinstance(raw_records, list) or any(
        not isinstance(record, Mapping) for record in raw_records
    ):
        raise TypeError("input artifact records must be a JSON object array")
    analysis = analyze_matched_failure(
        raw_records,
        candidate_method_id=CANDIDATE,
        baseline_method_ids=BASELINES,
    )
    artifact = {
        "artifact_type": "q_gapselect_ccfa_matched_failure_attribution",
        "schema_version": 1,
        "source_artifact": {
            "path": args.input.name,
            "sha256": _sha256(args.input),
            "campaign_manifest_hash": source.get("campaign_manifest_hash"),
        },
        "analysis": analysis,
        "ccf_a_quantum_advantage_claimable": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    maximum = analysis["maximum_cap_summary"]
    sys.stdout.write(
        f"wrote matched failure attribution to {args.output}\n"
        f"attempts={analysis['attempt_count']} max_cap={maximum['query_cap']} "
        f"observed_dominated_families={maximum['observed_dominated_family_count']} "
        f"idealized_dominated_families={maximum['idealized_dominated_family_count']}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
