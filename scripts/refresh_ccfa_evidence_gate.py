#!/usr/bin/env python3
"""Recompute the fail-closed evidence gate from an immutable attempt artifact."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from qgapselect.ccfa_evidence_gate import (  # noqa: E402
    CCFAEvidenceGateConfig,
    PreregisteredFixture,
    evaluate_ccfa_evidence_gate,
)
from qgapselect.ccfa_matched_benchmarking import (  # noqa: E402
    COHERENT_METHOD_ID,
    K_ONLY_INFORMATION_REGIME,
    PRIMARY_COMPARISON_GROUP,
)

REFRESH_SCHEMA_VERSION = 1
IMPLEMENTATION_RESOURCE_STATUSES = {
    "coherent_index_execution": "ANALYTIC_FINITE_STATE_IR_ONLY",
    "resource_accounting": "PARTIAL_LOGICAL_QUERY_AND_CANDIDATE_IR",
    "strongest_baseline_fidelity": "PAPER_INFORMED_STAND_INS_ONLY",
}


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _read(path: Path) -> dict[str, object]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    return dict(_mapping(value, "artifact"))


def _write(path: Path, artifact: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(artifact, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
        return
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def refresh_evidence_gate(artifact: Mapping[str, object]) -> dict[str, object]:
    """Re-evaluate only claims; raw attempts and campaign manifest are unchanged."""

    result = json.loads(json.dumps(dict(artifact), allow_nan=False))
    for record in _sequence(result.get("records"), "records"):
        mutable_record = _mapping(record, "record")
        uses_finite_state_tape = bool(mutable_record.get("direct_multi_output", False))
        mutable_record["finite_state_direct_output_tape"] = uses_finite_state_tape
        mutable_record["coherent_direct_multi_output_verified"] = False
        mutable_record["direct_multi_output_semantics"] = (
            "analytic_finite_state_output_tape_not_coherent_cross_index_union"
            if uses_finite_state_tape
            else "not_executed"
        )
    for aggregate in _sequence(result.get("aggregates"), "aggregates"):
        mutable_aggregate = _mapping(aggregate, "aggregate")
        uses_finite_state_tape = bool(
            mutable_aggregate.get("direct_multi_output", False)
        )
        mutable_aggregate["finite_state_direct_output_tape"] = uses_finite_state_tape
        mutable_aggregate["coherent_direct_multi_output_verified"] = False
        mutable_aggregate["direct_multi_output_semantics"] = (
            "analytic_finite_state_output_tape_not_coherent_cross_index_union"
            if uses_finite_state_tape
            else "not_executed"
        )
    records = tuple(
        _mapping(item, "record") for item in _sequence(result.get("records"), "records")
    )
    if not records:
        raise ValueError("artifact has no records")
    first = records[0]
    campaign_hash = str(result.get("campaign_manifest_hash", ""))
    if len(campaign_hash) != 64:
        raise ValueError("artifact has no valid campaign manifest hash")
    fixture_keys = tuple(
        (str(item[0]), str(item[1]))
        for item in _sequence(
            first.get("preregistered_fixture_keys"),
            "preregistered_fixture_keys",
        )
    )
    query_caps = tuple(
        int(item)
        for item in _sequence(
            first.get("preregistered_query_caps"),
            "preregistered_query_caps",
        )
    )
    repetitions = int(first.get("preregistered_repetitions", 0))
    baselines = tuple(
        sorted(
            {
                str(record["method_id"])
                for record in records
                if record.get("comparison_group") == PRIMARY_COMPARISON_GROUP
                and record.get("information_regime") == K_ONLY_INFORMATION_REGIME
                and record.get("method_id") != COHERENT_METHOD_ID
            }
        )
    )
    resolved = _mapping(result.get("resolved_config"), "resolved_config")
    theory = _mapping(result.get("theory_audit"), "theory_audit")
    statuses = {
        str(key): str(value)
        for key, value in _mapping(theory.get("statuses"), "theory statuses").items()
    }
    gate_config = CCFAEvidenceGateConfig(
        candidate_method_id=COHERENT_METHOD_ID,
        strongest_baseline_method_ids=baselines,
        information_regime=K_ONLY_INFORMATION_REGIME,
        preregistered_fixtures=tuple(
            PreregisteredFixture(family_id, instance_id)
            for family_id, instance_id in fixture_keys
        ),
        preregistered_query_caps=query_caps,
        repetitions_per_fixture=repetitions,
        preregistration_status="LOCKED_BEFORE_RUN",
        preregistration_manifest_sha256=campaign_hash,
        minimum_risk_difference=float(resolved["minimum_risk_difference"]),
        theory_statuses=statuses,
        implementation_resource_statuses=IMPLEMENTATION_RESOURCE_STATUSES,
        familywise_alpha=float(resolved["familywise_alpha"]),
        bootstrap_repetitions=int(resolved["bootstrap_repetitions"]),
        bootstrap_seed=int(resolved["bootstrap_seed"]),
        minimum_fixtures_per_family=2,
    )
    report = evaluate_ccfa_evidence_gate(records, gate_config)
    result["evidence_gate"] = report.as_dict()
    summary = dict(_mapping(result.get("summary"), "summary"))
    summary["blocker_count"] = len(report.blockers)
    summary["ccf_a_quantum_advantage_claimable"] = report.advantage_claimable
    summary["coherent_direct_multi_output_verified"] = False
    result["summary"] = summary
    status_suffix = "passed" if report.advantage_claimable else "blocked"
    result["claim_status"] = (
        "external_validity_diagnostic_advantage_gate_" + status_suffix
        if result.get("artifact_type")
        == "q_gapselect_uci_classifier_selection_campaign"
        else "fail_closed_ccfa_evidence_gate_" + status_suffix
    )
    provenance = dict(_mapping(result.get("provenance"), "provenance"))
    provenance.update(
        evidence_gate_refreshed_from_raw_records=True,
        evidence_gate_refresh_schema_version=REFRESH_SCHEMA_VERSION,
        raw_record_count_at_refresh=len(records),
    )
    result["provenance"] = provenance
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = refresh_evidence_gate(_read(args.input))
    _write(args.output, artifact)
    sys.stdout.write(
        f"refreshed evidence gate from {len(artifact['records'])} raw records "
        f"to {args.output}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
