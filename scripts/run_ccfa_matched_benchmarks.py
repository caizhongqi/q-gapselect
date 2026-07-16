#!/usr/bin/env python3
"""Run the preregistered fixed-cap, fixed-fixture Layer-C paper campaign."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
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
    CCFAMatchedBenchmarkConfig,
    CCFAMatchedTrialRecord,
    aggregate_ccfa_matched_trials,
    iter_ccfa_matched_trials,
    matched_campaign_manifest_hash,
    validate_complete_matched_panel,
)
from qgapselect.coherent_activity_history_core import (  # noqa: E402
    VariableTimeHistoryConfig,
)
from qgapselect.fixed_fixture_calibration import (  # noqa: E402
    SELECTION_RULE,
    select_hardness_quantile_anchors,
    summarize_fixed_fixture_calibration,
)
from qgapselect.frozen_quantum_instance_design import (  # noqa: E402
    generate_frozen_quantum_instance_design,
)
from qgapselect.frozen_quantum_reference_benchmarking import (  # noqa: E402
    FrozenQuantumReferenceInstance,
)
from qgapselect.matched_quantum_baselines import (  # noqa: E402
    MatchedBaselineConfig,
)
from qgapselect.models import IAEConfig  # noqa: E402

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "q_gapselect_ccfa_matched_layer_c_campaign"
DEFAULT_CONFIG = REPOSITORY / "configs" / "ccfa_matched_benchmarks.json"
DEFAULT_MIXTURE = REPOSITORY / "artifacts" / "frozen_quantum_reference_diagnostic.json"
DEFAULT_THEORY = REPOSITORY / "artifacts" / "theorem_closure_audit.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "ccfa_matched_benchmark_diagnostic.json.gz"
_INSTANCE_PATTERN = re.compile(r"^(?P<family>.+)/problem-(?P<index>[0-9]+)$")
_FORK_WORKER_INSTANCES: tuple[FrozenQuantumReferenceInstance, ...] | None = None
_FORK_WORKER_METHOD_CONFIG: CCFAMatchedBenchmarkConfig | None = None


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object with string keys")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(value: object, name: str) -> float:
    result = _number(value, name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return result


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _load_json(path: Path) -> Mapping[str, object]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return _mapping(json.load(handle), str(path))
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def load_config(path: Path) -> dict[str, object]:
    document = _load_json(path)
    if _integer(document.get("schema_version"), "schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    families = tuple(
        _string(value, "included family")
        for value in _sequence(document.get("included_families"), "included_families")
    )
    caps = tuple(
        _integer(value, "query cap", minimum=1)
        for value in _sequence(document.get("query_caps"), "query_caps")
    )
    if not families or len(set(families)) != len(families):
        raise ValueError("included_families must be non-empty and unique")
    if not caps or len(set(caps)) != len(caps):
        raise ValueError("query_caps must be non-empty and unique")
    selection_rule = _string(document.get("selection_rule"), "selection_rule")
    if selection_rule != SELECTION_RULE:
        raise ValueError(f"selection_rule must equal {SELECTION_RULE!r}")
    notes = tuple(_string(value, "note") for value in _sequence(document.get("notes"), "notes"))
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": _string(document.get("experiment_name"), "experiment_name"),
        "master_seed": _integer(document.get("master_seed"), "master_seed"),
        "anchors_per_family": _integer(
            document.get("anchors_per_family"), "anchors_per_family", minimum=2
        ),
        "repetitions_per_fixture": _integer(
            document.get("repetitions_per_fixture"),
            "repetitions_per_fixture",
            minimum=2,
        ),
        "query_caps": tuple(sorted(caps)),
        "failure_budget": _probability(document.get("failure_budget"), "failure_budget"),
        "included_families": families,
        "selection_rule": selection_rule,
        "coherent_config": dict(_mapping(document.get("coherent_config"), "coherent_config")),
        "baseline_config": dict(_mapping(document.get("baseline_config"), "baseline_config")),
        "coarse_partition_block_size": _integer(
            document.get("coarse_partition_block_size"),
            "coarse_partition_block_size",
            minimum=2,
        ),
        "coarse_partition_seed": _integer(
            document.get("coarse_partition_seed"), "coarse_partition_seed"
        ),
        "fixed_fixture_target_probability": _probability(
            document.get("fixed_fixture_target_probability"),
            "fixed_fixture_target_probability",
        ),
        "familywise_alpha": _probability(document.get("familywise_alpha"), "familywise_alpha"),
        "minimum_risk_difference": _number(
            document.get("minimum_risk_difference"), "minimum_risk_difference"
        ),
        "bootstrap_repetitions": _integer(
            document.get("bootstrap_repetitions"),
            "bootstrap_repetitions",
            minimum=1,
        ),
        "bootstrap_seed": _integer(document.get("bootstrap_seed"), "bootstrap_seed"),
        "workers": _integer(document.get("workers", 4), "workers", minimum=1),
        "dataset_protocol": dict(_mapping(document.get("dataset_protocol"), "dataset_protocol")),
        "notes": notes,
    }


def _derived_source_seed(master_seed: int, *parts: object) -> int:
    material = "\0".join(
        ("qgapselect-layer-c-campaign-v1", str(master_seed), *map(str, parts))
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _source_case_map(resolved: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for raw in _sequence(resolved.get("cases"), "resolved.cases"):
        case = _mapping(raw, "resolved case")
        name = _string(case.get("name"), "case name")
        if name in result:
            raise ValueError("source mixture contains duplicate case names")
        result[name] = case
    return result


def _selected_instances(
    mixture: Mapping[str, object], selections: Sequence[object]
) -> tuple[FrozenQuantumReferenceInstance, ...]:
    resolved = _mapping(mixture.get("resolved_config"), "resolved_config")
    source_master_seed = _integer(resolved.get("master_seed"), "source master seed")
    table_size = _integer(resolved.get("table_size"), "source table_size", minimum=1)
    threshold = _number(resolved.get("public_threshold"), "source public_threshold")
    rounding_margin = _number(resolved.get("rounding_angular_margin"), "source rounding margin")
    cases = _source_case_map(resolved)
    report = _mapping(mixture.get("report"), "report")
    documents = {
        (str(item["family_id"]), str(item["instance_id"])): item
        for item in (
            _mapping(value, "report instance")
            for value in _sequence(report.get("instances"), "report.instances")
        )
    }
    instances: list[FrozenQuantumReferenceInstance] = []
    for selection in selections:
        match = _INSTANCE_PATTERN.fullmatch(selection.instance_id)
        if match is None or match.group("family") != selection.family_id:
            raise ValueError("anchor instance_id does not encode family/index")
        problem_index = int(match.group("index"))
        case = cases[selection.family_id]
        public_floor = _number(case.get("angular_gap"), "case angular gap") - rounding_margin
        design = generate_frozen_quantum_instance_design(
            case_id=selection.family_id,
            family=str(case["family"]),
            n_arms=int(case["n"]),
            k=int(case["k"]),
            threshold=threshold,
            public_gap_floor=public_floor,
            table_size=table_size,
            case_seed=_derived_source_seed(
                source_master_seed, selection.family_id, problem_index, "case"
            ),
            permutation_seed=_derived_source_seed(
                source_master_seed, selection.family_id, problem_index, "assignment"
            ),
        )
        source = documents[(selection.family_id, selection.instance_id)]
        if design.difficulty_fingerprint != selection.difficulty_fingerprint:
            raise RuntimeError("regenerated difficulty fingerprint changed")
        if design.fixture.manifest_hash != source["fixture_manifest_hash"]:
            raise RuntimeError("regenerated fixture manifest changed")
        instances.append(
            FrozenQuantumReferenceInstance(
                family_id=selection.family_id,
                instance_id=selection.instance_id,
                fixture=design.fixture,
                public_threshold=threshold,
                public_gap_floor=public_floor,
                k=int(case["k"]),
                difficulty_fingerprint=design.difficulty_fingerprint,
                structure_metrics=_mapping(source["structure_metrics"], "structure metrics"),
            )
        )
    return tuple(instances)


def _method_config(config: Mapping[str, object]) -> CCFAMatchedBenchmarkConfig:
    baseline_document = _mapping(config["baseline_config"], "baseline_config")
    iae_document = _mapping(baseline_document.get("iae"), "baseline_config.iae")
    baseline = MatchedBaselineConfig(
        initial_angular_precision=float(baseline_document["initial_angular_precision"]),
        precision_decay=float(baseline_document["precision_decay"]),
        max_levels=int(baseline_document["max_levels"]),
        iae=IAEConfig(**dict(iae_document)),
    )
    return CCFAMatchedBenchmarkConfig(
        master_seed=int(config["master_seed"]),
        repetitions=int(config["repetitions_per_fixture"]),
        query_caps=tuple(config["query_caps"]),
        failure_budget=float(config["failure_budget"]),
        coherent=VariableTimeHistoryConfig(**dict(config["coherent_config"])),
        baselines=baseline,
        coarse_partition_block_size=int(config["coarse_partition_block_size"]),
        coarse_partition_seed=int(config["coarse_partition_seed"]),
    )


def _theory_statuses(theory: Mapping[str, object]) -> dict[str, str]:
    summary = _mapping(theory.get("summary"), "theory summary")
    return {
        "new_upper_bound": ("PROVED" if bool(summary.get("theorem_chain_closed")) else "OPEN"),
        "same_interface_composition_frontier": (
            "PROVED"
            if not bool(summary.get("public_partition_composition_matches_every_scale"))
            else "FALSIFIED_BY_MATCHING_COMPOSITION"
        ),
        "matching_lower_bound": (
            "PROVED"
            if not bool(summary.get("weighted_matching_lower_bound_open_every_scale"))
            else "OPEN"
        ),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY))
    except ValueError:
        return str(path)


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


def _fixture_shard(
    instances: tuple[FrozenQuantumReferenceInstance, ...],
    method_config: CCFAMatchedBenchmarkConfig,
    fixture_key: tuple[str, str],
) -> tuple[CCFAMatchedTrialRecord, ...]:
    """Execute one fixture while retaining the full campaign registration."""

    return tuple(
        iter_ccfa_matched_trials(
            instances,
            method_config,
            execution_fixture_keys=(fixture_key,),
        )
    )


def _fork_fixture_shard(
    fixture_key: tuple[str, str],
) -> tuple[CCFAMatchedTrialRecord, ...]:
    """Read immutable campaign state inherited by a POSIX fork worker."""

    if _FORK_WORKER_INSTANCES is None or _FORK_WORKER_METHOD_CONFIG is None:
        raise RuntimeError("fork worker campaign context was not initialized")
    return tuple(
        replace(record, query_counts=dict(record.query_counts))
        for record in _fixture_shard(
            _FORK_WORKER_INSTANCES,
            _FORK_WORKER_METHOD_CONFIG,
            fixture_key,
        )
    )


def _record_sort_key(record: CCFAMatchedTrialRecord) -> tuple[object, ...]:
    return (
        record.family_id,
        record.instance_id,
        record.repetition,
        record.query_cap,
        record.method_id,
    )


def _execute_matched_campaign(
    instances: tuple[FrozenQuantumReferenceInstance, ...],
    method_config: CCFAMatchedBenchmarkConfig,
    *,
    workers: int,
) -> tuple[CCFAMatchedTrialRecord, ...]:
    """Run serially or as one deterministic process task per fixture."""

    workers = _integer(workers, "workers", minimum=1)
    if workers == 1:
        rows = tuple(iter_ccfa_matched_trials(instances, method_config))
    else:
        fixture_keys = tuple((instance.family_id, instance.instance_id) for instance in instances)
        if os.name == "posix":
            global _FORK_WORKER_INSTANCES, _FORK_WORKER_METHOD_CONFIG
            _FORK_WORKER_INSTANCES = instances
            _FORK_WORKER_METHOD_CONFIG = method_config
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=get_context("fork"),
            ) as executor:
                futures = tuple(executor.submit(_fork_fixture_shard, key) for key in fixture_keys)
                rows = tuple(row for future in futures for row in future.result())
        else:
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="qgapselect-fixture",
            ) as executor:
                futures = tuple(
                    executor.submit(_fixture_shard, instances, method_config, key)
                    for key in fixture_keys
                )
                rows = tuple(row for future in futures for row in future.result())
    return tuple(sorted(rows, key=_record_sort_key))


def run_experiment(
    config: Mapping[str, object],
    *,
    mixture_path: Path,
    theory_path: Path,
) -> dict[str, object]:
    mixture = _load_json(mixture_path)
    source_summary = _mapping(mixture.get("summary"), "mixture summary")
    if bool(source_summary.get("llm_execution_performed")):
        raise ValueError("source mixture must be algorithm-only")
    public_documents = tuple(
        _mapping(value, "public instance")
        for value in _sequence(
            _mapping(mixture.get("report"), "mixture report").get("instances"),
            "report.instances",
        )
    )
    selections = select_hardness_quantile_anchors(
        public_documents,
        anchors_per_family=int(config["anchors_per_family"]),
        included_families=config["included_families"],
    )
    instances = _selected_instances(mixture, selections)
    method_config = _method_config(config)
    campaign_hash = matched_campaign_manifest_hash(instances, method_config)
    workers = _integer(config.get("workers", 4), "workers", minimum=1)
    records = _execute_matched_campaign(
        instances,
        method_config,
        workers=workers,
    )
    validate_complete_matched_panel(records)
    aggregates = aggregate_ccfa_matched_trials(records)

    maximum_cap = max(method_config.query_caps)
    calibration = summarize_fixed_fixture_calibration(
        tuple(record for record in records if record.query_cap == maximum_cap),
        target_success_probability=float(config["fixed_fixture_target_probability"]),
        familywise_alpha=float(config["familywise_alpha"]),
    )
    theory = _load_json(theory_path)
    primary_baselines = tuple(
        method_id for method_id in method_config.method_ids if method_id != COHERENT_METHOD_ID
    )
    evidence_config = CCFAEvidenceGateConfig(
        candidate_method_id=COHERENT_METHOD_ID,
        strongest_baseline_method_ids=primary_baselines,
        information_regime=K_ONLY_INFORMATION_REGIME,
        preregistered_fixtures=tuple(
            PreregisteredFixture(instance.family_id, instance.instance_id) for instance in instances
        ),
        preregistered_query_caps=method_config.query_caps,
        repetitions_per_fixture=method_config.repetitions,
        preregistration_status="LOCKED_BEFORE_RUN",
        preregistration_manifest_sha256=campaign_hash,
        minimum_risk_difference=float(config["minimum_risk_difference"]),
        theory_statuses=_theory_statuses(theory),
        implementation_resource_statuses={
            "coherent_index_execution": "ANALYTIC_FINITE_STATE_IR_ONLY",
            "resource_accounting": "PARTIAL_LOGICAL_QUERY_AND_CANDIDATE_IR",
            "strongest_baseline_fidelity": "PAPER_INFORMED_STAND_INS_ONLY",
        },
        familywise_alpha=float(config["familywise_alpha"]),
        bootstrap_repetitions=int(config["bootstrap_repetitions"]),
        bootstrap_seed=int(config["bootstrap_seed"]),
        minimum_fixtures_per_family=2,
    )
    evidence = evaluate_ccfa_evidence_gate(records, evidence_config)
    resolved_config = {
        **{key: value for key, value in dict(config).items() if key != "workers"},
        "query_caps": list(config["query_caps"]),
        "included_families": list(config["included_families"]),
        "notes": list(config["notes"]),
    }
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "experiment_name": config["experiment_name"],
        "claim_status": (
            "fail_closed_ccfa_evidence_gate_"
            + ("passed" if evidence.advantage_claimable else "blocked")
        ),
        "resolved_config": resolved_config,
        "source_mixture": {
            "path": _display_path(mixture_path),
            "sha256": _file_sha256(mixture_path),
        },
        "theory_audit": {
            "path": _display_path(theory_path),
            "sha256": _file_sha256(theory_path),
            "statuses": _theory_statuses(theory),
        },
        "campaign_manifest_hash": campaign_hash,
        "anchor_selections": [selection.as_dict() for selection in selections],
        "summary": {
            "family_count": len(config["included_families"]),
            "fixture_count": len(instances),
            "repetitions_per_fixture": method_config.repetitions,
            "query_cap_count": len(method_config.query_caps),
            "method_count": len(method_config.method_ids),
            "run_count": len(records),
            "all_attempts_in_denominator": True,
            "information_matched_primary_panel": True,
            "query_cap_matched_primary_panel": True,
            "fixed_fixture_multiseed_calibration": True,
            "llm_execution_performed": False,
            "commercial_api_execution_performed": False,
            "quantum_hardware_execution_performed": False,
            "coherent_index_execution_performed": False,
            "ccf_a_quantum_advantage_claimable": evidence.advantage_claimable,
            "blocker_count": len(evidence.blockers),
        },
        "aggregates": [aggregate.as_dict() for aggregate in aggregates],
        "fixed_fixture_calibration": calibration.as_dict(),
        "evidence_gate": evidence.as_dict(),
        "records": [record.as_dict() for record in records],
        "provenance": {
            **_git_provenance(),
            "workers": workers,
            "executor": (
                "serial_fixture_order"
                if workers == 1
                else (
                    "process_pool_fork_one_task_per_fixture"
                    if os.name == "posix"
                    else "thread_pool_one_task_per_fixture"
                )
            ),
            "fixture_task_count": len(instances),
        },
    }


def _write_artifact(path: Path, artifact: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(artifact, handle, ensure_ascii=False, allow_nan=False, sort_keys=True)
            handle.write("\n")
        return
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mixture-artifact", type=Path, default=DEFAULT_MIXTURE)
    parser.add_argument("--theory-artifact", type=Path, default=DEFAULT_THEORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--anchors-per-family", type=int)
    parser.add_argument("--repetitions-per-fixture", type=int)
    parser.add_argument("--query-caps", nargs="+", type=int)
    parser.add_argument("--bootstrap-repetitions", type=int)
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--workers", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.anchors_per_family is not None:
        config["anchors_per_family"] = _integer(
            args.anchors_per_family, "--anchors-per-family", minimum=2
        )
    if args.repetitions_per_fixture is not None:
        config["repetitions_per_fixture"] = _integer(
            args.repetitions_per_fixture,
            "--repetitions-per-fixture",
            minimum=2,
        )
    if args.query_caps is not None:
        config["query_caps"] = tuple(sorted(set(args.query_caps)))
    if args.bootstrap_repetitions is not None:
        config["bootstrap_repetitions"] = _integer(
            args.bootstrap_repetitions,
            "--bootstrap-repetitions",
            minimum=1,
        )
    if args.families is not None:
        config["included_families"] = tuple(args.families)
    if args.workers is not None:
        config["workers"] = _integer(args.workers, "--workers", minimum=1)
    artifact = run_experiment(
        config,
        mixture_path=args.mixture_artifact.resolve(),
        theory_path=args.theory_artifact.resolve(),
    )
    _write_artifact(args.output, artifact)
    summary = artifact["summary"]
    sys.stdout.write(
        f"wrote matched paper campaign to {args.output}\n"
        f"runs={summary['run_count']} fixtures={summary['fixture_count']} "
        f"claimable={summary['ccf_a_quantum_advantage_claimable']}\n"
        "scope=algorithm-only Layer-C experiment; no LLM, API, hardware, or security run\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
