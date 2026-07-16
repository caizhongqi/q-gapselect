#!/usr/bin/env python3
"""Run the frozen public-data classifier-selection external-validity panel."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
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
    summarize_fixed_fixture_calibration,
)
from qgapselect.matched_quantum_baselines import MatchedBaselineConfig  # noqa: E402
from qgapselect.models import IAEConfig  # noqa: E402
from qgapselect.uci_classifier_benchmarks import (  # noqa: E402
    CLAIM_SCOPE as UCI_CLAIM_SCOPE,
)
from qgapselect.uci_classifier_benchmarks import (  # noqa: E402
    LoadedUCIDataset,
    UCIClassifierBenchmark,
    build_uci_classifier_benchmark,
    load_covertype,
    load_letter_recognition,
    load_optdigits,
    load_sklearn_digits_offline,
)

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "q_gapselect_uci_classifier_selection_campaign"
DEFAULT_CONFIG = REPOSITORY / "configs" / "uci_classifier_benchmarks.json"
DEFAULT_DATA_ROOT = REPOSITORY / "data" / "uci"
DEFAULT_THEORY = REPOSITORY / "artifacts" / "theorem_closure_audit.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "uci_classifier_benchmark_diagnostic.json.gz"


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


def _load_json(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return dict(_mapping(document, str(path)))


def load_config(path: Path) -> dict[str, object]:
    document = _load_json(path)
    if _integer(document.get("schema_version"), "schema_version", minimum=1) != 1:
        raise ValueError("unsupported config schema_version")
    dataset = dict(_mapping(document.get("dataset"), "dataset"))
    notes = tuple(_string(item, "notes item") for item in _sequence(document.get("notes"), "notes"))
    caps = tuple(
        sorted(
            {
                _integer(item, "query cap", minimum=1)
                for item in _sequence(document.get("query_caps"), "query_caps")
            }
        )
    )
    if not caps:
        raise ValueError("query_caps cannot be empty")
    source_mode = _string(dataset.get("source_mode"), "dataset.source_mode")
    if source_mode not in {
        "sklearn_bundled_offline_diagnostic",
        "official_uci_local_files",
    }:
        raise ValueError("unsupported dataset.source_mode")
    official_confirmatory = dataset.get("official_confirmatory")
    if not isinstance(official_confirmatory, bool):
        raise TypeError("dataset.official_confirmatory must be bool")
    dataset.update(
        dataset_id=_string(dataset.get("dataset_id"), "dataset.dataset_id"),
        source_mode=source_mode,
        official_confirmatory=official_confirmatory,
        n_arms=_integer(dataset.get("n_arms"), "dataset.n_arms", minimum=2),
        k=_integer(dataset.get("k"), "dataset.k", minimum=1),
        n_shards=_integer(dataset.get("n_shards"), "dataset.n_shards", minimum=5),
    )
    if int(dataset["k"]) >= int(dataset["n_arms"]):
        raise ValueError("dataset.k must be smaller than dataset.n_arms")
    return {
        "schema_version": 1,
        "experiment_name": _string(document.get("experiment_name"), "experiment_name"),
        "dataset": dataset,
        "master_seed": _integer(document.get("master_seed"), "master_seed"),
        "repetitions_per_fixture": _integer(
            document.get("repetitions_per_fixture"),
            "repetitions_per_fixture",
            minimum=2,
        ),
        "query_caps": caps,
        "failure_budget": _probability(document.get("failure_budget"), "failure_budget"),
        "coherent_config": dict(
            _mapping(document.get("coherent_config"), "coherent_config")
        ),
        "baseline_config": dict(
            _mapping(document.get("baseline_config"), "baseline_config")
        ),
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
        "familywise_alpha": _probability(
            document.get("familywise_alpha"), "familywise_alpha"
        ),
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
        "notes": notes,
    }


def _one_local_file(data_root: Path, file_name: str) -> Path:
    matches = tuple(path for path in data_root.rglob(file_name) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {file_name!r} below {data_root}, found {len(matches)}"
        )
    return matches[0]


def _load_dataset(config: Mapping[str, object], data_root: Path) -> LoadedUCIDataset:
    dataset_id = str(config["dataset_id"])
    source_mode = str(config["source_mode"])
    if source_mode == "sklearn_bundled_offline_diagnostic":
        if dataset_id != "sklearn_digits_offline":
            raise ValueError("the sklearn bundled mode only supports sklearn_digits_offline")
        return load_sklearn_digits_offline()
    if dataset_id == "letter_recognition":
        return load_letter_recognition(_one_local_file(data_root, "letter-recognition.data"))
    if dataset_id == "optdigits":
        return load_optdigits(
            _one_local_file(data_root, "optdigits.tra"),
            _one_local_file(data_root, "optdigits.tes"),
        )
    if dataset_id == "covertype":
        compressed = tuple(
            path for path in data_root.rglob("covtype.data.gz") if path.is_file()
        )
        if len(compressed) > 1:
            raise FileNotFoundError(
                f"expected at most one 'covtype.data.gz' below {data_root}"
            )
        source = (
            compressed[0]
            if compressed
            else _one_local_file(data_root, "covtype.data")
        )
        return load_covertype(source)
    raise ValueError(f"unsupported official dataset_id {dataset_id!r}")


def _method_config(config: Mapping[str, object]) -> CCFAMatchedBenchmarkConfig:
    baseline_document = _mapping(config["baseline_config"], "baseline_config")
    iae_document = _mapping(baseline_document.get("iae"), "baseline_config.iae")
    return CCFAMatchedBenchmarkConfig(
        master_seed=int(config["master_seed"]),
        repetitions=int(config["repetitions_per_fixture"]),
        query_caps=tuple(config["query_caps"]),
        failure_budget=float(config["failure_budget"]),
        coherent=VariableTimeHistoryConfig(**dict(config["coherent_config"])),
        baselines=MatchedBaselineConfig(
            initial_angular_precision=float(baseline_document["initial_angular_precision"]),
            precision_decay=float(baseline_document["precision_decay"]),
            max_levels=int(baseline_document["max_levels"]),
            iae=IAEConfig(**dict(iae_document)),
        ),
        coarse_partition_block_size=int(config["coarse_partition_block_size"]),
        coarse_partition_seed=int(config["coarse_partition_seed"]),
    )


def _record_key(record: CCFAMatchedTrialRecord) -> tuple[object, ...]:
    return (
        record.family_id,
        record.instance_id,
        record.repetition,
        record.query_cap,
        record.method_id,
    )


def _fixture_shard(
    instances: tuple[object, ...],
    method_config: CCFAMatchedBenchmarkConfig,
    fixture_key: tuple[str, str],
) -> tuple[CCFAMatchedTrialRecord, ...]:
    return tuple(
        iter_ccfa_matched_trials(
            instances,
            method_config,
            execution_fixture_keys=(fixture_key,),
        )
    )


def _execute_campaign(
    benchmark: UCIClassifierBenchmark,
    method_config: CCFAMatchedBenchmarkConfig,
    *,
    workers: int,
) -> tuple[CCFAMatchedTrialRecord, ...]:
    instances = tuple(benchmark.instances)
    if workers == 1:
        rows = tuple(iter_ccfa_matched_trials(instances, method_config))
    else:
        keys = tuple((instance.family_id, instance.instance_id) for instance in instances)
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="qgapselect-uci-fixture",
        ) as executor:
            futures = tuple(
                executor.submit(_fixture_shard, instances, method_config, key)
                for key in keys
            )
            rows = tuple(row for future in futures for row in future.result())
    rows = tuple(sorted(rows, key=_record_key))
    validate_complete_matched_panel(rows)
    return rows


def _theory_statuses(theory: Mapping[str, object]) -> dict[str, str]:
    summary = _mapping(theory.get("summary"), "theory summary")
    return {
        "new_upper_bound": (
            "PROVED" if bool(summary.get("theorem_chain_closed")) else "OPEN"
        ),
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


def run_experiment(
    config: Mapping[str, object],
    *,
    data_root: Path,
    theory_path: Path,
) -> dict[str, object]:
    dataset_config = _mapping(config["dataset"], "dataset")
    dataset = _load_dataset(dataset_config, data_root)
    official_confirmatory = bool(dataset_config["official_confirmatory"])
    if official_confirmatory and not dataset.official_source:
        raise ValueError("a non-official source cannot be marked official_confirmatory")
    benchmark = build_uci_classifier_benchmark(
        dataset,
        n_arms=int(dataset_config["n_arms"]),
        k=int(dataset_config["k"]),
        n_shards=int(dataset_config["n_shards"]),
    )
    if len(benchmark.instances) < 2:
        raise ValueError("at least two accepted fixed fixtures are required for inference")
    if official_confirmatory and len(benchmark.instances) < 5:
        raise ValueError("an official confirmatory dataset requires at least five fixtures")

    method_config = _method_config(config)
    workers = int(config["workers"])
    campaign_hash = matched_campaign_manifest_hash(benchmark.instances, method_config)
    records = _execute_campaign(benchmark, method_config, workers=workers)
    aggregates = aggregate_ccfa_matched_trials(records)
    maximum_cap = max(method_config.query_caps)
    calibration = summarize_fixed_fixture_calibration(
        tuple(record for record in records if record.query_cap == maximum_cap),
        target_success_probability=float(config["fixed_fixture_target_probability"]),
        familywise_alpha=float(config["familywise_alpha"]),
    )
    theory = _load_json(theory_path)
    baselines = tuple(
        method_id
        for method_id in method_config.method_ids
        if method_id != COHERENT_METHOD_ID
    )
    evidence_config = CCFAEvidenceGateConfig(
        candidate_method_id=COHERENT_METHOD_ID,
        strongest_baseline_method_ids=baselines,
        information_regime=K_ONLY_INFORMATION_REGIME,
        preregistered_fixtures=tuple(
            PreregisteredFixture(instance.family_id, instance.instance_id)
            for instance in benchmark.instances
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
    gaps = tuple(float(instance.public_gap_floor) for instance in benchmark.instances)
    resolved = {
        **{key: value for key, value in dict(config).items() if key != "workers"},
        "query_caps": list(config["query_caps"]),
        "notes": list(config["notes"]),
    }
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "experiment_name": config["experiment_name"],
        "claim_scope": UCI_CLAIM_SCOPE,
        "claim_status": "external_validity_diagnostic_advantage_gate_blocked",
        "resolved_config": resolved,
        "dataset_source_manifest": dataset.source_document()
        | {"source_manifest_hash": dataset.source_manifest_hash},
        "benchmark_manifest": benchmark.manifest.as_dict()
        | {"manifest_hash": benchmark.manifest.manifest_hash},
        "campaign_manifest_hash": campaign_hash,
        "theory_audit": {
            "path": str(theory_path),
            "statuses": _theory_statuses(theory),
        },
        "summary": {
            "dataset_id": dataset.dataset_id,
            "official_source": dataset.official_source,
            "official_confirmatory": official_confirmatory,
            "semi_synthetic_external_validity": True,
            "accepted_fixture_count": len(benchmark.instances),
            "boundary_failure_count": len(benchmark.boundary_failures),
            "arm_count": len(benchmark.arm_selection.selected_arms),
            "k": int(dataset_config["k"]),
            "minimum_angular_boundary_gap": min(gaps),
            "maximum_angular_boundary_gap": max(gaps),
            "repetitions_per_fixture": method_config.repetitions,
            "query_cap_count": len(method_config.query_caps),
            "method_count": len(method_config.method_ids),
            "run_count": len(records),
            "information_matched_primary_panel": True,
            "query_cap_matched_primary_panel": True,
            "all_attempts_in_denominator": True,
            "fixed_fixture_multiseed_calibration": True,
            "coherent_index_execution_performed": False,
            "quantum_hardware_execution_performed": False,
            "llm_execution_performed": False,
            "commercial_api_execution_performed": False,
            "ccf_a_quantum_advantage_claimable": evidence.advantage_claimable,
            "blocker_count": len(evidence.blockers),
        },
        "boundary_failures": [asdict(item) for item in benchmark.boundary_failures],
        "aggregates": [item.as_dict() for item in aggregates],
        "fixed_fixture_calibration": calibration.as_dict(),
        "evidence_gate": evidence.as_dict(),
        "records": [record.as_dict() for record in records],
        "provenance": {
            **_git_provenance(),
            "workers": workers,
            "executor": (
                "serial_fixture_order"
                if workers == 1
                else "thread_pool_one_task_per_fixture"
            ),
            "fixture_task_count": len(benchmark.instances),
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
        json.dumps(artifact, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--theory-artifact", type=Path, default=DEFAULT_THEORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions-per-fixture", type=int)
    parser.add_argument("--query-caps", nargs="+", type=int)
    parser.add_argument("--bootstrap-repetitions", type=int)
    parser.add_argument("--workers", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.repetitions_per_fixture is not None:
        config["repetitions_per_fixture"] = _integer(
            args.repetitions_per_fixture,
            "--repetitions-per-fixture",
            minimum=2,
        )
    if args.query_caps is not None:
        config["query_caps"] = tuple(
            sorted({_integer(item, "--query-caps", minimum=1) for item in args.query_caps})
        )
    if args.bootstrap_repetitions is not None:
        config["bootstrap_repetitions"] = _integer(
            args.bootstrap_repetitions,
            "--bootstrap-repetitions",
            minimum=1,
        )
    if args.workers is not None:
        config["workers"] = _integer(args.workers, "--workers", minimum=1)
    artifact = run_experiment(
        config,
        data_root=args.data_root.resolve(),
        theory_path=args.theory_artifact.resolve(),
    )
    _write_artifact(args.output, artifact)
    summary = artifact["summary"]
    sys.stdout.write(
        f"wrote UCI classifier-selection diagnostic to {args.output}\n"
        f"runs={summary['run_count']} fixtures={summary['accepted_fixture_count']} "
        f"official={summary['official_source']} "
        f"claimable={summary['ccf_a_quantum_advantage_claimable']}\n"
        "scope=semi-synthetic external validity; no LLM, API, hardware, or security run\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
