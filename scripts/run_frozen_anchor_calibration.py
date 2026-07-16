#!/usr/bin/env python3
"""Run fixed-fixture, multi-seed Layer-C calibration without any LLM."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from qgapselect.fixed_fixture_calibration import (
    CLAIM_SCOPE,
    SELECTION_RULE,
    calibration_manifest_hash,
    select_hardness_quantile_anchors,
    summarize_fixed_fixture_calibration,
)
from qgapselect.frozen_quantum_instance_design import (
    generate_frozen_quantum_instance_design,
)
from qgapselect.frozen_quantum_reference_benchmarking import (
    FrozenQuantumMethodConfigs,
    FrozenQuantumReferenceInstance,
    run_frozen_quantum_reference_benchmark,
)
from qgapselect.models import GapSelectConfig, IAEConfig

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs" / "frozen_anchor_calibration.json"
DEFAULT_MIXTURE_ARTIFACT = (
    REPOSITORY / "artifacts" / "frozen_quantum_reference_diagnostic.json"
)
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "frozen_anchor_calibration.json.gz"
ARTIFACT_TYPE = "q_gapselect_fixed_fixture_multiseed_calibration"
SCHEMA_VERSION = 1
_INSTANCE_PATTERN = re.compile(r"^(?P<family>.+)/problem-(?P<index>[0-9]+)$")


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


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return result


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _only_keys(document: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")


def load_config(path: Path) -> dict[str, object]:
    document = _mapping(json.loads(path.read_text(encoding="utf-8")), "config")
    _only_keys(
        document,
        {
            "schema_version",
            "experiment_name",
            "master_seed",
            "anchors_per_family",
            "repetitions_per_anchor",
            "target_success_probability",
            "familywise_alpha",
            "included_families",
            "selection_rule",
            "notes",
        },
        "config",
    )
    if _integer(document.get("schema_version"), "schema_version", minimum=1) != 1:
        raise ValueError("schema_version must equal 1")
    selection_rule = _string(document.get("selection_rule"), "selection_rule")
    if selection_rule != SELECTION_RULE:
        raise ValueError(f"selection_rule must equal {SELECTION_RULE!r}")
    families = tuple(
        _string(item, f"included_families[{index}]")
        for index, item in enumerate(
            _sequence(document.get("included_families"), "included_families")
        )
    )
    if not families or len(set(families)) != len(families):
        raise ValueError("included_families must be non-empty and unique")
    notes = tuple(
        _string(item, f"notes[{index}]")
        for index, item in enumerate(_sequence(document.get("notes"), "notes"))
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": _string(document.get("experiment_name"), "experiment_name"),
        "master_seed": _integer(document.get("master_seed"), "master_seed"),
        "anchors_per_family": _integer(
            document.get("anchors_per_family"), "anchors_per_family", minimum=1
        ),
        "repetitions_per_anchor": _integer(
            document.get("repetitions_per_anchor"),
            "repetitions_per_anchor",
            minimum=1,
        ),
        "target_success_probability": _probability(
            document.get("target_success_probability"),
            "target_success_probability",
        ),
        "familywise_alpha": _probability(
            document.get("familywise_alpha"), "familywise_alpha"
        ),
        "included_families": families,
        "selection_rule": selection_rule,
        "notes": notes,
    }


def _derived_source_seed(master_seed: int, *parts: object) -> int:
    material = "\0".join(
        ("qgapselect-layer-c-campaign-v1", str(master_seed), *map(str, parts))
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _source_method_configs(resolved: Mapping[str, object]) -> FrozenQuantumMethodConfigs:
    document = _mapping(resolved.get("method_configs"), "method_configs")
    qgap = _mapping(document.get("qgapselect"), "method_configs.qgapselect")
    independent = _mapping(
        document.get("independent_iae"), "method_configs.independent_iae"
    )
    threshold = _mapping(
        document.get("known_threshold_iae"), "method_configs.known_threshold_iae"
    )
    method_ids = tuple(
        _string(item, f"method_ids[{index}]")
        for index, item in enumerate(
            _sequence(document.get("method_ids"), "method_configs.method_ids")
        )
    )
    return FrozenQuantumMethodConfigs(
        qgapselect=GapSelectConfig(**dict(qgap)),
        independent_iae=IAEConfig(**dict(independent)),
        known_threshold_iae=IAEConfig(**dict(threshold)),
        failure_probability=float(document["failure_probability"]),
        precision_fraction=float(document["precision_fraction"]),
        method_ids=method_ids,
    )


def _source_case_map(resolved: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for index, raw in enumerate(_sequence(resolved.get("cases"), "resolved.cases")):
        case = _mapping(raw, f"resolved.cases[{index}]")
        name = _string(case.get("name"), f"resolved.cases[{index}].name")
        if name in result:
            raise ValueError("source mixture config contains duplicate case names")
        result[name] = case
    return result


def _selected_instances(
    mixture: Mapping[str, object],
    selections: Sequence[object],
) -> tuple[FrozenQuantumReferenceInstance, ...]:
    resolved = _mapping(mixture.get("resolved_config"), "resolved_config")
    source_master_seed = _integer(resolved.get("master_seed"), "source master_seed")
    table_size = _integer(resolved.get("table_size"), "source table_size", minimum=1)
    threshold = float(resolved["public_threshold"])
    rounding_margin = float(resolved["rounding_angular_margin"])
    cases = _source_case_map(resolved)
    public_documents = {
        (str(document["family_id"]), str(document["instance_id"])): document
        for document in _sequence(
            _mapping(mixture.get("report"), "report").get("instances"),
            "report.instances",
        )
    }
    instances: list[FrozenQuantumReferenceInstance] = []
    for selection in selections:
        match = _INSTANCE_PATTERN.fullmatch(selection.instance_id)
        if match is None or match.group("family") != selection.family_id:
            raise ValueError("anchor instance_id does not encode its family and index")
        problem_index = int(match.group("index"))
        case = cases.get(selection.family_id)
        if case is None:
            raise ValueError("selected family is missing from the source mixture config")
        public_floor = float(case["angular_gap"]) - rounding_margin
        case_seed = _derived_source_seed(
            source_master_seed,
            selection.family_id,
            problem_index,
            "case",
        )
        assignment_seed = _derived_source_seed(
            source_master_seed,
            selection.family_id,
            problem_index,
            "assignment",
        )
        design = generate_frozen_quantum_instance_design(
            case_id=selection.family_id,
            family=str(case["family"]),
            n_arms=int(case["n"]),
            k=int(case["k"]),
            threshold=threshold,
            public_gap_floor=public_floor,
            table_size=table_size,
            case_seed=case_seed,
            permutation_seed=assignment_seed,
        )
        if design.difficulty_fingerprint != selection.difficulty_fingerprint:
            raise RuntimeError("regenerated anchor difficulty fingerprint changed")
        source_document = public_documents[(selection.family_id, selection.instance_id)]
        if design.fixture.manifest_hash != source_document["fixture_manifest_hash"]:
            raise RuntimeError("regenerated anchor fixture manifest changed")
        instances.append(
            FrozenQuantumReferenceInstance(
                family_id=selection.family_id,
                instance_id=selection.instance_id,
                fixture=design.fixture,
                public_threshold=threshold,
                public_gap_floor=public_floor,
                k=int(case["k"]),
                difficulty_fingerprint=design.difficulty_fingerprint,
                structure_metrics=source_document["structure_metrics"],
            )
        )
    return tuple(instances)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    """Prefer a repository-relative provenance path without requiring one."""

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


def run_experiment(
    config: Mapping[str, object],
    mixture_artifact: Path,
) -> dict[str, object]:
    mixture = _mapping(
        json.loads(mixture_artifact.read_text(encoding="utf-8")),
        "mixture artifact",
    )
    if bool(_mapping(mixture.get("summary"), "summary").get("llm_execution_performed")):
        raise ValueError("source mixture must be an algorithm-only artifact")
    report_document = _mapping(mixture.get("report"), "report")
    public_instances = tuple(
        _mapping(item, f"report.instances[{index}]")
        for index, item in enumerate(
            _sequence(report_document.get("instances"), "report.instances")
        )
    )
    selections = select_hardness_quantile_anchors(
        public_instances,
        anchors_per_family=int(config["anchors_per_family"]),
        included_families=config["included_families"],
    )
    instances = _selected_instances(mixture, selections)
    resolved_source = _mapping(mixture.get("resolved_config"), "resolved_config")
    method_configs = _source_method_configs(resolved_source)
    repetitions = int(config["repetitions_per_anchor"])
    benchmark = run_frozen_quantum_reference_benchmark(
        instances,
        method_configs,
        repetitions=repetitions,
        master_seed=int(config["master_seed"]),
        instance_chunk_size=1,
    )
    calibration = summarize_fixed_fixture_calibration(
        benchmark.runs,
        target_success_probability=float(config["target_success_probability"]),
        familywise_alpha=float(config["familywise_alpha"]),
    )
    resolved_config = {
        **dict(config),
        "included_families": list(config["included_families"]),
        "notes": list(config["notes"]),
    }
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "experiment_name": config["experiment_name"],
        "claim_status": CLAIM_SCOPE,
        "resolved_config": resolved_config,
        "source_mixture": {
            "path": _display_path(mixture_artifact),
            "sha256": _file_sha256(mixture_artifact),
            "manifest_hash": report_document["manifest_hash"],
        },
        "anchor_manifest_hash": calibration_manifest_hash(
            selections,
            repetitions=repetitions,
            master_seed=int(config["master_seed"]),
        ),
        "anchor_selections": [selection.as_dict() for selection in selections],
        "summary": {
            "family_count": len(config["included_families"]),
            "anchor_count": len(selections),
            "anchors_per_family": config["anchors_per_family"],
            "repetitions_per_anchor": repetitions,
            "method_count": len(method_configs.method_ids),
            "run_count": len(benchmark.runs),
            "selection_uses_algorithm_outcomes": False,
            "fixed_fixture_multiseed_calibration_performed": True,
            "fixture_is_independent_unit": True,
            "llm_execution_performed": False,
            "hardware_execution_performed": False,
            "quantum_advantage_claimed": False,
            "worst_case_fixed_confidence_claimed": False,
        },
        "calibration": calibration.as_dict(),
        "benchmark": benchmark.as_dict(),
        "provenance": _git_provenance(),
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
    parser.add_argument("--mixture-artifact", type=Path, default=DEFAULT_MIXTURE_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--anchors-per-family", type=int)
    parser.add_argument("--repetitions-per-anchor", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.anchors_per_family is not None:
        config["anchors_per_family"] = _integer(
            args.anchors_per_family, "--anchors-per-family", minimum=1
        )
    if args.repetitions_per_anchor is not None:
        config["repetitions_per_anchor"] = _integer(
            args.repetitions_per_anchor, "--repetitions-per-anchor", minimum=1
        )
    artifact = run_experiment(config, args.mixture_artifact.resolve())
    _write_artifact(args.output, artifact)
    summary = artifact["summary"]
    sys.stdout.write(
        f"wrote fixed-fixture calibration to {args.output}\n"
        f"runs={summary['run_count']} anchors={summary['anchor_count']}\n"
        "scope=fixed synthetic fixtures; no LLM, hardware, or advantage claim\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
