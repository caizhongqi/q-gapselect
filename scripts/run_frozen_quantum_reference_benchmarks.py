#!/usr/bin/env python3
"""Run synthetic frozen-empirical Layer-C quantum reference experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from qgapselect.frozen_paired_statistics import (
    analyze_frozen_quantum_reference_pairs,
)
from qgapselect.frozen_quantum_instance_design import (
    generate_frozen_quantum_instance_design,
)
from qgapselect.frozen_quantum_reference_benchmarking import (
    CLAIM_SCOPE,
    FrozenQuantumMethodConfigs,
    FrozenQuantumReferenceInstance,
    run_frozen_quantum_reference_benchmark,
)
from qgapselect.models import GapSelectConfig, IAEConfig

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs" / "frozen_quantum_reference_benchmarks.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "frozen_quantum_reference_diagnostic.json"
ARTIFACT_TYPE = "q_gapselect_frozen_empirical_layer_c_reference_diagnostic"
SCHEMA_VERSION = 1
INSTANCE_DISTRIBUTION_GENERATOR = "nonisomorphic_angular_gap_vector_v1"
CASE_SEED_DERIVATION = "sha256(master_seed,case_name,problem_index,'case')"
ASSIGNMENT_SEED_DERIVATION = "sha256(master_seed,case_name,problem_index,'assignment')"
DIFFICULTY_FINGERPRINT_SCHEMA = "permutation_invariant_exact_count_structure_v1"
VARIED_INSTANCE_AXES = (
    "boundary_gap",
    "nonboundary_gaps",
    "selected_active_count",
    "rejected_active_count",
    "heterogeneity",
)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object with string keys")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _number(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a JSON number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _open_probability(value: object, name: str) -> float:
    result = _number(value, name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return result


def _only_keys(document: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")


def _instance_distribution(document: Mapping[str, object]) -> dict[str, object]:
    _only_keys(
        document,
        {
            "generator",
            "case_seed_derivation",
            "assignment_seed_derivation",
            "varied_axes",
            "difficulty_fingerprint",
            "paper_scale_minimum_unique_fraction",
        },
        "instance_distribution",
    )
    expected_strings = {
        "generator": INSTANCE_DISTRIBUTION_GENERATOR,
        "case_seed_derivation": CASE_SEED_DERIVATION,
        "assignment_seed_derivation": ASSIGNMENT_SEED_DERIVATION,
        "difficulty_fingerprint": DIFFICULTY_FINGERPRINT_SCHEMA,
    }
    for key, expected in expected_strings.items():
        actual = _string(document.get(key), f"instance_distribution.{key}")
        if actual != expected:
            raise ValueError(f"instance_distribution.{key} must be {expected!r}")
    varied_axes = tuple(
        _string(item, f"instance_distribution.varied_axes[{index}]")
        for index, item in enumerate(
            _sequence(
                document.get("varied_axes"),
                "instance_distribution.varied_axes",
            )
        )
    )
    if varied_axes != VARIED_INSTANCE_AXES:
        raise ValueError("instance_distribution.varied_axes must enumerate the preregistered axes")
    minimum_unique_fraction = _number(
        document.get("paper_scale_minimum_unique_fraction"),
        "instance_distribution.paper_scale_minimum_unique_fraction",
    )
    if minimum_unique_fraction > 1.0:
        raise ValueError(
            "instance_distribution.paper_scale_minimum_unique_fraction must not exceed 1"
        )
    return {
        **expected_strings,
        "varied_axes": list(varied_axes),
        "paper_scale_minimum_unique_fraction": minimum_unique_fraction,
    }


def _derived_seed(master_seed: int, *parts: object) -> int:
    material = "\0".join(
        ("qgapselect-layer-c-campaign-v1", str(master_seed), *map(str, parts))
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _iae_config(document: Mapping[str, object], name: str) -> IAEConfig:
    _only_keys(
        document,
        {
            "target_angular_precision",
            "confidence",
            "shots_per_round",
            "max_rounds",
            "max_grover_power",
            "grid_points",
        },
        name,
    )
    return IAEConfig(
        target_angular_precision=_number(
            document.get("target_angular_precision"), f"{name}.target_angular_precision"
        ),
        confidence=_number(document.get("confidence"), f"{name}.confidence"),
        shots_per_round=_integer(
            document.get("shots_per_round"), f"{name}.shots_per_round", minimum=1
        ),
        max_rounds=_integer(document.get("max_rounds"), f"{name}.max_rounds", minimum=1),
        max_grover_power=_integer(document.get("max_grover_power"), f"{name}.max_grover_power"),
        grid_points=_integer(document.get("grid_points"), f"{name}.grid_points", minimum=257),
    )


def _gapselect_config(document: Mapping[str, object]) -> GapSelectConfig:
    _only_keys(
        document,
        {
            "confidence",
            "initial_angular_epsilon",
            "epsilon_decay",
            "max_rounds",
            "shots_per_iae_round",
            "iae_max_rounds",
            "iae_max_grover_power",
            "iae_grid_points",
        },
        "qgapselect",
    )
    return GapSelectConfig(
        confidence=_number(document.get("confidence"), "qgapselect.confidence"),
        initial_angular_epsilon=_number(
            document.get("initial_angular_epsilon"),
            "qgapselect.initial_angular_epsilon",
        ),
        epsilon_decay=_number(document.get("epsilon_decay"), "qgapselect.epsilon_decay"),
        max_rounds=_integer(document.get("max_rounds"), "qgapselect.max_rounds", minimum=1),
        shots_per_iae_round=_integer(
            document.get("shots_per_iae_round"),
            "qgapselect.shots_per_iae_round",
            minimum=1,
        ),
        iae_max_rounds=_integer(
            document.get("iae_max_rounds"), "qgapselect.iae_max_rounds", minimum=1
        ),
        iae_max_grover_power=_integer(
            document.get("iae_max_grover_power"), "qgapselect.iae_max_grover_power"
        ),
        iae_grid_points=_integer(
            document.get("iae_grid_points"),
            "qgapselect.iae_grid_points",
            minimum=257,
        ),
    )


def load_config(path: Path) -> dict[str, object]:
    root = _mapping(json.loads(path.read_text(encoding="utf-8")), "root")
    _only_keys(
        root,
        {
            "schema_version",
            "experiment_name",
            "master_seed",
            "problem_instances",
            "algorithm_repetitions",
            "table_size",
            "public_threshold",
            "rounding_angular_margin",
            "failure_probability",
            "precision_fraction",
            "paired_bootstrap_repetitions",
            "paired_bootstrap_confidence_level",
            "paired_holm_alpha",
            "method_ids",
            "qgapselect",
            "independent_iae",
            "known_threshold_iae",
            "instance_distribution",
            "cases",
            "notes",
        },
        "root",
    )
    if _integer(root.get("schema_version"), "schema_version", minimum=1) != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    failure_probability = _number(root.get("failure_probability"), "failure_probability")
    method_ids = tuple(
        _string(item, f"method_ids[{index}]")
        for index, item in enumerate(_sequence(root.get("method_ids"), "method_ids"))
    )
    method_configs = FrozenQuantumMethodConfigs(
        qgapselect=_gapselect_config(_mapping(root.get("qgapselect"), "qgapselect")),
        independent_iae=_iae_config(
            _mapping(root.get("independent_iae"), "independent_iae"),
            "independent_iae",
        ),
        known_threshold_iae=_iae_config(
            _mapping(root.get("known_threshold_iae"), "known_threshold_iae"),
            "known_threshold_iae",
        ),
        failure_probability=failure_probability,
        precision_fraction=_number(root.get("precision_fraction"), "precision_fraction"),
        method_ids=method_ids,
    )
    raw_cases = tuple(
        dict(_mapping(item, f"cases[{index}]"))
        for index, item in enumerate(_sequence(root.get("cases"), "cases"))
    )
    for index, case in enumerate(raw_cases):
        _only_keys(case, {"name", "family", "n", "k", "angular_gap"}, f"cases[{index}]")
        _string(case.get("name"), f"cases[{index}].name")
        family = _string(case.get("family"), f"cases[{index}].family")
        if family not in {"equal_gap", "dyadic_gap", "clustered_boundary"}:
            raise ValueError(f"unknown cases[{index}].family {family!r}")
        n_arms = _integer(case.get("n"), f"cases[{index}].n", minimum=2)
        k = _integer(case.get("k"), f"cases[{index}].k", minimum=1)
        if k >= n_arms:
            raise ValueError(f"cases[{index}].k must be smaller than n")
        gap = _number(case.get("angular_gap"), f"cases[{index}].angular_gap")
        if not 0.0 < gap < math.pi / 4.0:
            raise ValueError(f"cases[{index}].angular_gap must lie in (0, pi/4)")
    notes = tuple(
        _string(item, f"notes[{index}]")
        for index, item in enumerate(_sequence(root.get("notes", ()), "notes"))
    )
    instance_distribution = _instance_distribution(
        _mapping(root.get("instance_distribution"), "instance_distribution")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": _string(root.get("experiment_name"), "experiment_name"),
        "master_seed": _integer(root.get("master_seed"), "master_seed"),
        "problem_instances": _integer(
            root.get("problem_instances"), "problem_instances", minimum=1
        ),
        "algorithm_repetitions": _integer(
            root.get("algorithm_repetitions"), "algorithm_repetitions", minimum=1
        ),
        "table_size": _integer(root.get("table_size"), "table_size", minimum=1),
        "public_threshold": _number(root.get("public_threshold"), "public_threshold"),
        "rounding_angular_margin": _number(
            root.get("rounding_angular_margin"), "rounding_angular_margin"
        ),
        "paired_bootstrap_repetitions": _integer(
            root.get("paired_bootstrap_repetitions"),
            "paired_bootstrap_repetitions",
            minimum=1,
        ),
        "paired_bootstrap_confidence_level": _open_probability(
            root.get("paired_bootstrap_confidence_level"),
            "paired_bootstrap_confidence_level",
        ),
        "paired_holm_alpha": _open_probability(
            root.get("paired_holm_alpha"),
            "paired_holm_alpha",
        ),
        "instance_distribution": instance_distribution,
        "method_configs": method_configs,
        "cases": raw_cases,
        "notes": notes,
    }


def build_instances(
    config: Mapping[str, object],
) -> Iterator[FrozenQuantumReferenceInstance]:
    """Lazily construct deterministic fixtures in canonical campaign order."""

    threshold = float(config["public_threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError("public_threshold must lie in (0, 1)")
    margin = float(config["rounding_angular_margin"])
    master_seed = int(config["master_seed"])
    for case in config["cases"]:
        name = str(case["name"])
        family = str(case["family"])
        n_arms = int(case["n"])
        k = int(case["k"])
        angular_gap = float(case["angular_gap"])
        public_floor = angular_gap - margin
        if public_floor <= 0.0:
            raise ValueError(f"rounding margin consumes the public gap in case {name!r}")
        for problem_index in range(int(config["problem_instances"])):
            case_seed = _derived_seed(master_seed, name, problem_index, "case")
            assignment_seed = _derived_seed(master_seed, name, problem_index, "assignment")
            design = generate_frozen_quantum_instance_design(
                case_id=name,
                family=family,
                n_arms=n_arms,
                k=k,
                threshold=threshold,
                public_gap_floor=public_floor,
                table_size=int(config["table_size"]),
                case_seed=case_seed,
                permutation_seed=assignment_seed,
            )
            structure = design.structure
            instance = FrozenQuantumReferenceInstance(
                family_id=name,
                instance_id=f"{name}/problem-{problem_index:04d}",
                fixture=design.fixture,
                public_threshold=threshold,
                public_gap_floor=public_floor,
                k=k,
                difficulty_fingerprint=design.difficulty_fingerprint,
                structure_metrics={
                    "generator_family": structure.family,
                    "n_arms": structure.n_arms,
                    "selected_active_count": structure.selected_active_count,
                    "rejected_active_count": structure.rejected_active_count,
                    "active_count": structure.active_count,
                    "base_gap_scale": structure.base_gap_scale,
                    "heterogeneity": structure.heterogeneity,
                    "latent_boundary_gap": structure.latent_boundary_gap,
                    "empirical_boundary_gap": structure.empirical_boundary_gap,
                    "empirical_maximum_gap": structure.empirical_maximum_gap,
                    "empirical_mean_gap": structure.empirical_mean_gap,
                    "empirical_gap_cv": structure.empirical_gap_cv,
                    "distinct_empirical_gap_count": (structure.distinct_empirical_gap_count),
                },
            )
            yield instance
            # Generator frames retain local variables across yields.  Delete
            # complete fixture-bearing values before constructing the next one.
            del instance
            del design


def _resolved_config(config: Mapping[str, object]) -> dict[str, object]:
    method_configs = config["method_configs"]
    return {
        "schema_version": config["schema_version"],
        "experiment_name": config["experiment_name"],
        "master_seed": config["master_seed"],
        "problem_instances": config["problem_instances"],
        "algorithm_repetitions": config["algorithm_repetitions"],
        "table_size": config["table_size"],
        "public_threshold": config["public_threshold"],
        "rounding_angular_margin": config["rounding_angular_margin"],
        "paired_bootstrap_repetitions": config["paired_bootstrap_repetitions"],
        "paired_bootstrap_confidence_level": config[
            "paired_bootstrap_confidence_level"
        ],
        "paired_holm_alpha": config["paired_holm_alpha"],
        "instance_distribution": dict(config["instance_distribution"]),
        "method_configs": method_configs.as_dict(),
        "cases": list(config["cases"]),
        "notes": list(config["notes"]),
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


def _paired_gap_aided_reference_query_diagnostic(
    runs: Sequence[object],
) -> list[dict[str, object]]:
    panels: dict[tuple[str, str, int], dict[str, object]] = {}
    for run in runs:
        if run.method_id not in {"qgapselect", "independent_iae_topk"}:
            continue
        key = (run.family_id, run.instance_id, run.repetition)
        panels.setdefault(key, {})[run.method_id] = run
    by_family: dict[str, list[tuple[object, object]]] = {}
    for (family_id, _, _), methods in panels.items():
        if set(methods) == {"qgapselect", "independent_iae_topk"}:
            by_family.setdefault(family_id, []).append(
                (methods["qgapselect"], methods["independent_iae_topk"])
            )
    summaries: list[dict[str, object]] = []
    for family_id, pairs in sorted(by_family.items()):
        success_conditioned_pairs = [
            (first, second)
            for first, second in pairs
            if first.certified_exact_recovery and second.certified_exact_recovery
        ]
        ratios = [
            first.coherent_queries / second.coherent_queries
            for first, second in success_conditioned_pairs
            if second.coherent_queries > 0
        ]
        summaries.append(
            {
                "family_id": family_id,
                "paired_instance_count": len(pairs),
                "both_certified_exact_count": len(success_conditioned_pairs),
                "query_ratio_definition": "qgapselect_over_independent_iae_topk",
                "mean_query_ratio_on_both_certified": (
                    statistics.fmean(ratios) if ratios else None
                ),
                "median_query_ratio_on_both_certified": (
                    float(statistics.median(ratios)) if ratios else None
                ),
                "qgapselect_fewer_query_count_on_both_certified": sum(
                    first.coherent_queries < second.coherent_queries
                    for first, second in success_conditioned_pairs
                ),
                "success_conditioned_descriptive_only": True,
                "information_matched": False,
                "resource_advantage_claimed": False,
            }
        )
    return summaries


def _difficulty_fingerprint_diagnostic(
    instances: Sequence[Mapping[str, object]],
    *,
    expected_instances_per_family: int,
    paper_scale_minimum_unique_fraction: float,
) -> dict[str, object]:
    """Audit actual difficulty diversity, independently of arm assignment.

    A permutation-only campaign has a single invariant fingerprint even when
    every fixture manifest differs.  Paper-scale cells additionally have to
    satisfy the preregistered minimum unique-fingerprint fraction.
    """

    grouped: dict[str, list[str]] = {}
    for index, document in enumerate(instances):
        family_id = _string(document.get("family_id"), f"instances[{index}].family_id")
        fingerprint = _string(
            document.get("difficulty_fingerprint"),
            f"instances[{index}].difficulty_fingerprint",
        )
        try:
            valid_fingerprint = len(fingerprint) == 64 and int(fingerprint, 16) >= 0
        except ValueError:
            valid_fingerprint = False
        if not valid_fingerprint:
            raise RuntimeError("difficulty fingerprints must be SHA-256 digests")
        grouped.setdefault(family_id, []).append(fingerprint)

    rows: list[dict[str, object]] = []
    for family_id, fingerprints in sorted(grouped.items()):
        instance_count = len(fingerprints)
        if instance_count != expected_instances_per_family:
            raise RuntimeError(
                f"family {family_id!r} has {instance_count} instances; "
                f"expected {expected_instances_per_family}"
            )
        unique_count = len(set(fingerprints))
        unique_fraction = unique_count / instance_count
        permutation_only = instance_count > 1 and unique_count == 1
        paper_scale = instance_count >= 500
        required_unique_count = (
            math.ceil(instance_count * paper_scale_minimum_unique_fraction) if paper_scale else None
        )
        paper_scale_gate_passed = not paper_scale or unique_count >= int(required_unique_count or 0)
        if permutation_only:
            raise RuntimeError(
                f"family {family_id!r} repeats one difficulty under arm permutations"
            )
        if not paper_scale_gate_passed:
            raise RuntimeError(
                f"family {family_id!r} has only {unique_count}/{instance_count} "
                "unique permutation-invariant difficulty fingerprints"
            )
        rows.append(
            {
                "family_id": family_id,
                "instance_count": instance_count,
                "unique_difficulty_fingerprint_count": unique_count,
                "duplicate_difficulty_fingerprint_count": (instance_count - unique_count),
                "unique_difficulty_fingerprint_fraction": unique_fraction,
                "permutation_only_repetition_detected": permutation_only,
                "paper_scale": paper_scale,
                "paper_scale_required_unique_count": required_unique_count,
                "paper_scale_nonisomorphic_gate_passed": paper_scale_gate_passed,
            }
        )
    return {
        "fingerprint_is_permutation_invariant": True,
        "family_count": len(rows),
        "families": rows,
        "permutation_only_repetition_detected": any(
            bool(row["permutation_only_repetition_detected"]) for row in rows
        ),
        "paper_scale_minimum_unique_fraction": (paper_scale_minimum_unique_fraction),
        "paper_scale_nonisomorphic_gate_passed": all(
            bool(row["paper_scale_nonisomorphic_gate_passed"]) for row in rows
        ),
    }


def run_experiment(
    config: Mapping[str, object],
    *,
    instance_chunk_size: int = 1,
) -> dict[str, object]:
    instances = build_instances(config)
    report = run_frozen_quantum_reference_benchmark(
        instances,
        config["method_configs"],
        repetitions=int(config["algorithm_repetitions"]),
        master_seed=int(config["master_seed"]),
        instance_chunk_size=instance_chunk_size,
    )
    panel_width = int(config["algorithm_repetitions"]) * len(config["method_configs"].method_ids)
    if len(report.runs) % panel_width:
        raise RuntimeError("the streamed run matrix is not panel-complete")
    problem_instance_count = len(report.runs) // panel_width
    if problem_instance_count != len(report.instances):
        raise RuntimeError("each problem instance must have exactly one public document")
    distribution = config["instance_distribution"]
    fingerprint_diagnostic = _difficulty_fingerprint_diagnostic(
        report.instances,
        expected_instances_per_family=int(config["problem_instances"]),
        paper_scale_minimum_unique_fraction=float(
            distribution["paper_scale_minimum_unique_fraction"]
        ),
    )
    paired_analysis = analyze_frozen_quantum_reference_pairs(
        report.runs,
        master_seed=int(config["master_seed"]),
        bootstrap_repetitions=int(config["paired_bootstrap_repetitions"]),
        confidence_level=float(config["paired_bootstrap_confidence_level"]),
        holm_alpha=float(config["paired_holm_alpha"]),
    )
    resolved = _resolved_config(config)
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "experiment_name": config["experiment_name"],
        "claim_status": CLAIM_SCOPE,
        "config_sha256": hashlib.sha256(
            json.dumps(resolved, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "resolved_config": resolved,
        "summary": {
            "problem_instance_count": problem_instance_count,
            "problem_instances_per_case": config["problem_instances"],
            "algorithm_repetitions": config["algorithm_repetitions"],
            "case_count": len(config["cases"]),
            "method_count": len(config["method_configs"].method_ids),
            "run_count": len(report.runs),
            "same_layer_c_oracle": True,
            "same_output_set": True,
            "known_threshold_control_has_stronger_information": True,
            "fully_information_matched_primary_baseline_available": False,
            "independent_problem_instances": True,
            "instance_distribution_generator": distribution["generator"],
            "difficulty_fingerprint_diagnostic": fingerprint_diagnostic,
            "permutation_only_repetition_detected": fingerprint_diagnostic[
                "permutation_only_repetition_detected"
            ],
            "hardware_execution_performed": False,
            "llm_execution_performed": False,
            "quantum_advantage_claimed": False,
            "fixed_confidence_calibration_claimed": False,
            "paired_gap_aided_reference_query_diagnostic": (
                _paired_gap_aided_reference_query_diagnostic(report.runs)
            ),
            "paired_gap_aided_reference_statistics": paired_analysis.as_dict(),
        },
        "report": report.as_dict(),
        "provenance": _git_provenance(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--problem-instances",
        type=int,
        help="override the independent fixture count per case",
    )
    parser.add_argument(
        "--fixture-chunk-size",
        type=int,
        default=1,
        help="maximum complete frozen fixtures retained by the benchmark runner",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.problem_instances is not None:
        config["problem_instances"] = _integer(
            args.problem_instances,
            "--problem-instances",
            minimum=1,
        )
    fixture_chunk_size = _integer(
        args.fixture_chunk_size,
        "--fixture-chunk-size",
        minimum=1,
    )
    artifact = run_experiment(
        config,
        instance_chunk_size=fixture_chunk_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = artifact["summary"]
    sys.stdout.write(
        f"wrote frozen Layer-C reference diagnostic to {args.output}\n"
        f"runs={summary['run_count']} problem_instances={summary['problem_instance_count']}\n"
        "scope=analytic measurement-law references; no hardware, LLM, or advantage claim\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
