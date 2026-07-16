#!/usr/bin/env python3
"""Run the same-interface, hard-query-cap S3 composition controls."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import math
import operator
import platform
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from qgapselect.hidden_frontier_fixtures import (  # noqa: E402
    F_PUBLIC_PARTITION,
    FAMILY_IDS,
    generate_hidden_frontier_fixture,
)
from qgapselect.oracles import CanonicalBernoulliOracleSimulator  # noqa: E402
from qgapselect.strong_composition_registry import (  # noqa: E402
    REQUIRED_BASELINE_IDS,
)
from qgapselect.strong_composition_s3 import (  # noqa: E402
    CLAIM_SCOPE,
    FIDELITY_STATUS,
    INFORMATION_REGIME,
    FixedPrecisionGlobalTopKBAI,
    PublicCapUnknownTimeSearchComposition,
    RepeatedFixedPrecisionPhaseBAI,
    S3AttemptScore,
    S3ExecutionConfig,
    aggregate_s3_attempts,
    score_s3_attempt,
)

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "q_gapselect_same_interface_strong_composition_s3"
DEFAULT_CONFIG = REPOSITORY / "configs" / "strong_composition_s3.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "strong_composition_s3.json.gz"
RUNTIME_INPUT_FIELDS = ("n", "k", "delta", "oracle", "atomic_query_cap")
FORBIDDEN_RUNTIME_INPUTS = frozenset(
    {
        "gap",
        "design_gap",
        "family",
        "family_id",
        "truth",
        "expected_top_k",
        "schedule",
        "stop_levels",
        "threshold",
        "partition",
        "fixture_seed",
        "measurement_seed",
    }
)
BLIND_FAMILY_IDS = tuple(
    family_id for family_id in FAMILY_IDS if family_id != F_PUBLIC_PARTITION
)

METHOD_INVENTORY = (
    {
        "method_id": "fixed_precision_global_topk_bai",
        "registry_comparison_targets": [
            "miqae_per_arm_sort",
            "rall_coherent_ae_rounding",
            "wang_qbai_repeated",
        ],
        "implementation_boundary": (
            "fixed controlled-Grover measurement schedule plus a global interval "
            "certificate; no coherent QPE register and no official QBAI reproduction"
        ),
    },
    {
        "method_id": "repeated_fixed_precision_phase_bai",
        "registry_comparison_targets": [
            "rall_coherent_ae_rounding",
            "wang_qbai_repeated",
        ],
        "implementation_boundary": (
            "repeated certified one-best-arm interval composition; no coherent "
            "exclusion circuit and no official QBAI reproduction"
        ),
    },
    {
        "method_id": "public_cap_unknown_time_search_composition",
        "registry_comparison_targets": [
            "vihrovs_unknown_time_vts",
            "jeffery_subroutine_composition",
            "jeffery_loop_composition",
            "low_su_tunable_vtaa",
        ],
        "implementation_boundary": (
            "serial public geometric schedule with data-dependent arm removal; "
            "the L2 field is a proxy and no coherent variable-time search is executed"
        ),
    },
)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object with string keys")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    return value


def _integer(value: object, name: str, *, minimum: int) -> int:
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
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ValueError(
            f"{name} fields mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _unique_integers(value: object, name: str, *, minimum: int, at_least: int) -> tuple[int, ...]:
    result = tuple(_integer(item, name, minimum=minimum) for item in _sequence(value, name))
    if len(result) < at_least or len(result) != len(set(result)):
        raise ValueError(f"{name} needs at least {at_least} distinct values")
    return result


def load_config(path: Path) -> tuple[dict[str, object], bytes]:
    config_bytes = path.read_bytes()
    root = _mapping(json.loads(config_bytes), "config")
    _exact_keys(
        root,
        {
            "schema_version",
            "experiment_name",
            "families",
            "execution_config",
            "panels",
            "claim_boundary",
        },
        "config",
    )
    if _integer(root["schema_version"], "schema_version", minimum=1) != SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SCHEMA_VERSION}")
    experiment_name = root["experiment_name"]
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("experiment_name must be a nonempty string")

    raw_families = _sequence(root["families"], "families")
    if not all(isinstance(value, str) for value in raw_families):
        raise TypeError("families entries must be strings")
    families = tuple(raw_families)
    if len(families) != len(set(families)) or set(families) != set(BLIND_FAMILY_IDS):
        raise ValueError(
            "families must cover every registered blind-interface family exactly once; "
            "F-PUBLIC-PARTITION is a stronger interface and must be excluded"
        )

    execution = _mapping(root["execution_config"], "execution_config")
    _exact_keys(
        execution,
        {
            "phase_powers",
            "fixed_shots_per_power",
            "unknown_time_shots_per_level",
            "grid_points",
        },
        "execution_config",
    )
    resolved_execution = S3ExecutionConfig(
        phase_powers=tuple(
            _integer(value, "phase_powers", minimum=0)
            for value in _sequence(execution["phase_powers"], "phase_powers")
        ),
        fixed_shots_per_power=_integer(
            execution["fixed_shots_per_power"],
            "fixed_shots_per_power",
            minimum=1,
        ),
        unknown_time_shots_per_level=_integer(
            execution["unknown_time_shots_per_level"],
            "unknown_time_shots_per_level",
            minimum=1,
        ),
        grid_points=_integer(execution["grid_points"], "grid_points", minimum=257),
    )

    panels: list[dict[str, object]] = []
    panel_ids: set[str] = set()
    for index, raw_panel in enumerate(_sequence(root["panels"], "panels")):
        panel = _mapping(raw_panel, f"panels[{index}]")
        _exact_keys(
            panel,
            {
                "panel_id",
                "n",
                "k",
                "design_gap",
                "fixture_seeds",
                "measurement_seeds",
                "delta",
                "atomic_query_caps",
            },
            f"panels[{index}]",
        )
        panel_id = panel["panel_id"]
        if not isinstance(panel_id, str) or not panel_id.strip() or panel_id in panel_ids:
            raise ValueError("panel_id values must be nonempty and unique")
        panel_ids.add(panel_id)
        n = _integer(panel["n"], f"panels[{index}].n", minimum=4)
        k = _integer(panel["k"], f"panels[{index}].k", minimum=1)
        if k >= n:
            raise ValueError("panel k must be smaller than n")
        design_gap = _number(panel["design_gap"], f"panels[{index}].design_gap")
        if not 0.0 < design_gap <= math.pi / 96.0:
            raise ValueError("panel design_gap must lie in (0, pi/96]")
        delta = _number(panel["delta"], f"panels[{index}].delta")
        if not 0.0 < delta < 1.0:
            raise ValueError("panel delta must lie in (0, 1)")
        panels.append(
            {
                "panel_id": panel_id,
                "n": n,
                "k": k,
                "design_gap": design_gap,
                "fixture_seeds": list(
                    _unique_integers(
                        panel["fixture_seeds"],
                        f"panels[{index}].fixture_seeds",
                        minimum=0,
                        at_least=2,
                    )
                ),
                "measurement_seeds": list(
                    _unique_integers(
                        panel["measurement_seeds"],
                        f"panels[{index}].measurement_seeds",
                        minimum=0,
                        at_least=2,
                    )
                ),
                "delta": delta,
                "atomic_query_caps": list(
                    _unique_integers(
                        panel["atomic_query_caps"],
                        f"panels[{index}].atomic_query_caps",
                        minimum=1,
                        at_least=1,
                    )
                ),
            }
        )
    if not panels:
        raise ValueError("panels cannot be empty")

    claim_boundary = tuple(_sequence(root["claim_boundary"], "claim_boundary"))
    if not claim_boundary or not all(
        isinstance(value, str) and value.strip() for value in claim_boundary
    ):
        raise ValueError("claim_boundary must contain nonempty strings")
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_name": experiment_name,
            "families": list(families),
            "execution_config": {
                "phase_powers": list(resolved_execution.phase_powers),
                "fixed_shots_per_power": resolved_execution.fixed_shots_per_power,
                "unknown_time_shots_per_level": (resolved_execution.unknown_time_shots_per_level),
                "grid_points": resolved_execution.grid_points,
            },
            "panels": panels,
            "claim_boundary": list(claim_boundary),
        },
        config_bytes,
    )


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
        "source_status_capture": "before_s3_artifact_write",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY).as_posix()
    except ValueError:
        return str(resolved)


def _method_runtime_signature(method: object) -> list[str]:
    return [name for name in inspect.signature(method.run).parameters if name != "self"]


def build_artifact(config_path: Path) -> dict[str, object]:
    resolved, config_bytes = load_config(config_path)
    execution_raw = _mapping(resolved["execution_config"], "execution_config")
    execution = S3ExecutionConfig(
        phase_powers=tuple(execution_raw["phase_powers"]),  # type: ignore[arg-type]
        fixed_shots_per_power=int(execution_raw["fixed_shots_per_power"]),
        unknown_time_shots_per_level=int(execution_raw["unknown_time_shots_per_level"]),
        grid_points=int(execution_raw["grid_points"]),
    )
    methods = (
        FixedPrecisionGlobalTopKBAI(execution),
        RepeatedFixedPrecisionPhaseBAI(execution),
        PublicCapUnknownTimeSearchComposition(execution),
    )
    globally_preregistered_schedule_shared = all(
        method.config is execution for method in methods
    )
    signatures = {method.method_id: _method_runtime_signature(method) for method in methods}
    if any(tuple(fields) != RUNTIME_INPUT_FIELDS for fields in signatures.values()):
        raise RuntimeError("an S3 method runtime signature is not the canonical five fields")
    if any(FORBIDDEN_RUNTIME_INPUTS.intersection(fields) for fields in signatures.values()):
        raise RuntimeError("an S3 method runtime signature contains a forbidden field")
    comparison_targets = {
        target for row in METHOD_INVENTORY for target in row["registry_comparison_targets"]
    }
    if not comparison_targets <= set(REQUIRED_BASELINE_IDS):
        raise RuntimeError("an S3 registry comparison target is not registered")

    attempts: list[dict[str, object]] = []
    grouped: defaultdict[tuple[str, str, int], list[S3AttemptScore]] = defaultdict(list)
    family_grouped: defaultdict[
        tuple[str, str, str, int], list[S3AttemptScore]
    ] = defaultdict(list)
    expected_attempt_count = 0
    for raw_panel in _sequence(resolved["panels"], "panels"):
        panel = _mapping(raw_panel, "panel")
        panel_id = str(panel["panel_id"])
        n = int(panel["n"])
        k = int(panel["k"])
        design_gap = float(panel["design_gap"])
        delta = float(panel["delta"])
        fixture_seeds = tuple(panel["fixture_seeds"])  # type: ignore[arg-type]
        measurement_seeds = tuple(panel["measurement_seeds"])  # type: ignore[arg-type]
        caps = tuple(panel["atomic_query_caps"])  # type: ignore[arg-type]
        expected_attempt_count += (
            len(resolved["families"])
            * len(fixture_seeds)
            * len(measurement_seeds)
            * len(caps)
            * len(methods)
        )
        for family_id in resolved["families"]:  # type: ignore[union-attr]
            for fixture_seed in fixture_seeds:
                for cap in caps:
                    fixture = generate_hidden_frontier_fixture(
                        family_id=str(family_id),
                        n=n,
                        k=k,
                        design_gap=design_gap,
                        fixture_seed=int(fixture_seed),
                        delta=delta,
                        hard_query_cap=int(cap),
                    )
                    # Trusted harness boundary: these amplitudes construct a
                    # completed oracle.  No angle or fixture reference is
                    # retained by, or passed to, an algorithm.
                    means = tuple(math.sin(angle) ** 2 for angle in fixture.angles)
                    for measurement_seed in measurement_seeds:
                        for method in methods:
                            oracle = CanonicalBernoulliOracleSimulator(
                                means, seed=int(measurement_seed)
                            )
                            result = method.run(
                                n=n,
                                k=k,
                                delta=delta,
                                oracle=oracle,
                                atomic_query_cap=int(cap),
                            )
                            score = score_s3_attempt(result, fixture.top_k_membership)
                            grouped[(panel_id, method.method_id, int(cap))].append(score)
                            family_grouped[
                                (
                                    panel_id,
                                    str(family_id),
                                    method.method_id,
                                    int(cap),
                                )
                            ].append(score)
                            attempts.append(
                                {
                                    "attempt_id": (
                                        f"{panel_id}/{family_id}/fixture-{fixture_seed}/"
                                        f"measurement-{measurement_seed}/cap-{cap}/"
                                        f"{method.method_id}"
                                    ),
                                    "trusted_scoring_stratum": {
                                        "panel_id": panel_id,
                                        "family_id": family_id,
                                        "fixture_seed": fixture_seed,
                                        "measurement_seed": measurement_seed,
                                        "strict_instance": (fixture.top_k_membership is not None),
                                        "fixture_hash": fixture.fixture_hash,
                                        "interface_id": fixture.interface_id,
                                    },
                                    "algorithm_runtime_input_audit": {
                                        "fields": list(RUNTIME_INPUT_FIELDS),
                                        "n": n,
                                        "k": k,
                                        "delta": delta,
                                        "atomic_query_cap": cap,
                                        "oracle_handle": "fresh_canonical_bernoulli_oracle",
                                        "private_fixture_object_supplied": False,
                                        "truth_supplied": False,
                                        "gap_supplied": False,
                                        "family_supplied": False,
                                        "stopping_schedule_supplied": False,
                                    },
                                    "result": result.as_dict(),
                                    "trusted_score": score.as_dict(),
                                }
                            )

    aggregates = [
        {
            "panel_id": panel_id,
            **aggregate_s3_attempts(scores).as_dict(),
        }
        for (panel_id, _method_id, _cap), scores in sorted(grouped.items())
    ]
    per_family_aggregates = [
        {
            "panel_id": panel_id,
            "family_id": family_id,
            **aggregate_s3_attempts(scores).as_dict(),
        }
        for (
            panel_id,
            family_id,
            _method_id,
            _cap,
        ), scores in sorted(family_grouped.items())
    ]
    budget_violations = sum(
        not bool(row["trusted_score"]["budget_valid"])  # type: ignore[index]
        for row in attempts
    )
    incorrect_certificates = sum(
        bool(row["trusted_score"]["incorrect_certificate"])  # type: ignore[index]
        for row in attempts
    )
    inconclusive = sum(
        bool(row["trusted_score"]["inconclusive"])  # type: ignore[index]
        for row in attempts
    )
    certified_exact = sum(
        bool(row["trusted_score"]["certified_exact_success"])  # type: ignore[index]
        for row in attempts
    )
    checks = {
        "attempt_count_matches_full_cartesian_design": (len(attempts) == expected_attempt_count),
        "every_attempt_retained_in_denominator": all(
            row["trusted_score"]["included_in_all_attempt_denominator"]  # type: ignore[index]
            for row in attempts
        ),
        "runtime_signatures_are_canonical_five_field_interface": all(
            tuple(fields) == RUNTIME_INPUT_FIELDS for fields in signatures.values()
        ),
        "runtime_audits_withhold_private_fixture_and_truth": all(
            not row["algorithm_runtime_input_audit"][field]  # type: ignore[index]
            for row in attempts
            for field in (
                "private_fixture_object_supplied",
                "truth_supplied",
                "gap_supplied",
                "family_supplied",
                "stopping_schedule_supplied",
            )
        ),
        "all_hard_query_caps_respected": budget_violations == 0,
        "no_incorrect_certificates": incorrect_certificates == 0,
        "every_uncertified_result_has_no_output_mask": all(
            row["result"]["certified"]  # type: ignore[index]
            or (
                row["result"]["output_relation"] == "INCONCLUSIVE"  # type: ignore[index]
                and row["result"]["output_indices"] is None  # type: ignore[index]
                and row["result"]["output_mask"] is None  # type: ignore[index]
            )
            for row in attempts
        ),
        "registry_coverage_remains_fail_closed": all(
            not row["result"]["registry_coverage_activated"]  # type: ignore[index]
            for row in attempts
        ),
        "registry_comparison_targets_are_registered": (
            comparison_targets <= set(REQUIRED_BASELINE_IDS)
        ),
        "stronger_public_partition_interface_excluded": (
            F_PUBLIC_PARTITION not in resolved["families"]
            and all(
                row["trusted_scoring_stratum"]["family_id"] != F_PUBLIC_PARTITION  # type: ignore[index]
                for row in attempts
            )
        ),
        "one_global_preregistered_schedule_shared_by_all_methods": (
            globally_preregistered_schedule_shared
        ),
        "no_hardware_or_advantage_flags": all(
            not row["result"]["hardware_claimable"]  # type: ignore[index]
            and not row["result"]["quantum_advantage_claimable"]  # type: ignore[index]
            for row in attempts
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "experiment_name": resolved["experiment_name"],
        "resolved_config": resolved,
        "canonical_runtime_contract": {
            "fields": list(RUNTIME_INPUT_FIELDS),
            "forbidden_fields": sorted(FORBIDDEN_RUNTIME_INPUTS),
            "information_regime": INFORMATION_REGIME,
            "method_signatures": signatures,
            "execution_schedule_scope": "global_preregistered_not_instance_derived",
            "excluded_stronger_information_families": [F_PUBLIC_PARTITION],
        },
        "method_inventory": [
            {
                **row,
                "fidelity_status": FIDELITY_STATUS,
                "strongest_registry_baseline_covered": False,
            }
            for row in METHOD_INVENTORY
        ],
        "registry_reconciliation": {
            "registered_required_baseline_ids": list(REQUIRED_BASELINE_IDS),
            "comparison_target_ids": sorted(comparison_targets),
            "all_comparison_targets_registered": True,
            "coverage_activated_by_s3_controls": False,
        },
        "attempts": attempts,
        "all_attempt_aggregates": aggregates,
        "per_family_all_attempt_aggregates": per_family_aggregates,
        "aggregate_audit": {
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "attempt_count": len(attempts),
            "expected_attempt_count": expected_attempt_count,
            "certified_exact_count": certified_exact,
            "inconclusive_count": inconclusive,
            "incorrect_certificate_count": incorrect_certificates,
            "budget_violation_count": budget_violations,
            "all_attempt_denominator_used": True,
        },
        "claim_boundary": {
            "claim_scope": CLAIM_SCOPE,
            "fidelity_status": FIDELITY_STATUS,
            "exact_measurement_probability_law_sampled": ("sin^2((2m+1) theta)"),
            "qpe_circuit_executed": False,
            "coherent_variable_time_search_executed": False,
            "official_literature_reproduction": False,
            "candidate_included_in_cer_panel": False,
            "paired_candidate_cer_superiority_verified": False,
            "claim_bearing_sample_size_met": False,
            "strongest_registry_coverage_claimed": False,
            "new_upper_bound_claimed": False,
            "matching_lower_bound_claimed": False,
            "hardware_claimed": False,
            "quantum_advantage_claimed": False,
            "ccf_a_claimable": False,
        },
        "provenance": {
            **_git_provenance(),
            "config_path": _portable_path(config_path),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
    }


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_artifact(artifact: Mapping[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(artifact)
    if output_path.suffix == ".gz":
        with output_path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                stream.write(payload)
    else:
        output_path.write_bytes(payload)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run same-interface strong-composition S3 controls.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = build_artifact(args.config)
    output = write_artifact(artifact, args.output)
    audit = artifact["aggregate_audit"]
    print(
        "S3 controls complete: "
        f"attempts={audit['attempt_count']} "
        f"certified_exact={audit['certified_exact_count']} "
        f"inconclusive={audit['inconclusive_count']} "
        f"incorrect_certificates={audit['incorrect_certificate_count']} "
        f"budget_violations={audit['budget_violation_count']}"
    )
    print(
        "Claim boundary: executable same-interface controls only; no full "
        "literature reproduction, new theorem, quantum advantage, or CCF-A claim."
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
