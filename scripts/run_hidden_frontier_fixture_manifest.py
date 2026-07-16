#!/usr/bin/env python3
"""Freeze and audit the hidden-frontier unknown-boundary fixture manifest."""

from __future__ import annotations

import argparse
import hashlib
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
    F_HIDDEN_FRONTIER,
    F_PUBLIC_PARTITION,
    F_TIE_NC,
    FAMILY_IDS,
    GENERATOR_VERSION,
    FrozenHiddenFrontierFixture,
    algorithmic_fixture_deduplication_key,
    deduplicate_isomorphic_fixtures,
    generate_hidden_frontier_fixture,
)

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "q_gapselect_hidden_frontier_fixture_manifest"
DEFAULT_CONFIG = REPOSITORY / "configs" / "hidden_frontier_fixture_manifest.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "hidden_frontier_fixture_manifest.json"

_TOP_LEVEL_KEYS = {"schema_version", "generator_version", "families", "panels"}
_PANEL_KEYS = {
    "panel_id",
    "n",
    "k",
    "design_gap",
    "fixture_seeds",
    "delta",
    "hard_query_cap",
}
_FORBIDDEN_PUBLIC_KEY_TOKENS = {
    "angle",
    "beta",
    "center",
    "ranking",
    "membership",
    "active",
    "schedule",
    "seed",
    "family",
    "permutation",
    "stopping",
    "radius",
    "radii",
}


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY).as_posix()
    except ValueError:
        return str(resolved)


def _canonical_json(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(document: object) -> str:
    return hashlib.sha256(_canonical_json(document)).hexdigest()


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        resolved = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if resolved < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return resolved


def _finite_real(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number, not bool")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    if positive and resolved <= 0.0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    return value


def _validate_exact_keys(document: Mapping[str, object], expected: set[str], name: str) -> None:
    missing = expected - set(document)
    unknown = set(document) - expected
    if missing or unknown:
        raise ValueError(
            f"{name} fields mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _load_config(config_path: Path) -> tuple[dict[str, object], bytes]:
    config_bytes = config_path.read_bytes()
    document = _mapping(json.loads(config_bytes), "config")
    _validate_exact_keys(document, _TOP_LEVEL_KEYS, "config")
    if _integer(document["schema_version"], "schema_version", minimum=1) != SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SCHEMA_VERSION}")
    if document["generator_version"] != GENERATOR_VERSION:
        raise ValueError("generator_version does not match the imported generator")

    raw_families = _sequence(document["families"], "families")
    if not all(isinstance(value, str) for value in raw_families):
        raise TypeError("families entries must be strings")
    families = tuple(raw_families)
    if len(families) != len(set(families)):
        raise ValueError("families must not contain duplicates")
    if set(families) != set(FAMILY_IDS):
        raise ValueError("families must cover every registered family exactly once")

    raw_panels = _sequence(document["panels"], "panels")
    if not raw_panels:
        raise ValueError("panels cannot be empty")
    panels: list[dict[str, object]] = []
    panel_ids: set[str] = set()
    has_default_scale = False
    for position, raw_panel in enumerate(raw_panels):
        panel = _mapping(raw_panel, f"panels[{position}]")
        _validate_exact_keys(panel, _PANEL_KEYS, f"panels[{position}]")
        panel_id = panel["panel_id"]
        if not isinstance(panel_id, str) or not panel_id.strip():
            raise ValueError(f"panels[{position}].panel_id must be a nonempty string")
        if panel_id in panel_ids:
            raise ValueError("panel_id values must be unique")
        panel_ids.add(panel_id)
        n = _integer(panel["n"], f"panels[{position}].n", minimum=4)
        k = _integer(panel["k"], f"panels[{position}].k", minimum=1)
        if k >= n:
            raise ValueError(f"panels[{position}].k must be smaller than n")
        design_gap = _finite_real(
            panel["design_gap"], f"panels[{position}].design_gap", positive=True
        )
        if design_gap > math.pi / 96.0:
            raise ValueError(f"panels[{position}].design_gap must not exceed pi/96")
        delta = _finite_real(panel["delta"], f"panels[{position}].delta", positive=True)
        if delta >= 1.0:
            raise ValueError(f"panels[{position}].delta must lie in (0, 1)")
        hard_query_cap = _integer(
            panel["hard_query_cap"], f"panels[{position}].hard_query_cap", minimum=1
        )
        raw_seeds = _sequence(panel["fixture_seeds"], f"panels[{position}].fixture_seeds")
        seeds = tuple(
            _integer(value, f"panels[{position}].fixture_seeds", minimum=0) for value in raw_seeds
        )
        if len(seeds) < 2 or len(seeds) != len(set(seeds)):
            raise ValueError("each panel requires at least two distinct fixture seeds")
        has_default_scale |= n == 32 and k == 6
        panels.append(
            {
                "panel_id": panel_id,
                "n": n,
                "k": k,
                "design_gap": design_gap,
                "fixture_seeds": list(seeds),
                "delta": delta,
                "hard_query_cap": hard_query_cap,
            }
        )
    if not has_default_scale:
        raise ValueError("at least one n=32, k=6 panel is required")
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "families": list(families),
        "panels": panels,
    }, config_bytes


def _walk_keys(value: object) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(child))
    elif isinstance(value, list | tuple):
        for child in value:
            keys.extend(_walk_keys(child))
    return keys


def _public_view_is_isolated(view: Mapping[str, object]) -> bool:
    lowered_keys = (key.lower() for key in _walk_keys(view))
    return all(
        not any(token in key for token in _FORBIDDEN_PUBLIC_KEY_TOKENS) for key in lowered_keys
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
        "source_status_capture": "before_fixture_manifest_artifact_write",
    }


def _fixture_key(panel_id: str, family_id: str, seed: int) -> str:
    return f"{panel_id}/{family_id}/seed-{seed}"


def _trusted_summary(panel_id: str, fixture: FrozenHiddenFrontierFixture) -> dict[str, object]:
    replayed = fixture.replay()
    return {
        "fixture_key": _fixture_key(panel_id, fixture.family_id, fixture.fixture_seed),
        "panel_id": panel_id,
        "family_id": fixture.family_id,
        "fixture_seed": fixture.fixture_seed,
        "n": fixture.n,
        "k": fixture.k,
        "fixture_hash": fixture.fixture_hash,
        "orbit_hash": fixture.orbit_hash,
        "interface_id": fixture.interface_id,
        "angle_order_commitment_sha256": _sha256_json([value.hex() for value in fixture.angles]),
        "angle_multiset_commitment_sha256": _sha256_json(
            sorted(value.hex() for value in fixture.angles)
        ),
        "permutation_commitment_sha256": _sha256_json(list(fixture.hidden_permutation)),
        "replay_fixture_hash": replayed.fixture_hash,
        "replay_passed": replayed == fixture,
        "non_unique_output": fixture.non_unique_output,
        "tie_label_scope": "trusted_harness_only",
        "top_k_truth_status": (
            "non_unique_reject_control"
            if fixture.non_unique_output
            else "unique_truth_committed_in_fixture_hash"
        ),
    }


def _pair_audit(
    panel_id: str,
    seed: int,
    hidden: FrozenHiddenFrontierFixture,
    public: FrozenHiddenFrontierFixture,
) -> dict[str, object]:
    checks = {
        "same_ordered_angles": hidden.angles == public.angles,
        "same_angle_multiset": sorted(hidden.angles) == sorted(public.angles),
        "same_hidden_permutation": hidden.hidden_permutation == public.hidden_permutation,
        "same_orbit_hash": hidden.orbit_hash == public.orbit_hash,
        "same_count_profile": hidden.nested_count_profile == public.nested_count_profile,
        "distinct_interface_ids": hidden.interface_id != public.interface_id,
        "hidden_view_has_no_partition": "static_partition" not in hidden.algorithm_view(),
        "public_view_has_static_partition": "static_partition" in public.algorithm_view(),
    }
    return {
        "panel_id": panel_id,
        "fixture_seed": seed,
        "hidden_fixture_hash": hidden.fixture_hash,
        "public_partition_fixture_hash": public.fixture_hash,
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_artifact(config_path: Path) -> dict[str, object]:
    """Generate, replay, isolate, and hash-audit all preregistered fixtures."""

    config, config_bytes = _load_config(config_path)
    families = tuple(config["families"])
    fixtures: list[tuple[str, FrozenHiddenFrontierFixture]] = []
    fixture_lookup: dict[tuple[str, int, str], FrozenHiddenFrontierFixture] = {}
    for panel_value in config["panels"]:
        panel = _mapping(panel_value, "normalized panel")
        panel_id = str(panel["panel_id"])
        for seed_value in panel["fixture_seeds"]:
            seed = int(seed_value)
            for family_id in families:
                fixture = generate_hidden_frontier_fixture(
                    family_id=str(family_id),
                    n=int(panel["n"]),
                    k=int(panel["k"]),
                    design_gap=float(panel["design_gap"]),
                    fixture_seed=seed,
                    delta=float(panel["delta"]),
                    hard_query_cap=int(panel["hard_query_cap"]),
                )
                fixture.validate()
                fixtures.append((panel_id, fixture))
                fixture_lookup[(panel_id, seed, str(family_id))] = fixture

    trusted = [_trusted_summary(panel_id, fixture) for panel_id, fixture in fixtures]
    public_by_interface: dict[str, dict[str, object]] = {}
    for _, fixture in fixtures:
        view = fixture.algorithm_view()
        if not _public_view_is_isolated(view):
            raise RuntimeError("algorithm view failed private-field isolation")
        existing = public_by_interface.setdefault(
            fixture.interface_id,
            {"interface_id": fixture.interface_id, "algorithm_view": view},
        )
        if existing["algorithm_view"] != view:
            raise RuntimeError("one interface_id resolved to inconsistent public views")

    pair_audits: list[dict[str, object]] = []
    tie_audits: list[dict[str, object]] = []
    for panel_value in config["panels"]:
        panel = _mapping(panel_value, "normalized panel")
        panel_id = str(panel["panel_id"])
        for seed_value in panel["fixture_seeds"]:
            seed = int(seed_value)
            hidden = fixture_lookup[(panel_id, seed, F_HIDDEN_FRONTIER)]
            public = fixture_lookup[(panel_id, seed, F_PUBLIC_PARTITION)]
            tie = fixture_lookup[(panel_id, seed, F_TIE_NC)]
            pair_audits.append(_pair_audit(panel_id, seed, hidden, public))
            tie_checks = {
                "same_blind_algorithm_view_as_positive": tie.algorithm_view()
                == hidden.algorithm_view(),
                "same_blind_interface_id_as_positive": tie.interface_id == hidden.interface_id,
                "tie_label_absent_from_algorithm_view": "tie"
                not in json.dumps(tie.algorithm_view(), sort_keys=True).lower(),
                "non_unique_label_is_trusted_true": tie.non_unique_output,
                "top_k_truth_withheld_for_tie": tie.top_k_membership is None,
            }
            tie_audits.append(
                {
                    "panel_id": panel_id,
                    "fixture_seed": seed,
                    "tie_fixture_hash": tie.fixture_hash,
                    "checks": tie_checks,
                    "passed": all(tie_checks.values()),
                }
            )

    fixture_hashes = [fixture.fixture_hash for _, fixture in fixtures]
    orbit_groups: defaultdict[str, list[str]] = defaultdict(list)
    for panel_id, fixture in fixtures:
        orbit_groups[fixture.orbit_hash].append(
            _fixture_key(panel_id, fixture.family_id, fixture.fixture_seed)
        )
    raw_duplicate_orbit_groups = [
        {"orbit_hash": orbit_hash, "fixture_keys": sorted(keys)}
        for orbit_hash, keys in sorted(orbit_groups.items())
        if len(keys) > 1
    ]
    expected_duplicate_groups = {
        frozenset(
            {
                _fixture_key(str(panel["panel_id"]), F_HIDDEN_FRONTIER, int(seed)),
                _fixture_key(str(panel["panel_id"]), F_PUBLIC_PARTITION, int(seed)),
            }
        )
        for panel in config["panels"]
        for seed in panel["fixture_seeds"]
    }
    observed_duplicate_groups = {
        frozenset(group["fixture_keys"]) for group in raw_duplicate_orbit_groups
    }
    fixture_objects = tuple(fixture for _, fixture in fixtures)
    deduplicated = deduplicate_isomorphic_fixtures(fixture_objects)
    algorithmic_keys = tuple(
        algorithmic_fixture_deduplication_key(fixture) for fixture in fixture_objects
    )
    unique_algorithmic_keys = tuple(dict.fromkeys(algorithmic_keys))
    retained_hashes = {fixture.fixture_hash for fixture in deduplicated}
    paired_fixture_hashes = {
        fixture.fixture_hash
        for panel in config["panels"]
        for seed in panel["fixture_seeds"]
        for fixture in (
            fixture_lookup[(str(panel["panel_id"]), int(seed), F_HIDDEN_FRONTIER)],
            fixture_lookup[(str(panel["panel_id"]), int(seed), F_PUBLIC_PARTITION)],
        )
    }
    orbit_checks = {
        "all_fixture_hashes_unique": len(fixture_hashes) == len(set(fixture_hashes)),
        "raw_duplicate_orbits_are_only_hidden_public_pairs": observed_duplicate_groups
        == expected_duplicate_groups,
        "algorithmic_key_count_includes_interface": len(unique_algorithmic_keys)
        == len(fixtures),
        "algorithmic_deduplicator_count_matches_unique_geometry_interface_keys": len(
            deduplicated
        )
        == len(unique_algorithmic_keys),
        "algorithmic_deduplicator_preserves_first_fixture_per_key": tuple(
            algorithmic_fixture_deduplication_key(fixture) for fixture in deduplicated
        )
        == unique_algorithmic_keys,
        "hidden_public_pair_members_are_both_retained": paired_fixture_hashes
        <= retained_hashes,
    }

    aggregate_checks = {
        "registered_family_coverage_complete": set(families) == set(FAMILY_IDS),
        "multiple_fixed_seeds_per_panel": all(
            len(panel["fixture_seeds"]) >= 2 for panel in config["panels"]
        ),
        "n32_k6_default_scale_present": any(
            panel["n"] == 32 and panel["k"] == 6 for panel in config["panels"]
        ),
        "all_fixture_validation_and_replay_passed": all(
            row["replay_passed"] and row["replay_fixture_hash"] == row["fixture_hash"]
            for row in trusted
        ),
        "all_public_algorithm_views_isolated": all(
            _public_view_is_isolated(row["algorithm_view"]) for row in public_by_interface.values()
        ),
        "all_hidden_public_pair_audits_passed": all(row["passed"] for row in pair_audits),
        "all_tie_labels_trusted_only": all(row["passed"] for row in tie_audits),
        "raw_orbit_grouping_and_algorithmic_deduplication_passed": all(
            orbit_checks.values()
        ),
    }
    all_checks_passed = all(aggregate_checks.values())
    if not all_checks_passed:
        failed = [name for name, passed in aggregate_checks.items() if not passed]
        raise RuntimeError(f"hidden-frontier manifest audit failed: {failed}")

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "claim_status": "fixture_isolation_audit_only_not_algorithm_evidence",
        "resolved_config": config,
        "public_algorithm_views": [public_by_interface[key] for key in sorted(public_by_interface)],
        "trusted_fixture_summary": trusted,
        "hidden_public_pair_audits": pair_audits,
        "tie_trusted_only_audits": tie_audits,
        "orbit_hash_audit": {
            "checks": orbit_checks,
            "deduplication_key": ["orbit_hash", "interface_id"],
            "fixture_count": len(fixtures),
            "unique_fixture_hash_count": len(set(fixture_hashes)),
            "raw_unique_orbit_hash_count": len(orbit_groups),
            "unique_algorithmic_deduplication_key_count": len(
                unique_algorithmic_keys
            ),
            "deduplicated_algorithmic_fixture_count": len(deduplicated),
            "raw_duplicate_orbit_group_count": len(raw_duplicate_orbit_groups),
            "raw_duplicate_orbit_groups": raw_duplicate_orbit_groups,
        },
        "aggregate_audit": {
            "checks": aggregate_checks,
            "all_checks_passed": all_checks_passed,
            "panel_count": len(config["panels"]),
            "family_count": len(families),
            "fixture_count": len(fixtures),
            "algorithm_performance_measured": False,
            "theorem_claimed": False,
            "quantum_advantage_claimed": False,
            "ccf_a_claimable": False,
        },
        "claim_boundaries": {
            "supports": [
                "deterministic replay of seven frozen fixture families",
                "blind/public-partition interface isolation auditing",
                "trusted-only tie-control labeling",
                "raw permutation-orbit grouping without conflating information interfaces",
                "algorithmic fixture deduplication by orbit hash and interface ID",
            ],
            "does_not_support": [
                "performance of any classical or quantum algorithm",
                "a coherent unknown-boundary implementation",
                "an upper or lower bound theorem",
                "a quantum advantage or CCF-A publication claim",
            ],
        },
        "provenance": {
            "config_path": _portable_path(config_path),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            **_git_provenance(),
        },
    }


def write_artifact(artifact: Mapping[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
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
    return output_path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = build_artifact(args.config)
    written = write_artifact(artifact, args.output)
    audit = artifact["aggregate_audit"]
    orbit = artifact["orbit_hash_audit"]
    sys.stdout.write(
        f"wrote hidden-frontier fixture manifest to {written}\n"
        f"fixtures={audit['fixture_count']} families={audit['family_count']} "
        f"raw_unique_orbits={orbit['raw_unique_orbit_hash_count']} "
        f"algorithmic_fixtures_after_dedup="
        f"{orbit['deduplicated_algorithmic_fixture_count']} "
        f"audit_passed={str(audit['all_checks_passed']).lower()}\n"
        "This fixture manifest is not a theorem, algorithm result, quantum-advantage "
        "claim, or CCF-A claim.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
