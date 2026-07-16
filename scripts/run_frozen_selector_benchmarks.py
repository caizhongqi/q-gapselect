#!/usr/bin/env python3
"""Run the preregistered synthetic frozen-oracle selector benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from qgapselect.attack_oracles import CandidateEdge
from qgapselect.frozen_selector_benchmarking import (
    CLAIM_SCOPE,
    DEFAULT_SELECTOR_IDS,
    SelectorBudget,
    SelectorLandscape,
    run_frozen_selector_benchmark,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs" / "frozen_selector_benchmarks.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "frozen_selector_benchmark_diagnostic.json"
ARTIFACT_TYPE = "q_gapselect_frozen_selector_algorithm_diagnostic"
SCHEMA_VERSION = 1

# Fixed permutation prevents candidate order from revealing latent rank.
P32 = (
    17,
    3,
    26,
    8,
    30,
    11,
    0,
    22,
    14,
    6,
    29,
    19,
    1,
    24,
    10,
    31,
    4,
    16,
    27,
    7,
    21,
    13,
    2,
    28,
    9,
    18,
    25,
    5,
    23,
    12,
    20,
    15,
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


def _only_keys(value: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")


def _bidirectional_ring(candidate_ids: Sequence[str]) -> tuple[CandidateEdge, ...]:
    values = tuple(candidate_ids)
    edges: list[CandidateEdge] = []
    for index, candidate_id in enumerate(values):
        successor = values[(index + 1) % len(values)]
        edges.append(CandidateEdge(candidate_id, successor, "ring_neighbor"))
        edges.append(CandidateEdge(successor, candidate_id, "ring_neighbor"))
    return tuple(edges)


def _rank_gap_means(gap: float) -> dict[str, float]:
    rank_means = [0.90, 0.82, 0.74, 0.60 + gap / 2.0, 0.60 - gap / 2.0]
    rank_means.extend(0.46 - 0.008 * (rank - 5) for rank in range(5, 32))
    return {
        f"c{candidate_index:02d}": rank_means[rank]
        for rank, candidate_index in enumerate(P32)
    }


def _rank_costs(profile: str) -> dict[str, float]:
    candidate_ids = tuple(f"c{index:02d}" for index in range(32))
    if profile == "unit":
        return {candidate_id: 1.0 for candidate_id in candidate_ids}
    if profile == "independent":
        levels = (0.25, 0.5, 1.0, 2.0, 4.0)
        return {
            f"c{index:02d}": levels[(13 * index + 2) % len(levels)]
            for index in range(32)
        }
    if profile == "boundary_expensive":
        costs: dict[str, float] = {}
        for rank, candidate_index in enumerate(P32):
            if rank in {3, 4}:
                cost = 8.0
            elif rank in {2, 5}:
                cost = 4.0
            elif rank in {1, 6}:
                cost = 2.0
            else:
                cost = 1.0
            costs[f"c{candidate_index:02d}"] = cost
        return costs
    raise ValueError(f"unknown cost_profile {profile!r}")


def _rank_gap_landscape(case: Mapping[str, object]) -> SelectorLandscape:
    name = _string(case.get("name"), "case.name")
    gap = _number(case.get("gap"), f"{name}.gap")
    if not 0.0 < gap <= 0.5:
        raise ValueError(f"{name}.gap must lie in (0, 0.5]")
    profile = _string(case.get("cost_profile", "unit"), f"{name}.cost_profile")
    jitter = _number(case.get("cost_jitter", 0.0), f"{name}.cost_jitter")
    if jitter >= 1.0:
        raise ValueError(f"{name}.cost_jitter must be less than 1")
    candidate_ids = tuple(f"c{index:02d}" for index in range(32))
    return SelectorLandscape.from_means(
        name,
        _rank_gap_means(gap),
        candidate_costs=_rank_costs(profile),
        cost_jitter=jitter,
        edges=_bidirectional_ring(candidate_ids),
    )


def _smooth_grid_landscape(name: str) -> SelectorLandscape:
    means: dict[str, float] = {}
    edges: list[CandidateEdge] = []
    for x in range(6):
        for y in range(6):
            candidate_id = f"v{x}{y}"
            distance = abs(x - 1) + abs(y - 1)
            # The small deterministic tilt removes rank ties without revealing
            # membership through candidate order.
            means[candidate_id] = 0.15 + 0.10 * max(0, 5 - distance) + 0.0002 * (
                35 - (6 * x + y)
            )
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 6 and 0 <= ny < 6:
                    edges.append(CandidateEdge(candidate_id, f"v{nx}{ny}", "grid_neighbor"))
    return SelectorLandscape.from_means(
        name,
        means,
        candidate_costs=1.0,
        edges=tuple(edges),
    )


def _disconnected_multipeak_landscape(name: str) -> SelectorLandscape:
    a_ids = tuple(f"A{index:02d}" for index in range(16))
    b_ids = tuple(f"B{index:02d}" for index in range(16))
    means: dict[str, float] = {}
    for rank in range(16):
        a_id = f"A{(5 * rank + 3) % 16:02d}"
        b_id = f"B{(7 * rank + 1) % 16:02d}"
        means[a_id] = (0.82, 0.78)[rank] if rank < 2 else 0.38 - 0.01 * (rank - 2)
        means[b_id] = (0.80, 0.76)[rank] if rank < 2 else 0.37 - 0.01 * (rank - 2)
    return SelectorLandscape.from_means(
        name,
        means,
        candidate_costs=1.0,
        edges=(*_bidirectional_ring(a_ids), *_bidirectional_ring(b_ids)),
    )


def _landscape(case: Mapping[str, object], index: int) -> SelectorLandscape:
    _only_keys(
        case,
        {"name", "family", "gap", "cost_profile", "cost_jitter"},
        f"cases[{index}]",
    )
    family = _string(case.get("family"), f"cases[{index}].family")
    if family == "rank_gap_ring":
        return _rank_gap_landscape(case)
    name = _string(case.get("name"), f"cases[{index}].name")
    unexpected = set(case) - {"name", "family"}
    if unexpected:
        raise ValueError(f"{family} does not accept fields {sorted(unexpected)}")
    if family == "smooth_grid":
        return _smooth_grid_landscape(name)
    if family == "disconnected_multipeak":
        return _disconnected_multipeak_landscape(name)
    raise ValueError(f"unknown landscape family {family!r}")


def load_config(path: Path) -> dict[str, object]:
    root = _mapping(json.loads(path.read_text(encoding="utf-8")), "root")
    _only_keys(
        root,
        {
            "schema_version",
            "experiment_name",
            "master_seed",
            "trials",
            "k",
            "samples_per_candidate",
            "selector_ids",
            "budgets",
            "cases",
            "notes",
        },
        "root",
    )
    if _integer(root.get("schema_version"), "schema_version", minimum=1) != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    selectors = tuple(
        _string(value, f"selector_ids[{index}]")
        for index, value in enumerate(_sequence(root.get("selector_ids"), "selector_ids"))
    )
    if not selectors or len(set(selectors)) != len(selectors):
        raise ValueError("selector_ids must be non-empty and unique")
    unknown = set(selectors) - set(DEFAULT_SELECTOR_IDS)
    if unknown:
        raise ValueError(f"unknown selector IDs: {sorted(unknown)}")
    budgets: list[SelectorBudget] = []
    for index, raw_budget in enumerate(_sequence(root.get("budgets"), "budgets")):
        budget = _mapping(raw_budget, f"budgets[{index}]")
        _only_keys(budget, {"budget_id", "max_queries", "max_cost"}, f"budgets[{index}]")
        budgets.append(
            SelectorBudget(
                _string(budget.get("budget_id"), f"budgets[{index}].budget_id"),
                _integer(budget.get("max_queries"), f"budgets[{index}].max_queries"),
                _number(budget.get("max_cost"), f"budgets[{index}].max_cost"),
            )
        )
    if not budgets:
        raise ValueError("budgets cannot be empty")
    cases = tuple(
        _landscape(_mapping(raw_case, f"cases[{index}]"), index)
        for index, raw_case in enumerate(_sequence(root.get("cases"), "cases"))
    )
    if not cases:
        raise ValueError("cases cannot be empty")
    notes = tuple(
        _string(value, f"notes[{index}]")
        for index, value in enumerate(_sequence(root.get("notes", ()), "notes"))
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": _string(root.get("experiment_name"), "experiment_name"),
        "master_seed": _integer(root.get("master_seed"), "master_seed"),
        "trials": _integer(root.get("trials"), "trials", minimum=1),
        "k": _integer(root.get("k"), "k", minimum=1),
        "samples_per_candidate": _integer(
            root.get("samples_per_candidate"), "samples_per_candidate", minimum=1
        ),
        "selector_ids": selectors,
        "budgets": tuple(budgets),
        "landscapes": cases,
        "case_specs": tuple(dict(_mapping(value, "case")) for value in root["cases"]),
        "notes": notes,
    }


def _config_document(config: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": config["schema_version"],
        "experiment_name": config["experiment_name"],
        "master_seed": config["master_seed"],
        "trials": config["trials"],
        "k": config["k"],
        "samples_per_candidate": config["samples_per_candidate"],
        "selector_ids": list(config["selector_ids"]),
        "budgets": [budget.as_dict() for budget in config["budgets"]],
        "cases": list(config["case_specs"]),
        "notes": list(config["notes"]),
    }


def _leader_rows(aggregates: Sequence[object]) -> list[dict[str, object]]:
    panels: dict[tuple[str, str], list[object]] = {}
    for row in aggregates:
        panels.setdefault((row.landscape_id, row.budget_id), []).append(row)
    leaders: list[dict[str, object]] = []
    for (landscape_id, budget_id), rows in sorted(panels.items()):
        best = max(row.exact_match_rate for row in rows)
        leaders.append(
            {
                "landscape_id": landscape_id,
                "budget_id": budget_id,
                "best_exact_match_rate": best,
                "leader_selector_ids": sorted(
                    row.selector_id
                    for row in rows
                    if math.isclose(row.exact_match_rate, best, abs_tol=1e-15)
                ),
            }
        )
    return leaders


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


def run_experiment(config: Mapping[str, object]) -> dict[str, object]:
    report = run_frozen_selector_benchmark(
        landscapes=config["landscapes"],
        budgets=config["budgets"],
        trials=int(config["trials"]),
        k=int(config["k"]),
        samples_per_candidate=int(config["samples_per_candidate"]),
        master_seed=int(config["master_seed"]),
        selector_ids=config["selector_ids"],
    )
    runs = report.runs
    resolved = _config_document(config)
    config_sha = hashlib.sha256(
        json.dumps(resolved, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "claim_status": CLAIM_SCOPE,
        "experiment_name": config["experiment_name"],
        "config_sha256": config_sha,
        "resolved_config": resolved,
        "summary": {
            "landscape_count": len(config["landscapes"]),
            "budget_count": len(config["budgets"]),
            "selector_count": len(config["selector_ids"]),
            "trial_count": config["trials"],
            "run_count": len(runs),
            "aggregate_count": len(report.aggregates),
            "all_runs_within_query_budget": all(
                run.queries_used <= run.query_budget for run in runs
            ),
            "all_runs_within_cost_budget": all(
                run.cost_used <= run.cost_budget + 1e-12 for run in runs
            ),
            "random_selector_zero_oracle_queries": all(
                run.queries_used == 0 for run in runs if run.selector_id == "random"
            ),
            "leader_rows": _leader_rows(report.aggregates),
            "empirical_superiority_claimed": False,
            "quantum_advantage_claimed": False,
            "llm_execution_performed": False,
        },
        "report": report.as_dict(),
        "claim_boundaries": [
            "Synthetic frozen Bernoulli reward/cost streams only.",
            "Selectors are classical baselines; no LLM or commercial API is executed.",
            "No hardware quantum execution or asymptotic quantum advantage is claimed.",
            "Finite diagnostic repetitions do not replace the preregistered paper campaign.",
        ],
        "provenance": _git_provenance(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials", type=int, help="override the diagnostic repetition count")
    parser.add_argument("--seed", type=int, help="override the master seed")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.trials is not None:
        config["trials"] = _integer(args.trials, "--trials", minimum=1)
    if args.seed is not None:
        config["master_seed"] = _integer(args.seed, "--seed")
    artifact = run_experiment(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    summary = artifact["summary"]
    sys.stdout.write(
        f"wrote frozen selector diagnostic to {args.output}\n"
        f"runs={summary['run_count']} landscapes={summary['landscape_count']} "
        f"budgets={summary['budget_count']} trials={summary['trial_count']}\n"
        "scope=synthetic algorithm experiment; no LLM execution and no quantum-advantage claim\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
