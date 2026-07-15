#!/usr/bin/env python3
"""Run direct unknown-oracle coherent threshold-search experiments."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import operator
import sys
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from qgapselect.coherent import CanonicalRyStatevectorOracle

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs" / "direct_search.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "direct_search_results.json"

ARTIFACT_TYPE = "direct_unknown_oracle_threshold_search_execution"
METHOD = "full_workspace_bbht_qpe_threshold_search"
ORACLE_MODEL = "canonical_ry_unknown_to_search"
CLAIM_STATUS = (
    "exact_state_direct_oracle_execution_no_hardware_or_complexity_advantage_claim"
)


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _number(
    value: object,
    name: str,
    *,
    minimum: float,
    maximum: float,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a JSON number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a JSON number") from error
    lower_ok = result > minimum if strict_minimum else result >= minimum
    if not math.isfinite(result) or not lower_ok or result > maximum:
        left = "(" if strict_minimum else "["
        raise ValueError(f"{name} must lie in {left}{minimum}, {maximum}]")
    return result


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"all keys in {name} must be strings")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _only_keys(value: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")


def _jsonable(value: Any) -> Any:
    """Convert result dataclasses and immutable ledgers to JSON values."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    raise TypeError(f"result contains a non-JSON value of type {type(value).__name__}")


def _attribute(result: object, names: Sequence[str], label: str) -> object:
    for name in names:
        if hasattr(result, name):
            return getattr(result, name)
    expected = ", ".join(names)
    raise AttributeError(f"search result must expose {label} via one of: {expected}")


def _resolved_config(document: object) -> dict[str, object]:
    root = _mapping(document, "configuration")
    _only_keys(
        root,
        {
            "schema_version",
            "experiment_name",
            "master_seed",
            "trials",
            "search",
            "scenarios",
            "notes",
        },
        "top-level configuration",
    )
    schema_version = _integer(root.get("schema_version"), "schema_version", minimum=1)
    if schema_version != 1:
        raise ValueError("schema_version must be 1")
    experiment_name = _string(root.get("experiment_name"), "experiment_name")
    master_seed = _integer(root.get("master_seed"), "master_seed", minimum=0)
    trials = _integer(root.get("trials"), "trials", minimum=1)

    search = _mapping(root.get("search"), "search")
    _only_keys(
        search,
        {
            "phase_qubits",
            "max_attempts_per_output",
            "verification_shots",
            "verification_confidence",
            "max_statevector_dimension",
        },
        "search",
    )
    resolved_search = {
        "phase_qubits": _integer(
            search.get("phase_qubits"), "search.phase_qubits", minimum=1
        ),
        "max_attempts_per_output": _integer(
            search.get("max_attempts_per_output"),
            "search.max_attempts_per_output",
            minimum=1,
        ),
        "verification_shots": _integer(
            search.get("verification_shots"),
            "search.verification_shots",
            minimum=1,
        ),
        "verification_confidence": _number(
            search.get("verification_confidence"),
            "search.verification_confidence",
            minimum=0.0,
            maximum=1.0,
            strict_minimum=True,
        ),
        "max_statevector_dimension": _integer(
            search.get("max_statevector_dimension"),
            "search.max_statevector_dimension",
            minimum=1,
        ),
    }

    raw_scenarios = _sequence(root.get("scenarios"), "scenarios")
    if not raw_scenarios:
        raise ValueError("scenarios cannot be empty")
    scenarios: list[dict[str, object]] = []
    names: set[str] = set()
    for position, raw_scenario in enumerate(raw_scenarios):
        scenario = _mapping(raw_scenario, f"scenarios[{position}]")
        _only_keys(
            scenario,
            {"name", "means", "threshold", "expected_count", "relation", "metadata"},
            f"scenarios[{position}]",
        )
        name = _string(scenario.get("name"), f"scenarios[{position}].name")
        if name in names:
            raise ValueError(f"duplicate scenario name: {name}")
        names.add(name)
        raw_means = _sequence(scenario.get("means"), f"scenarios[{position}].means")
        if not raw_means:
            raise ValueError(f"scenarios[{position}].means cannot be empty")
        means = [
            _number(
                mean,
                f"scenarios[{position}].means[{index}]",
                minimum=0.0,
                maximum=1.0,
            )
            for index, mean in enumerate(raw_means)
        ]
        threshold = _number(
            scenario.get("threshold"),
            f"scenarios[{position}].threshold",
            minimum=0.0,
            maximum=1.0,
        )
        expected_count = _integer(
            scenario.get("expected_count"),
            f"scenarios[{position}].expected_count",
            minimum=1,
        )
        if expected_count > len(means):
            raise ValueError(
                f"scenarios[{position}].expected_count cannot exceed the arm count"
            )
        relation = _string(
            scenario.get("relation"), f"scenarios[{position}].relation"
        )
        if relation not in {"above", "below"}:
            raise ValueError(f"scenarios[{position}].relation must be 'above' or 'below'")
        metadata = _mapping(scenario.get("metadata", {}), f"scenarios[{position}].metadata")
        scenarios.append(
            {
                "name": name,
                "means": means,
                "threshold": threshold,
                "expected_count": expected_count,
                "relation": relation,
                "metadata": _jsonable(metadata),
            }
        )

    raw_notes = _sequence(root.get("notes", []), "notes")
    notes = [_string(note, f"notes[{index}]") for index, note in enumerate(raw_notes)]
    return {
        "schema_version": schema_version,
        "experiment_name": experiment_name,
        "master_seed": master_seed,
        "trials": trials,
        "search": resolved_search,
        "scenarios": scenarios,
        "notes": notes,
    }


def _config_hash(config: Mapping[str, object]) -> str:
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trial_seed(master_seed: int, scenario: str, trial: int) -> int:
    payload = f"{master_seed}:direct-search:{scenario}:{trial}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _run_to_terminal(search: object) -> object:
    """Resume a bounded-per-call search until it reaches a real terminal state.

    ``FullWorkspaceBBHT.run()`` intentionally advances only one configured
    per-output attempt budget.  A multi-output experiment runner must therefore
    resume a paused result instead of recording that intermediate checkpoint as
    the final outcome.
    """

    run = getattr(search, "run", None)
    if not callable(run):
        raise TypeError("search must expose a callable run method")
    while True:
        result = run()
        status = _attribute(result, ("status",), "status")
        if status != "paused_resumable":
            return result


def load_config(
    path: Path,
    *,
    trials_override: int | None = None,
    seed_override: int | None = None,
    scenario_names: Sequence[str] | None = None,
) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        resolved = _resolved_config(json.load(stream))
    if trials_override is not None:
        resolved["trials"] = _integer(trials_override, "trials", minimum=1)
    if seed_override is not None:
        resolved["master_seed"] = _integer(seed_override, "seed", minimum=0)
    if scenario_names is not None:
        requested = tuple(scenario_names)
        available = {scenario["name"] for scenario in resolved["scenarios"]}  # type: ignore[index]
        missing = sorted(set(requested) - available)
        if missing:
            raise ValueError(f"unknown scenarios: {missing}")
        resolved["scenarios"] = [
            scenario
            for scenario in resolved["scenarios"]  # type: ignore[union-attr]
            if scenario["name"] in requested
        ]
    return resolved


def run(config: Mapping[str, object]) -> dict[str, object]:
    # Delayed so configuration inspection remains usable while the search module
    # is developed independently from this experiment entry point.
    from qgapselect.direct_search import FullWorkspaceBBHT

    config_hash = _config_hash(config)
    search = config["search"]
    if not isinstance(search, Mapping):
        raise TypeError("resolved search configuration is invalid")
    records: list[dict[str, object]] = []
    for scenario in config["scenarios"]:  # type: ignore[union-attr]
        if not isinstance(scenario, Mapping):
            raise TypeError("resolved scenario is invalid")
        for trial in range(int(config["trials"])):
            seed = _trial_seed(int(config["master_seed"]), str(scenario["name"]), trial)
            oracle = CanonicalRyStatevectorOracle(scenario["means"], seed=seed)
            search_instance = FullWorkspaceBBHT(
                oracle,
                scenario["threshold"],
                scenario["expected_count"],
                phase_qubits=search["phase_qubits"],
                relation=scenario["relation"],
                seed=seed,
                max_attempts_per_output=search["max_attempts_per_output"],
                verification_shots=search["verification_shots"],
                verification_confidence=search["verification_confidence"],
                max_statevector_dimension=search["max_statevector_dimension"],
            )
            result = _run_to_terminal(search_instance)
            selected = _attribute(
                result,
                ("selected_indices", "found_indices", "indices", "selected"),
                "selected indices",
            )
            status = _attribute(result, ("status",), "status")
            resources = _attribute(result, ("resources",), "resources")
            snapshot = oracle.query_snapshot()
            records.append(
                {
                    "scenario": scenario["name"],
                    "trial": trial,
                    "seed": seed,
                    "method": METHOD,
                    "backend": "numpy_exact_statevector_small_scale",
                    "oracle_model": ORACLE_MODEL,
                    "direct_unknown_oracle_search": True,
                    "selected_indices": _jsonable(selected),
                    "status": _jsonable(status),
                    "runner_reached_terminal": status != "paused_resumable",
                    "executed_resources": _jsonable(resources),
                    "oracle_query_ledger": {
                        "coherent_total": snapshot.coherent_total,
                        "by_kind": _jsonable(snapshot.flat()),
                        "by_tag": _jsonable(snapshot.by_tag),
                    },
                    "result": _jsonable(result),
                    "config_hash": config_hash,
                }
            )
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "method": METHOD,
        "backend": "numpy_exact_statevector_small_scale",
        "oracle_model": ORACLE_MODEL,
        "claim_status": CLAIM_STATUS,
        "config_hash": config_hash,
        "resolved_config": _jsonable(config),
        "raw_execution_records": records,
        "provenance": {
            "means_used_only_to_construct_oracle": True,
            "search_receives_no_marked_index_set": True,
            "query_resources_are_executed_ledger_counts": True,
            "statevector_simulation_is_not_hardware_evidence": True,
        },
    }


def write_report(report: Mapping[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(
            args.config,
            trials_override=args.trials,
            seed_override=args.seed,
            scenario_names=args.scenarios,
        )
        report = run(config)
        output = write_report(report, args.output)
    except (AttributeError, ImportError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"direct-search experiment failed: {error}") from error

    sys.stdout.write(
        f"wrote {len(report['raw_execution_records'])} execution records to {output}\n"
        f"method={METHOD}\n"
        f"claim_status={CLAIM_STATUS}\n"
        "Search receives an unknown canonical reward oracle, not a marked-index set; "
        "this exact-state audit is not hardware or advantage evidence.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
