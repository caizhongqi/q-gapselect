#!/usr/bin/env python3
"""Run an authorization-scoped LLM attack study from offline replay records."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qgapselect.llm_attack import (
    SCHEMA_VERSION,
    AttackStudyPlan,
    EvaluatedGeneration,
    FunctionalityState,
    GenerationRecord,
    GenerationStatus,
    OfflineReplayBackend,
    QueryBudget,
    SecurityState,
    Seed,
    SemanticVariant,
    Task,
    ValidatorResult,
    evaluated_generation_to_dict,
    run_attack_study,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs" / "attack_study.json"
DEFAULT_OUTPUT = REPOSITORY / "artifacts" / "attack_study_results.json"
DEFAULT_RAW_OUTPUT = REPOSITORY / "artifacts" / "attack_study_raw.jsonl"
CLAIM_STATUS = "offline_pipeline_output_not_evidence_of_attack_superiority"
FIXTURE_STATUS = "synthetic_state_only_fixture_not_research_evidence"

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "study_name",
    "master_seed",
    "seeds",
    "source_models",
    "victim_models",
    "budgets",
    "portfolio_size",
    "tasks",
    "variants",
    "replay_files",
    "use_safe_example_fixture",
    "authorization",
    "notes",
}
_TASK_FIELDS = {"task_id", "clean_prompt_ref", "validator_id", "metadata"}
_VARIANT_FIELDS = {
    "task_id",
    "variant_id",
    "prompt_ref",
    "transformation",
    "metadata",
}


def _canonical_json(document: object) -> str:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _reject_unknown(
    document: Mapping[str, object], allowed: set[str], context: str
) -> None:
    unknown = set(document) - allowed
    if unknown:
        raise ValueError(f"unknown {context} fields: {', '.join(sorted(unknown))}")


def resolve_plan(document: Mapping[str, Any]) -> AttackStudyPlan:
    """Validate an attack-study config into strong runtime models."""

    _reject_unknown(document, _TOP_LEVEL_FIELDS, "top-level")
    authorization = _require_mapping(
        document.get("authorization"), "authorization"
    )
    if _require_bool(
        authorization.get("external_services"),
        "authorization.external_services",
    ):
        raise ValueError("external services are outside this offline CLI")
    if _require_bool(
        authorization.get("deploy_or_exploit"),
        "authorization.deploy_or_exploit",
    ):
        raise ValueError("deployment or exploitation is outside this CLI")
    replay_files = _require_list(document.get("replay_files", []), "replay_files")
    for index, value in enumerate(replay_files):
        _require_str(value, f"replay_files[{index}]")
    notes = _require_list(document.get("notes", []), "notes")
    for index, value in enumerate(notes):
        _require_str(value, f"notes[{index}]")
    _require_bool(
        document.get("use_safe_example_fixture", False),
        "use_safe_example_fixture",
    )
    tasks: list[Task] = []
    for index, raw in enumerate(_require_list(document.get("tasks"), "tasks")):
        item = _require_mapping(raw, f"tasks[{index}]")
        _reject_unknown(item, _TASK_FIELDS, f"tasks[{index}]")
        metadata = _require_mapping(item.get("metadata", {}), "task metadata")
        tasks.append(
            Task(
                task_id=_require_str(item["task_id"], "task_id"),
                clean_prompt_ref=_require_str(
                    item["clean_prompt_ref"], "clean_prompt_ref"
                ),
                validator_id=_require_str(item["validator_id"], "validator_id"),
                metadata=metadata,
            )
        )
    variants: list[SemanticVariant] = []
    for index, raw in enumerate(
        _require_list(document.get("variants"), "variants")
    ):
        item = _require_mapping(raw, f"variants[{index}]")
        _reject_unknown(item, _VARIANT_FIELDS, f"variants[{index}]")
        metadata = _require_mapping(item.get("metadata", {}), "variant metadata")
        variants.append(
            SemanticVariant(
                task_id=_require_str(item["task_id"], "task_id"),
                variant_id=_require_str(item["variant_id"], "variant_id"),
                prompt_ref=_require_str(item["prompt_ref"], "prompt_ref"),
                transformation=_require_str(
                    item["transformation"], "transformation"
                ),
                metadata=metadata,
            )
        )
    return AttackStudyPlan(
        schema_version=_require_int(
            document.get("schema_version", SCHEMA_VERSION), "schema_version"
        ),
        study_name=_require_str(document["study_name"], "study_name"),
        master_seed=_require_int(document.get("master_seed", 0), "master_seed"),
        tasks=tuple(tasks),
        variants=tuple(variants),
        seeds=tuple(
            Seed(_require_int(value, "seed"))
            for value in _require_list(document["seeds"], "seeds")
        ),
        source_model_ids=tuple(
            _require_str(value, "source model_id")
            for value in _require_list(document["source_models"], "source_models")
        ),
        victim_model_ids=tuple(
            _require_str(value, "victim model_id")
            for value in _require_list(document["victim_models"], "victim_models")
        ),
        budgets=tuple(
            QueryBudget(_require_int(value, "budget"))
            for value in _require_list(document["budgets"], "budgets")
        ),
        portfolio_size=_require_int(document["portfolio_size"], "portfolio_size"),
    )


def _fixture_evaluation(
    *,
    record_id: str,
    model_id: str,
    task_id: str,
    variant_id: str | None,
    seed: Seed,
    query_index: int,
    functionality: FunctionalityState,
    security: SecurityState,
) -> EvaluatedGeneration:
    query_cost = 0 if variant_id is None else 1
    generation = GenerationRecord(
        record_id=record_id,
        model_id=model_id,
        task_id=task_id,
        variant_id=variant_id,
        seed=seed,
        query_index=query_index,
        query_cost=query_cost,
        status=GenerationStatus.COMPLETE,
        output_ref=f"fixture://{record_id}",
        provenance={"data_status": FIXTURE_STATUS},
    )
    return EvaluatedGeneration(
        generation,
        ValidatorResult(
            record_id=record_id,
            validator_id="fixture-state-validator",
            functionality=functionality,
            security=security,
            details={"data_status": FIXTURE_STATUS, "contains_payload": False},
        ),
    )


def build_safe_example_replay(plan: AttackStudyPlan) -> tuple[EvaluatedGeneration, ...]:
    """Create deterministic state-only rows that exercise the complete pipeline.

    The fixture contains labels and opaque output references only.  It contains
    no prompts, generated code, vulnerability recipe, or service interaction.
    """

    records: list[EvaluatedGeneration] = []
    variants_by_task = {
        task.task_id: tuple(
            variant for variant in plan.variants if variant.task_id == task.task_id
        )
        for task in plan.tasks
    }
    for model_id in (*plan.source_model_ids, *plan.victim_model_ids):
        is_source = model_id in set(plan.source_model_ids)
        for task in plan.tasks:
            query_index = 1
            for seed_index, seed in enumerate(plan.seeds):
                clean_id = f"fixture:{model_id}:{task.task_id}:clean:{seed.value}"
                records.append(
                    _fixture_evaluation(
                        record_id=clean_id,
                        model_id=model_id,
                        task_id=task.task_id,
                        variant_id=None,
                        seed=seed,
                        query_index=0,
                        functionality=FunctionalityState.FUNCTIONAL,
                        security=SecurityState.SAFE,
                    )
                )
                for variant_index, variant in enumerate(
                    variants_by_task[task.task_id]
                ):
                    # The first source variant is selected.  On held-out replay,
                    # it transfers only for the first paired seed.  These are
                    # state labels for pipeline tests, not empirical findings.
                    vulnerable = variant_index == 0 and (is_source or seed_index == 0)
                    attack_id = (
                        f"fixture:{model_id}:{task.task_id}:"
                        f"{variant.variant_id}:{seed.value}"
                    )
                    records.append(
                        _fixture_evaluation(
                            record_id=attack_id,
                            model_id=model_id,
                            task_id=task.task_id,
                            variant_id=variant.variant_id,
                            seed=seed,
                            query_index=query_index,
                            functionality=FunctionalityState.FUNCTIONAL,
                            security=(
                                SecurityState.VULNERABLE
                                if vulnerable
                                else SecurityState.SAFE
                            ),
                        )
                    )
                    query_index += 1
    return tuple(records)


def _selection_dict(result: object) -> dict[str, object]:
    selection = result.selection
    return {
        "selector_id": selection.selector_id,
        "source_model_ids": list(selection.source_model_ids),
        "source_record_ids": list(selection.source_record_ids),
        "proof_status": selection.proof_status,
        "entries": [
            {
                "task_id": entry.task_id,
                "variant_id": entry.variant_id,
                "rank": entry.rank,
                "source_score": entry.source_score,
            }
            for entry in selection.entries
        ],
    }


def _raw_records(result: object) -> list[dict[str, object]]:
    selected = {
        (entry.task_id, entry.variant_id) for entry in result.selection.entries
    }
    records: list[dict[str, object]] = []
    for phase, evidence in (
        ("source_selection", result.source_records),
        ("held_out_victim_evaluation", result.victim_records),
    ):
        for item in evidence:
            document = evaluated_generation_to_dict(item)
            key = (item.generation.task_id, item.generation.variant_id)
            records.append(
                {
                    "phase": phase,
                    "selected_variant": (
                        item.generation.is_clean or key in selected
                    ),
                    **document,
                }
            )
    return records


def _plan_dict(plan: AttackStudyPlan) -> dict[str, object]:
    return {
        "schema_version": plan.schema_version,
        "study_name": plan.study_name,
        "master_seed": plan.master_seed,
        "seeds": [seed.value for seed in plan.seeds],
        "source_models": list(plan.source_model_ids),
        "victim_models": list(plan.victim_model_ids),
        "budgets": [budget.max_queries for budget in plan.budgets],
        "portfolio_size": plan.portfolio_size,
        "tasks": [
            {
                "task_id": task.task_id,
                "clean_prompt_ref": task.clean_prompt_ref,
                "validator_id": task.validator_id,
                "metadata": dict(task.metadata),
            }
            for task in plan.tasks
        ],
        "variants": [
            {
                "task_id": variant.task_id,
                "variant_id": variant.variant_id,
                "prompt_ref": variant.prompt_ref,
                "transformation": variant.transformation,
                "metadata": dict(variant.metadata),
            }
            for variant in plan.variants
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--replay",
        type=Path,
        action="append",
        help="offline replay JSONL; repeat to combine shards",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument(
        "--write-example-replay",
        type=Path,
        help="write and use a safe state-only JSONL fixture",
    )
    return parser


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(_canonical_json(record) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_bytes = args.config.read_bytes()
        document = _require_mapping(json.loads(config_bytes), "config")
        plan = resolve_plan(document)
        config_replay = tuple(
            args.config.parent / _require_str(path, "replay file")
            for path in _require_list(document.get("replay_files", []), "replay_files")
        )
        replay_paths = tuple(args.replay) if args.replay else config_replay
        fixture_requested = _require_bool(
            document.get("use_safe_example_fixture", False),
            "use_safe_example_fixture",
        )
        generated_fixture_path = False

        if args.write_example_replay is not None:
            fixture = build_safe_example_replay(plan)
            _write_jsonl(
                args.write_example_replay,
                [evaluated_generation_to_dict(item) for item in fixture],
            )
            replay_paths = (args.write_example_replay,)
            fixture_requested = False
            generated_fixture_path = True

        if replay_paths:
            backend = OfflineReplayBackend.from_jsonl(replay_paths)
            data_status = (
                FIXTURE_STATUS
                if generated_fixture_path
                else "user_supplied_offline_replay"
            )
        elif fixture_requested:
            backend = OfflineReplayBackend(build_safe_example_replay(plan))
            data_status = FIXTURE_STATUS
        else:
            raise ValueError(
                "no replay input: provide --replay, configure replay_files, or "
                "enable the state-only example fixture"
            )

        result = run_attack_study(plan, backend)
        raw_records = _raw_records(result)
        input_hashes = {
            str(path): _sha256_bytes(path.read_bytes()) for path in replay_paths
        }
        report = {
            "schema_version": SCHEMA_VERSION,
            "study_name": plan.study_name,
            "claim_status": CLAIM_STATUS,
            "data_status": data_status,
            "authorization": document.get("authorization"),
            "resolved_plan": _plan_dict(plan),
            "selection": _selection_dict(result),
            "metrics_by_budget": {
                str(budget): metrics.as_dict()
                for budget, metrics in result.metrics_by_budget.items()
            },
            "metrics_by_victim_model": {
                model_id: {
                    str(budget): metrics.as_dict()
                    for budget, metrics in by_budget.items()
                }
                for model_id, by_budget in result.metrics_by_victim_model.items()
            },
            "raw_records": raw_records,
            "provenance": {
                "config_sha256": _sha256_bytes(config_bytes),
                "resolved_plan_sha256": _sha256_bytes(
                    _canonical_json(_plan_dict(plan)).encode("utf-8")
                ),
                "replay_file_sha256": input_hashes,
                "backend_id": result.backend_id,
                "python_version": sys.version.split()[0],
                "implementation": "qgapselect.offline_attack_study_v1",
                "network_access": "not_implemented",
                "contains_builtin_attack_payload": False,
                "raw_jsonl": str(args.raw_output),
            },
            "notes": list(document.get("notes", [])),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        _write_jsonl(args.raw_output, raw_records)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"attack study configuration error: {error}") from error

    sys.stdout.write(
        f"wrote task-level report to {args.output}\n"
        f"wrote {len(raw_records)} raw evidence rows to {args.raw_output}\n"
        f"claim_status={CLAIM_STATUS}\n"
        f"data_status={data_status}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
