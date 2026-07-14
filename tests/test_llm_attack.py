from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from qgapselect.llm_attack import (
    AttackStudyPlan,
    CallableGenerationValidator,
    CallableLocalModelAdapter,
    EvaluatedGeneration,
    FunctionalityState,
    GenerationRecord,
    GenerationStatus,
    OfflineReplayBackend,
    PortfolioEntry,
    PortfolioSelection,
    QGapSelectPortfolioAdapter,
    QueryBudget,
    SecurityState,
    Seed,
    SemanticVariant,
    StaticPortfolioSelector,
    Task,
    ValidatorResult,
    collect_local_records,
    evaluated_generation_to_dict,
    paired_counterfactual_event,
    run_attack_study,
    source_counterfactual_statistics,
)


def _evaluated(
    record_id: str,
    model: str,
    task: str,
    variant: str | None,
    seed: int,
    query_index: int,
    functionality: FunctionalityState,
    security: SecurityState,
) -> EvaluatedGeneration:
    record = GenerationRecord(
        record_id=record_id,
        model_id=model,
        task_id=task,
        variant_id=variant,
        seed=Seed(seed),
        query_index=query_index,
        query_cost=0 if variant is None else 1,
        status=GenerationStatus.COMPLETE,
        output_ref=f"test://{record_id}",
    )
    return EvaluatedGeneration(
        record,
        ValidatorResult(
            record_id=record_id,
            validator_id="validator",
            functionality=functionality,
            security=security,
        ),
    )


def _plan() -> AttackStudyPlan:
    return AttackStudyPlan(
        study_name="unit",
        tasks=(Task("task", "ref://clean", "validator"),),
        variants=(
            SemanticVariant("task", "a", "ref://a", "semantic-a"),
            SemanticVariant("task", "b", "ref://b", "semantic-b"),
        ),
        seeds=(Seed(7),),
        source_model_ids=("source",),
        victim_model_ids=("victim",),
        budgets=(QueryBudget(1), QueryBudget(2)),
        portfolio_size=1,
    )


def _study_records() -> tuple[EvaluatedGeneration, ...]:
    return (
        _evaluated(
            "s-clean", "source", "task", None, 7, 0,
            FunctionalityState.FUNCTIONAL, SecurityState.SAFE,
        ),
        _evaluated(
            "s-a", "source", "task", "a", 7, 1,
            FunctionalityState.FUNCTIONAL, SecurityState.VULNERABLE,
        ),
        _evaluated(
            "s-b", "source", "task", "b", 7, 2,
            FunctionalityState.FUNCTIONAL, SecurityState.SAFE,
        ),
        _evaluated(
            "v-clean", "victim", "task", None, 7, 0,
            FunctionalityState.FUNCTIONAL, SecurityState.SAFE,
        ),
        _evaluated(
            "v-a", "victim", "task", "a", 7, 1,
            FunctionalityState.FUNCTIONAL, SecurityState.VULNERABLE,
        ),
        _evaluated(
            "v-b", "victim", "task", "b", 7, 2,
            FunctionalityState.FUNCTIONAL, SecurityState.VULNERABLE,
        ),
    )


def test_counterfactual_event_requires_the_same_paired_seed() -> None:
    clean = _evaluated(
        "clean",
        "victim",
        "task",
        None,
        7,
        0,
        FunctionalityState.FUNCTIONAL,
        SecurityState.SAFE,
    )
    matching = _evaluated(
        "attack",
        "victim",
        "task",
        "a",
        7,
        1,
        FunctionalityState.FUNCTIONAL,
        SecurityState.VULNERABLE,
    )
    wrong_seed = _evaluated(
        "wrong-seed",
        "victim",
        "task",
        "a",
        8,
        2,
        FunctionalityState.FUNCTIONAL,
        SecurityState.VULNERABLE,
    )

    assert paired_counterfactual_event(clean, matching)
    assert not paired_counterfactual_event(clean, wrong_seed)


def test_preexisting_clean_vulnerability_is_not_induced() -> None:
    clean = _evaluated(
        "clean",
        "victim",
        "task",
        None,
        7,
        0,
        FunctionalityState.FUNCTIONAL,
        SecurityState.VULNERABLE,
    )
    attack = _evaluated(
        "attack",
        "victim",
        "task",
        "a",
        7,
        1,
        FunctionalityState.FUNCTIONAL,
        SecurityState.VULNERABLE,
    )

    assert not paired_counterfactual_event(clean, attack)


def test_source_selection_and_victim_replay_are_disjoint() -> None:
    records = _study_records()
    plan = _plan()
    result = run_attack_study(plan, OfflineReplayBackend(records))

    assert result.selection.variants_for_task("task") == ("a",)
    assert {item.generation.record_id for item in result.source_records} == {
        "s-clean",
        "s-a",
        "s-b",
    }
    assert {item.generation.record_id for item in result.victim_records} == {
        "v-clean",
        "v-a",
    }
    assert not (
        set(result.selection.source_record_ids)
        & {item.generation.record_id for item in result.victim_records}
    )
    assert result.metrics_by_victim_model["victim"][1].task_model_unit_count == 1

    statistics = source_counterfactual_statistics(
        tasks=plan.tasks,
        variants=plan.variants,
        source_records=result.source_records,
        source_model_ids=plan.source_model_ids,
        seeds=plan.seeds,
        selection_budget=QueryBudget(2),
    )
    assert [(item.variant_id, item.empirical_rate) for item in statistics] == [
        ("a", 1.0),
        ("b", 0.0),
    ]


def test_static_selector_rejects_non_source_evidence_reference() -> None:
    plan = _plan()
    selection = PortfolioSelection(
        selector_id="external",
        source_model_ids=("source",),
        entries=(PortfolioEntry("task", "a", 1, 0.5),),
        source_record_ids=("victim-record",),
    )
    backend = OfflineReplayBackend(
        [
            _evaluated(
                "source-record",
                "source",
                "task",
                None,
                7,
                0,
                FunctionalityState.FUNCTIONAL,
                SecurityState.SAFE,
            )
        ]
    )

    with pytest.raises(ValueError, match="outside source"):
        run_attack_study(plan, backend, StaticPortfolioSelector(selection))


def test_offline_jsonl_round_trip(tmp_path: Path) -> None:
    item = _evaluated(
        "record",
        "source",
        "task",
        None,
        7,
        0,
        FunctionalityState.FUNCTIONAL,
        SecurityState.SAFE,
    )
    path = tmp_path / "replay.jsonl"
    path.write_text(json.dumps(evaluated_generation_to_dict(item)) + "\n")

    backend = OfflineReplayBackend.from_jsonl([path])

    assert backend.records == (item,)


def test_callable_local_adapters_collect_replay_ready_records() -> None:
    plan = _plan()

    def generate(request, task, variant):
        del task, variant
        return GenerationRecord(
            record_id=(
                f"{request.model_id}:{request.task_id}:"
                f"{request.variant_id}:{request.seed.value}"
            ),
            model_id=request.model_id,
            task_id=request.task_id,
            variant_id=request.variant_id,
            seed=request.seed,
            query_index=request.query_index,
            query_cost=0 if request.variant_id is None else 1,
            status=GenerationStatus.COMPLETE,
            output_ref="local://opaque-output",
        )

    def validate(task, record):
        del task
        return ValidatorResult(
            record_id=record.record_id,
            validator_id="validator",
            functionality=FunctionalityState.FUNCTIONAL,
            security=SecurityState.SAFE,
        )

    adapters = {
        model_id: CallableLocalModelAdapter(f"adapter:{model_id}", generate)
        for model_id in ("source", "victim")
    }
    validators = {"validator": CallableGenerationValidator("validator", validate)}

    records = collect_local_records(plan, adapters=adapters, validators=validators)

    assert len(records) == 6
    assert {item.generation.model_id for item in records} == {"source", "victim"}
    assert all(item.validation is not None for item in records)


@dataclass(frozen=True)
class _QResult:
    selected: tuple[int, ...]
    accepted_by_intervals: tuple[int, ...]
    unresolved_at_stop: tuple[int, ...] = ()
    interval_resolved: bool = True


def test_qgapselect_result_adapter_preserves_arm_mapping() -> None:
    candidates = _plan().variants
    selection = QGapSelectPortfolioAdapter.from_result(
        task_id="task",
        candidates=candidates,
        result=_QResult((1,), (1,)),
        source_model_ids=("source",),
        source_record_ids=("source-record",),
    )

    assert selection.variants_for_task("task") == ("b",)
    assert selection.proof_status == (
        "complete_certificate_checked_adapter_no_quantum_advantage_claim"
    )


def test_qgapselect_result_adapter_rejects_unresolved_or_uncertified_output() -> None:
    candidates = _plan().variants
    with pytest.raises(ValueError, match="incomplete or unresolved"):
        QGapSelectPortfolioAdapter.from_result(
            task_id="task",
            candidates=candidates,
            result=_QResult((1,), (), (1,), False),
            source_model_ids=("source",),
            source_record_ids=("source-record",),
        )

    with pytest.raises(ValueError, match="complete valid certificate"):
        QGapSelectPortfolioAdapter.from_result(
            task_id="task",
            candidates=candidates,
            result=_QResult((1,), (), (), True),
            source_model_ids=("source",),
            source_record_ids=("source-record",),
        )


def test_portfolio_ranks_must_be_unique_and_contiguous() -> None:
    with pytest.raises(ValueError, match="unique and contiguous"):
        PortfolioSelection(
            selector_id="invalid-ranks",
            source_model_ids=("source",),
            entries=(
                PortfolioEntry("task", "a", 1, 0.7),
                PortfolioEntry("task", "b", 3, 0.6),
            ),
            source_record_ids=("source-record",),
        )


def test_attack_study_requires_exact_portfolio_size_for_every_task() -> None:
    plan = _plan()
    selection = PortfolioSelection(
        selector_id="empty",
        source_model_ids=("source",),
        entries=(),
        source_record_ids=("source-record",),
    )
    backend = OfflineReplayBackend(
        [
            _evaluated(
                "source-record",
                "source",
                "task",
                None,
                7,
                0,
                FunctionalityState.FUNCTIONAL,
                SecurityState.SAFE,
            )
        ]
    )

    with pytest.raises(ValueError, match="exactly portfolio_size"):
        run_attack_study(plan, backend, StaticPortfolioSelector(selection))


def test_attack_study_rejects_wrong_validator_identity() -> None:
    plan = _plan()
    records = list(_study_records())
    source = records[0]
    assert source.validation is not None
    records[0] = EvaluatedGeneration(
        source.generation,
        ValidatorResult(
            record_id=source.generation.record_id,
            validator_id="wrong-validator",
            functionality=source.validation.functionality,
            security=source.validation.security,
        ),
    )

    with pytest.raises(ValueError, match="expected 'validator'"):
        run_attack_study(plan, OfflineReplayBackend(records))


def test_static_selection_must_cite_source_evidence() -> None:
    plan = _plan()
    selection = PortfolioSelection(
        selector_id="no-evidence",
        source_model_ids=("source",),
        entries=(PortfolioEntry("task", "a", 1, 0.0),),
        source_record_ids=(),
    )

    with pytest.raises(ValueError, match="non-empty source evidence"):
        run_attack_study(
            plan,
            OfflineReplayBackend(_study_records()),
            StaticPortfolioSelector(selection),
        )


def test_plan_forbids_source_victim_model_leakage() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        AttackStudyPlan(
            study_name="bad",
            tasks=(Task("task", "ref://clean", "validator"),),
            variants=(SemanticVariant("task", "a", "ref://a", "semantic"),),
            seeds=(Seed(1),),
            source_model_ids=("same",),
            victim_model_ids=("same",),
            budgets=(QueryBudget(1),),
            portfolio_size=1,
        )


def test_cli_runs_safe_state_only_fixture_end_to_end(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "report.json"
    raw_output = tmp_path / "raw.jsonl"
    environment = dict(os.environ)
    source = str(repository / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )

    completed = subprocess.run(
        [
            sys._base_executable,
            str(repository / "scripts" / "run_attack_study.py"),
            "--config",
            str(repository / "configs" / "attack_study.json"),
            "--output",
            str(output),
            "--raw-output",
            str(raw_output),
        ],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    raw = [
        json.loads(line)
        for line in raw_output.read_text(encoding="utf-8").splitlines()
    ]
    assert report["data_status"] == (
        "synthetic_state_only_fixture_not_research_evidence"
    )
    assert report["provenance"]["network_access"] == "not_implemented"
    assert report["provenance"]["contains_builtin_attack_payload"] is False
    assert set(report["metrics_by_victim_model"]) == {"held-out-local-fixture"}
    assert raw == report["raw_records"]
    assert {record["phase"] for record in raw} == {
        "source_selection",
        "held_out_victim_evaluation",
    }
