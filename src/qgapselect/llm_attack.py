"""Offline, authorization-scoped replay pipeline for LLM attack research.

This module deliberately contains no network client, service credential, attack
payload, or exploit executor.  It operates on records produced by an explicitly
authorized local-model experiment and makes the source/held-out-victim split a
checked part of the data model.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

SCHEMA_VERSION = 1
CLEAN_VARIANT_ID = None
_REPLAY_FIELDS = {"generation", "validation"}
_GENERATION_FIELDS = {
    "record_id",
    "model_id",
    "task_id",
    "variant_id",
    "seed",
    "query_index",
    "query_cost",
    "status",
    "output_ref",
    "output_text",
    "provenance",
}
_VALIDATION_FIELDS = {
    "record_id",
    "validator_id",
    "functionality",
    "security",
    "details",
}


def _frozen_mapping(value: Mapping[str, object] | None) -> dict[str, object]:
    return {} if value is None else {str(key): item for key, item in value.items()}


def _require_identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or not value or not value.strip():
        raise ValueError(f"{name} cannot be empty")


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _identifier_value(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or null")
    return value


def _reject_unknown_fields(
    document: Mapping[str, object], allowed: set[str], context: str
) -> None:
    unknown = set(document) - allowed
    if unknown:
        raise ValueError(f"unknown {context} fields: {', '.join(sorted(unknown))}")


@dataclass(frozen=True, slots=True, order=True)
class Seed:
    """A paired generation seed shared by clean and attack generations."""

    value: int

    def __post_init__(self) -> None:
        _require_int(self.value, "seed")
        if self.value < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True, slots=True, order=True)
class QueryBudget:
    """Maximum cumulative attack-query cost for one task/model unit."""

    max_queries: int

    def __post_init__(self) -> None:
        _require_int(self.max_queries, "max_queries")
        if self.max_queries < 0:
            raise ValueError("max_queries cannot be negative")


@dataclass(frozen=True, slots=True)
class Task:
    """An authorized evaluation task identified by opaque prompt references."""

    task_id: str
    clean_prompt_ref: str
    validator_id: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.clean_prompt_ref, "clean_prompt_ref")
        _require_identifier(self.validator_id, "validator_id")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class SemanticVariant:
    """A pre-approved semantic prompt variant, represented by an opaque ref."""

    task_id: str
    variant_id: str
    prompt_ref: str
    transformation: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.variant_id, "variant_id")
        _require_identifier(self.prompt_ref, "prompt_ref")
        _require_identifier(self.transformation, "transformation")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @property
    def key(self) -> tuple[str, str]:
        return self.task_id, self.variant_id


class GenerationStatus(str, Enum):
    """Terminal state of one local generation request."""

    COMPLETE = "complete"
    ERROR = "error"
    TIMEOUT = "timeout"


class FunctionalityState(str, Enum):
    """Outcome of an authorized task functionality validator."""

    FUNCTIONAL = "functional"
    NONFUNCTIONAL = "nonfunctional"
    INDETERMINATE = "indeterminate"


class SecurityState(str, Enum):
    """Outcome of a non-deploying vulnerability validator."""

    SAFE = "safe"
    VULNERABLE = "vulnerable"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Request contract for an authorized local-model adapter."""

    model_id: str
    task_id: str
    variant_id: str | None
    seed: Seed
    query_index: int

    def __post_init__(self) -> None:
        _require_identifier(self.model_id, "model_id")
        _require_identifier(self.task_id, "task_id")
        if self.variant_id is not None:
            _require_identifier(self.variant_id, "variant_id")
        _require_int(self.query_index, "query_index")
        if self.variant_id is None and self.query_index != 0:
            raise ValueError("clean generations must use query_index=0")
        if self.variant_id is not None and self.query_index <= 0:
            raise ValueError("attack generations must use a positive query_index")


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """One generation emitted by a local model or loaded from offline replay.

    ``query_index`` is global within one ``(model_id, task_id)`` attack stream.
    Clean records use index zero and do not consume the attack-query budget.
    """

    record_id: str
    model_id: str
    task_id: str
    variant_id: str | None
    seed: Seed
    query_index: int
    query_cost: int
    status: GenerationStatus
    output_ref: str | None = None
    output_text: str | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.record_id, "record_id")
        _require_identifier(self.model_id, "model_id")
        _require_identifier(self.task_id, "task_id")
        _require_int(self.query_index, "query_index")
        _require_int(self.query_cost, "query_cost")
        if not isinstance(self.status, GenerationStatus):
            raise TypeError("status must be a GenerationStatus")
        if self.variant_id is None:
            if self.query_index != 0:
                raise ValueError("clean generations must use query_index=0")
            if self.query_cost != 0:
                raise ValueError("clean generations must have query_cost=0")
        else:
            _require_identifier(self.variant_id, "variant_id")
            if self.query_index <= 0:
                raise ValueError("attack query_index must be positive")
            if self.query_cost <= 0:
                raise ValueError("attack query_cost must be positive")
        if self.status is GenerationStatus.COMPLETE and not (
            self.output_ref or self.output_text is not None
        ):
            raise ValueError("complete generations need output_ref or output_text")
        if self.output_ref is not None and not isinstance(self.output_ref, str):
            raise TypeError("output_ref must be a string or null")
        if self.output_text is not None and not isinstance(self.output_text, str):
            raise TypeError("output_text must be a string or null")
        object.__setattr__(self, "provenance", _frozen_mapping(self.provenance))

    @property
    def is_clean(self) -> bool:
        return self.variant_id is CLEAN_VARIANT_ID


@dataclass(frozen=True, slots=True)
class ValidatorResult:
    """Functionality and vulnerability labels for exactly one generation."""

    record_id: str
    validator_id: str
    functionality: FunctionalityState
    security: SecurityState
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.record_id, "record_id")
        _require_identifier(self.validator_id, "validator_id")
        if not isinstance(self.functionality, FunctionalityState):
            raise TypeError("functionality must be a FunctionalityState")
        if not isinstance(self.security, SecurityState):
            raise TypeError("security must be a SecurityState")
        object.__setattr__(self, "details", _frozen_mapping(self.details))

    @property
    def functional_and_safe(self) -> bool:
        return (
            self.functionality is FunctionalityState.FUNCTIONAL
            and self.security is SecurityState.SAFE
        )

    @property
    def functional_and_vulnerable(self) -> bool:
        return (
            self.functionality is FunctionalityState.FUNCTIONAL
            and self.security is SecurityState.VULNERABLE
        )

    @property
    def indeterminate(self) -> bool:
        return (
            self.functionality is FunctionalityState.INDETERMINATE
            or self.security is SecurityState.INDETERMINATE
        )


@dataclass(frozen=True, slots=True)
class EvaluatedGeneration:
    """A generation and its optional offline validation result."""

    generation: GenerationRecord
    validation: ValidatorResult | None

    def __post_init__(self) -> None:
        if (
            self.validation is not None
            and self.validation.record_id != self.generation.record_id
        ):
            raise ValueError("validation record_id does not match generation")
        if (
            self.generation.status is not GenerationStatus.COMPLETE
            and self.validation is not None
        ):
            raise ValueError("only complete generations may have validation")


@runtime_checkable
class LocalModelAdapter(Protocol):
    """Extension point for explicitly authorized, local open-weight models."""

    adapter_id: str

    def generate(
        self,
        request: GenerationRequest,
        task: Task,
        variant: SemanticVariant | None,
    ) -> GenerationRecord:
        """Generate locally without contacting an external victim service."""


@runtime_checkable
class GenerationValidator(Protocol):
    """Extension point for sandboxed functionality/security validators."""

    validator_id: str

    def validate(self, task: Task, record: GenerationRecord) -> ValidatorResult:
        """Validate one local output without deploying or exploiting it."""


class CallableLocalModelAdapter:
    """Concrete adapter around a caller-owned local generation callable.

    The package supplies no HTTP client and never resolves ``prompt_ref``.  The
    callable is responsible for looking up authorized local inputs and returning
    a fully accounted :class:`GenerationRecord`.
    """

    def __init__(
        self,
        adapter_id: str,
        generate: Callable[
            [GenerationRequest, Task, SemanticVariant | None], GenerationRecord
        ],
    ) -> None:
        _require_identifier(adapter_id, "adapter_id")
        if not callable(generate):
            raise TypeError("generate must be callable")
        self.adapter_id = adapter_id
        self._generate = generate

    def generate(
        self,
        request: GenerationRequest,
        task: Task,
        variant: SemanticVariant | None,
    ) -> GenerationRecord:
        record = self._generate(request, task, variant)
        if not isinstance(record, GenerationRecord):
            raise TypeError("local generation callable must return GenerationRecord")
        expected = (
            request.model_id,
            request.task_id,
            request.variant_id,
            request.seed,
            request.query_index,
        )
        observed = (
            record.model_id,
            record.task_id,
            record.variant_id,
            record.seed,
            record.query_index,
        )
        if observed != expected:
            raise ValueError("local adapter returned a record for a different request")
        return record


class CallableGenerationValidator:
    """Concrete validator adapter around an offline sandboxed callable."""

    def __init__(
        self,
        validator_id: str,
        validate: Callable[[Task, GenerationRecord], ValidatorResult],
    ) -> None:
        _require_identifier(validator_id, "validator_id")
        if not callable(validate):
            raise TypeError("validate must be callable")
        self.validator_id = validator_id
        self._validate = validate

    def validate(self, task: Task, record: GenerationRecord) -> ValidatorResult:
        result = self._validate(task, record)
        if not isinstance(result, ValidatorResult):
            raise TypeError("validator callable must return ValidatorResult")
        if result.record_id != record.record_id:
            raise ValueError("validator returned a result for a different record")
        if result.validator_id != self.validator_id:
            raise ValueError("validator_id does not match adapter")
        return result


@runtime_checkable
class ReplayBackend(Protocol):
    """Read-only evidence source used by the attack-study engine."""

    backend_id: str

    def records_for(
        self,
        *,
        model_id: str,
        task_id: str,
        seed: Seed | None = None,
    ) -> tuple[EvaluatedGeneration, ...]:
        """Return deterministic replay records matching the requested key."""


class OfflineReplayBackend:
    """Deterministic in-memory/JSONL backend with no network capability."""

    backend_id = "offline_jsonl_replay_v1"

    def __init__(self, records: Iterable[EvaluatedGeneration]) -> None:
        materialized = tuple(records)
        record_ids: set[str] = set()
        attack_indices: set[tuple[str, str, int]] = set()
        clean_keys: set[tuple[str, str, Seed]] = set()
        for evaluated in materialized:
            record = evaluated.generation
            if record.record_id in record_ids:
                raise ValueError(f"duplicate record_id {record.record_id!r}")
            record_ids.add(record.record_id)
            if record.is_clean:
                clean_key = (record.model_id, record.task_id, record.seed)
                if clean_key in clean_keys:
                    raise ValueError(
                        "only one clean record is allowed per model/task/seed"
                    )
                clean_keys.add(clean_key)
            else:
                index_key = (record.model_id, record.task_id, record.query_index)
                if index_key in attack_indices:
                    raise ValueError(
                        "attack query_index must be unique within model/task"
                    )
                attack_indices.add(index_key)
        self._records = tuple(
            sorted(
                materialized,
                key=lambda item: (
                    item.generation.model_id,
                    item.generation.task_id,
                    item.generation.query_index,
                    item.generation.seed.value,
                    item.generation.record_id,
                ),
            )
        )

    @classmethod
    def from_jsonl(cls, paths: Sequence[str | Path]) -> OfflineReplayBackend:
        records: list[EvaluatedGeneration] = []
        for raw_path in paths:
            path = Path(raw_path)
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        document = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"invalid replay JSON at {path}:{line_number}: {error}"
                        ) from error
                    try:
                        records.append(evaluated_generation_from_dict(document))
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError(
                            f"invalid replay record at {path}:{line_number}: {error}"
                        ) from error
        return cls(records)

    @property
    def records(self) -> tuple[EvaluatedGeneration, ...]:
        return self._records

    def records_for(
        self,
        *,
        model_id: str,
        task_id: str,
        seed: Seed | None = None,
    ) -> tuple[EvaluatedGeneration, ...]:
        return tuple(
            record
            for record in self._records
            if record.generation.model_id == model_id
            and record.generation.task_id == task_id
            and (seed is None or record.generation.seed == seed)
        )


@dataclass(frozen=True, slots=True)
class PortfolioEntry:
    """One source-selected semantic variant for a particular task."""

    task_id: str
    variant_id: str
    rank: int
    source_score: float

    def __post_init__(self) -> None:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.variant_id, "variant_id")
        _require_int(self.rank, "rank")
        if isinstance(self.source_score, bool) or not isinstance(
            self.source_score, (int, float)
        ):
            raise TypeError("source_score must be numeric")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 <= self.source_score <= 1.0:
            raise ValueError("source_score must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class PortfolioSelection:
    """Auditable output of source-only portfolio selection."""

    selector_id: str
    source_model_ids: tuple[str, ...]
    entries: tuple[PortfolioEntry, ...]
    source_record_ids: tuple[str, ...]
    proof_status: str = "empirical_source_only_selection"

    def __post_init__(self) -> None:
        _require_identifier(self.selector_id, "selector_id")
        _require_identifier(self.proof_status, "proof_status")
        for model_id in self.source_model_ids:
            _require_identifier(model_id, "source_model_id")
        for record_id in self.source_record_ids:
            _require_identifier(record_id, "source_record_id")
        if len(set(self.source_model_ids)) != len(self.source_model_ids):
            raise ValueError("source_model_ids must be unique")
        if len(set(self.source_record_ids)) != len(self.source_record_ids):
            raise ValueError("source_record_ids must be unique")
        entry_keys = {(entry.task_id, entry.variant_id) for entry in self.entries}
        if len(entry_keys) != len(self.entries):
            raise ValueError("portfolio entries must be unique by task/variant")
        tasks = {entry.task_id for entry in self.entries}
        for task_id in tasks:
            ranks = sorted(
                entry.rank for entry in self.entries if entry.task_id == task_id
            )
            if ranks != list(range(1, len(ranks) + 1)):
                raise ValueError(
                    "portfolio ranks must be unique and contiguous from one "
                    f"within task {task_id!r}"
                )

    def variants_for_task(self, task_id: str) -> tuple[str, ...]:
        return tuple(
            entry.variant_id
            for entry in sorted(self.entries, key=lambda item: item.rank)
            if entry.task_id == task_id
        )


@dataclass(frozen=True, slots=True)
class SourceVariantStatistic:
    """Source-only Bernoulli statistic that can define a selector arm."""

    task_id: str
    variant_id: str
    counterfactual_successes: int
    eligible_paired_seeds: int

    def __post_init__(self) -> None:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.variant_id, "variant_id")
        _require_int(self.counterfactual_successes, "counterfactual_successes")
        _require_int(self.eligible_paired_seeds, "eligible_paired_seeds")
        if not 0 <= self.counterfactual_successes <= self.eligible_paired_seeds:
            raise ValueError("success count must lie within eligible paired seeds")

    @property
    def empirical_rate(self) -> float:
        return (
            self.counterfactual_successes / self.eligible_paired_seeds
            if self.eligible_paired_seeds
            else 0.0
        )


def source_counterfactual_statistics(
    *,
    tasks: Sequence[Task],
    variants: Sequence[SemanticVariant],
    source_records: Sequence[EvaluatedGeneration],
    source_model_ids: Sequence[str],
    seeds: Sequence[Seed],
    selection_budget: QueryBudget,
) -> tuple[SourceVariantStatistic, ...]:
    """Build source-only arm statistics for classical or QGapSelect selection."""

    allowed_models = set(source_model_ids)
    if any(item.generation.model_id not in allowed_models for item in source_records):
        raise ValueError("source statistics received a non-source record")
    indexed: dict[tuple[str, str, Seed], list[EvaluatedGeneration]] = defaultdict(
        list
    )
    for item in source_records:
        record = item.generation
        indexed[(record.model_id, record.task_id, record.seed)].append(item)

    statistics: list[SourceVariantStatistic] = []
    for task in tasks:
        for variant in (item for item in variants if item.task_id == task.task_id):
            successes = 0
            eligible = 0
            for model_id in source_model_ids:
                for seed in seeds:
                    group = indexed.get((model_id, task.task_id, seed), [])
                    clean = next(
                        (item for item in group if item.generation.is_clean), None
                    )
                    if (
                        clean is None
                        or clean.generation.status is not GenerationStatus.COMPLETE
                        or clean.validation is None
                        or not clean.validation.functional_and_safe
                    ):
                        continue
                    eligible += 1
                    attacks = _budget_prefix(
                        (
                            item
                            for item in group
                            if item.generation.variant_id == variant.variant_id
                        ),
                        selection_budget,
                    )
                    if any(
                        paired_counterfactual_event(clean, attack)
                        for attack, _ in attacks
                    ):
                        successes += 1
            statistics.append(
                SourceVariantStatistic(
                    task_id=task.task_id,
                    variant_id=variant.variant_id,
                    counterfactual_successes=successes,
                    eligible_paired_seeds=eligible,
                )
            )
    return tuple(statistics)


@runtime_checkable
class PortfolioSelector(Protocol):
    """Source-only selection interface, including quantum selector adapters."""

    selector_id: str

    def select(
        self,
        *,
        tasks: Sequence[Task],
        variants: Sequence[SemanticVariant],
        source_records: Sequence[EvaluatedGeneration],
        source_model_ids: Sequence[str],
        seeds: Sequence[Seed],
        portfolio_size: int,
        selection_budget: QueryBudget,
    ) -> PortfolioSelection:
        """Select variants without access to held-out victim records."""


class CounterfactualRateSelector:
    """Deterministic classical selector using paired source-only events."""

    selector_id = "paired_counterfactual_rate_v1"

    def select(
        self,
        *,
        tasks: Sequence[Task],
        variants: Sequence[SemanticVariant],
        source_records: Sequence[EvaluatedGeneration],
        source_model_ids: Sequence[str],
        seeds: Sequence[Seed],
        portfolio_size: int,
        selection_budget: QueryBudget,
    ) -> PortfolioSelection:
        if portfolio_size <= 0:
            raise ValueError("portfolio_size must be positive")
        statistics = source_counterfactual_statistics(
            tasks=tasks,
            variants=variants,
            source_records=source_records,
            source_model_ids=source_model_ids,
            seeds=seeds,
            selection_budget=selection_budget,
        )

        entries: list[PortfolioEntry] = []
        for task in tasks:
            scores = [
                (statistic.empirical_rate, statistic.variant_id)
                for statistic in statistics
                if statistic.task_id == task.task_id
            ]

            chosen = sorted(scores, key=lambda item: (-item[0], item[1]))[
                :portfolio_size
            ]
            entries.extend(
                PortfolioEntry(task.task_id, variant_id, rank, score)
                for rank, (score, variant_id) in enumerate(chosen, start=1)
            )

        return PortfolioSelection(
            selector_id=self.selector_id,
            source_model_ids=tuple(source_model_ids),
            entries=tuple(entries),
            source_record_ids=tuple(
                sorted(item.generation.record_id for item in source_records)
            ),
        )


class StaticPortfolioSelector:
    """Adapter for externally computed selections, including QGapSelect output."""

    selector_id = "static_precomputed_portfolio_v1"

    def __init__(self, selection: PortfolioSelection) -> None:
        self.selection = selection

    def select(
        self,
        *,
        tasks: Sequence[Task],
        variants: Sequence[SemanticVariant],
        source_records: Sequence[EvaluatedGeneration],
        source_model_ids: Sequence[str],
        seeds: Sequence[Seed],
        portfolio_size: int,
        selection_budget: QueryBudget,
    ) -> PortfolioSelection:
        del tasks, variants, seeds, portfolio_size, selection_budget
        if tuple(source_model_ids) != self.selection.source_model_ids:
            raise ValueError("precomputed selection source-model split mismatch")
        available = {item.generation.record_id for item in source_records}
        if not set(self.selection.source_record_ids).issubset(available):
            raise ValueError("precomputed selection cites records outside source split")
        return self.selection


class QGapSelectPortfolioAdapter:
    """Convert QGapSelect arm indices to a checked portfolio selection."""

    @staticmethod
    def _result_is_complete(result: object) -> bool:
        """Accept only an explicit successful completion marker.

        The analytic reference result exposes ``interval_resolved`` while the
        exact-state controller exposes ``complete``.  Missing markers are not
        interpreted optimistically.
        """

        markers: list[bool] = []
        for attribute in ("interval_resolved", "complete"):
            if hasattr(result, attribute):
                value = getattr(result, attribute)
                if not isinstance(value, bool):
                    raise TypeError(f"QGapSelect result {attribute} must be boolean")
                markers.append(value)
        return bool(markers) and all(markers)

    @staticmethod
    def _explicit_certificate_is_complete(certificate: object) -> bool:
        if certificate is None:
            return False
        if isinstance(certificate, Mapping):
            markers = {
                name: certificate[name]
                for name in ("complete", "valid", "verified")
                if name in certificate
            }
        elif isinstance(certificate, Sequence) and not isinstance(
            certificate, (str, bytes)
        ):
            return any(
                QGapSelectPortfolioAdapter._explicit_certificate_is_complete(item)
                for item in certificate
            )
        else:
            markers = {
                name: getattr(certificate, name)
                for name in ("complete", "valid", "verified")
                if hasattr(certificate, name)
            }
        if "complete" not in markers:
            return False
        if any(not isinstance(value, bool) for value in markers.values()):
            raise TypeError("QGapSelect certificate status markers must be boolean")
        return all(markers.values())

    @staticmethod
    def _result_has_complete_certificate(result: object) -> bool:
        """Check native certificates or the reference result's interval trace."""

        for attribute in ("certificate", "certificates"):
            if hasattr(result, attribute):
                return QGapSelectPortfolioAdapter._explicit_certificate_is_complete(
                    getattr(result, attribute)
                )

        # GapSelectResult represents its certificate through a fully resolved
        # interval trace rather than a separate certificate object.
        required = ("accepted_by_intervals", "unresolved_at_stop", "selected")
        if all(hasattr(result, attribute) for attribute in required):
            selected = tuple(result.selected)
            accepted = tuple(result.accepted_by_intervals)
            unresolved = tuple(result.unresolved_at_stop)
            return not unresolved and set(selected) == set(accepted)
        return False

    @staticmethod
    def from_result(
        *,
        task_id: str,
        candidates: Sequence[SemanticVariant],
        result: object,
        source_model_ids: Sequence[str],
        source_record_ids: Sequence[str],
    ) -> PortfolioSelection:
        if not QGapSelectPortfolioAdapter._result_is_complete(result):
            raise ValueError("QGapSelect result is incomplete or unresolved")
        if not QGapSelectPortfolioAdapter._result_has_complete_certificate(result):
            raise ValueError("QGapSelect result lacks a complete valid certificate")
        selected = getattr(result, "selected", None)
        if selected is None:
            raise TypeError("QGapSelect result must expose a selected attribute")
        if any(candidate.task_id != task_id for candidate in candidates):
            raise ValueError("all QGapSelect candidates must belong to task_id")
        indices = tuple(_require_int(index, "selected arm index") for index in selected)
        if len(set(indices)) != len(indices) or any(
            index < 0 or index >= len(candidates) for index in indices
        ):
            raise ValueError("QGapSelect selected invalid candidate indices")
        entries = tuple(
            PortfolioEntry(
                task_id=task_id,
                variant_id=candidates[index].variant_id,
                rank=rank,
                source_score=0.0,
            )
            for rank, index in enumerate(indices, start=1)
        )
        return PortfolioSelection(
            selector_id="qgapselect_arm_adapter_v1",
            source_model_ids=tuple(source_model_ids),
            entries=entries,
            source_record_ids=tuple(sorted(source_record_ids)),
            proof_status=(
                "complete_certificate_checked_adapter_no_quantum_advantage_claim"
            ),
        )

    @staticmethod
    def from_results(
        *,
        candidates_by_task: Mapping[str, Sequence[SemanticVariant]],
        results_by_task: Mapping[str, object],
        source_model_ids: Sequence[str],
        source_record_ids: Sequence[str],
    ) -> PortfolioSelection:
        """Adapt one QGapSelect result per task into a complete portfolio."""

        if set(candidates_by_task) != set(results_by_task):
            raise ValueError("candidate and QGapSelect result task keys must match")
        entries: list[PortfolioEntry] = []
        for task_id in sorted(candidates_by_task):
            partial = QGapSelectPortfolioAdapter.from_result(
                task_id=task_id,
                candidates=candidates_by_task[task_id],
                result=results_by_task[task_id],
                source_model_ids=source_model_ids,
                source_record_ids=source_record_ids,
            )
            entries.extend(partial.entries)
        return PortfolioSelection(
            selector_id="qgapselect_arm_adapter_v1",
            source_model_ids=tuple(source_model_ids),
            entries=tuple(entries),
            source_record_ids=tuple(sorted(source_record_ids)),
            proof_status=(
                "complete_certificate_checked_adapter_no_quantum_advantage_claim"
            ),
        )


@dataclass(frozen=True, slots=True)
class AttackStudyPlan:
    """Resolved source-selection and held-out-victim study plan."""

    study_name: str
    tasks: tuple[Task, ...]
    variants: tuple[SemanticVariant, ...]
    seeds: tuple[Seed, ...]
    source_model_ids: tuple[str, ...]
    victim_model_ids: tuple[str, ...]
    budgets: tuple[QueryBudget, ...]
    portfolio_size: int
    master_seed: int = 0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_int(self.schema_version, "schema_version")
        _require_int(self.master_seed, "master_seed")
        _require_int(self.portfolio_size, "portfolio_size")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version}")
        _require_identifier(self.study_name, "study_name")
        if self.master_seed < 0:
            raise ValueError("master_seed must be non-negative")
        if not self.tasks or not self.variants:
            raise ValueError("study requires tasks and semantic variants")
        if not self.seeds:
            raise ValueError("study requires at least one seed")
        if not self.source_model_ids or not self.victim_model_ids:
            raise ValueError("source and held-out victim models are required")
        for model_id in (*self.source_model_ids, *self.victim_model_ids):
            _require_identifier(model_id, "model_id")
        if set(self.source_model_ids) & set(self.victim_model_ids):
            raise ValueError("source and held-out victim model IDs must be disjoint")
        if len(set(self.source_model_ids)) != len(self.source_model_ids):
            raise ValueError("source_model_ids must be unique")
        if len(set(self.victim_model_ids)) != len(self.victim_model_ids):
            raise ValueError("victim_model_ids must be unique")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("budgets must be sorted and unique")
        if not self.budgets:
            raise ValueError("at least one query budget is required")
        if self.portfolio_size <= 0:
            raise ValueError("portfolio_size must be positive")
        task_ids = [task.task_id for task in self.tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("task_id values must be unique")
        task_set = set(task_ids)
        variant_keys = [variant.key for variant in self.variants]
        if len(set(variant_keys)) != len(variant_keys):
            raise ValueError("semantic variants must be unique within each task")
        if any(variant.task_id not in task_set for variant in self.variants):
            raise ValueError("semantic variant refers to an unknown task")
        for task_id in task_set:
            count = sum(variant.task_id == task_id for variant in self.variants)
            if count < self.portfolio_size:
                raise ValueError(
                    f"task {task_id!r} has fewer variants than portfolio_size"
                )


@dataclass(frozen=True, slots=True)
class AttackStudyResult:
    """Source selection, held-out evidence, and budget-indexed metrics."""

    plan: AttackStudyPlan
    selection: PortfolioSelection
    source_records: tuple[EvaluatedGeneration, ...]
    victim_records: tuple[EvaluatedGeneration, ...]
    metrics_by_budget: Mapping[int, object]
    metrics_by_victim_model: Mapping[str, Mapping[int, object]]
    backend_id: str


def collect_local_records(
    plan: AttackStudyPlan,
    *,
    adapters: Mapping[str, LocalModelAdapter],
    validators: Mapping[str, GenerationValidator],
) -> tuple[EvaluatedGeneration, ...]:
    """Execute a fully specified local study and return replay-ready records.

    This collection function is intentionally separate from held-out evaluation:
    callers first persist the returned rows, then run :func:`run_attack_study` on
    the resulting immutable replay.  Attack ``query_index`` values are assigned
    globally within each model/task stream so every ``@Q`` value is a true prefix.
    """

    requested_models = (*plan.source_model_ids, *plan.victim_model_ids)
    missing_adapters = set(requested_models) - set(adapters)
    if missing_adapters:
        raise ValueError(
            "missing local adapters: " + ", ".join(sorted(missing_adapters))
        )
    missing_validators = {task.validator_id for task in plan.tasks} - set(validators)
    if missing_validators:
        raise ValueError(
            "missing validators: " + ", ".join(sorted(missing_validators))
        )

    variants_by_task: dict[str, tuple[SemanticVariant, ...]] = {
        task.task_id: tuple(
            variant for variant in plan.variants if variant.task_id == task.task_id
        )
        for task in plan.tasks
    }
    collected: list[EvaluatedGeneration] = []
    for model_id in requested_models:
        adapter = adapters[model_id]
        for task in plan.tasks:
            validator = validators[task.validator_id]
            query_index = 1
            for seed in plan.seeds:
                clean_request = GenerationRequest(
                    model_id=model_id,
                    task_id=task.task_id,
                    variant_id=None,
                    seed=seed,
                    query_index=0,
                )
                clean = adapter.generate(clean_request, task, None)
                clean_validation = (
                    validator.validate(task, clean)
                    if clean.status is GenerationStatus.COMPLETE
                    else None
                )
                collected.append(EvaluatedGeneration(clean, clean_validation))
                for variant in variants_by_task[task.task_id]:
                    request = GenerationRequest(
                        model_id=model_id,
                        task_id=task.task_id,
                        variant_id=variant.variant_id,
                        seed=seed,
                        query_index=query_index,
                    )
                    generation = adapter.generate(request, task, variant)
                    validation = (
                        validator.validate(task, generation)
                        if generation.status is GenerationStatus.COMPLETE
                        else None
                    )
                    collected.append(EvaluatedGeneration(generation, validation))
                    query_index += 1
    # Constructor validation catches duplicate IDs and malformed query streams.
    return OfflineReplayBackend(collected).records


def paired_counterfactual_event(
    clean: EvaluatedGeneration,
    attack: EvaluatedGeneration,
) -> bool:
    """Return the strict functional-safe -> functional-vulnerable event."""

    clean_record = clean.generation
    attack_record = attack.generation
    if not clean_record.is_clean or attack_record.is_clean:
        return False
    if (
        clean_record.model_id,
        clean_record.task_id,
        clean_record.seed,
    ) != (
        attack_record.model_id,
        attack_record.task_id,
        attack_record.seed,
    ):
        return False
    return bool(
        clean_record.status is GenerationStatus.COMPLETE
        and attack_record.status is GenerationStatus.COMPLETE
        and clean.validation is not None
        and attack.validation is not None
        and clean.validation.functional_and_safe
        and attack.validation.functional_and_vulnerable
    )


def _budget_prefix(
    records: Iterable[EvaluatedGeneration],
    budget: QueryBudget,
) -> tuple[tuple[EvaluatedGeneration, int], ...]:
    cumulative = 0
    included: list[tuple[EvaluatedGeneration, int]] = []
    for item in sorted(records, key=lambda value: value.generation.query_index):
        cost = item.generation.query_cost
        if cumulative + cost > budget.max_queries:
            break
        cumulative += cost
        included.append((item, cumulative))
    return tuple(included)


def _load_split_records(
    backend: ReplayBackend,
    *,
    model_ids: Sequence[str],
    tasks: Sequence[Task],
    seeds: Sequence[Seed],
) -> tuple[EvaluatedGeneration, ...]:
    allowed_seeds = set(seeds)
    loaded: list[EvaluatedGeneration] = []
    for model_id in model_ids:
        for task in tasks:
            records = backend.records_for(model_id=model_id, task_id=task.task_id)
            for item in records:
                record = item.generation
                if record.model_id != model_id or record.task_id != task.task_id:
                    raise RuntimeError("replay backend returned a mismatched lookup key")
                if record.seed in allowed_seeds:
                    if (
                        item.validation is not None
                        and item.validation.validator_id != task.validator_id
                    ):
                        raise ValueError(
                            f"record {record.record_id!r} uses validator "
                            f"{item.validation.validator_id!r}; expected "
                            f"{task.validator_id!r}"
                        )
                    loaded.append(item)
    record_ids = [item.generation.record_id for item in loaded]
    if len(set(record_ids)) != len(record_ids):
        raise RuntimeError("replay backend returned duplicate records within a split")
    return tuple(loaded)


def run_attack_study(
    plan: AttackStudyPlan,
    backend: ReplayBackend,
    selector: PortfolioSelector | None = None,
) -> AttackStudyResult:
    """Select on source replay and evaluate only afterward on held-out replay."""

    source_records = _load_split_records(
        backend,
        model_ids=plan.source_model_ids,
        tasks=plan.tasks,
        seeds=plan.seeds,
    )
    actual_source_models = {item.generation.model_id for item in source_records}
    if not actual_source_models.issubset(set(plan.source_model_ids)):
        raise RuntimeError("backend returned a held-out record to source selection")
    expected_source_units = {
        (model_id, task.task_id)
        for model_id in plan.source_model_ids
        for task in plan.tasks
    }
    actual_source_units = {
        (item.generation.model_id, item.generation.task_id)
        for item in source_records
    }
    missing_source_units = expected_source_units - actual_source_units
    if missing_source_units:
        rendered = ", ".join(
            f"{model_id}/{task_id}"
            for model_id, task_id in sorted(missing_source_units)
        )
        raise ValueError("source replay is missing preregistered units: " + rendered)

    active_selector = selector if selector is not None else CounterfactualRateSelector()
    selection = active_selector.select(
        tasks=plan.tasks,
        variants=plan.variants,
        source_records=source_records,
        source_model_ids=plan.source_model_ids,
        seeds=plan.seeds,
        portfolio_size=plan.portfolio_size,
        selection_budget=plan.budgets[-1],
    )
    if selection.source_model_ids != plan.source_model_ids:
        raise ValueError("selector returned a mismatched source-model split")
    source_record_ids = {item.generation.record_id for item in source_records}
    if not selection.source_record_ids:
        raise ValueError("selection must cite non-empty source evidence")
    if not set(selection.source_record_ids).issubset(source_record_ids):
        raise ValueError("selection cites evidence outside the source split")

    known_variants = {variant.key for variant in plan.variants}
    if any(
        (entry.task_id, entry.variant_id) not in known_variants
        for entry in selection.entries
    ):
        raise ValueError("selector returned a variant outside the study plan")
    selected_by_task = {
        task.task_id: selection.variants_for_task(task.task_id)
        for task in plan.tasks
    }
    for task in plan.tasks:
        selected = selected_by_task[task.task_id]
        if len(selected) != plan.portfolio_size:
            raise ValueError(
                f"selector must return exactly portfolio_size entries for "
                f"task {task.task_id!r}"
            )
        ranks = sorted(
            entry.rank
            for entry in selection.entries
            if entry.task_id == task.task_id
        )
        if ranks != list(range(1, plan.portfolio_size + 1)):
            raise ValueError(
                f"selector ranks for task {task.task_id!r} must be exactly "
                "1..portfolio_size"
            )

    selected_sets = {
        task_id: set(variant_ids)
        for task_id, variant_ids in selected_by_task.items()
    }

    # Held-out records are intentionally requested only after selection is fixed.
    all_victim_records = _load_split_records(
        backend,
        model_ids=plan.victim_model_ids,
        tasks=plan.tasks,
        seeds=plan.seeds,
    )
    victim_records = tuple(
        item
        for item in all_victim_records
        if (
            item.generation.is_clean
            or item.generation.variant_id
            in selected_sets[item.generation.task_id]
        )
    )
    victim_record_ids = {item.generation.record_id for item in victim_records}
    if source_record_ids & victim_record_ids:
        raise RuntimeError("source and held-out victim evidence overlap")
    if any(
        item.generation.model_id not in set(plan.victim_model_ids)
        for item in victim_records
    ):
        raise RuntimeError("backend returned a source record to victim evaluation")

    from .attack_metrics import aggregate_attack_metrics

    expected_units = tuple(
        (model_id, task.task_id)
        for model_id in plan.victim_model_ids
        for task in plan.tasks
    )
    metrics = {
        budget.max_queries: aggregate_attack_metrics(
            victim_records,
            budget=budget,
            expected_task_model_units=expected_units,
            selected_variants=selected_by_task,
        )
        for budget in plan.budgets
    }
    metrics_by_model = {
        model_id: {
            budget.max_queries: aggregate_attack_metrics(
                tuple(
                    item
                    for item in victim_records
                    if item.generation.model_id == model_id
                ),
                budget=budget,
                expected_task_model_units=tuple(
                    (model_id, task.task_id) for task in plan.tasks
                ),
                selected_variants=selected_by_task,
            )
            for budget in plan.budgets
        }
        for model_id in plan.victim_model_ids
    }
    return AttackStudyResult(
        plan=plan,
        selection=selection,
        source_records=source_records,
        victim_records=victim_records,
        metrics_by_budget=metrics,
        metrics_by_victim_model=metrics_by_model,
        backend_id=backend.backend_id,
    )


def generation_record_to_dict(record: GenerationRecord) -> dict[str, object]:
    return {
        "record_id": record.record_id,
        "model_id": record.model_id,
        "task_id": record.task_id,
        "variant_id": record.variant_id,
        "seed": record.seed.value,
        "query_index": record.query_index,
        "query_cost": record.query_cost,
        "status": record.status.value,
        "output_ref": record.output_ref,
        "output_text": record.output_text,
        "provenance": dict(record.provenance),
    }


def validator_result_to_dict(result: ValidatorResult) -> dict[str, object]:
    return {
        "record_id": result.record_id,
        "validator_id": result.validator_id,
        "functionality": result.functionality.value,
        "security": result.security.value,
        "details": dict(result.details),
    }


def evaluated_generation_to_dict(item: EvaluatedGeneration) -> dict[str, object]:
    return {
        "generation": generation_record_to_dict(item.generation),
        "validation": (
            None
            if item.validation is None
            else validator_result_to_dict(item.validation)
        ),
    }


def evaluated_generation_from_dict(document: Mapping[str, Any]) -> EvaluatedGeneration:
    """Strictly parse one replay JSONL object."""

    if not isinstance(document, Mapping):
        raise TypeError("replay line must be a JSON object")
    _reject_unknown_fields(document, _REPLAY_FIELDS, "replay")
    generation = document.get("generation")
    if not isinstance(generation, Mapping):
        raise TypeError("generation must be a JSON object")
    _reject_unknown_fields(generation, _GENERATION_FIELDS, "generation")
    record = GenerationRecord(
        record_id=_identifier_value(generation["record_id"], "record_id"),
        model_id=_identifier_value(generation["model_id"], "model_id"),
        task_id=_identifier_value(generation["task_id"], "task_id"),
        variant_id=(
            None
            if generation.get("variant_id") is None
            else _identifier_value(generation["variant_id"], "variant_id")
        ),
        seed=Seed(_require_int(generation["seed"], "seed")),
        query_index=_require_int(generation["query_index"], "query_index"),
        query_cost=_require_int(generation["query_cost"], "query_cost"),
        status=GenerationStatus(
            _identifier_value(generation["status"], "generation status")
        ),
        output_ref=_optional_string(generation.get("output_ref"), "output_ref"),
        output_text=_optional_string(generation.get("output_text"), "output_text"),
        provenance=_mapping_field(generation.get("provenance"), "provenance"),
    )
    raw_validation = document.get("validation")
    validation = None
    if raw_validation is not None:
        if not isinstance(raw_validation, Mapping):
            raise TypeError("validation must be null or a JSON object")
        _reject_unknown_fields(raw_validation, _VALIDATION_FIELDS, "validation")
        validation = ValidatorResult(
            record_id=_identifier_value(raw_validation["record_id"], "record_id"),
            validator_id=_identifier_value(
                raw_validation["validator_id"], "validator_id"
            ),
            functionality=FunctionalityState(
                _identifier_value(raw_validation["functionality"], "functionality")
            ),
            security=SecurityState(
                _identifier_value(raw_validation["security"], "security")
            ),
            details=_mapping_field(raw_validation.get("details"), "details"),
        )
    return EvaluatedGeneration(record, validation)


def _mapping_field(value: object, name: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value
