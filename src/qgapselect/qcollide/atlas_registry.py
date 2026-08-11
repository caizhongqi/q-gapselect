"""Validation and compact summaries for the universal neural collision atlas registry."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

_REQUIRED_SYSTEM_FIELDS = {
    "id",
    "task",
    "dataset",
    "model",
    "family",
    "tier",
    "execution",
}
_REQUIRED_DOMAINS = {
    "vision_classification",
    "language_understanding",
    "audio",
    "graph_learning",
    "time_series",
    "multimodal",
    "guard_llm",
    "policy_control",
}
_ALLOWED_TIERS = {"core", "external"}
_ALLOWED_EXECUTION = {"existing", "phase1", "phase2", "phase3", "phase4"}


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def validate_atlas_registry(registry: Mapping[str, object]) -> dict[str, object]:
    """Validate the frozen experiment registry and return coverage diagnostics."""

    if int(registry.get("schema_version", 0)) != 1:
        raise ValueError("only atlas schema_version=1 is supported")
    domains = list(_sequence(registry.get("domains"), "domains"))
    if not domains:
        raise ValueError("domains must be non-empty")

    domain_names: list[str] = []
    system_ids: set[str] = set()
    task_names: set[str] = set()
    dataset_names: set[str] = set()
    model_names: set[str] = set()
    family_names: set[str] = set()
    tiers: Counter[str] = Counter()
    execution: Counter[str] = Counter()
    domain_sizes: dict[str, int] = {}
    systems: list[Mapping[str, Any]] = []

    for domain_index, raw_domain in enumerate(domains):
        if not isinstance(raw_domain, Mapping):
            raise ValueError(f"domains[{domain_index}] must be a mapping")
        domain = str(raw_domain.get("domain", ""))
        if not domain:
            raise ValueError(f"domains[{domain_index}] lacks domain")
        if domain in domain_names:
            raise ValueError(f"duplicate domain {domain!r}")
        domain_names.append(domain)
        control = str(raw_domain.get("control_interface", ""))
        if not control:
            raise ValueError(f"domain {domain!r} lacks control_interface")
        raw_systems = list(_sequence(raw_domain.get("systems"), f"{domain}.systems"))
        if not raw_systems:
            raise ValueError(f"domain {domain!r} has no systems")
        domain_sizes[domain] = len(raw_systems)
        for system_index, raw_system in enumerate(raw_systems):
            if not isinstance(raw_system, Mapping):
                raise ValueError(f"{domain}.systems[{system_index}] must be a mapping")
            missing = sorted(_REQUIRED_SYSTEM_FIELDS - set(raw_system))
            if missing:
                raise ValueError(
                    f"{domain}.systems[{system_index}] missing fields {missing}"
                )
            system = {key: raw_system[key] for key in raw_system}
            system_id = str(system["id"])
            if system_id in system_ids:
                raise ValueError(f"duplicate system id {system_id!r}")
            system_ids.add(system_id)
            tier = str(system["tier"])
            status = str(system["execution"])
            if tier not in _ALLOWED_TIERS:
                raise ValueError(f"invalid tier {tier!r} for {system_id}")
            if status not in _ALLOWED_EXECUTION:
                raise ValueError(f"invalid execution status {status!r} for {system_id}")
            tiers[tier] += 1
            execution[status] += 1
            task_names.add(str(system["task"]))
            dataset_names.add(str(system["dataset"]))
            model_names.add(str(system["model"]))
            family_names.add(str(system["family"]))
            systems.append(system)

    observed_domains = set(domain_names)
    missing_domains = sorted(_REQUIRED_DOMAINS - observed_domains)
    if missing_domains:
        raise ValueError(f"atlas misses required domains: {missing_domains}")

    core_metrics = tuple(
        str(value)
        for value in _sequence(
            registry.get("core_metrics"),
            "core_metrics",
        )
    )
    if "capacity_fraction" not in core_metrics or "basin_density" not in core_metrics:
        raise ValueError("core_metrics must include capacity_fraction and basin_density")
    cross_cutting = tuple(
        str(value)
        for value in _sequence(
            registry.get("cross_cutting_experiments"),
            "cross_cutting_experiments",
        )
    )
    if len(set(cross_cutting)) != len(cross_cutting):
        raise ValueError("cross_cutting_experiments must be unique")

    return {
        "artifact_type": "qcollide_neural_atlas_registry_audit",
        "schema_version": 1,
        "atlas_name": str(registry.get("atlas_name", "")),
        "domain_count": len(domain_names),
        "system_cell_count": len(systems),
        "unique_task_count": len(task_names),
        "unique_dataset_count": len(dataset_names),
        "unique_model_name_count": len(model_names),
        "architecture_family_count": len(family_names),
        "domain_sizes": dict(sorted(domain_sizes.items())),
        "tier_counts": dict(sorted(tiers.items())),
        "execution_counts": dict(sorted(execution.items())),
        "domains": sorted(domain_names),
        "architecture_families": sorted(family_names),
        "core_metrics": list(core_metrics),
        "cross_cutting_experiment_count": len(cross_cutting),
        "cross_cutting_experiments": list(cross_cutting),
        "gates": {
            "all_required_domains_present": True,
            "unique_system_ids": True,
            "all_systems_have_required_fields": True,
            "at_least_40_system_cells": len(systems) >= 40,
            "at_least_20_model_names": len(model_names) >= 20,
            "at_least_12_datasets": len(dataset_names) >= 12,
            "at_least_12_architecture_families": len(family_names) >= 12,
        },
    }


__all__ = ["validate_atlas_registry"]
