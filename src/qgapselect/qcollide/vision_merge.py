"""Merge independent real-vision components and construct compact summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .vision_common import adaptive_rank_summary, summarize_rows


def _merged_claim_scope(artifacts: Sequence[Mapping[str, object]]) -> dict[str, object]:
    scopes = [artifact["claim_scope"] for artifact in artifacts]
    first = dict(scopes[0])
    datasets = {str(scope["dataset"]) for scope in scopes}
    image_sizes = {int(scope["image_size"]) for scope in scopes}
    class_counts = {int(scope["class_count"]) for scope in scopes}
    if len(datasets) != 1 or len(image_sizes) != 1 or len(class_counts) != 1:
        raise ValueError("component artifacts must share dataset, image size, and class count")
    architectures = sorted(
        {
            str(architecture)
            for scope in scopes
            for architecture in scope["architectures"]
        }
    )
    first["architectures"] = architectures
    return first


def merge_real_vision_artifacts(
    artifacts: Sequence[Mapping[str, object]],
    *,
    packing_target: float,
) -> dict[str, object]:
    """Merge raw independent jobs without averaging pre-averaged rows."""

    if not artifacts:
        raise ValueError("artifacts must be non-empty")
    schema_versions = {artifact["schema_version"] for artifact in artifacts}
    master_seeds = {artifact["master_seed"] for artifact in artifacts}
    artifact_types = {artifact["artifact_type"] for artifact in artifacts}
    if len(schema_versions) != 1 or len(master_seeds) != 1 or len(artifact_types) != 1:
        raise ValueError("component artifacts must share schema, seed, and artifact type")
    rows = [row for artifact in artifacts for row in artifact["rows"]]
    training = [row for artifact in artifacts for row in artifact["training"]]
    spectra = [row for artifact in artifacts for row in artifact["spectra"]]
    summary = summarize_rows(rows)
    first = artifacts[0]
    return {
        "artifact_type": first["artifact_type"],
        "schema_version": first["schema_version"],
        "master_seed": first["master_seed"],
        "claim_scope": _merged_claim_scope(artifacts),
        "component_count": len(artifacts),
        "training": training,
        "spectra": spectra,
        "rows": rows,
        "summary": summary,
        "adaptive_rank_summary": adaptive_rank_summary(
            summary,
            spectra,
            packing_target=packing_target,
        ),
    }


def compact_real_vision_summary(artifact: Mapping[str, object]) -> dict[str, object]:
    return {
        "artifact_type": artifact["artifact_type"],
        "schema_version": artifact["schema_version"],
        "master_seed": artifact["master_seed"],
        "claim_scope": artifact["claim_scope"],
        "training": artifact["training"],
        "summary": artifact["summary"],
        "adaptive_rank_summary": artifact["adaptive_rank_summary"],
    }


__all__ = ["compact_real_vision_summary", "merge_real_vision_artifacts"]
