"""Merge independent real-vision components and construct compact summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .vision_common import adaptive_rank_summary, summarize_rows


def merge_real_vision_artifacts(
    artifacts: Sequence[Mapping[str, object]],
    *,
    packing_target: float,
) -> dict[str, object]:
    """Merge raw independent jobs without averaging pre-averaged rows."""

    if not artifacts:
        raise ValueError("artifacts must be non-empty")
    rows = [row for artifact in artifacts for row in artifact["rows"]]
    training = [row for artifact in artifacts for row in artifact["training"]]
    spectra = [row for artifact in artifacts for row in artifact["spectra"]]
    summary = summarize_rows(rows)
    first = artifacts[0]
    return {
        "artifact_type": "qcollide_real_digits_cnn_transformer_causal_diagnostic",
        "schema_version": first["schema_version"],
        "master_seed": first["master_seed"],
        "claim_scope": first["claim_scope"],
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
