"""F4 geometry-to-packing scaling panel."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .fixtures import geometry_to_packing
from .graph import packing_statistics
from .scaling_common import (
    derived_seed,
    float_values,
    int_values,
    number,
    positive_int,
)


def run_f4(config: Mapping[str, object], master_seed: int) -> dict[str, object]:
    dimensions = int_values(config.get("manifold_dimensions"), "f4.manifold_dimensions")
    rank_fractions = float_values(
        config.get("control_rank_fractions"),
        "f4.control_rank_fractions",
    )
    openness_values = float_values(config.get("openness_values"), "f4.openness_values")
    curvature_values = float_values(config.get("curvature_values"), "f4.curvature_values")
    anchors = positive_int(config.get("anchors"), "f4.anchors")
    repetitions = positive_int(config.get("repetitions"), "f4.repetitions")
    control_epsilon = number(config.get("control_epsilon"), "f4.control_epsilon")
    payload_delta = number(config.get("payload_delta"), "f4.payload_delta")
    behavior_gamma = number(config.get("behavior_gamma"), "f4.behavior_gamma")
    local_radius = number(config.get("local_radius"), "f4.local_radius")
    rows: list[dict[str, object]] = []
    for dimension in dimensions:
        for rank_fraction in rank_fractions:
            rank = max(0, min(dimension, int(round(rank_fraction * dimension))))
            for openness in openness_values:
                for curvature in curvature_values:
                    for repetition in range(repetitions):
                        instance = geometry_to_packing(
                            manifold_dimension=dimension,
                            control_rank=rank,
                            openness=openness,
                            anchors=anchors,
                            control_epsilon=control_epsilon,
                            payload_delta=payload_delta,
                            behavior_gamma=behavior_gamma,
                            control_curvature=curvature,
                            behavior_curvature=curvature,
                            payload_curvature=curvature,
                            local_radius=local_radius,
                            seed=derived_seed(
                                master_seed,
                                "F4",
                                dimension,
                                rank,
                                openness,
                                curvature,
                                repetition,
                            ),
                        )
                        stats = packing_statistics(instance)
                        certificates = instance.metadata["certificates"]
                        margins = np.asarray(
                            [float(certificate["margin"]) for certificate in certificates],
                            dtype=float,
                        )
                        finite_margins = margins[np.isfinite(margins)]
                        mean_margin = (
                            float(finite_margins.mean()) if finite_margins.size else None
                        )
                        certified = np.asarray(
                            [bool(certificate["certified"]) for certificate in certificates],
                            dtype=bool,
                        )
                        rows.append(
                            {
                                "manifold_dimension": dimension,
                                "control_rank": rank,
                                "control_rank_fraction": rank / dimension,
                                "tunnel_dimension": dimension - rank,
                                "openness": openness,
                                "curvature": curvature,
                                "anchors": anchors,
                                "certified_fraction": float(certified.mean()),
                                "mean_certificate_margin": mean_margin,
                                "finite_margin_fraction": float(np.mean(np.isfinite(margins))),
                                "positive_margin_fraction": float(np.mean(margins >= 0.0)),
                                "edge_count": stats.edge_count,
                                "matching_size": stats.matching_size,
                                "packing_fraction": stats.matching_size / anchors,
                                "fingerprint": instance.metadata["fingerprint"],
                            }
                        )
    return {
        "fixture": "F4",
        "semantics": "local geometry certificate to independent packing",
        "rows": rows,
    }
