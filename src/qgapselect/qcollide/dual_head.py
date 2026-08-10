"""Trainable synthetic dual-head causal campaign for Q-COLLIDE."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import sqrt
from typing import Any

import numpy as np

from .dual_head_geometry import (
    _calibrate_epsilon,
    _evaluate_projection,
    _generate_attacks,
    _interventions,
)
from .dual_head_model import (
    DualHeadModel,
    _model_seeds,
    _seed,
    train_dual_head_model,
)
from .geometry import tunnel_geometry
from .scaling_common import int_values, number, positive_int


def _summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    ranks = sorted({int(row["visible_rank"]) for row in rows})
    interventions = sorted({str(row["intervention"]) for row in rows})
    for rank in ranks:
        for intervention in interventions:
            cell = [
                row
                for row in rows
                if row["visible_rank"] == rank and row["intervention"] == intervention
            ]
            packing = np.asarray([row["packing_fraction"] for row in cell], dtype=float)
            output.append(
                {
                    "visible_rank": rank,
                    "intervention": intervention,
                    "mean_packing_fraction": float(packing.mean()),
                    "standard_error": float(packing.std(ddof=1) / sqrt(len(packing)))
                    if len(packing) > 1
                    else 0.0,
                    "mean_benign_acceptance": float(
                        np.mean([row["benign_acceptance"] for row in cell])
                    ),
                    "mean_openness": float(np.mean([row["mean_openness"] for row in cell])),
                    "mean_tunnel_dimension": float(
                        np.mean([row["mean_tunnel_dimension"] for row in cell])
                    ),
                    "mean_candidate_fraction": float(
                        np.mean([row["candidate_fraction"] for row in cell])
                    ),
                }
            )
    return output


def run_dual_head_causal_campaign(config: Mapping[str, object]) -> dict[str, object]:
    """Run disjoint calibration/evaluation intervention experiments."""

    schema_version = positive_int(config.get("schema_version"), "schema_version")
    if schema_version != 1:
        raise ValueError("only schema_version=1 is supported")
    master_seed = positive_int(config.get("master_seed"), "master_seed")
    model_seeds = _model_seeds(config.get("model_seeds"))
    latent_dimension = positive_int(config.get("latent_dimension"), "latent_dimension")
    hidden_dimension = positive_int(config.get("hidden_dimension"), "hidden_dimension")
    visible_ranks = int_values(config.get("visible_ranks"), "visible_ranks")
    if latent_dimension not in visible_ranks:
        raise ValueError("visible_ranks must include the full latent dimension")
    if not any(rank < latent_dimension for rank in visible_ranks):
        raise ValueError("visible_ranks must include at least one non-full rank")
    if any(rank > latent_dimension for rank in visible_ranks):
        raise ValueError("visible_ranks must not exceed latent_dimension")
    integer_names = (
        "training_samples",
        "training_steps",
        "batch_size",
        "calibration_anchors",
        "evaluation_anchors",
        "support_samples",
        "benign_repetitions",
        "attack_steps",
        "closure_rank",
    )
    integers = {name: positive_int(config.get(name), name) for name in integer_names}
    numeric_names = (
        "learning_rate",
        "minimum_latent_scale",
        "anchor_bound",
        "latent_bound",
        "benign_radius",
        "benign_quantile",
        "payload_delta",
        "behavior_gamma",
        "local_radius",
    )
    numbers = {name: number(config.get(name), name) for name in numeric_names}

    training_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for model_seed in model_seeds:
        model, diagnostics = train_dual_head_model(
            seed=_seed(master_seed, "model", model_seed),
            latent_dimension=latent_dimension,
            hidden_dimension=hidden_dimension,
            training_samples=integers["training_samples"],
            training_steps=integers["training_steps"],
            batch_size=integers["batch_size"],
            learning_rate=numbers["learning_rate"],
            minimum_latent_scale=numbers["minimum_latent_scale"],
        )
        training_rows.append({"model_seed": model_seed, **diagnostics})
        rng = np.random.default_rng(_seed(master_seed, "anchors", model_seed))
        calibration_anchors = rng.uniform(
            -numbers["anchor_bound"],
            numbers["anchor_bound"],
            size=(integers["calibration_anchors"], latent_dimension),
        )
        evaluation_anchors = rng.uniform(
            -numbers["anchor_bound"],
            numbers["anchor_bound"],
            size=(integers["evaluation_anchors"], latent_dimension),
        )
        support_latent = rng.uniform(
            -1.0,
            1.0,
            size=(integers["support_samples"], latent_dimension),
        )
        hidden_support = model.hidden(support_latent)

        for visible_rank in visible_ranks:
            projection = model.control_head[:, :visible_rank].T
            calibration_standardized, calibration_epsilon, _ = _calibrate_epsilon(
                model,
                calibration_anchors,
                projection,
                hidden_support,
                seed=_seed(master_seed, "calibration", model_seed, visible_rank),
                benign_radius=numbers["benign_radius"],
                benign_repetitions=integers["benign_repetitions"],
                benign_quantile=numbers["benign_quantile"],
            )
            calibration_attacks = _generate_attacks(
                model,
                calibration_anchors,
                visible_rank,
                calibration_standardized,
                calibration_epsilon,
                payload_delta=numbers["payload_delta"],
                behavior_gamma=numbers["behavior_gamma"],
                local_radius=numbers["local_radius"],
                attack_steps=integers["attack_steps"],
                latent_bound=numbers["latent_bound"],
            )
            projections = _interventions(
                model,
                projection,
                calibration_anchors,
                calibration_attacks,
                hidden_support,
                closure_rank=integers["closure_rank"],
                seed=_seed(master_seed, "intervention", model_seed, visible_rank),
            )
            evaluation_standardized, evaluation_epsilon, _ = _calibrate_epsilon(
                model,
                evaluation_anchors,
                projection,
                hidden_support,
                seed=_seed(master_seed, "evaluation", model_seed, visible_rank),
                benign_radius=numbers["benign_radius"],
                benign_repetitions=integers["benign_repetitions"],
                benign_quantile=numbers["benign_quantile"],
            )
            evaluation_attacks = _generate_attacks(
                model,
                evaluation_anchors,
                visible_rank,
                evaluation_standardized,
                evaluation_epsilon,
                payload_delta=numbers["payload_delta"],
                behavior_gamma=numbers["behavior_gamma"],
                local_radius=numbers["local_radius"],
                attack_steps=integers["attack_steps"],
                latent_bound=numbers["latent_bound"],
            )
            geometries = []
            for anchor in evaluation_anchors:
                hidden_jacobian = model.hidden_jacobian(anchor)
                geometries.append(
                    tunnel_geometry(
                        projection @ hidden_jacobian,
                        model.main_head @ hidden_jacobian,
                    )
                )
            common = {
                "model_seed": model_seed,
                "visible_rank": visible_rank,
                "mean_openness": float(np.mean([item.openness for item in geometries])),
                "mean_tunnel_dimension": float(
                    np.mean([item.tunnel_dimension for item in geometries])
                ),
                "calibration_candidate_fraction": float(
                    np.mean([item is not None for item in calibration_attacks])
                ),
                "candidate_fraction": float(
                    np.mean([item is not None for item in evaluation_attacks])
                ),
            }
            for index, (intervention, candidate_projection) in enumerate(
                projections.items()
            ):
                result = _evaluate_projection(
                    model,
                    evaluation_anchors,
                    evaluation_attacks,
                    candidate_projection,
                    hidden_support,
                    intervention=intervention,
                    seed=_seed(
                        master_seed,
                        "projection",
                        model_seed,
                        visible_rank,
                        index,
                    ),
                    benign_radius=numbers["benign_radius"],
                    benign_repetitions=integers["benign_repetitions"],
                    benign_quantile=numbers["benign_quantile"],
                    payload_delta=numbers["payload_delta"],
                    behavior_gamma=numbers["behavior_gamma"],
                )
                rows.append({**common, **result})

    summary = _summary(rows)
    target_nonfull = [
        row
        for row in summary
        if row["intervention"] == "targeted"
        and row["visible_rank"] < latent_dimension
    ]
    full_baseline = [
        row
        for row in summary
        if row["intervention"] == "baseline"
        and row["visible_rank"] == latent_dimension
    ]
    benign_values = np.asarray([row["benign_acceptance"] for row in rows], dtype=float)
    return {
        "artifact_type": "qcollide_trainable_dual_head_causal_diagnostic",
        "schema_version": schema_version,
        "master_seed": master_seed,
        "claim_scope": {
            "trained_neural_network": True,
            "synthetic_task_only": True,
            "pretrained_model_claimed": False,
            "real_world_attack_claimed": False,
            "quantum_values_are": "analytic endpoint-query proxies",
            "coherent_quantum_execution": False,
            "new_lower_bound_claimed": False,
        },
        "config": dict(config),
        "training": training_rows,
        "rows": rows,
        "summary": summary,
        "gates": {
            "maximum_targeted_nonfull_packing": max(
                row["mean_packing_fraction"] for row in target_nonfull
            ),
            "maximum_full_rank_baseline_packing": max(
                row["mean_packing_fraction"] for row in full_baseline
            ),
            "mean_benign_acceptance": float(benign_values.mean()),
            "maximum_benign_acceptance_deviation": float(
                np.max(np.abs(benign_values - numbers["benign_quantile"]))
            ),
        },
    }


def compact_dual_head_summary(artifact: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "artifact_type",
        "schema_version",
        "master_seed",
        "claim_scope",
        "training",
        "summary",
        "gates",
    )
    return {key: artifact[key] for key in keys}


__all__ = [
    "DualHeadModel",
    "compact_dual_head_summary",
    "run_dual_head_causal_campaign",
    "train_dual_head_model",
]
