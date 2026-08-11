"""Matched-rank random-subspace falsification for the text collision atlas."""

from __future__ import annotations

from collections.abc import Mapping
from math import sqrt
from typing import Any

import numpy as np

from .text_atlas import (
    _clean_dataset,
    _encode_texts,
    _nearest_same_label_indices,
    _optional_dependencies,
    _select_correct_per_class,
    _stratified_take,
    control_output,
    control_projection_from_head,
    pairwise_distances,
    residual_hidden,
    standardized_projection,
    text_surface_descriptors,
)
from .topology_persistence import collision_filtration_profile


def random_orthogonal_projection(
    hidden_dimension: int,
    visible_rank: int,
    *,
    seed: int,
) -> np.ndarray:
    """Draw an orthonormal row basis without using labels, controls, or behavior."""

    if hidden_dimension <= 0:
        raise ValueError("hidden_dimension must be positive")
    if not 1 <= visible_rank <= hidden_dimension:
        raise ValueError("visible_rank must lie in [1, hidden_dimension]")
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(hidden_dimension, visible_rank))
    q, _ = np.linalg.qr(raw, mode="reduced")
    return q.T.copy()


def _projection_metrics(
    *,
    projection: np.ndarray,
    train_hidden: np.ndarray,
    benign_hidden: np.ndarray,
    benign_logits: np.ndarray,
    benign_labels: np.ndarray,
    anchor_hidden: np.ndarray,
    candidate_hidden: np.ndarray,
    anchor_logits: np.ndarray,
    candidate_logits: np.ndarray,
    anchor_labels: np.ndarray,
    candidate_labels: np.ndarray,
    config: Mapping[str, object],
) -> dict[str, float]:
    standardized = standardized_projection(projection, train_hidden)
    benign_control = control_output(benign_hidden, standardized)
    benign_residual = residual_hidden(benign_hidden, projection)
    nearest = _nearest_same_label_indices(benign_control, benign_labels)
    nominal_epsilon = max(
        float(
            np.quantile(
                np.linalg.norm(benign_control - benign_control[nearest], axis=1),
                float(config["control_quantile"]),
            )
        ),
        1e-8,
    )
    payload_delta = float(
        np.quantile(
            np.linalg.norm(benign_residual - benign_residual[nearest], axis=1),
            float(config["payload_quantile"]),
        )
    )
    behavior_gamma = float(
        np.quantile(
            np.linalg.norm(benign_logits - benign_logits[nearest], axis=1)
            / sqrt(benign_logits.shape[1]),
            float(config["behavior_quantile"]),
        )
    )
    thresholds = tuple(
        sorted(
            {
                max(1e-10, nominal_epsilon * float(multiplier))
                for multiplier in config["epsilon_multipliers"]
            }
        )
    )

    anchor_control = control_output(anchor_hidden, standardized)
    candidate_control = control_output(candidate_hidden, standardized)
    anchor_residual = residual_hidden(anchor_hidden, projection)
    candidate_residual = residual_hidden(candidate_hidden, projection)
    control_distances = pairwise_distances(candidate_control, anchor_control)
    payload_distances = pairwise_distances(candidate_residual, anchor_residual)
    behavior_distances = pairwise_distances(candidate_logits, anchor_logits) / sqrt(
        anchor_logits.shape[1]
    )
    valid_pairs = candidate_labels[:, None] != anchor_labels[None, :]
    displacements = candidate_residual[:, None, :] - anchor_residual[None, :, :]
    profile = collision_filtration_profile(
        control_distances,
        payload_distances,
        behavior_distances,
        thresholds,
        nominal_epsilon=nominal_epsilon,
        payload_delta=payload_delta,
        behavior_gamma=behavior_gamma,
        valid_pairs=valid_pairs,
        edge_displacements=displacements,
    )
    nominal_point = min(
        profile.points,
        key=lambda point: abs(float(point.control_epsilon) - nominal_epsilon),
    )
    metrics = nominal_point.metrics
    denominator = max(1, min(int(metrics.n_left), int(metrics.n_right)))
    beta0 = int(metrics.beta0_active)
    return {
        "nominal_epsilon": nominal_epsilon,
        "payload_delta": payload_delta,
        "behavior_gamma": behavior_gamma,
        "capacity_fraction": float(metrics.capacity_fraction),
        "basin_density": float(beta0) / denominator,
        "within_basin_multiplicity": (
            float(metrics.matching_size) / beta0 if beta0 > 0 else 0.0
        ),
        "cycle_density": float(metrics.beta1_active) / max(1, int(metrics.edge_count)),
        "capacity_auc": float(profile.summary.capacity_auc),
        "capacity_robustness_ratio": float(profile.summary.capacity_robustness_ratio),
        "persistent_basin_lifetime": float(
            profile.basin_persistence.normalized_total_lifetime
        ),
    }


def _null_summary(
    learned: Mapping[str, float],
    null_rows: list[Mapping[str, float]],
) -> dict[str, object]:
    metrics = (
        "capacity_fraction",
        "basin_density",
        "within_basin_multiplicity",
        "cycle_density",
        "capacity_auc",
        "capacity_robustness_ratio",
        "persistent_basin_lifetime",
    )
    summary: dict[str, object] = {"repetition_count": len(null_rows)}
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in null_rows], dtype=float)
        learned_value = float(learned[metric])
        summary[metric] = {
            "random_mean": float(values.mean()),
            "random_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "random_min": float(values.min()),
            "random_max": float(values.max()),
            "learned_value": learned_value,
            "learned_minus_random_mean": learned_value - float(values.mean()),
            "learned_percentile_among_random": float(np.mean(values <= learned_value)),
        }
    return summary


def run_text_random_projection_null_component(
    config: Mapping[str, object],
    *,
    model_name: str,
    fixture_seed: int,
) -> dict[str, object]:
    """Pair the learned control subspace with matched-rank random orthogonal nulls."""

    (
        torch,
        fetch_20newsgroups,
        LogisticRegression,
        Ridge,
        train_test_split,
        StandardScaler,
        AutoModel,
        AutoTokenizer,
    ) = _optional_dependencies()
    if model_name not in tuple(str(value) for value in config["models"]):
        raise ValueError(f"model {model_name!r} is not configured")
    if fixture_seed not in tuple(int(value) for value in config["fixture_seeds"]):
        raise ValueError(f"fixture_seed={fixture_seed} is not configured")

    torch.manual_seed(fixture_seed)
    torch.set_num_threads(2)
    np.random.seed(fixture_seed % (2**32))
    train_bundle = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
        shuffle=False,
    )
    test_bundle = fetch_20newsgroups(
        subset="test",
        remove=("headers", "footers", "quotes"),
        shuffle=False,
    )
    train_texts_all, train_labels_all = _clean_dataset(train_bundle.data, train_bundle.target)
    test_texts_all, test_labels_all = _clean_dataset(test_bundle.data, test_bundle.target)

    full_indices = np.arange(len(train_labels_all))
    train_pool, calibration_indices = train_test_split(
        full_indices,
        test_size=int(config["calibration_samples"]),
        random_state=fixture_seed,
        stratify=train_labels_all,
    )
    train_local = _stratified_take(
        train_labels_all[train_pool],
        int(config["train_samples"]),
        fixture_seed + 1,
        train_test_split,
    )
    evaluation_indices = _stratified_take(
        test_labels_all,
        int(config["evaluation_samples"]),
        fixture_seed + 2,
        train_test_split,
    )
    train_indices = train_pool[train_local]
    train_texts = [train_texts_all[int(index)] for index in train_indices]
    calibration_texts = [train_texts_all[int(index)] for index in calibration_indices]
    evaluation_texts = [test_texts_all[int(index)] for index in evaluation_indices]
    train_labels = train_labels_all[train_indices]
    calibration_labels = train_labels_all[calibration_indices]
    evaluation_labels = test_labels_all[evaluation_indices]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    batch_size = int(config["batch_size"])
    max_length = int(config["max_length"])
    train_hidden_raw = _encode_texts(
        model,
        tokenizer,
        train_texts,
        batch_size=batch_size,
        max_length=max_length,
        torch=torch,
    )
    calibration_hidden_raw = _encode_texts(
        model,
        tokenizer,
        calibration_texts,
        batch_size=batch_size,
        max_length=max_length,
        torch=torch,
    )
    evaluation_hidden_raw = _encode_texts(
        model,
        tokenizer,
        evaluation_texts,
        batch_size=batch_size,
        max_length=max_length,
        torch=torch,
    )
    hidden_scaler = StandardScaler().fit(train_hidden_raw)
    train_hidden = hidden_scaler.transform(train_hidden_raw)
    calibration_hidden = hidden_scaler.transform(calibration_hidden_raw)
    evaluation_hidden = hidden_scaler.transform(evaluation_hidden_raw)

    classifier = LogisticRegression(
        C=float(config.get("classifier_c", 1.0)),
        max_iter=int(config.get("classifier_max_iter", 800)),
        solver="lbfgs",
    ).fit(train_hidden, train_labels)
    calibration_logits = classifier.decision_function(calibration_hidden)
    evaluation_logits = classifier.decision_function(evaluation_hidden)
    calibration_predictions = calibration_logits.argmax(axis=1)
    evaluation_predictions = evaluation_logits.argmax(axis=1)

    descriptor_scaler = StandardScaler().fit(text_surface_descriptors(train_texts))
    train_controls = descriptor_scaler.transform(text_surface_descriptors(train_texts))
    calibration_controls = descriptor_scaler.transform(
        text_surface_descriptors(calibration_texts)
    )
    control_head = Ridge(alpha=float(config.get("control_ridge_alpha", 1.0))).fit(
        train_hidden,
        train_controls,
    )
    control_r2 = float(control_head.score(calibration_hidden, calibration_controls))
    coefficients = np.asarray(control_head.coef_, dtype=float)

    correct_calibration = np.flatnonzero(calibration_predictions == calibration_labels)
    benign_hidden = calibration_hidden[correct_calibration]
    benign_logits = calibration_logits[correct_calibration]
    benign_labels = calibration_labels[correct_calibration]
    anchors_per_class = int(config["anchors_per_class"])
    anchor_indices = _select_correct_per_class(
        calibration_logits,
        calibration_labels,
        per_class=anchors_per_class,
    )
    candidate_indices = _select_correct_per_class(
        evaluation_logits,
        evaluation_labels,
        per_class=anchors_per_class,
    )
    shared = {
        "train_hidden": train_hidden,
        "benign_hidden": benign_hidden,
        "benign_logits": benign_logits,
        "benign_labels": benign_labels,
        "anchor_hidden": calibration_hidden[anchor_indices],
        "candidate_hidden": evaluation_hidden[candidate_indices],
        "anchor_logits": calibration_logits[anchor_indices],
        "candidate_logits": evaluation_logits[candidate_indices],
        "anchor_labels": calibration_labels[anchor_indices],
        "candidate_labels": evaluation_labels[candidate_indices],
        "config": config,
    }

    repetitions = int(config.get("random_projection_repetitions", 8))
    if repetitions < 2:
        raise ValueError("random_projection_repetitions must be at least 2")
    rows: list[dict[str, Any]] = []
    for visible_rank in (int(value) for value in config["visible_ranks"]):
        learned_projection = control_projection_from_head(coefficients, visible_rank)
        learned = _projection_metrics(projection=learned_projection, **shared)
        random_rows = []
        for repetition in range(repetitions):
            seed = fixture_seed * 100003 + visible_rank * 1009 + repetition
            projection = random_orthogonal_projection(
                train_hidden.shape[1],
                visible_rank,
                seed=seed,
            )
            random_rows.append(_projection_metrics(projection=projection, **shared))
        rows.append(
            {
                "visible_rank": visible_rank,
                "learned_control": learned,
                "random_projection_summary": _null_summary(learned, random_rows),
                "random_projection_rows": random_rows,
            }
        )

    return {
        "artifact_type": "qcollide_text_random_projection_null_component",
        "schema_version": 1,
        "domain": "language_understanding",
        "dataset": "20newsgroups",
        "model": model_name,
        "fixture_seed": fixture_seed,
        "model_diagnostics": {
            "hidden_dimension": int(train_hidden.shape[1]),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "calibration_accuracy": float(
                np.mean(calibration_predictions == calibration_labels)
            ),
            "evaluation_accuracy": float(np.mean(evaluation_predictions == evaluation_labels)),
            "control_head_calibration_r2": control_r2,
        },
        "random_projection_repetitions": repetitions,
        "rows": rows,
        "claim_boundary": {
            "learned_and_random_use_same_hidden_representations": True,
            "learned_and_random_use_same_classifier": True,
            "learned_and_random_use_same_anchor_candidate_fixture": True,
            "each_projection_recalibrates_thresholds_on_benign_calibration_only": True,
            "random_subspaces_are_label_behavior_control_blind": True,
            "generated_attack_text": False,
        },
    }


__all__ = [
    "random_orthogonal_projection",
    "run_text_random_projection_null_component",
]
