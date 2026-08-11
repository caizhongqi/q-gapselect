"""Cross-model text functional-collision atlas on public text classification data."""

from __future__ import annotations

import hashlib
import string
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from math import sqrt
from typing import Any

import numpy as np

from .topology_persistence import collision_filtration_profile


def text_surface_descriptors(texts: Sequence[str]) -> np.ndarray:
    """Return label-agnostic surface/style controls for raw text.

    These descriptors are deliberately independent of the downstream class label:
    log character length, log word count, mean word length, type-token ratio,
    digit fraction, punctuation fraction, uppercase fraction, and sentence-mark
    density. They define the domain control target, not the task behavior.
    """

    output = np.zeros((len(texts), 8), dtype=np.float64)
    punctuation = set(string.punctuation)
    sentence_marks = set(".!?")
    for index, raw in enumerate(texts):
        text = str(raw)
        words = text.split()
        characters = max(len(text), 1)
        word_count = len(words)
        alpha = [character for character in text if character.isalpha()]
        output[index, 0] = np.log1p(len(text))
        output[index, 1] = np.log1p(word_count)
        output[index, 2] = (
            float(np.mean([len(word) for word in words])) if words else 0.0
        )
        output[index, 3] = len({word.lower() for word in words}) / max(word_count, 1)
        output[index, 4] = sum(character.isdigit() for character in text) / characters
        output[index, 5] = sum(character in punctuation for character in text) / characters
        output[index, 6] = (
            sum(character.isupper() for character in alpha) / max(len(alpha), 1)
        )
        output[index, 7] = sum(character in sentence_marks for character in text) / characters
    return output


def row_basis(matrix: np.ndarray, *, tolerance: float = 1e-9) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if values.shape[0] == 0:
        return np.zeros((0, values.shape[1]), dtype=float)
    _, singular_values, vh = np.linalg.svd(values, full_matrices=False)
    scale = max(float(singular_values[0]) if singular_values.size else 0.0, 1.0)
    rank = int(np.count_nonzero(singular_values > tolerance * scale))
    return vh[:rank].copy()


def standardized_projection(projection: np.ndarray, hidden_support: np.ndarray) -> np.ndarray:
    basis = row_basis(projection)
    if basis.shape[0] == 0:
        return basis
    support = np.asarray(hidden_support, dtype=float)
    outputs = support @ basis.T
    centered = outputs - outputs.mean(axis=0, keepdims=True)
    denominator = max(len(centered) - 1, 1)
    covariance = centered.T @ centered / denominator
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = max(float(eigenvalues.max()) if eigenvalues.size else 0.0, 1.0)
    inverse_root = eigenvectors @ np.diag(
        1.0 / np.sqrt(np.maximum(eigenvalues, 1e-8 * largest))
    ) @ eigenvectors.T
    return inverse_root @ basis


def control_output(hidden: np.ndarray, standardized: np.ndarray) -> np.ndarray:
    values = np.asarray(hidden, dtype=float)
    if standardized.shape[0] == 0:
        return np.zeros(values.shape[:-1] + (0,), dtype=float)
    return values @ standardized.T / sqrt(standardized.shape[0])


def residual_hidden(hidden: np.ndarray, projection: np.ndarray) -> np.ndarray:
    values = np.asarray(hidden, dtype=float)
    basis = row_basis(projection)
    if basis.shape[0] == 0:
        return values.copy()
    return values - (values @ basis.T) @ basis


def pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lhs = np.asarray(left, dtype=float)
    rhs = np.asarray(right, dtype=float)
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[1] != rhs.shape[1]:
        raise ValueError("left and right must be aligned matrices")
    lhs_square = np.sum(lhs * lhs, axis=1)[:, None]
    rhs_square = np.sum(rhs * rhs, axis=1)[None, :]
    return np.sqrt(np.maximum(lhs_square + rhs_square - 2.0 * lhs @ rhs.T, 0.0))


def control_projection_from_head(control_coefficients: np.ndarray, visible_rank: int) -> np.ndarray:
    coefficients = np.asarray(control_coefficients, dtype=float)
    if coefficients.ndim != 2:
        raise ValueError("control_coefficients must be a matrix")
    _, singular_values, vh = np.linalg.svd(coefficients, full_matrices=False)
    scale = max(float(singular_values[0]) if singular_values.size else 0.0, 1.0)
    numerical_rank = int(np.count_nonzero(singular_values > 1e-9 * scale))
    if not 1 <= visible_rank <= numerical_rank:
        raise ValueError(
            f"visible_rank={visible_rank} exceeds numerical control rank {numerical_rank}"
        )
    return vh[:visible_rank].copy()


def _clean_dataset(texts: Sequence[str], labels: Sequence[int]) -> tuple[list[str], np.ndarray]:
    kept_texts: list[str] = []
    kept_labels: list[int] = []
    for text, label in zip(texts, labels, strict=True):
        normalized = " ".join(str(text).split())
        if len(normalized) < 20:
            continue
        kept_texts.append(normalized)
        kept_labels.append(int(label))
    if not kept_texts:
        raise RuntimeError("text dataset is empty after cleaning")
    return kept_texts, np.asarray(kept_labels, dtype=np.int64)


def _stratified_take(labels: np.ndarray, count: int, seed: int, train_test_split) -> np.ndarray:
    if count <= 0 or count > len(labels):
        raise ValueError("sample count must lie in [1,dataset-size]")
    indices = np.arange(len(labels))
    if count == len(labels):
        return indices
    selected, _ = train_test_split(
        indices,
        train_size=count,
        random_state=seed,
        stratify=labels,
    )
    return np.sort(selected)


def _select_correct_per_class(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    per_class: int,
) -> np.ndarray:
    predictions = np.asarray(logits).argmax(axis=1)
    selected: list[int] = []
    for label in sorted(int(value) for value in np.unique(labels)):
        candidates = np.flatnonzero((labels == label) & (predictions == label))
        if len(candidates) < per_class:
            raise RuntimeError(
                f"class {label} has {len(candidates)} correct samples; {per_class} required"
            )
        selected.extend(int(value) for value in candidates[:per_class])
    return np.asarray(selected, dtype=np.int64)


def _nearest_same_label_indices(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    distances = pairwise_distances(values, values)
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    distances[~same] = np.inf
    if np.any(~np.isfinite(np.min(distances, axis=1))):
        raise RuntimeError("each calibration class needs at least two correctly classified samples")
    return np.argmin(distances, axis=1)


def _encode_texts(model, tokenizer, texts: Sequence[str], *, batch_size: int, max_length: int, torch):
    parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            output = model(**encoded, return_dict=True)
            hidden = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            parts.append(pooled.cpu().float().numpy())
    return np.concatenate(parts, axis=0)


def _optional_dependencies():
    try:
        import torch
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - executable dependency guard
        raise RuntimeError("install text atlas extras with: pip install -e '.[atlas_text]'") from exc
    return (
        torch,
        fetch_20newsgroups,
        LogisticRegression,
        Ridge,
        train_test_split,
        StandardScaler,
        AutoModel,
        AutoTokenizer,
    )


def _hash_fixture(texts: Sequence[str], labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for label, text in zip(labels.tolist(), texts, strict=True):
        digest.update(str(int(label)).encode("ascii"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def run_text_atlas_component(
    config: Mapping[str, object],
    *,
    model_name: str,
    fixture_seed: int,
) -> dict[str, object]:
    """Run one pretrained text-encoder collision-topology component."""

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
    torch.manual_seed(fixture_seed)
    torch.set_num_threads(2)
    np.random.seed(fixture_seed % (2**32))

    allowed_models = tuple(str(value) for value in config["models"])
    if model_name not in allowed_models:
        raise ValueError(f"model {model_name!r} is not configured")
    allowed_seeds = tuple(int(value) for value in config["fixture_seeds"])
    if fixture_seed not in allowed_seeds:
        raise ValueError(f"fixture_seed={fixture_seed} is not configured")

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

    calibration_samples = int(config["calibration_samples"])
    train_samples = int(config["train_samples"])
    evaluation_samples = int(config["evaluation_samples"])
    full_indices = np.arange(len(train_labels_all))
    train_pool, calibration_indices = train_test_split(
        full_indices,
        test_size=calibration_samples,
        random_state=fixture_seed,
        stratify=train_labels_all,
    )
    train_local = _stratified_take(
        train_labels_all[train_pool],
        train_samples,
        fixture_seed + 1,
        train_test_split,
    )
    evaluation_indices = _stratified_take(
        test_labels_all,
        evaluation_samples,
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
    if calibration_logits.ndim != 2 or evaluation_logits.ndim != 2:
        raise RuntimeError("text atlas expects multiclass decision logits")
    calibration_predictions = calibration_logits.argmax(axis=1)
    evaluation_predictions = evaluation_logits.argmax(axis=1)
    calibration_accuracy = float(np.mean(calibration_predictions == calibration_labels))
    evaluation_accuracy = float(np.mean(evaluation_predictions == evaluation_labels))

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
    control_coefficients = np.asarray(control_head.coef_, dtype=float)

    correct_calibration = np.flatnonzero(calibration_predictions == calibration_labels)
    if len(correct_calibration) < 2 * len(np.unique(calibration_labels)):
        raise RuntimeError("too few correctly classified calibration texts")
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
    anchor_hidden = calibration_hidden[anchor_indices]
    candidate_hidden = evaluation_hidden[candidate_indices]
    anchor_logits = calibration_logits[anchor_indices]
    candidate_logits = evaluation_logits[candidate_indices]
    anchor_labels = calibration_labels[anchor_indices]
    candidate_labels = evaluation_labels[candidate_indices]

    epsilon_multipliers = tuple(float(value) for value in config["epsilon_multipliers"])
    control_quantile = float(config["control_quantile"])
    payload_quantile = float(config["payload_quantile"])
    behavior_quantile = float(config["behavior_quantile"])
    rows: list[dict[str, Any]] = []
    for visible_rank in (int(value) for value in config["visible_ranks"]):
        projection = control_projection_from_head(control_coefficients, visible_rank)
        standardized = standardized_projection(projection, train_hidden)
        benign_control = control_output(benign_hidden, standardized)
        benign_residual = residual_hidden(benign_hidden, projection)
        benign_nearest = _nearest_same_label_indices(benign_control, benign_labels)
        benign_control_distances = np.linalg.norm(
            benign_control - benign_control[benign_nearest],
            axis=1,
        )
        benign_payload_distances = np.linalg.norm(
            benign_residual - benign_residual[benign_nearest],
            axis=1,
        )
        benign_behavior_distances = np.linalg.norm(
            benign_logits - benign_logits[benign_nearest],
            axis=1,
        ) / sqrt(benign_logits.shape[1])
        nominal_epsilon = max(
            float(np.quantile(benign_control_distances, control_quantile)),
            1e-8,
        )
        payload_delta = float(np.quantile(benign_payload_distances, payload_quantile))
        behavior_gamma = float(np.quantile(benign_behavior_distances, behavior_quantile))
        thresholds = tuple(
            sorted({max(1e-10, nominal_epsilon * value) for value in epsilon_multipliers})
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
        edge_displacements = (
            candidate_residual[:, None, :] - anchor_residual[None, :, :]
        )
        profile = collision_filtration_profile(
            control_distances,
            payload_distances,
            behavior_distances,
            thresholds,
            nominal_epsilon=nominal_epsilon,
            payload_delta=payload_delta,
            behavior_gamma=behavior_gamma,
            valid_pairs=valid_pairs,
            edge_displacements=edge_displacements,
        )
        rows.append(
            {
                "visible_rank": visible_rank,
                "nominal_epsilon": nominal_epsilon,
                "payload_delta": payload_delta,
                "behavior_gamma": behavior_gamma,
                "filtration_summary": asdict(profile.summary),
                "basin_persistence": asdict(profile.basin_persistence),
                "filtration_points": [
                    {"control_epsilon": point.control_epsilon, **asdict(point.metrics)}
                    for point in profile.points
                ],
            }
        )

    return {
        "artifact_type": "qcollide_text_functional_collision_atlas_component",
        "schema_version": 1,
        "domain": "language_understanding",
        "dataset": "20newsgroups",
        "model": model_name,
        "fixture_seed": fixture_seed,
        "fixture_sha256": _hash_fixture(
            calibration_texts + evaluation_texts,
            np.concatenate([calibration_labels, evaluation_labels]),
        ),
        "sample_counts": {
            "train": len(train_texts),
            "calibration": len(calibration_texts),
            "evaluation": len(evaluation_texts),
        },
        "model_diagnostics": {
            "hidden_dimension": int(train_hidden.shape[1]),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "calibration_accuracy": calibration_accuracy,
            "evaluation_accuracy": evaluation_accuracy,
            "control_head_calibration_r2": control_r2,
        },
        "selection": {
            "anchors_per_class": anchors_per_class,
            "anchor_count": len(anchor_indices),
            "candidate_count": len(candidate_indices),
        },
        "rows": rows,
        "claim_boundary": {
            "pretrained_encoder_frozen": True,
            "task_classifier_is_linear_probe": True,
            "control_head_is_label_agnostic_surface_descriptor_probe": True,
            "cross_class_collision_graph": True,
            "generated_attack_text": False,
            "quantum_execution": False,
        },
    }


__all__ = [
    "control_output",
    "control_projection_from_head",
    "pairwise_distances",
    "residual_hidden",
    "row_basis",
    "run_text_atlas_component",
    "standardized_projection",
    "text_surface_descriptors",
]
