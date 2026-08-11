#!/usr/bin/env python3
"""Run the real prompt-guard / downstream-LLM functional-collision benchmark."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from qgapselect.qcollide.guard_application import (
    build_guard_collision_graph,
    calibrate_collision_thresholds,
    nearest_control_discovery,
    packed_classical_query_proxy,
    packed_quantum_query_proxy,
    query_budget_grid,
    random_pair_discovery,
)

_PROTECTAI_VALIDATION_DATASET = "protectai/prompt-injection-validation"
_PROTECTAI_VALIDATION_REVISION = "c581eb48bf461df1cf8fe916af6eb971c8e5d296"
_PROTECTAI_VALIDATION_SPLITS = (
    "InjecGuard_valid",
    "spikee",
    "bipia_code",
    "bipia_text",
    "not_inject",
    "wildguard",
    "deepset",
)


def _load_optional_dependencies():
    try:
        import torch
        from datasets import concatenate_datasets, load_dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:  # pragma: no cover - executable dependency guard
        raise RuntimeError("install the guard extras with: pip install -e '.[guard]'") from exc
    return (
        torch,
        concatenate_datasets,
        load_dataset,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )


def _download_public_parquet(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "q-gapselect-guard-benchmark/0.1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def _load_dataset_bundle(load_dataset, dataset_name: str):
    """Load a dataset, bypassing the Hub API for the pinned ProtectAI fallback."""
    try:
        if dataset_name == _PROTECTAI_VALIDATION_DATASET:
            dataset = load_dataset(
                dataset_name,
                revision=_PROTECTAI_VALIDATION_REVISION,
            )
            return dataset, "hub_dataset", _PROTECTAI_VALIDATION_REVISION
        return load_dataset(dataset_name), "hub_dataset", None
    except Exception:  # pragma: no cover - network/runtime fallback
        if dataset_name != _PROTECTAI_VALIDATION_DATASET:
            raise

        with tempfile.TemporaryDirectory(prefix="qcollide-protectai-") as temporary_dir:
            root = Path(temporary_dir)
            local_files = {}
            for split in _PROTECTAI_VALIDATION_SPLITS:
                filename = f"{split}-00000-of-00001.parquet"
                url = (
                    f"https://huggingface.co/datasets/{dataset_name}/resolve/"
                    f"{_PROTECTAI_VALIDATION_REVISION}/data/{filename}?download=true"
                )
                destination = root / filename
                _download_public_parquet(url, destination)
                local_files[split] = str(destination)

            dataset = load_dataset(
                "parquet",
                data_files=local_files,
                keep_in_memory=True,
            )
        return dataset, "direct_http_parquet", _PROTECTAI_VALIDATION_REVISION


def _masked_mean(hidden, attention_mask, torch):
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _unsafe_index(config) -> int:
    mapping = getattr(config, "id2label", None) or {}
    for index, label in mapping.items():
        text = str(label).upper()
        if any(token in text for token in ("INJECTION", "UNSAFE", "MALICIOUS", "TOXIC")):
            return int(index)
    if getattr(config, "num_labels", 2) == 2:
        return 1
    raise RuntimeError("could not infer unsafe guard class from model config")


def _batched_guard_features(model, tokenizer, texts, *, batch_size, max_length, torch):
    embeddings = []
    logits = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            output = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
            )
            embeddings.append(
                _masked_mean(output.hidden_states[-1], encoded["attention_mask"], torch)
                .cpu()
                .float()
                .numpy()
            )
            logits.append(output.logits.cpu().float().numpy())
    return np.concatenate(embeddings), np.concatenate(logits)


def _batched_behavior_features(model, tokenizer, texts, *, batch_size, max_length, torch):
    embeddings = []
    next_logits = []
    model.eval()
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            output = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = output.hidden_states[-1]
            embeddings.append(
                _masked_mean(hidden, encoded["attention_mask"], torch)
                .cpu()
                .float()
                .numpy()
            )
            last_index = encoded["attention_mask"].sum(dim=1) - 1
            batch_index = torch.arange(len(batch))
            next_logits.append(
                output.logits[batch_index, last_index]
                .cpu()
                .float()
                .numpy()
            )
    return np.concatenate(embeddings), np.concatenate(next_logits)


def _select_logit_features(values: np.ndarray, feature_count: int) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("logits must be a matrix")
    count = min(feature_count, values.shape[1])
    variance = np.var(values, axis=0)
    indices = np.argpartition(variance, -count)[-count:]
    return values[:, np.sort(indices)]


def _balanced_indices(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for label in (0, 1):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < per_class:
            raise RuntimeError(
                f"dataset has only {len(candidates)} examples for label {label}; "
                f"requested {per_class}"
            )
        selected.extend(rng.choice(candidates, size=per_class, replace=False).tolist())
    return np.asarray(selected, dtype=int)


def _edge_quality(graph) -> dict[str, float | None]:
    pairs = [(u, v) for u, row in enumerate(graph.adjacency) for v in row]
    if not pairs:
        return {
            "mean_control_distance": None,
            "mean_payload_distance": None,
            "mean_behavior_distance": None,
        }
    left = np.asarray([u for u, _ in pairs], dtype=int)
    right = np.asarray([v for _, v in pairs], dtype=int)
    return {
        "mean_control_distance": float(np.mean(graph.control_distances[left, right])),
        "mean_payload_distance": float(np.mean(graph.payload_distances[left, right])),
        "mean_behavior_distance": float(np.mean(graph.behavior_distances[left, right])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard-model", default="protectai/deberta-v3-small-prompt-injection-v2")
    parser.add_argument("--behavior-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--dataset", default=_PROTECTAI_VALIDATION_DATASET)
    parser.add_argument("--per-class", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--logit-features", type=int, default=512)
    parser.add_argument("--benign-acceptance", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qcollide_guard_application.json"),
    )
    args = parser.parse_args()

    (
        torch,
        concatenate_datasets,
        load_dataset,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    ) = _load_optional_dependencies()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)

    dataset, dataset_access, dataset_revision = _load_dataset_bundle(
        load_dataset,
        args.dataset,
    )
    split_names = sorted(dataset.keys())
    combined = concatenate_datasets([dataset[name] for name in split_names])
    texts_all = np.asarray(combined["text"], dtype=object)
    labels_all = np.asarray(combined["label"], dtype=int)
    selected = _balanced_indices(labels_all, args.per_class, args.seed)
    texts = texts_all[selected].tolist()
    labels = labels_all[selected]

    guard_tokenizer = AutoTokenizer.from_pretrained(args.guard_model)
    guard_model = AutoModelForSequenceClassification.from_pretrained(args.guard_model)
    guard_embeddings, guard_logits = _batched_guard_features(
        guard_model,
        guard_tokenizer,
        texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        torch=torch,
    )
    unsafe_index = _unsafe_index(guard_model.config)
    guard_probabilities = torch.softmax(torch.from_numpy(guard_logits), dim=1).numpy()
    unsafe_scores = guard_probabilities[:, unsafe_index]

    behavior_tokenizer = AutoTokenizer.from_pretrained(args.behavior_model)
    behavior_model = AutoModelForCausalLM.from_pretrained(args.behavior_model)
    behavior_embeddings, behavior_logits_full = _batched_behavior_features(
        behavior_model,
        behavior_tokenizer,
        texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        torch=torch,
    )
    behavior_logits = _select_logit_features(behavior_logits_full, args.logit_features)
    del behavior_logits_full

    benign = np.flatnonzero(labels == 0)
    injections = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(args.seed ^ 0x51AFE)
    rng.shuffle(benign)
    benign_split = max(2, len(benign) // 2)
    benign_calibration = benign[:benign_split]
    benign_evaluation = benign[benign_split:]

    thresholds = calibrate_collision_thresholds(
        benign_guard_scores=unsafe_scores[benign_calibration],
        benign_guard_embeddings=guard_embeddings[benign_calibration],
        benign_behavior_embeddings=behavior_embeddings[benign_calibration],
        benign_behavior_logits=behavior_logits[benign_calibration],
        benign_acceptance=args.benign_acceptance,
    )
    accepted = unsafe_scores <= thresholds.acceptance_threshold
    accepted_injections = injections[accepted[injections]]
    accepted_benign = benign_evaluation[accepted[benign_evaluation]]

    graph = build_guard_collision_graph(
        left_guard_embeddings=guard_embeddings[accepted_injections],
        right_guard_embeddings=guard_embeddings[accepted_benign],
        left_behavior_embeddings=behavior_embeddings[accepted_injections],
        right_behavior_embeddings=behavior_embeddings[accepted_benign],
        left_behavior_logits=behavior_logits[accepted_injections],
        right_behavior_logits=behavior_logits[accepted_benign],
        thresholds=thresholds,
    )
    budgets = query_budget_grid(graph.n_left, graph.n_right)
    discovery = []
    for budget in budgets:
        random_values = [
            random_pair_discovery(
                graph,
                query_budget=budget,
                seed=args.seed + 1009 * repetition + budget,
            )
            for repetition in range(16)
        ]
        discovery.append(
            {
                "query_budget": budget,
                "random_pair_mean_capacity": float(np.mean(random_values)),
                "random_pair_std_capacity": float(np.std(random_values, ddof=1)),
                "nearest_control_capacity": nearest_control_discovery(
                    graph,
                    query_budget=budget,
                ),
            }
        )

    benign_acceptance_observed = float(np.mean(accepted[benign_evaluation]))
    injection_acceptance_observed = float(np.mean(accepted[injections]))
    exact_matching = graph.matching_size
    payload: dict[str, Any] = {
        "artifact_type": "qcollide_real_guard_functional_collision_benchmark",
        "schema_version": 1,
        "guard_model": args.guard_model,
        "behavior_model": args.behavior_model,
        "dataset": args.dataset,
        "dataset_access": dataset_access,
        "dataset_revision": dataset_revision,
        "dataset_splits": split_names,
        "seed": args.seed,
        "sampled_per_class": args.per_class,
        "unsafe_class_index": unsafe_index,
        "thresholds": {
            "acceptance_threshold": thresholds.acceptance_threshold,
            "control_epsilon": thresholds.control_epsilon,
            "payload_delta": thresholds.payload_delta,
            "behavior_gamma": thresholds.behavior_gamma,
        },
        "domain": {
            "accepted_injection_prompts": int(len(accepted_injections)),
            "accepted_benign_prompts": int(len(accepted_benign)),
            "benign_acceptance_observed": benign_acceptance_observed,
            "injection_acceptance_observed": injection_acceptance_observed,
        },
        "collision_graph": {
            "edge_count": graph.edge_count,
            "matching_capacity": exact_matching,
            "capacity_fraction": (
                exact_matching / min(graph.n_left, graph.n_right)
                if min(graph.n_left, graph.n_right)
                else 0.0
            ),
            **_edge_quality(graph),
        },
        "query_proxies": {
            "all_pair_classical_queries": graph.n_left * graph.n_right,
            "packed_classical_discovery_proxy": packed_classical_query_proxy(
                graph.n_left,
                graph.n_right,
                exact_matching,
            ),
            "packed_quantum_discovery_proxy": packed_quantum_query_proxy(
                graph.n_left,
                graph.n_right,
                exact_matching,
            ),
            "quantum_values_are": "analytic endpoint-query proxies",
        },
        "discovery_curves": discovery,
        "claim_boundary": {
            "uses_public_existing_prompts_only": True,
            "generates_jailbreak_content": False,
            "guard_acceptance_is_empirical": True,
            "functional_divergence_is_next_token_logit_distance": True,
            "coherent_quantum_execution": False,
            "hardware_runtime_advantage": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
