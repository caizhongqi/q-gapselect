#!/usr/bin/env python3
"""Run the prospective query-budget utility experiment on a real Guard/LLM graph."""

from __future__ import annotations

import argparse
import json
import random
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

from qgapselect.qcollide.guard_application import (
    build_guard_collision_graph,
    calibrate_collision_thresholds,
    nearest_control_discovery,
    random_pair_discovery,
)
from qgapselect.qcollide.guard_utility import collision_diversified_control_discovery
from scripts.run_qcollide_guard_application import (
    _balanced_indices,
    _batched_behavior_features,
    _batched_guard_features,
    _fixture_sha256,
    _load_dataset_bundle,
    _load_optional_dependencies,
    _select_logit_features,
    _unsafe_index,
)

_QUERY_FRACTIONS = (0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--guard-model",
        default="protectai/deberta-v3-base-prompt-injection-v2",
    )
    parser.add_argument(
        "--behavior-model",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
    )
    parser.add_argument("--dataset", default="deepset/prompt-injections")
    parser.add_argument("--per-class", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--logit-features", type=int, default=512)
    parser.add_argument("--benign-acceptance", type=float, default=0.95)
    parser.add_argument("--random-repetitions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output", required=True, type=Path)
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
    fixture_sha256 = _fixture_sha256(texts, labels)

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
    unsafe_scores = torch.softmax(torch.from_numpy(guard_logits), dim=1).numpy()[:, unsafe_index]

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

    total_pairs = graph.n_left * graph.n_right
    full_capacity = graph.matching_size
    curves: list[dict[str, Any]] = []
    for fraction in _QUERY_FRACTIONS:
        budget = min(total_pairs, max(1, ceil(total_pairs * fraction))) if total_pairs else 0
        random_values = [
            random_pair_discovery(
                graph,
                query_budget=budget,
                seed=args.seed + 1009 * repetition + budget,
            )
            for repetition in range(args.random_repetitions)
        ]
        nearest = nearest_control_discovery(graph, query_budget=budget)
        diversified = collision_diversified_control_discovery(graph, query_budget=budget)
        denominator = max(full_capacity, 1)
        curves.append(
            {
                "query_fraction": fraction,
                "query_budget": budget,
                "random_pair_mean_capacity": float(np.mean(random_values)),
                "random_pair_std_capacity": float(np.std(random_values, ddof=1)),
                "nearest_control_capacity": nearest,
                "collision_diversified_capacity": diversified,
                "random_utility_fraction": float(np.mean(random_values)) / denominator,
                "nearest_control_utility_fraction": nearest / denominator,
                "collision_diversified_utility_fraction": diversified / denominator,
            }
        )

    payload: dict[str, Any] = {
        "artifact_type": "qcollide_guard_query_budget_utility_component",
        "schema_version": 1,
        "guard_model": args.guard_model,
        "behavior_model": args.behavior_model,
        "dataset": args.dataset,
        "dataset_access": dataset_access,
        "dataset_revision": dataset_revision,
        "dataset_splits": split_names,
        "fixture_sha256": fixture_sha256,
        "seed": args.seed,
        "sampled_per_class": args.per_class,
        "domain": {
            "accepted_injection_prompts": int(len(accepted_injections)),
            "accepted_benign_prompts": int(len(accepted_benign)),
            "benign_acceptance_observed": float(np.mean(accepted[benign_evaluation])),
            "injection_acceptance_observed": float(np.mean(accepted[injections])),
        },
        "collision_graph": {
            "n_left": graph.n_left,
            "n_right": graph.n_right,
            "all_pairs": total_pairs,
            "edge_count": graph.edge_count,
            "matching_capacity": full_capacity,
            "capacity_fraction": (
                full_capacity / min(graph.n_left, graph.n_right)
                if min(graph.n_left, graph.n_right)
                else 0.0
            ),
        },
        "discovery_curves": curves,
        "information_contract": {
            "random_prequery_information": "domain dimensions only",
            "nearest_control_prequery_information": "full control-distance matrix",
            "collision_diversified_prequery_information": "same full control-distance matrix",
            "collision_diversified_adaptive_information": "outcomes of previously queried pairs only",
            "unqueried_collision_labels_visible": False,
            "query_budget_identical": True,
        },
        "claim_boundary": {
            "public_existing_prompts_only": True,
            "generated_jailbreak_content": False,
            "this_experiment_is_classical_query_policy_comparison": True,
            "quantum_runtime_claimed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "seed": args.seed,
                "full_capacity": full_capacity,
                "all_pairs": total_pairs,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
