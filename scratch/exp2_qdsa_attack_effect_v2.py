#!/usr/bin/env python3
"""Experiment 2 v2: QDSA attack effect with unbiased feasible-target selection.

The failed v1 run exposed a protocol feasibility issue: a randomly selected target
may have fewer than the fixed 512 candidates satisfying the pre-registered semantic
gate.  v2 does NOT relax Delta and does NOT shrink the pool.  Instead it selects
50 class-balanced targets from the set that is feasible under the semantic gate.
Feasibility uses only labels, independent semantic-reference embeddings, and
reference correctness; it never uses victim collision distance or attack outcome.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from exp1_mnist_qco import (
    BUDGETS, SIM_SEEDS, SmallCNN, embed_dataset, l2_distances,
    qco_min_find, random_min_find, seed_all, train_victim,
)
from exp1b_mnist_strong_baselines import knn_graph, pixel_rank, simulated_annealing
from exp1c_qdsa_semantic_gate import (
    CALIBRATION_N, SemanticMLP, calibrate_delta, raw_flat,
)
from exp2_qdsa_attack_effect import (
    EPSILON_QUANTILE, calibrate_collision_epsilon, summarize,
)

TARGET_SELECTION_SEED = 2026080912
POOL_SELECTION_SEED_BASE = 90000


def legal_mask_for_target(
    t: int,
    labels: np.ndarray,
    ref_labels: np.ndarray,
    ref_preds: np.ndarray,
    ref_emb: np.ndarray,
    semantic_delta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    all_idx = np.arange(len(labels))
    y0 = int(labels[t])
    sem_all = l2_distances(ref_emb[t], ref_emb)
    legal = (
        (all_idx != int(t))
        & (labels != y0)
        & (ref_labels == ref_preds)
        & (sem_all >= semantic_delta)
    )
    return legal, sem_all


def select_balanced_feasible_targets(
    labels: np.ndarray,
    preds: np.ndarray,
    ref_labels: np.ndarray,
    ref_preds: np.ndarray,
    ref_emb: np.ndarray,
    semantic_delta: float,
    pool_size: int,
    n_targets: int,
) -> Tuple[np.ndarray, Dict[int, int], int]:
    """Pre-register targets using only feasibility, never victim collision scores."""
    if n_targets <= 0:
        raise ValueError("n_targets must be positive")
    rng = np.random.default_rng(TARGET_SELECTION_SEED)
    target_ok = (labels == preds) & (ref_labels == ref_preds)

    base = n_targets // 10
    rem = n_targets % 10
    quota = {c: base + (1 if c < rem else 0) for c in range(10)}
    selected: List[int] = []
    feasible_counts: Dict[int, int] = {}
    rejected_for_capacity = 0

    for c in range(10):
        candidates = np.flatnonzero(target_ok & (labels == c))
        candidates = rng.permutation(candidates)
        taken = 0
        for t in candidates:
            legal, _ = legal_mask_for_target(
                int(t), labels, ref_labels, ref_preds, ref_emb, semantic_delta
            )
            count = int(np.sum(legal))
            if count < pool_size:
                rejected_for_capacity += 1
                continue
            selected.append(int(t))
            feasible_counts[int(t)] = count
            taken += 1
            if taken >= quota[c]:
                break
        if taken < quota[c]:
            raise RuntimeError(
                f"Class {c}: only {taken} feasible targets for quota {quota[c]} "
                f"at pool={pool_size} and fixed semantic Delta"
            )

    return np.asarray(selected, dtype=int), feasible_counts, rejected_for_capacity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp2_v2_out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-seed", type=int, default=20260809)
    ap.add_argument("--targets", type=int, default=50)
    ap.add_argument("--pool", type=int, default=512)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    tf = transforms.ToTensor()
    train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)

    # Victim: identical to Experiments 1/1B/1C.
    seed_all(args.train_seed)
    vg = torch.Generator().manual_seed(args.train_seed)
    victim_loader = DataLoader(train_ds, batch_size=256, shuffle=True, generator=vg, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)
    victim = SmallCNN()
    train_victim(victim, victim_loader, device, args.epochs)
    labels, preds, emb = embed_dataset(victim, test_loader, device)
    victim_acc = float(np.mean(labels == preds))
    if victim_acc < 0.97:
        raise RuntimeError(f"Victim accuracy gate failed: {victim_acc:.4f} < 0.97")

    # Independent semantic reference.
    ref_seed = args.train_seed + 101
    seed_all(ref_seed)
    rg = torch.Generator().manual_seed(ref_seed)
    ref_loader = DataLoader(train_ds, batch_size=256, shuffle=True, generator=rg, num_workers=2)
    ref = SemanticMLP()
    train_victim(ref, ref_loader, device, args.epochs)
    ref_labels, ref_preds, ref_emb_test = embed_dataset(ref, test_loader, device)
    ref_acc = float(np.mean(ref_labels == ref_preds))
    if ref_acc < 0.95:
        raise RuntimeError(f"Reference accuracy gate failed: {ref_acc:.4f} < 0.95")

    # Pre-test calibration of both thresholds.
    cal_ds = Subset(train_ds, range(CALIBRATION_N))
    cal_loader = DataLoader(cal_ds, batch_size=512, shuffle=False, num_workers=2)
    vcal_labels, vcal_preds, vcal_emb = embed_dataset(victim, cal_loader, device)
    rcal_labels, rcal_preds, rcal_emb = embed_dataset(ref, cal_loader, device)
    ref_good = rcal_labels == rcal_preds
    semantic_delta, n_sem_pairs = calibrate_delta(rcal_emb[ref_good], rcal_labels[ref_good])
    collision_eps, n_collision_pairs = calibrate_collision_epsilon(
        vcal_emb, rcal_emb, vcal_labels,
        vcal_labels == vcal_preds,
        rcal_labels == rcal_preds,
        semantic_delta,
    )

    calibration = {
        "victim_accuracy": victim_acc,
        "semantic_reference_accuracy": ref_acc,
        "semantic_delta": semantic_delta,
        "collision_epsilon": collision_eps,
        "collision_epsilon_quantile": EPSILON_QUANTILE,
        "semantic_calibration_pairs": n_sem_pairs,
        "collision_calibration_pairs": n_collision_pairs,
    }
    (out / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print("CALIBRATION " + json.dumps(calibration, sort_keys=True), flush=True)

    target_indices, feasible_counts, rejected_capacity = select_balanced_feasible_targets(
        labels, preds, ref_labels, ref_preds, ref_emb_test,
        semantic_delta, args.pool, args.targets,
    )
    target_manifest = {
        "selection_seed": TARGET_SELECTION_SEED,
        "selection_rule": "class-balanced among correctly classified targets with >=pool semantic-legal candidates; no victim collision distance used",
        "target_indices": [int(x) for x in target_indices],
        "target_labels": [int(labels[x]) for x in target_indices],
        "eligible_counts": {str(k): int(v) for k, v in feasible_counts.items()},
        "rejected_for_capacity_before_quotas_filled": int(rejected_capacity),
    }
    (out / "target_manifest.json").write_text(json.dumps(target_manifest, indent=2), encoding="utf-8")

    images = torch.cat([x for x, _ in test_loader], dim=0)
    pix = raw_flat(images)
    rows: List[dict] = []

    for target_ord, t in enumerate(target_indices):
        t = int(t)
        y0 = int(labels[t])
        target_pred = int(preds[t])
        legal, sem_all = legal_mask_for_target(
            t, labels, ref_labels, ref_preds, ref_emb_test, semantic_delta
        )
        eligible = np.flatnonzero(legal)
        if len(eligible) < args.pool:
            raise AssertionError("Feasibility preselection invariant broken")

        pool_rng = np.random.default_rng(POOL_SELECTION_SEED_BASE + t * 23)
        pool_idx = pool_rng.choice(eligible, size=args.pool, replace=False)
        distances = l2_distances(emb[t], emb[pool_idx])
        sem_pool = sem_all[pool_idx]
        pool_pix = pix[pool_idx]
        p_order = pixel_rank(pix[t], pool_pix)
        s_order = np.argsort(sem_pool, kind="stable")
        graph = knn_graph(pool_pix)

        for sim_seed in SIM_SEEDS:
            for budget in BUDGETS:
                base_seed = int(sim_seed * 1_000_003 + t * 97 + budget * 13 + 193)
                q_rng = np.random.default_rng(base_seed)
                r_rng = np.random.default_rng(base_seed + 1)
                sa_rng = np.random.default_rng(base_seed + 2)

                qres = qco_min_find(distances, budget, q_rng)
                ridx, rbest, rq = random_min_find(distances, budget, r_rng)
                q = min(int(budget), len(distances))
                psel = p_order[:q]
                plocal = int(psel[np.argmin(distances[psel])])
                ssel = s_order[:q]
                slocal = int(ssel[np.argmin(distances[ssel])])
                saidx, sabest, saq = simulated_annealing(distances, graph, budget, sa_rng)

                selected = {
                    "dh_bbht": (qres.best_index, qres.best_distance, qres.strict_queries),
                    "random": (ridx, rbest, rq),
                    "pixel_nn": (plocal, float(distances[plocal]), q),
                    "semantic_boundary": (slocal, float(distances[slocal]), q),
                    "simulated_annealing": (saidx, sabest, saq),
                }
                row = {
                    "target_order": target_ord,
                    "target_test_index": t,
                    "target_label": y0,
                    "target_pred": target_pred,
                    "budget": budget,
                    "sim_seed": sim_seed,
                    "semantic_delta": semantic_delta,
                    "collision_epsilon": collision_eps,
                    "eligible_semantic_candidates": int(len(eligible)),
                    "pool_size": args.pool,
                }
                for m, (local_idx, best_d, q_used) in selected.items():
                    local_idx = int(local_idx)
                    global_idx = int(pool_idx[local_idx])
                    pred_a = int(preds[global_idx])
                    truth_a = int(labels[global_idx])
                    deep = bool(float(best_d) <= collision_eps)
                    hard = bool(pred_a == target_pred)
                    success = bool(deep and hard)
                    row.update({
                        f"{m}_best_distance": float(best_d),
                        f"{m}_queries": int(q_used),
                        f"{m}_candidate_test_index": global_idx,
                        f"{m}_candidate_truth": truth_a,
                        f"{m}_candidate_pred": pred_a,
                        f"{m}_candidate_semantic_distance": float(sem_pool[local_idx]),
                        f"{m}_deep_collision": int(deep),
                        f"{m}_hard_label_collision": int(hard),
                        f"{m}_attack_success": int(success),
                    })
                rows.append(row)

    with (out / "trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    summary = summarize(rows)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    protocol = {
        "name": "QDSA-Experiment-2-v2-Attack-Effect",
        "attack_success": "semantic-legal second preimage AND victim deep distance <= epsilon_h AND victim hard label equals target hard label",
        "semantic_delta": semantic_delta,
        "collision_epsilon": collision_eps,
        "collision_epsilon_quantile": EPSILON_QUANTILE,
        "victim_accuracy": victim_acc,
        "semantic_reference_accuracy": ref_acc,
        "targets": int(len(target_indices)),
        "target_class_balance": {str(c): int(np.sum(labels[target_indices] == c)) for c in range(10)},
        "pool_size": args.pool,
        "budgets": list(BUDGETS),
        "methods": ["dh_bbht", "random", "pixel_nn", "semantic_boundary", "simulated_annealing"],
        "mean_eligible_candidates": float(np.mean(list(feasible_counts.values()))),
        "min_eligible_candidates": int(np.min(list(feasible_counts.values()))),
        "rejected_for_capacity_before_quotas_filled": int(rejected_capacity),
        "target_selection_bias_control": "feasibility only; victim collision distances and attack outcomes are never consulted",
        "quantum_novelty_claim": "none; dh_bbht is a standard quantum-search baseline",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"protocol": protocol, "summary": summary}, indent=2), encoding="utf-8")
    print("SUMMARY_JSON " + json.dumps({"protocol": protocol, "summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
