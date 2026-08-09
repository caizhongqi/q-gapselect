#!/usr/bin/env python3
"""Experiment 2: QDSA attack effect / hard-label impersonation on MNIST.

A search result is a successful semantic second-preimage attack only if:

  1) x_a is legal by the strict semantic gate D_sem(x_a,x_t) >= Delta and
     has a different ground-truth class (guaranteed by candidate construction),
  2) victim representation distance d_h(x_a,x_t) <= epsilon_h, where epsilon_h
     is calibrated BEFORE test targets from the lower 5% tail of legal
     cross-semantic training pairs, and
  3) victim hard label f(x_a) == f(x_t).

Because targets are required to be correctly classified and candidate truth labels
differ from the target truth label, condition (3) is a genuine target-class
impersonation rather than a same-class nearest neighbor.

This experiment still treats Durr-Hoyer/BBHT as a standard quantum-search
baseline, not as a novel quantum primitive.
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
    BUDGETS,
    SIM_SEEDS,
    SmallCNN,
    embed_dataset,
    l2_distances,
    qco_min_find,
    random_min_find,
    seed_all,
    train_victim,
)
from exp1b_mnist_strong_baselines import knn_graph, pixel_rank, simulated_annealing
from exp1c_qdsa_semantic_gate import (
    CALIBRATION_N,
    CALIBRATION_PAIRS,
    SemanticMLP,
    calibrate_delta,
    raw_flat,
)

EPSILON_QUANTILE = 0.05
COLLISION_CALIBRATION_PAIRS = 50000


def calibrate_collision_epsilon(
    victim_emb: np.ndarray,
    ref_emb: np.ndarray,
    labels: np.ndarray,
    victim_correct: np.ndarray,
    ref_correct: np.ndarray,
    semantic_delta: float,
    seed: int = 2026080911,
) -> Tuple[float, int]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals: List[float] = []
    attempts = 0
    while len(vals) < COLLISION_CALIBRATION_PAIRS and attempts < COLLISION_CALIBRATION_PAIRS * 30:
        batch = min(10000, COLLISION_CALIBRATION_PAIRS - len(vals))
        a = rng.integers(0, n, size=batch)
        b = rng.integers(0, n, size=batch)
        basic = (
            (a != b)
            & (labels[a] != labels[b])
            & victim_correct[a]
            & victim_correct[b]
            & ref_correct[a]
            & ref_correct[b]
        )
        if np.any(basic):
            aa, bb = a[basic], b[basic]
            sem_d = np.linalg.norm(ref_emb[aa] - ref_emb[bb], axis=1)
            legal = sem_d >= semantic_delta
            if np.any(legal):
                aaa, bbb = aa[legal], bb[legal]
                d = np.linalg.norm(victim_emb[aaa] - victim_emb[bbb], axis=1)
                vals.extend(d.tolist())
        attempts += batch
    arr = np.asarray(vals[:COLLISION_CALIBRATION_PAIRS], dtype=float)
    if len(arr) < 5000:
        raise RuntimeError(f"Only {len(arr)} legal collision-calibration pairs")
    return float(np.quantile(arr, EPSILON_QUANTILE)), int(len(arr))


def summarize(rows: List[dict]) -> List[dict]:
    methods = ["dh_bbht", "random", "pixel_nn", "semantic_boundary", "simulated_annealing"]
    out: List[dict] = []
    for budget in sorted({r["budget"] for r in rows}):
        cell = [r for r in rows if r["budget"] == budget]
        targets = sorted({r["target_test_index"] for r in cell})
        row: Dict[str, float] = {"budget": budget, "n_independent_targets": len(targets)}
        for m in methods:
            asr_target = []
            deep_target = []
            label_target = []
            dist_target = []
            query_target = []
            for t in targets:
                g = [r for r in cell if r["target_test_index"] == t]
                asr_target.append(float(np.mean([r[f"{m}_attack_success"] for r in g])))
                deep_target.append(float(np.mean([r[f"{m}_deep_collision"] for r in g])))
                label_target.append(float(np.mean([r[f"{m}_hard_label_collision"] for r in g])))
                dist_target.append(float(np.mean([r[f"{m}_best_distance"] for r in g])))
                query_target.append(float(np.mean([r[f"{m}_queries"] for r in g])))
            row.update({
                f"{m}_asr": float(np.mean(asr_target)),
                f"{m}_targets_with_nonzero_success": int(np.sum(np.asarray(asr_target) > 0.0)),
                f"{m}_deep_collision_rate": float(np.mean(deep_target)),
                f"{m}_hard_label_collision_rate": float(np.mean(label_target)),
                f"{m}_mean_best_distance": float(np.mean(dist_target)),
                f"{m}_mean_queries": float(np.mean(query_target)),
            })
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp2_out")
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

    seed_all(args.train_seed)
    vg = torch.Generator().manual_seed(args.train_seed)
    victim_train = DataLoader(train_ds, batch_size=256, shuffle=True, generator=vg, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)
    victim = SmallCNN()
    train_victim(victim, victim_train, device, args.epochs)
    labels, preds, emb = embed_dataset(victim, test_loader, device)
    victim_acc = float(np.mean(labels == preds))
    if victim_acc < 0.97:
        raise RuntimeError(f"Victim accuracy gate failed: {victim_acc:.4f} < 0.97")

    ref_seed = args.train_seed + 101
    seed_all(ref_seed)
    rg = torch.Generator().manual_seed(ref_seed)
    ref_train = DataLoader(train_ds, batch_size=256, shuffle=True, generator=rg, num_workers=2)
    ref = SemanticMLP()
    train_victim(ref, ref_train, device, args.epochs)
    ref_labels, ref_preds, ref_emb_test = embed_dataset(ref, test_loader, device)
    ref_acc = float(np.mean(ref_labels == ref_preds))
    if ref_acc < 0.95:
        raise RuntimeError(f"Reference accuracy gate failed: {ref_acc:.4f} < 0.95")

    # Independent training calibration subset for BOTH semantic Delta and collision epsilon_h.
    cal_ds = Subset(train_ds, range(CALIBRATION_N))
    cal_loader = DataLoader(cal_ds, batch_size=512, shuffle=False, num_workers=2)
    vcal_labels, vcal_preds, vcal_emb = embed_dataset(victim, cal_loader, device)
    rcal_labels, rcal_preds, rcal_emb = embed_dataset(ref, cal_loader, device)
    ref_good = rcal_labels == rcal_preds
    semantic_delta, n_sem_pairs = calibrate_delta(rcal_emb[ref_good], rcal_labels[ref_good])
    collision_eps, n_collision_pairs = calibrate_collision_epsilon(
        vcal_emb,
        rcal_emb,
        vcal_labels,
        vcal_labels == vcal_preds,
        rcal_labels == rcal_preds,
        semantic_delta,
    )
    print(
        f"victim_acc={victim_acc:.5f} ref_acc={ref_acc:.5f} "
        f"semantic_delta={semantic_delta:.6f} collision_eps={collision_eps:.6f}",
        flush=True,
    )

    images = torch.cat([x for x, _ in test_loader], dim=0)
    pix = raw_flat(images)
    all_indices = np.arange(len(test_ds))
    correct_targets = np.flatnonzero((labels == preds) & (ref_labels == ref_preds))
    target_rng = np.random.default_rng(2026080912)
    target_indices = target_rng.choice(correct_targets, size=min(args.targets, len(correct_targets)), replace=False)

    rows: List[dict] = []
    eligible_counts: List[int] = []
    for target_ord, t in enumerate(target_indices):
        y0 = int(labels[t])
        target_pred = int(preds[t])
        sem_all = l2_distances(ref_emb_test[t], ref_emb_test)
        legal = (
            (all_indices != int(t))
            & (labels != y0)
            & (ref_labels == ref_preds)
            & (sem_all >= semantic_delta)
        )
        eligible = all_indices[legal]
        eligible_counts.append(int(len(eligible)))
        if len(eligible) < args.pool:
            raise RuntimeError(f"Target {t}: only {len(eligible)} legal candidates")
        pool_rng = np.random.default_rng(90_000 + int(t) * 23)
        pool_idx = pool_rng.choice(eligible, size=args.pool, replace=False)
        distances = l2_distances(emb[t], emb[pool_idx])
        sem_pool = sem_all[pool_idx]
        pool_pix = pix[pool_idx]
        p_order = pixel_rank(pix[t], pool_pix)
        s_order = np.argsort(sem_pool, kind="stable")
        graph = knn_graph(pool_pix)

        for sim_seed in SIM_SEEDS:
            for budget in BUDGETS:
                base_seed = int(sim_seed * 1_000_003 + int(t) * 97 + budget * 13 + 193)
                q_rng = np.random.default_rng(base_seed)
                r_rng = np.random.default_rng(base_seed + 1)
                sa_rng = np.random.default_rng(base_seed + 2)

                qres = qco_min_find(distances, budget, q_rng)
                ridx, rbest, rq = random_min_find(distances, budget, r_rng)
                q = min(int(budget), len(distances))
                psel = p_order[:q]; plocal = int(psel[np.argmin(distances[psel])])
                ssel = s_order[:q]; slocal = int(ssel[np.argmin(distances[ssel])])
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
                    "target_test_index": int(t),
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
                    global_idx = int(pool_idx[int(local_idx)])
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
                        f"{m}_candidate_semantic_distance": float(sem_pool[int(local_idx)]),
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
        "name": "QDSA-Experiment-2-Attack-Effect",
        "attack_success": "semantic-legal second preimage AND victim deep distance <= epsilon_h AND victim hard label equals target hard label",
        "semantic_delta": semantic_delta,
        "semantic_delta_calibration": "median cross-class independent-reference distance on fixed training calibration subset",
        "collision_epsilon": collision_eps,
        "collision_epsilon_quantile": EPSILON_QUANTILE,
        "collision_epsilon_calibration": "lower-tail victim-distance quantile among legal cross-semantic pairs on fixed training calibration subset",
        "victim_accuracy": victim_acc,
        "semantic_reference_accuracy": ref_acc,
        "targets": int(len(target_indices)),
        "pool_size": args.pool,
        "budgets": list(BUDGETS),
        "methods": ["dh_bbht", "random", "pixel_nn", "semantic_boundary", "simulated_annealing"],
        "mean_eligible_candidates": float(np.mean(eligible_counts)),
        "min_eligible_candidates": int(np.min(eligible_counts)),
        "semantic_calibration_pairs": n_sem_pairs,
        "collision_calibration_pairs": n_collision_pairs,
        "quantum_novelty_claim": "none; dh_bbht is a standard quantum-search baseline",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"protocol": protocol, "summary": summary}, indent=2), encoding="utf-8")
    print("SUMMARY_JSON " + json.dumps({"protocol": protocol, "summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
