#!/usr/bin/env python3
"""Experiment 1C: QDSA strict semantic second-preimage gate on MNIST.

QDSA scientific predicate
-------------------------
Given target x_t, search only candidates x_a satisfying

    x_a != x_t,
    D_sem(x_a, x_t) >= Delta,

and minimize the victim penultimate-representation distance

    ||h_v(x_a) - h_v(x_t)||_2.

D_sem is NOT the victim representation.  It is defined by a separately trained,
architecturally distinct frozen semantic reference encoder.  Delta is calibrated
once from cross-class pairs in a fixed training calibration subset before target
selection.  This prevents the public-pixel-nearest control from winning merely
by selecting visually near cross-label samples.

Strong controls
---------------
- random
- pixel_nn: raw-pixel nearest among semantically legal candidates
- semantic_boundary: closest candidate in the independent semantic embedding
  subject to D_sem >= Delta (a stronger public-geometry control)
- simulated_annealing on a public-pixel kNN graph
- dh_bbht: statevector Durr-Hoyer/BBHT minimum finding with strict oracle cost

Inference is target-level: simulator seeds are averaged within each target.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from exp1_mnist_qco import (
    BUDGETS,
    SIM_SEEDS,
    EPS,
    SmallCNN,
    embed_dataset,
    l2_distances,
    qco_min_find,
    random_min_find,
    seed_all,
    train_victim,
)
from exp1b_mnist_strong_baselines import knn_graph, pixel_rank, simulated_annealing

CALIBRATION_N = 5000
CALIBRATION_PAIRS = 50000
DELTA_QUANTILE = 0.50


class SemanticMLP(nn.Module):
    """Independent semantic encoder; deliberately unlike the convolutional victim."""
    def __init__(self):
        super().__init__()
        self.f1 = nn.Linear(784, 256)
        self.f2 = nn.Linear(256, 64)
        self.cls = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor, return_feature: bool = False):
        z = x.flatten(1)
        z = F.relu(self.f1(z))
        h = F.relu(self.f2(z))
        logits = self.cls(h)
        if return_feature:
            return logits, F.normalize(h, p=2, dim=1)
        return logits


def raw_flat(images: torch.Tensor) -> np.ndarray:
    return images.numpy().reshape(len(images), -1).astype(np.float64)


def calibrate_delta(ref_emb: np.ndarray, labels: np.ndarray, seed: int = 2026080910) -> Tuple[float, int]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals: List[float] = []
    attempts = 0
    while len(vals) < CALIBRATION_PAIRS and attempts < CALIBRATION_PAIRS * 10:
        batch = min(10000, CALIBRATION_PAIRS - len(vals))
        a = rng.integers(0, n, size=batch)
        b = rng.integers(0, n, size=batch)
        keep = (a != b) & (labels[a] != labels[b])
        if np.any(keep):
            aa, bb = a[keep], b[keep]
            d = np.linalg.norm(ref_emb[aa] - ref_emb[bb], axis=1)
            vals.extend(d.tolist())
        attempts += batch
    arr = np.asarray(vals[:CALIBRATION_PAIRS], dtype=float)
    if len(arr) < 1000:
        raise RuntimeError("Insufficient calibration cross-class pairs")
    return float(np.quantile(arr, DELTA_QUANTILE)), int(len(arr))


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def target_level_summary(rows: List[dict]) -> List[dict]:
    methods = ["dh_bbht", "random", "pixel_nn", "semantic_boundary", "simulated_annealing"]
    result: List[dict] = []
    for budget in sorted({r["budget"] for r in rows}):
        cell = [r for r in rows if r["budget"] == budget]
        targets = sorted({r["target_test_index"] for r in cell})
        means: Dict[str, List[float]] = {m: [] for m in methods}
        opt: List[float] = []
        sem: List[float] = []
        for t in targets:
            g = [r for r in cell if r["target_test_index"] == t]
            opt.append(float(g[0]["global_best_distance"]))
            sem.append(float(g[0]["global_best_semantic_distance"]))
            for m in methods:
                means[m].append(float(np.mean([r[f"{m}_best_distance"] for r in g])))

        q = np.asarray(means["dh_bbht"], float)
        opt_arr = np.asarray(opt, float)
        row = {
            "budget": budget,
            "n_independent_targets": len(targets),
            "dh_bbht_mean": float(q.mean()),
            "dh_bbht_median": float(np.median(q)),
            "dh_bbht_mean_opt_gap": float(np.mean(q - opt_arr)),
            "mean_semantic_distance_of_global_collision": float(np.mean(sem)),
        }
        for m in ["random", "pixel_nn", "semantic_boundary", "simulated_annealing"]:
            c = np.asarray(means[m], float)
            w = int(np.sum(q < c - EPS))
            l = int(np.sum(c < q - EPS))
            ties = int(len(q) - w - l)
            row.update({
                f"{m}_mean": float(c.mean()),
                f"dh_vs_{m}_mean_reduction_pct": float(100.0 * (c.mean() - q.mean()) / max(c.mean(), 1e-12)),
                f"dh_vs_{m}_target_wins": w,
                f"{m}_target_wins": l,
                f"dh_vs_{m}_ties": ties,
                f"dh_vs_{m}_sign_p": exact_sign_p(w, l),
                f"{m}_mean_opt_gap": float(np.mean(c - opt_arr)),
            })
        result.append(row)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp1c_out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-seed", type=int, default=20260809)
    ap.add_argument("--targets", type=int, default=20)
    ap.add_argument("--pool", type=int, default=512)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    tf = transforms.ToTensor()
    train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)

    # Train victim.
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

    # Train independent semantic reference with a disjoint architecture and seed.
    ref_seed = args.train_seed + 101
    seed_all(ref_seed)
    rg = torch.Generator().manual_seed(ref_seed)
    ref_train = DataLoader(train_ds, batch_size=256, shuffle=True, generator=rg, num_workers=2)
    ref = SemanticMLP()
    train_victim(ref, ref_train, device, args.epochs)
    ref_labels, ref_preds, ref_emb_test = embed_dataset(ref, test_loader, device)
    ref_acc = float(np.mean(ref_labels == ref_preds))
    if ref_acc < 0.95:
        raise RuntimeError(f"Semantic reference accuracy gate failed: {ref_acc:.4f} < 0.95")

    # Frozen Delta from a training calibration subset, independent of test targets.
    cal_ds = Subset(train_ds, range(CALIBRATION_N))
    cal_loader = DataLoader(cal_ds, batch_size=512, shuffle=False, num_workers=2)
    cal_labels, cal_preds, cal_emb = embed_dataset(ref, cal_loader, device)
    good = cal_labels == cal_preds
    delta, n_cal_pairs = calibrate_delta(cal_emb[good], cal_labels[good])
    print(f"victim_acc={victim_acc:.5f} ref_acc={ref_acc:.5f} semantic_delta={delta:.6f}", flush=True)

    images = torch.cat([x for x, _ in test_loader], dim=0)
    pix = raw_flat(images)
    correct_targets = np.flatnonzero((labels == preds) & (ref_labels == ref_preds))
    target_rng = np.random.default_rng(2026080903)
    target_indices = target_rng.choice(correct_targets, size=min(args.targets, len(correct_targets)), replace=False)
    all_indices = np.arange(len(test_ds))

    rows: List[dict] = []
    eligible_counts: List[int] = []
    for target_ord, t in enumerate(target_indices):
        y0 = int(labels[t])
        sem_all = l2_distances(ref_emb_test[t], ref_emb_test)
        legal = (
            (all_indices != int(t))
            & (labels != y0)
            & (ref_labels == ref_preds)
            & (sem_all >= delta)
        )
        eligible = all_indices[legal]
        eligible_counts.append(int(len(eligible)))
        if len(eligible) < args.pool:
            raise RuntimeError(f"Target {t}: only {len(eligible)} legal candidates < pool {args.pool}")

        pool_rng = np.random.default_rng(70_000 + int(t) * 19)
        pool_idx = pool_rng.choice(eligible, size=args.pool, replace=False)
        distances = l2_distances(emb[t], emb[pool_idx])
        sem_pool = sem_all[pool_idx]
        global_local = int(np.argmin(distances))
        global_best = float(distances[global_local])
        global_sem = float(sem_pool[global_local])

        pool_pix = pix[pool_idx]
        p_order = pixel_rank(pix[t], pool_pix)
        # Strong semantic public control: pick candidates just beyond the legal boundary.
        s_order = np.argsort(sem_pool, kind="stable")
        graph = knn_graph(pool_pix)

        for sim_seed in SIM_SEEDS:
            for budget in BUDGETS:
                base_seed = int(sim_seed * 1_000_003 + int(t) * 97 + budget * 13 + 91)
                q_rng = np.random.default_rng(base_seed)
                r_rng = np.random.default_rng(base_seed + 1)
                sa_rng = np.random.default_rng(base_seed + 2)

                qres = qco_min_find(distances, budget, q_rng)
                ridx, rbest, rq = random_min_find(distances, budget, r_rng)
                q = min(int(budget), len(distances))

                psel = p_order[:q]
                plocal = int(psel[np.argmin(distances[psel])])
                pbest = float(distances[plocal])

                ssel = s_order[:q]
                slocal = int(ssel[np.argmin(distances[ssel])])
                sbest = float(distances[slocal])

                saidx, sabest, saq = simulated_annealing(distances, graph, budget, sa_rng)

                rows.append({
                    "target_order": target_ord,
                    "target_test_index": int(t),
                    "target_label": y0,
                    "eligible_semantic_candidates": int(len(eligible)),
                    "semantic_delta": delta,
                    "pool_size": args.pool,
                    "budget": budget,
                    "sim_seed": sim_seed,
                    "global_best_distance": global_best,
                    "global_best_semantic_distance": global_sem,
                    "dh_bbht_best_distance": qres.best_distance,
                    "dh_bbht_best_semantic_distance": float(sem_pool[qres.best_index]),
                    "dh_bbht_strict_queries": qres.strict_queries,
                    "dh_bbht_logical_queries": qres.logical_queries,
                    "random_best_distance": rbest,
                    "random_best_semantic_distance": float(sem_pool[ridx]),
                    "random_queries": rq,
                    "pixel_nn_best_distance": pbest,
                    "pixel_nn_best_semantic_distance": float(sem_pool[plocal]),
                    "pixel_nn_queries": q,
                    "semantic_boundary_best_distance": sbest,
                    "semantic_boundary_best_semantic_distance": float(sem_pool[slocal]),
                    "semantic_boundary_queries": q,
                    "simulated_annealing_best_distance": sabest,
                    "simulated_annealing_best_semantic_distance": float(sem_pool[saidx]),
                    "simulated_annealing_queries": saq,
                })

    with (out / "trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    summary = target_level_summary(rows)
    with (out / "target_level_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    protocol = {
        "name": "QDSA-Experiment-1C-Strict-Semantic-Gate",
        "scientific_predicate": "x_a != x_t; D_sem(x_a,x_t) >= Delta; minimize victim penultimate L2",
        "dataset": "MNIST",
        "victim": "SmallCNN conv32-conv64-fc128",
        "victim_accuracy": victim_acc,
        "semantic_reference": "independent MLP 784-256-64",
        "semantic_reference_accuracy": ref_acc,
        "semantic_delta_definition": f"q={DELTA_QUANTILE} of cross-class reference-embedding distances on fixed training calibration subset",
        "semantic_delta": delta,
        "semantic_calibration_n": CALIBRATION_N,
        "semantic_calibration_pairs": n_cal_pairs,
        "mean_eligible_candidates": float(np.mean(eligible_counts)),
        "min_eligible_candidates": int(np.min(eligible_counts)),
        "targets": int(len(target_indices)),
        "pool_size": args.pool,
        "budgets": list(BUDGETS),
        "methods": ["dh_bbht", "random", "pixel_nn", "semantic_boundary", "simulated_annealing"],
        "independent_unit": "target; simulator seeds averaged within target",
        "attack_success_measured": False,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"protocol": protocol, "summary": summary}, indent=2), encoding="utf-8")
    print("SUMMARY_JSON " + json.dumps({"protocol": protocol, "summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
