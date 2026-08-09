#!/usr/bin/env python3
"""Experiment 1B: strong-baseline audit for MNIST representation-collision search.

This extends Experiment 1 before any attack-success experiment.  The quantum
routine is deliberately labelled by its actual algorithmic ingredients
(Durr-Hoyer adaptive minimum finding with BBHT/Grover subroutines) rather than
claiming a new quantum primitive.

Fair search domain
------------------
All query-limited methods search the same fixed pool of 512 public MNIST test
candidates.  The victim objective is L2 distance between L2-normalized
penultimate representations.  A victim-oracle query reveals the objective for
one selected candidate; public input pixels may be preprocessed for free.

Methods
-------
1. random: uniform sampling without replacement.
2. pixel_nn: strongest simple public-geometry control; rank candidates by raw
   pixel L2 distance to x0, then query the first q.
3. simulated_annealing: black-box search on a public-pixel kNN graph.  Each new
   candidate objective evaluation costs one victim query.
4. dh_bbht: statevector-simulated Durr-Hoyer/BBHT minimum finding from
   Experiment 1, with conservative compute+uncompute oracle accounting.

The headline inferential unit is the TARGET, not repeated simulator seeds.
Stochastic seeds are averaged within target before paired sign tests.
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
from torch.utils.data import DataLoader
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

KNN_K = 16
SA_RESTART_PROB = 0.08
SA_TEMP_START = 0.12
SA_TEMP_END = 0.008


def raw_flat(images: torch.Tensor) -> np.ndarray:
    return images.numpy().reshape(len(images), -1).astype(np.float64)


def pixel_rank(target: np.ndarray, pool: np.ndarray) -> np.ndarray:
    d2 = np.sum((pool - target[None, :]) ** 2, axis=1)
    return np.argsort(d2, kind="stable")


def knn_graph(pool: np.ndarray, k: int = KNN_K) -> np.ndarray:
    # Public-input preprocessing: no victim calls. N=512, D=784.
    norms = np.sum(pool * pool, axis=1, keepdims=True)
    d2 = norms + norms.T - 2.0 * (pool @ pool.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, np.inf)
    kk = min(k, len(pool) - 1)
    nbr = np.argpartition(d2, kth=kk - 1, axis=1)[:, :kk]
    # Sort each row for deterministic tie handling.
    row = np.arange(len(pool))[:, None]
    ord_local = np.argsort(d2[row, nbr], axis=1, kind="stable")
    return nbr[row, ord_local]


def simulated_annealing(
    distances: np.ndarray,
    graph: np.ndarray,
    budget: int,
    rng: np.random.Generator,
) -> Tuple[int, float, int]:
    n = len(distances)
    budget = min(int(budget), n)
    current = int(rng.integers(n))
    current_d = float(distances[current])
    best_i, best_d = current, current_d
    seen = {current}
    queries = 1

    while queries < budget:
        frac = queries / max(1, budget - 1)
        temp = SA_TEMP_START * ((SA_TEMP_END / SA_TEMP_START) ** frac)

        # Mostly exploit public-pixel neighborhoods, occasionally restart.
        if rng.random() < SA_RESTART_PROB:
            unseen = np.array([i for i in range(n) if i not in seen], dtype=int)
            if len(unseen) == 0:
                break
            cand = int(rng.choice(unseen))
        else:
            local = [int(j) for j in graph[current] if int(j) not in seen]
            if local:
                cand = int(rng.choice(local))
            else:
                unseen = np.array([i for i in range(n) if i not in seen], dtype=int)
                if len(unseen) == 0:
                    break
                cand = int(rng.choice(unseen))

        seen.add(cand)
        d = float(distances[cand])
        queries += 1
        if d < best_d - EPS:
            best_i, best_d = cand, d
        delta = d - current_d
        if delta <= 0.0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            current, current_d = cand, d

    return best_i, best_d, queries


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def target_level_summary(rows: List[dict]) -> List[dict]:
    methods = ["dh_bbht", "random", "pixel_nn", "simulated_annealing"]
    result = []
    for condition in sorted({r["condition"] for r in rows}):
        for budget in sorted({r["budget"] for r in rows}):
            cell = [r for r in rows if r["condition"] == condition and r["budget"] == budget]
            targets = sorted({r["target_test_index"] for r in cell})
            target_means: Dict[str, List[float]] = {m: [] for m in methods}
            target_opt: List[float] = []
            for t in targets:
                g = [r for r in cell if r["target_test_index"] == t]
                target_opt.append(float(g[0]["global_best_distance"]))
                for m in methods:
                    target_means[m].append(float(np.mean([r[f"{m}_best_distance"] for r in g])))

            q = np.asarray(target_means["dh_bbht"], float)
            opt = np.asarray(target_opt, float)
            row = {
                "condition": condition,
                "budget": budget,
                "n_independent_targets": len(targets),
                "dh_bbht_mean": float(q.mean()),
                "dh_bbht_median": float(np.median(q)),
                "dh_bbht_mean_opt_gap": float(np.mean(q - opt)),
            }
            for m in ["random", "pixel_nn", "simulated_annealing"]:
                c = np.asarray(target_means[m], float)
                wins = int(np.sum(q < c - EPS))
                losses = int(np.sum(c < q - EPS))
                ties = int(len(q) - wins - losses)
                row.update({
                    f"{m}_mean": float(c.mean()),
                    f"dh_vs_{m}_mean_reduction_pct": float(100.0 * (c.mean() - q.mean()) / max(c.mean(), 1e-12)),
                    f"dh_vs_{m}_target_wins": wins,
                    f"{m}_target_wins": losses,
                    f"dh_vs_{m}_ties": ties,
                    f"dh_vs_{m}_sign_p": exact_sign_p(wins, losses),
                    f"{m}_mean_opt_gap": float(np.mean(c - opt)),
                })
            result.append(row)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp1b_out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-seed", type=int, default=20260809)
    ap.add_argument("--targets", type=int, default=20)
    ap.add_argument("--pool", type=int, default=512)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.train_seed)
    device = torch.device("cpu")

    tf = transforms.ToTensor()
    train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(args.train_seed)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, generator=g, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)

    model = SmallCNN()
    train_victim(model, train_loader, device, args.epochs)
    labels, preds, emb = embed_dataset(model, test_loader, device)
    images = torch.cat([x for x, _ in test_loader], dim=0)
    pix = raw_flat(images)
    test_acc = float(np.mean(labels == preds))
    print(f"test_accuracy={test_acc:.5f}", flush=True)
    if test_acc < 0.97:
        raise RuntimeError(f"Victim accuracy gate failed: {test_acc:.4f} < 0.97")

    correct = np.flatnonzero(labels == preds)
    target_rng = np.random.default_rng(2026080903)
    target_indices = target_rng.choice(correct, size=min(args.targets, len(correct)), replace=False)
    all_indices = np.arange(len(test_ds))

    rows: List[dict] = []
    for target_ord, t in enumerate(target_indices):
        h0 = emb[t]
        y0 = int(labels[t])
        for condition in ("all_candidates", "cross_semantic"):
            mask = all_indices != int(t)
            if condition == "cross_semantic":
                mask &= labels != y0
            eligible = all_indices[mask]
            pool_rng = np.random.default_rng(50_000 + int(t) * 17 + (0 if condition == "all_candidates" else 1))
            pool_idx = pool_rng.choice(eligible, size=args.pool, replace=False)
            distances = l2_distances(h0, emb[pool_idx])
            global_best = float(np.min(distances))

            pool_pix = pix[pool_idx]
            p_order = pixel_rank(pix[t], pool_pix)
            graph = knn_graph(pool_pix)

            for sim_seed in SIM_SEEDS:
                for budget in BUDGETS:
                    base_seed = int(sim_seed * 1_000_003 + int(t) * 97 + budget * 13 + (0 if condition == "all_candidates" else 7))
                    q_rng = np.random.default_rng(base_seed)
                    r_rng = np.random.default_rng(base_seed + 1)
                    sa_rng = np.random.default_rng(base_seed + 2)

                    qres = qco_min_find(distances, budget, q_rng)
                    ridx, rbest, rq = random_min_find(distances, budget, r_rng)
                    pq = min(int(budget), len(distances))
                    psel = p_order[:pq]
                    plocal = int(psel[np.argmin(distances[psel])])
                    pbest = float(distances[plocal])
                    saidx, sabest, saq = simulated_annealing(distances, graph, budget, sa_rng)

                    rows.append({
                        "target_order": target_ord,
                        "target_test_index": int(t),
                        "target_label": y0,
                        "condition": condition,
                        "pool_size": args.pool,
                        "budget": budget,
                        "sim_seed": sim_seed,
                        "global_best_distance": global_best,
                        "dh_bbht_best_distance": qres.best_distance,
                        "dh_bbht_best_test_index": int(pool_idx[qres.best_index]),
                        "dh_bbht_strict_queries": qres.strict_queries,
                        "dh_bbht_logical_queries": qres.logical_queries,
                        "random_best_distance": rbest,
                        "random_best_test_index": int(pool_idx[ridx]),
                        "random_queries": rq,
                        "pixel_nn_best_distance": pbest,
                        "pixel_nn_best_test_index": int(pool_idx[plocal]),
                        "pixel_nn_queries": pq,
                        "simulated_annealing_best_distance": sabest,
                        "simulated_annealing_best_test_index": int(pool_idx[saidx]),
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
        "name": "Experiment-1B-Strong-Baseline-Audit",
        "dataset": "MNIST",
        "victim": "SmallCNN conv32-conv64-fc128",
        "test_accuracy": test_acc,
        "targets": int(len(target_indices)),
        "pool_size": args.pool,
        "budgets": list(BUDGETS),
        "methods": ["dh_bbht", "random", "pixel_nn", "simulated_annealing"],
        "independent_unit": "target; simulator seeds averaged within target",
        "public_preprocessing_free": ["pixel distances", "pixel kNN graph"],
        "dh_bbht_query_accounting": "2 victim calls per coherent phase-mark iteration + 1 measured-candidate verification",
        "attack_success_measured": False,
        "white_box_gradient_pgd": "deferred to continuous attack stage because it searches a different domain and is not query-matched to the public candidate pool",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"protocol": protocol, "summary": summary}, indent=2), encoding="utf-8")
    print("SUMMARY_JSON " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
