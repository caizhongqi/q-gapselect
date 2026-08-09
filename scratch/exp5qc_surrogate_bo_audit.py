#!/usr/bin/env python3
"""Experiment 5QC: learned-surrogate classical audit for QDSA seed search.

Purpose
-------
5QB showed that the large-N fixed-budget DH/BBHT crossover survives fused public
geometry and an adaptive graph search.  This experiment gives the classical side
a substantially stronger capability: it may learn the expensive victim collision
objective online from already queried candidates and use that surrogate to choose
future queries.

Classical learned baseline: ExtraTrees-LCB
-----------------------------------------
For every public candidate we expose, for free:
  - the independent semantic-reference embedding (64-D),
  - a deterministic 7x7 average-pooled pixel descriptor (49-D),
  - semantic and raw-pixel distances to the target,
  - normalized pixel, semantic, min-rank, and Borda ranks.
The victim raw-penultimate collision distance remains expensive and is revealed
only when a candidate is queried.

The algorithm queries 32 hybrid-min warm-start candidates, then repeatedly fits
an ExtraTrees ensemble to queried victim distances.  A batch contains mostly the
lowest lower-confidence-bound (mean - 1.5*tree_std) candidates plus a smaller
hybrid-rank exploration tranche.  The full query path is cumulative, so B=64,
128,256,512 are checkpoints of the same online optimizer rather than independent
retuned runs.

Quantum baseline
----------------
Fixed-accounting DH/BBHT from 5Q/5QB: every run consumes the full strict budget;
there is no free empty-marked-set or global-optimum detection.

Important scope
---------------
This is still an oracle-model mechanism study.  DH/BBHT is a standard quantum
minimum-finding ingredient, not our novelty, and coherent victim-representation
oracle access is an explicit strong assumption.
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
from sklearn.ensemble import ExtraTreesRegressor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from exp1_mnist_qco import SmallCNN, embed_dataset, seed_all, train_victim
from exp1c_qdsa_semantic_gate import CALIBRATION_N, SemanticMLP, calibrate_delta, raw_flat
from exp2_qdsa_attack_effect_v2 import legal_mask_for_target, select_balanced_feasible_targets
from exp3a_certified_raw_collision import embed_raw_victim, raw_distances
from exp5q_query_scaling_fixed_accounting import qco_min_find_fixed_budget, quality_metrics
from exp5qb_strong_classical_structure_audit import (
    graph_best_first,
    knn_from_submatrix,
    master_semantic_distance_matrix,
    fusion_orders,
    pixel_order,
    semantic_order,
    best_from_order,
)

N_VALUES = (512, 1024, 2048)
BUDGETS = (64, 128, 256, 512)
QCO_SEEDS = (11, 22, 33, 44, 55)
RF_SEEDS = (701, 1701)
RF_TREES = 64
RF_KAPPA = 1.5
WARM_START = 32
BATCH_SIZE = 32
EXPLORE_PER_BATCH = 8
KNN_K = 16
EPS = 1e-12


def avgpool7_descriptor(flat_pixels: np.ndarray) -> np.ndarray:
    """28x28 -> 7x7 public descriptor by deterministic 4x4 block averaging."""
    x = flat_pixels.reshape(-1, 28, 28)
    return x.reshape(len(x), 7, 4, 7, 4).mean(axis=(2, 4)).reshape(len(x), 49)


def normalized_rank(order: np.ndarray) -> np.ndarray:
    n = len(order)
    rank = np.empty(n, dtype=np.float64)
    rank[order] = np.arange(n, dtype=np.float64)
    return rank / max(1.0, float(n - 1))


def public_features(
    target_pix: np.ndarray,
    pool_pix: np.ndarray,
    target_ref: np.ndarray,
    pool_ref: np.ndarray,
    p_order: np.ndarray,
    s_order: np.ndarray,
    fmin_order: np.ndarray,
    fborda_order: np.ndarray,
) -> np.ndarray:
    pooled = avgpool7_descriptor(pool_pix)
    sem_d = np.linalg.norm(pool_ref - target_ref[None, :], axis=1)
    pix_d = np.linalg.norm(pool_pix - target_pix[None, :], axis=1)
    feats = np.concatenate([
        pool_ref.astype(np.float64),
        pooled.astype(np.float64),
        sem_d[:, None],
        pix_d[:, None],
        normalized_rank(p_order)[:, None],
        normalized_rank(s_order)[:, None],
        normalized_rank(fmin_order)[:, None],
        normalized_rank(fborda_order)[:, None],
    ], axis=1)
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (feats - mu) / sd


def add_unique(order: List[int], proposed: np.ndarray, seen: np.ndarray, limit: int) -> None:
    for j in proposed:
        j = int(j)
        if not seen[j]:
            order.append(j)
            seen[j] = True
            if len(order) >= limit:
                return


def surrogate_lcb_path(
    distances: np.ndarray,
    features: np.ndarray,
    hybrid_order: np.ndarray,
    max_budget: int,
    rf_seed: int,
) -> Tuple[List[int], Dict[int, Tuple[int, float]]]:
    """Return cumulative queried order and best-so-far at budget checkpoints."""
    n = len(distances)
    max_budget = min(int(max_budget), n)
    queried_order: List[int] = []
    seen = np.zeros(n, dtype=bool)
    add_unique(queried_order, hybrid_order[:min(WARM_START, max_budget)], seen, max_budget)

    hybrid_cursor = min(WARM_START, n)
    round_id = 0
    while len(queried_order) < max_budget:
        qidx = np.asarray(queried_order, dtype=int)
        y = distances[qidx]
        unq = np.flatnonzero(~seen)
        if len(unq) == 0:
            break

        model = ExtraTreesRegressor(
            n_estimators=RF_TREES,
            criterion="squared_error",
            max_features="sqrt",
            min_samples_leaf=1,
            bootstrap=False,
            n_jobs=-1,
            random_state=rf_seed + round_id * 7919,
        )
        model.fit(features[qidx], y)

        # Tree-wise predictions provide an ensemble uncertainty proxy.
        tree_pred = np.stack([tree.predict(features[unq]) for tree in model.estimators_], axis=0)
        mean = tree_pred.mean(axis=0)
        std = tree_pred.std(axis=0)
        lcb = mean - RF_KAPPA * std
        exploit_local = np.argsort(lcb, kind="stable")
        exploit = unq[exploit_local]

        remaining = max_budget - len(queried_order)
        batch = min(BATCH_SIZE, remaining)
        explore = min(EXPLORE_PER_BATCH, batch // 2)
        exploit_n = batch - explore

        before = len(queried_order)
        add_unique(queried_order, exploit[:max(exploit_n * 4, exploit_n)], seen, before + exploit_n)

        # Explicit global public-geometry exploration keeps the learned surrogate
        # from becoming trapped by its current ensemble extrapolation.
        if explore > 0 and len(queried_order) < before + batch:
            candidates = []
            while hybrid_cursor < n and len(candidates) < explore * 8:
                j = int(hybrid_order[hybrid_cursor]); hybrid_cursor += 1
                if not seen[j]:
                    candidates.append(j)
            add_unique(queried_order, np.asarray(candidates, dtype=int), seen, before + batch)

        # If exploit/explore overlapped too heavily, fill deterministically from
        # the best remaining LCB candidates, then hybrid order.
        if len(queried_order) < before + batch:
            add_unique(queried_order, exploit, seen, before + batch)
        if len(queried_order) < before + batch:
            add_unique(queried_order, hybrid_order, seen, before + batch)
        round_id += 1

    checkpoints: Dict[int, Tuple[int, float]] = {}
    for b in BUDGETS:
        if b > max_budget:
            continue
        q = np.asarray(queried_order[:b], dtype=int)
        loc = int(q[np.argmin(distances[q])])
        checkpoints[b] = (loc, float(distances[loc]))
    return queried_order, checkpoints


def summarize(rows: List[dict]) -> List[dict]:
    out = []
    methods = sorted({r["method"] for r in rows})
    for n in N_VALUES:
        for b in [x for x in BUDGETS if x <= n]:
            for m in methods:
                cell = [r for r in rows if r["pool_size"] == n and r["budget"] == b and r["method"] == m]
                if not cell:
                    continue
                targets = sorted({r["target_test_index"] for r in cell})
                exact_t, near_t, regret_t, dist_t, rank_t = [], [], [], [], []
                for t in targets:
                    g = [r for r in cell if r["target_test_index"] == t]
                    exact_t.append(float(np.mean([r["exact_global_hit"] for r in g])))
                    near_t.append(float(np.mean([r["within_5pct_global"] for r in g])))
                    regret_t.append(float(np.mean([r["normalized_regret"] for r in g])))
                    dist_t.append(float(np.mean([r["best_distance"] for r in g])))
                    rank_t.append(float(np.mean([r["rank_fraction"] for r in g])))
                out.append({
                    "pool_size": n,
                    "budget": b,
                    "method": m,
                    "n_independent_targets": len(targets),
                    "exact_global_hit_rate": float(np.mean(exact_t)),
                    "within_5pct_global_rate": float(np.mean(near_t)),
                    "mean_normalized_regret": float(np.mean(regret_t)),
                    "mean_best_distance": float(np.mean(dist_t)),
                    "mean_rank_fraction": float(np.mean(rank_t)),
                    "mean_queries": float(np.mean([r["queries"] for r in cell])),
                })
    return out


def paired_bootstrap(rows: List[dict], n: int, b: int, classical: str, reps: int = 10000) -> Dict[str, float]:
    targets = sorted({r["target_test_index"] for r in rows})
    de, dr = [], []
    for t in targets:
        qg = [r for r in rows if r["target_test_index"] == t and r["pool_size"] == n and r["budget"] == b and r["method"] == "qco_fixed"]
        cg = [r for r in rows if r["target_test_index"] == t and r["pool_size"] == n and r["budget"] == b and r["method"] == classical]
        de.append(float(np.mean([r["exact_global_hit"] for r in qg]) - np.mean([r["exact_global_hit"] for r in cg])))
        dr.append(float(np.mean([r["normalized_regret"] for r in qg]) - np.mean([r["normalized_regret"] for r in cg])))
    de = np.asarray(de); dr = np.asarray(dr)
    rng = np.random.default_rng(2026080953 + n + b + sum(ord(x) for x in classical))
    be = np.empty(reps); br = np.empty(reps)
    for i in range(reps):
        ix = rng.integers(0, len(de), size=len(de))
        be[i] = de[ix].mean(); br[i] = dr[ix].mean()
    return {
        "pool_size": n,
        "budget": b,
        "classical_method": classical,
        "qco_minus_classical_exact_hit": float(de.mean()),
        "exact_bootstrap_lo": float(np.quantile(be, .025)),
        "exact_bootstrap_hi": float(np.quantile(be, .975)),
        "qco_minus_classical_normalized_regret": float(dr.mean()),
        "regret_bootstrap_lo": float(np.quantile(br, .025)),
        "regret_bootstrap_hi": float(np.quantile(br, .975)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp5qc_out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-seed", type=int, default=20260809)
    ap.add_argument("--targets", type=int, default=20)
    ap.add_argument("--max-pool", type=int, default=2048)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    tf = transforms.ToTensor()
    train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)

    seed_all(args.train_seed)
    vg = torch.Generator().manual_seed(args.train_seed)
    victim_loader = DataLoader(train_ds, batch_size=256, shuffle=True, generator=vg, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)
    victim = SmallCNN().to(device)
    train_victim(victim, victim_loader, device, args.epochs)
    labels, preds, _, raw_h = embed_raw_victim(victim, test_loader, device)
    victim_acc = float(np.mean(labels == preds))

    ref_seed = args.train_seed + 101
    seed_all(ref_seed)
    rg = torch.Generator().manual_seed(ref_seed)
    ref_loader = DataLoader(train_ds, batch_size=256, shuffle=True, generator=rg, num_workers=2)
    ref = SemanticMLP().to(device)
    train_victim(ref, ref_loader, device, args.epochs)
    ref_labels, ref_preds, ref_emb = embed_dataset(ref, test_loader, device)
    ref_acc = float(np.mean(ref_labels == ref_preds))
    if victim_acc < 0.97 or ref_acc < 0.95:
        raise RuntimeError("accuracy gate failed")

    cal_ds = Subset(train_ds, range(CALIBRATION_N))
    cal_loader = DataLoader(cal_ds, batch_size=512, shuffle=False, num_workers=2)
    rcal_labels, rcal_preds, rcal_emb = embed_dataset(ref, cal_loader, device)
    good = rcal_labels == rcal_preds
    delta, n_sem_pairs = calibrate_delta(rcal_emb[good], rcal_labels[good])
    targets, feasible_counts, rejected = select_balanced_feasible_targets(
        labels, preds, ref_labels, ref_preds, ref_emb,
        delta, args.max_pool, args.targets,
    )
    test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))], dim=0)
    pix = raw_flat(test_x)

    rows: List[dict] = []
    for target_ord, t0 in enumerate(targets):
        t = int(t0)
        legal, _ = legal_mask_for_target(t, labels, ref_labels, ref_preds, ref_emb, delta)
        eligible = np.flatnonzero(legal)
        master_rng = np.random.default_rng(12_000_000 + t * 31)
        master_pool = master_rng.choice(eligible, size=args.max_pool, replace=False)
        master_ref = ref_emb[master_pool]
        master_d2 = master_semantic_distance_matrix(master_ref)

        for n in N_VALUES:
            pool_idx = master_pool[:n]
            distances = raw_distances(raw_h[t], raw_h[pool_idx])
            pool_pix = pix[pool_idx]
            pool_ref = ref_emb[pool_idx]
            pord = pixel_order(pix[t], pool_pix)
            sord = semantic_order(ref_emb[t], pool_ref)
            fmin, fborda = fusion_orders(pord, sord)
            nbr, edge = knn_from_submatrix(master_d2, n, KNN_K)
            feats = public_features(pix[t], pool_pix, ref_emb[t], pool_ref, pord, sord, fmin, fborda)
            max_budget = min(max(BUDGETS), n)

            # Learned surrogate optimizer: two independent RF seeds, each one
            # cumulative online path shared by all budget checkpoints.
            for rf_seed in RF_SEEDS:
                _, cps = surrogate_lcb_path(distances, feats, fmin, max_budget, rf_seed)
                for b, (li, best) in cps.items():
                    qm = quality_metrics(best, distances)
                    rows.append({
                        "target_order": target_ord,
                        "target_test_index": t,
                        "target_class": int(labels[t]),
                        "pool_size": n,
                        "budget": b,
                        "sim_seed": rf_seed,
                        "method": "rf_lcb",
                        "queries": b,
                        "selected_local_index": li,
                        **qm,
                    })

            # QCO fixed-accounting simulations.
            for b in [x for x in BUDGETS if x <= n]:
                for ss in QCO_SEEDS:
                    base = int(ss * 1_000_003 + t * 97 + n * 17 + b * 13)
                    qres = qco_min_find_fixed_budget(distances, b, np.random.default_rng(base))
                    qm = quality_metrics(qres.best_distance, distances)
                    rows.append({
                        "target_order": target_ord,
                        "target_test_index": t,
                        "target_class": int(labels[t]),
                        "pool_size": n,
                        "budget": b,
                        "sim_seed": ss,
                        "method": "qco_fixed",
                        "queries": qres.strict_queries,
                        "selected_local_index": qres.best_index,
                        **qm,
                    })

                # Retain the strongest non-learned controls from 5QB.
                li, best, q = best_from_order(distances, fmin, b)
                qm = quality_metrics(best, distances)
                rows.append({
                    "target_order": target_ord, "target_test_index": t, "target_class": int(labels[t]),
                    "pool_size": n, "budget": b, "sim_seed": -1, "method": "hybrid_min_rank",
                    "queries": q, "selected_local_index": li, **qm,
                })
                li, best, q = graph_best_first(distances, fmin, nbr, edge, b)
                qm = quality_metrics(best, distances)
                rows.append({
                    "target_order": target_ord, "target_test_index": t, "target_class": int(labels[t]),
                    "pool_size": n, "budget": b, "sim_seed": -1, "method": "graph_best_first",
                    "queries": q, "selected_local_index": li, **qm,
                })
        print(f"target={t} done", flush=True)

    with (out / "trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = summarize(rows)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    contrasts = []
    for n in N_VALUES:
        for b in [x for x in BUDGETS if x <= n]:
            for c in ("rf_lcb", "hybrid_min_rank", "graph_best_first"):
                contrasts.append(paired_bootstrap(rows, n, b, c))
    with (out / "paired_contrasts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(contrasts[0].keys())); w.writeheader(); w.writerows(contrasts)

    result = {
        "name": "QDSA-Experiment-5QC-Learned-Surrogate-Classical-Audit",
        "role": "final strong classical audit before cross-victim replication",
        "victim_accuracy": victim_acc,
        "reference_accuracy": ref_acc,
        "semantic_delta": delta,
        "targets": int(len(targets)),
        "pool_sizes": list(N_VALUES),
        "budgets": list(BUDGETS),
        "methods": ["qco_fixed", "rf_lcb", "hybrid_min_rank", "graph_best_first"],
        "rf_seeds": list(RF_SEEDS),
        "rf_trees": RF_TREES,
        "rf_kappa": RF_KAPPA,
        "rf_warm_start": WARM_START,
        "rf_batch_size": BATCH_SIZE,
        "rf_explore_per_batch": EXPLORE_PER_BATCH,
        "rf_public_features": [
            "independent semantic embedding",
            "7x7 average-pooled pixels",
            "semantic distance to target",
            "pixel distance to target",
            "pixel/semantic/min-rank/Borda normalized ranks",
        ],
        "query_accounting": "same victim raw-representation-distance query budget; QCO consumes the full strict budget with no free optimum detection",
        "coherent_oracle_assumption": "QCO assumes coherent victim raw-representation distance predicate access",
        "quantum_novelty_claim": "none for DH/BBHT",
        "semantic_calibration_pairs": n_sem_pairs,
        "mean_eligible_candidates": float(np.mean(list(feasible_counts.values()))),
        "rejected_for_capacity": int(rejected),
        "summary": summary,
        "paired_contrasts": contrasts,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("SUMMARY_JSON " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
