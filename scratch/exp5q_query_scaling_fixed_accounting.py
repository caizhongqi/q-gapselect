#!/usr/bin/env python3
"""Experiment 5Q: fixed-accounting quantum seed-search scaling audit.

This experiment corrects an accounting flaw in the earlier simulator.  Previous
runs stopped for free when the precomputed simulator table showed that no entry
was better than the incumbent.  A real oracle does not reveal that global fact.
Here the quantum routine NEVER inspects whether the marked set is empty in order
to terminate.  It consumes the full pre-registered strict oracle budget.

The scientific question is therefore only:
  At the SAME fixed representation-oracle budget, how does DH/BBHT seed quality
  and global-minimum hit probability scale with candidate-pool size N compared
  with classical search baselines?

Strict quantum accounting:
  - one measured-candidate verification = 1 victim representation oracle call
  - one Grover phase-mark iteration = 2 victim representation oracle calls
    (compute + uncompute)
  - no free global-optimum / empty-marked-set detection

This is an oracle-model scaling experiment.  It does not claim that coherent
access to a deployed classical neural network is currently practical.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from exp1_mnist_qco import SmallCNN, embed_dataset, grover_measure, l2_distances, seed_all, train_victim
from exp1c_qdsa_semantic_gate import CALIBRATION_N, SemanticMLP, calibrate_delta, raw_flat
from exp2_qdsa_attack_effect_v2 import legal_mask_for_target, select_balanced_feasible_targets
from exp3a_certified_raw_collision import embed_raw_victim, raw_distances
from exp1b_mnist_strong_baselines import pixel_rank

N_VALUES = (128, 256, 512, 1024, 2048)
BUDGET_GRID = (8, 16, 32, 64, 128, 256, 512, 1024)
SIM_SEEDS = (11, 22, 33, 44, 55)
BBHT_LAMBDA = 6.0 / 5.0
EPS = 1e-12


@dataclass
class FixedQCOResult:
    best_index: int
    best_distance: float
    strict_queries: int
    logical_queries: int
    improvements: int
    phase_iterations: int
    verification_queries: int


def qco_min_find_fixed_budget(distances: np.ndarray, budget: int, rng: np.random.Generator) -> FixedQCOResult:
    """Budget-fair DH/BBHT-style minimum search with no free optimum detection."""
    n = len(distances)
    if budget < 1:
        raise ValueError("budget must be >=1")
    best_idx = int(rng.integers(n))
    best = float(distances[best_idx])
    strict_q = 1
    logical_q = 1
    phase_iters = 0
    verifications = 1
    improvements = 0
    m = 1.0

    while strict_q < budget:
        # This table is needed by the statevector simulator to emulate the phase
        # oracle.  Crucially, we do NOT inspect np.any(marked) to decide whether
        # to stop; an empty marked set is indistinguishable without queries.
        marked = distances < (best - EPS)
        remaining = budget - strict_q
        max_r_budget = max(0, (remaining - 1) // 2)
        width = max(1, int(math.ceil(m)))
        width = min(width, max(1, int(math.ceil(math.sqrt(n)))))
        max_r = min(width - 1, max_r_budget)
        r = int(rng.integers(0, max_r + 1)) if max_r > 0 else 0

        idx = grover_measure(marked, r, rng)
        cost = 2 * r + 1
        strict_q += cost
        logical_q += r + 1
        phase_iters += r
        verifications += 1
        d = float(distances[idx])
        if d < best - EPS:
            best = d
            best_idx = idx
            improvements += 1
            m = 1.0
        else:
            m = min(BBHT_LAMBDA * m, math.sqrt(n))

    if strict_q != budget:
        raise AssertionError(f"fixed-budget invariant failed: {strict_q} != {budget}")
    return FixedQCOResult(best_idx, best, strict_q, logical_q, improvements, phase_iters, verifications)


def random_fixed(distances: np.ndarray, budget: int, rng: np.random.Generator) -> Tuple[int, float, int]:
    q = min(int(budget), len(distances))
    idxs = rng.choice(len(distances), size=q, replace=False)
    loc = int(np.argmin(distances[idxs]))
    idx = int(idxs[loc])
    return idx, float(distances[idx]), q


def quality_metrics(best: float, distances: np.ndarray) -> Dict[str, float]:
    g = float(np.min(distances))
    med = float(np.median(distances))
    rank = int(np.sum(distances < best - EPS) + 1)
    denom = max(med - g, 1e-12)
    return {
        "global_best": g,
        "exact_global_hit": int(abs(best - g) <= 1e-12),
        "within_5pct_global": int(best <= 1.05 * g + 1e-12),
        "best_distance": float(best),
        "relative_to_global": float(best / max(g, 1e-12)),
        "normalized_regret": float((best - g) / denom),
        "rank": rank,
        "rank_fraction": float(rank / len(distances)),
    }


def summarize_trials(rows: List[dict]) -> List[dict]:
    out: List[dict] = []
    methods = sorted({r["method"] for r in rows})
    for n in N_VALUES:
        for budget in [b for b in BUDGET_GRID if b <= n]:
            for method in methods:
                cell = [r for r in rows if r["pool_size"] == n and r["budget"] == budget and r["method"] == method]
                if not cell:
                    continue
                targets = sorted({r["target_test_index"] for r in cell})
                exact_t, near_t, dist_t, regret_t, rank_t, q_t = [], [], [], [], [], []
                for t in targets:
                    g = [r for r in cell if r["target_test_index"] == t]
                    exact_t.append(float(np.mean([r["exact_global_hit"] for r in g])))
                    near_t.append(float(np.mean([r["within_5pct_global"] for r in g])))
                    dist_t.append(float(np.mean([r["best_distance"] for r in g])))
                    regret_t.append(float(np.mean([r["normalized_regret"] for r in g])))
                    rank_t.append(float(np.mean([r["rank_fraction"] for r in g])))
                    q_t.append(float(np.mean([r["queries"] for r in g])))
                out.append({
                    "pool_size": n,
                    "budget": budget,
                    "method": method,
                    "n_independent_targets": len(targets),
                    "exact_global_hit_rate": float(np.mean(exact_t)),
                    "within_5pct_global_rate": float(np.mean(near_t)),
                    "mean_best_distance": float(np.mean(dist_t)),
                    "mean_normalized_regret": float(np.mean(regret_t)),
                    "mean_rank_fraction": float(np.mean(rank_t)),
                    "mean_queries": float(np.mean(q_t)),
                })
    return out


def threshold_budget(summary: List[dict], method: str, n: int, metric: str, threshold: float):
    g = sorted([r for r in summary if r["method"] == method and r["pool_size"] == n], key=lambda r: r["budget"])
    for r in g:
        if float(r[metric]) >= threshold:
            return int(r["budget"])
    return None


def fit_power(ns: List[int], qs: List[int]):
    if len(ns) < 3:
        return None
    x = np.log(np.asarray(ns, dtype=float))
    y = np.log(np.asarray(qs, dtype=float))
    slope, intercept = np.polyfit(x, y, deg=1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {"exponent": float(slope), "prefactor": float(math.exp(intercept)), "loglog_r2": r2, "n_points": len(ns)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp5q_out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-seed", type=int, default=20260809)
    ap.add_argument("--targets", type=int, default=20)
    ap.add_argument("--max-pool", type=int, default=max(N_VALUES))
    args = ap.parse_args()
    if args.max_pool < max(N_VALUES):
        raise ValueError("max-pool must cover N_VALUES")

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

    # Target feasibility is decided at the maximum pool size, before any victim
    # collision distances are inspected.
    target_indices, feasible_counts, rejected = select_balanced_feasible_targets(
        labels, preds, ref_labels, ref_preds, ref_emb,
        delta, args.max_pool, args.targets,
    )
    test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))], dim=0)
    pix = raw_flat(test_x)

    rows: List[dict] = []
    methods = ("qco_fixed", "random", "pixel_nn", "semantic_boundary")

    for target_ord, t0 in enumerate(target_indices):
        t = int(t0)
        legal, sem_all = legal_mask_for_target(t, labels, ref_labels, ref_preds, ref_emb, delta)
        eligible = np.flatnonzero(legal)
        master_rng = np.random.default_rng(12_000_000 + t * 31)
        master_pool = master_rng.choice(eligible, size=args.max_pool, replace=False)

        for n in N_VALUES:
            pool_idx = master_pool[:n]
            distances = raw_distances(raw_h[t], raw_h[pool_idx])
            sem_pool = sem_all[pool_idx]
            p_order = pixel_rank(pix[t], pix[pool_idx])
            s_order = np.argsort(sem_pool, kind="stable")

            for budget in [b for b in BUDGET_GRID if b <= n]:
                # Stochastic methods: five paired simulator seeds.
                for sim_seed in SIM_SEEDS:
                    base = int(sim_seed * 1_000_003 + t * 97 + n * 17 + budget * 13)
                    qrng = np.random.default_rng(base)
                    rrng = np.random.default_rng(base + 1)
                    qres = qco_min_find_fixed_budget(distances, budget, qrng)
                    ridx, rbest, rq = random_fixed(distances, budget, rrng)
                    for method, idx, best, queries in (
                        ("qco_fixed", qres.best_index, qres.best_distance, qres.strict_queries),
                        ("random", ridx, rbest, rq),
                    ):
                        qm = quality_metrics(float(best), distances)
                        rows.append({
                            "target_order": target_ord,
                            "target_test_index": t,
                            "target_class": int(labels[t]),
                            "pool_size": n,
                            "budget": budget,
                            "sim_seed": sim_seed,
                            "method": method,
                            "queries": int(queries),
                            "selected_local_index": int(idx),
                            **qm,
                        })

                # Deterministic public-geometry controls, recorded once per target.
                q = min(budget, n)
                for method, order in (("pixel_nn", p_order), ("semantic_boundary", s_order)):
                    queried = order[:q]
                    li = int(queried[np.argmin(distances[queried])])
                    best = float(distances[li])
                    qm = quality_metrics(best, distances)
                    rows.append({
                        "target_order": target_ord,
                        "target_test_index": t,
                        "target_class": int(labels[t]),
                        "pool_size": n,
                        "budget": budget,
                        "sim_seed": -1,
                        "method": method,
                        "queries": q,
                        "selected_local_index": li,
                        **qm,
                    })
        print(f"target={t} done", flush=True)

    with (out / "trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    summary = summarize_trials(rows)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    scaling = []
    fits: Dict[str, dict] = {}
    for method in methods:
        ns, q50s = [], []
        for n in N_VALUES:
            q50 = threshold_budget(summary, method, n, "exact_global_hit_rate", 0.50)
            q80near = threshold_budget(summary, method, n, "within_5pct_global_rate", 0.80)
            scaling.append({
                "method": method,
                "pool_size": n,
                "budget_for_50pct_exact_global_hit": q50,
                "budget_for_80pct_within_5pct_global": q80near,
            })
            if q50 is not None:
                ns.append(n); q50s.append(q50)
        fit = fit_power(ns, q50s)
        if fit is not None:
            fits[method] = fit

    with (out / "scaling_thresholds.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scaling[0].keys())); w.writeheader(); w.writerows(scaling)

    result = {
        "name": "QDSA-Experiment-5Q-Fixed-Accounting-Query-Scaling",
        "role": "audit corrected quantum representation-oracle query scaling",
        "accounting_correction": "no np.any(marked) or other free global-optimum detection; QCO consumes the full strict budget",
        "victim_accuracy": victim_acc,
        "reference_accuracy": ref_acc,
        "semantic_delta": delta,
        "targets": int(len(target_indices)),
        "target_class_balance": {str(c): int(np.sum(labels[target_indices] == c)) for c in range(10)},
        "pool_sizes": list(N_VALUES),
        "budget_grid": list(BUDGET_GRID),
        "simulator_seeds": list(SIM_SEEDS),
        "methods": list(methods),
        "qco_phase_iteration_cost": 2,
        "qco_verification_cost": 1,
        "coherent_oracle_assumption": "coherent access to victim raw penultimate representation-distance predicate",
        "precomputed_distance_table_role": "statevector oracle emulation and evaluation only; never used for free stopping",
        "scaling_thresholds": scaling,
        "loglog_q50_fits": fits,
        "semantic_calibration_pairs": n_sem_pairs,
        "mean_eligible_candidates": float(np.mean(list(feasible_counts.values()))),
        "rejected_for_capacity": int(rejected),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("SUMMARY_JSON " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
