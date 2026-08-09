#!/usr/bin/env python3
"""Experiment 3A: margin-certified raw penultimate representation collisions.

Why this experiment exists
--------------------------
Experiments 1/2 searched L2 distance between *normalized* penultimate features,
while the victim classifier consumes the *raw* penultimate vector.  Therefore a
small normalized distance need not imply close logits or the same hard label.

Here the collision metric is the actual raw penultimate representation h.  For a
correct target with predicted class y and logits z, the linear classifier head
provides a target-specific certified radius

    r_t = min_{j != y} (z_y - z_j) / ||W_y - W_j||_2.

If ||h_a - h_t||_2 < r_t, then every pairwise target-class margin stays positive
by Cauchy-Schwarz, so f(x_a)=f(x_t)=y is guaranteed.  This links the representation
collision predicate to output behavior without tuning a post-hoc epsilon.

The candidate set remains the strict QDSA semantic second-preimage set.  Search
methods and conservative quantum-oracle accounting are unchanged.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from exp1_mnist_qco import (
    BUDGETS, SIM_SEEDS, SmallCNN, embed_dataset, qco_min_find,
    random_min_find, seed_all, train_victim,
)
from exp1b_mnist_strong_baselines import knn_graph, pixel_rank, simulated_annealing
from exp1c_qdsa_semantic_gate import CALIBRATION_N, SemanticMLP, calibrate_delta, raw_flat
from exp2_qdsa_attack_effect_v2 import (
    POOL_SELECTION_SEED_BASE, legal_mask_for_target, select_balanced_feasible_targets,
)

EPS = 1e-12


@torch.inference_mode()
def embed_raw_victim(model: SmallCNN, loader: DataLoader, device: torch.device):
    model.eval()
    ys, preds, logits_all, hs = [], [], [], []
    for x, y in loader:
        x = x.to(device)
        z = model.features(x)
        h = model.fc(z.flatten(1))
        logits = model.classifier(h)
        ys.append(y.numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        logits_all.append(logits.cpu().numpy().astype(np.float64))
        hs.append(h.cpu().numpy().astype(np.float64))
    return (
        np.concatenate(ys), np.concatenate(preds),
        np.concatenate(logits_all), np.concatenate(hs),
    )


def certified_radius(model: SmallCNN, logits_t: np.ndarray, y: int) -> float:
    w = model.classifier.weight.detach().cpu().numpy().astype(np.float64)
    radii = []
    for j in range(w.shape[0]):
        if j == y:
            continue
        margin = float(logits_t[y] - logits_t[j])
        denom = float(np.linalg.norm(w[y] - w[j]))
        if margin <= 0:
            return 0.0
        if denom <= EPS:
            continue
        radii.append(margin / denom)
    return float(min(radii)) if radii else float("inf")


def raw_distances(h0: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.linalg.norm(h - h0[None, :], axis=1)


def summarize(rows: List[dict]) -> List[dict]:
    methods = ["dh_bbht", "random", "pixel_nn", "semantic_boundary", "simulated_annealing"]
    out: List[dict] = []
    for budget in sorted({r["budget"] for r in rows}):
        cell = [r for r in rows if r["budget"] == budget]
        targets = sorted({r["target_test_index"] for r in cell})
        row: Dict[str, float] = {
            "budget": budget,
            "n_independent_targets": len(targets),
            "pool_oracle_attackable_target_rate": float(np.mean([
                [r for r in cell if r["target_test_index"] == t][0]["pool_has_target_label"]
                for t in targets
            ])),
        }
        for m in methods:
            empirical, certified, dists, ratios, qs, global_hits = [], [], [], [], [], []
            for t in targets:
                g = [r for r in cell if r["target_test_index"] == t]
                empirical.append(float(np.mean([r[f"{m}_hard_label_collision"] for r in g])))
                certified.append(float(np.mean([r[f"{m}_certified_collision"] for r in g])))
                dists.append(float(np.mean([r[f"{m}_best_raw_distance"] for r in g])))
                ratios.append(float(np.mean([r[f"{m}_distance_to_radius_ratio"] for r in g])))
                qs.append(float(np.mean([r[f"{m}_queries"] for r in g])))
                global_hits.append(float(np.mean([r[f"{m}_global_min_hit"] for r in g])))
            row.update({
                f"{m}_empirical_asr": float(np.mean(empirical)),
                f"{m}_certified_asr": float(np.mean(certified)),
                f"{m}_mean_best_raw_distance": float(np.mean(dists)),
                f"{m}_mean_distance_to_cert_radius_ratio": float(np.mean(ratios)),
                f"{m}_mean_queries": float(np.mean(qs)),
                f"{m}_global_min_hit_rate": float(np.mean(global_hits)),
            })
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp3a_out")
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
    victim_loader = DataLoader(train_ds, batch_size=256, shuffle=True, generator=vg, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)
    victim = SmallCNN()
    train_victim(victim, victim_loader, device, args.epochs)
    labels, preds, logits, raw_h = embed_raw_victim(victim, test_loader, device)
    victim_acc = float(np.mean(labels == preds))
    if victim_acc < 0.97:
        raise RuntimeError(f"Victim accuracy gate failed: {victim_acc:.4f}")

    ref_seed = args.train_seed + 101
    seed_all(ref_seed)
    rg = torch.Generator().manual_seed(ref_seed)
    ref_loader = DataLoader(train_ds, batch_size=256, shuffle=True, generator=rg, num_workers=2)
    ref = SemanticMLP()
    train_victim(ref, ref_loader, device, args.epochs)
    ref_labels, ref_preds, ref_emb_test = embed_dataset(ref, test_loader, device)
    ref_acc = float(np.mean(ref_labels == ref_preds))
    if ref_acc < 0.95:
        raise RuntimeError(f"Reference accuracy gate failed: {ref_acc:.4f}")

    # Same fixed semantic Delta protocol as Experiment 2.
    from torch.utils.data import Subset
    cal_ds = Subset(train_ds, range(CALIBRATION_N))
    cal_loader = DataLoader(cal_ds, batch_size=512, shuffle=False, num_workers=2)
    rcal_labels, rcal_preds, rcal_emb = embed_dataset(ref, cal_loader, device)
    ref_good = rcal_labels == rcal_preds
    semantic_delta, n_sem_pairs = calibrate_delta(rcal_emb[ref_good], rcal_labels[ref_good])

    target_indices, feasible_counts, rejected_capacity = select_balanced_feasible_targets(
        labels, preds, ref_labels, ref_preds, ref_emb_test,
        semantic_delta, args.pool, args.targets,
    )
    images = torch.cat([x for x, _ in test_loader], dim=0)
    pix = raw_flat(images)

    rows: List[dict] = []
    target_diag: List[dict] = []
    for target_ord, t0 in enumerate(target_indices):
        t = int(t0)
        y0 = int(preds[t])
        radius = certified_radius(victim, logits[t], y0)
        legal, sem_all = legal_mask_for_target(
            t, labels, ref_labels, ref_preds, ref_emb_test, semantic_delta
        )
        eligible = np.flatnonzero(legal)
        pool_rng = np.random.default_rng(POOL_SELECTION_SEED_BASE + t * 23)
        pool_idx = pool_rng.choice(eligible, size=args.pool, replace=False)
        distances = raw_distances(raw_h[t], raw_h[pool_idx])
        global_best_local = int(np.argmin(distances))
        global_best = float(distances[global_best_local])
        pool_preds = preds[pool_idx]
        impersonator = pool_preds == y0
        pool_has_target = bool(np.any(impersonator))
        if pool_has_target:
            imp_local_candidates = np.flatnonzero(impersonator)
            imp_local = int(imp_local_candidates[np.argmin(distances[imp_local_candidates])])
            best_imp_d = float(distances[imp_local])
            rank = int(np.sum(distances < best_imp_d - EPS) + 1)
        else:
            best_imp_d = float("inf")
            rank = -1

        target_diag.append({
            "target_test_index": t,
            "target_label": int(labels[t]),
            "certified_radius": radius,
            "global_min_raw_distance": global_best,
            "global_min_to_radius_ratio": global_best / max(radius, EPS),
            "global_min_pred": int(pool_preds[global_best_local]),
            "pool_has_target_label": int(pool_has_target),
            "pool_target_label_count": int(np.sum(impersonator)),
            "best_impersonator_raw_distance": best_imp_d,
            "best_impersonator_rank_by_raw_distance": rank,
        })

        sem_pool = sem_all[pool_idx]
        pool_pix = pix[pool_idx]
        p_order = pixel_rank(pix[t], pool_pix)
        s_order = np.argsort(sem_pool, kind="stable")
        graph = knn_graph(pool_pix)

        for sim_seed in SIM_SEEDS:
            for budget in BUDGETS:
                base_seed = int(sim_seed * 1_000_003 + t * 97 + budget * 13 + 307)
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
                    "target_test_index": t,
                    "target_label": int(labels[t]),
                    "target_pred": y0,
                    "budget": budget,
                    "sim_seed": sim_seed,
                    "semantic_delta": semantic_delta,
                    "certified_radius": radius,
                    "global_best_raw_distance": global_best,
                    "pool_has_target_label": int(pool_has_target),
                }
                for m, (local_idx, best_d, q_used) in selected.items():
                    local_idx = int(local_idx)
                    global_idx = int(pool_idx[local_idx])
                    hard = bool(preds[global_idx] == y0)
                    cert = bool(float(best_d) < radius)
                    if cert and not hard:
                        raise AssertionError("Certified collision bound violated")
                    row.update({
                        f"{m}_best_raw_distance": float(best_d),
                        f"{m}_distance_to_radius_ratio": float(best_d) / max(radius, EPS),
                        f"{m}_queries": int(q_used),
                        f"{m}_candidate_test_index": global_idx,
                        f"{m}_candidate_truth": int(labels[global_idx]),
                        f"{m}_candidate_pred": int(preds[global_idx]),
                        f"{m}_hard_label_collision": int(hard),
                        f"{m}_certified_collision": int(cert),
                        f"{m}_global_min_hit": int(abs(float(best_d) - global_best) <= 1e-10),
                    })
                rows.append(row)

    with (out / "trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with (out / "target_diagnostics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(target_diag[0].keys()))
        w.writeheader(); w.writerows(target_diag)

    summary = summarize(rows)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    protocol = {
        "name": "QDSA-Experiment-3A-Certified-Raw-Collision",
        "collision_metric": "L2 distance on the actual unnormalized penultimate vector consumed by the linear classifier",
        "certified_radius": "min_j (z_y-z_j)/||W_y-W_j||_2",
        "certificate": "raw feature distance below target radius is sufficient for identical target hard label",
        "semantic_delta": semantic_delta,
        "victim_accuracy": victim_acc,
        "reference_accuracy": ref_acc,
        "targets": int(len(target_indices)),
        "pool_size": args.pool,
        "mean_certified_radius": float(np.mean([d["certified_radius"] for d in target_diag])),
        "median_global_min_to_radius_ratio": float(np.median([d["global_min_to_radius_ratio"] for d in target_diag])),
        "pool_oracle_attackable_target_rate": float(np.mean([d["pool_has_target_label"] for d in target_diag])),
        "semantic_calibration_pairs": n_sem_pairs,
        "mean_eligible_candidates": float(np.mean(list(feasible_counts.values()))),
        "rejected_for_capacity": int(rejected_capacity),
        "quantum_novelty_claim": "none; DH/BBHT remains a standard quantum-search baseline",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"protocol": protocol, "summary": summary}, indent=2), encoding="utf-8")
    print("SUMMARY_JSON " + json.dumps({"protocol": protocol, "summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
