#!/usr/bin/env python3
"""Experiment 3B: continuous QDSA second-preimage existence gate.

This is deliberately a WHITE-BOX UPPER BOUND, not the final quantum attack.
Experiment 3A showed that the discrete 512-point public pool almost never contains
a valid target-label second preimage even when quantum minimum finding reaches the
pool optimum.  Before designing a quantum tunnelling/refinement algorithm, we ask
whether the continuous input domain contains such solutions at all.

For each target we take the three semantically legal pool points with the smallest
RAW victim penultimate distance and jointly refine them with Adam.  The optimizer
minimizes victim raw-feature collision while penalizing violations of two semantic
constraints measured by an independent reference network:

  (i) D_sem(x, x_t) >= Delta,
  (ii) reference hard label remains the source class.

A successful empirical second preimage must therefore be classified by the victim
as the target class while the independent semantic model still classifies it as
the source class and the pre-registered semantic distance gate remains satisfied.
We additionally report whether the point enters the target-specific certified raw
feature radius from Experiment 3A.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils as tvutils

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from exp1_mnist_qco import SmallCNN, embed_dataset, seed_all, train_victim
from exp1c_qdsa_semantic_gate import CALIBRATION_N, SemanticMLP, calibrate_delta
from exp2_qdsa_attack_effect_v2 import (
    POOL_SELECTION_SEED_BASE, legal_mask_for_target, select_balanced_feasible_targets,
)
from exp3a_certified_raw_collision import embed_raw_victim, certified_radius, raw_distances


def victim_raw(model: SmallCNN, x: torch.Tensor):
    z = model.features(x)
    h = model.fc(z.flatten(1))
    logits = model.classifier(h)
    return logits, h


def reference_forward(model: SemanticMLP, x: torch.Tensor):
    return model(x, return_feature=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp3b_out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-seed", type=int, default=20260809)
    ap.add_argument("--targets", type=int, default=20)
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--lambda-sem", type=float, default=120.0)
    ap.add_argument("--lambda-cls", type=float, default=2.0)
    ap.add_argument("--lambda-src", type=float, default=0.02)
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
    victim = SmallCNN().to(device)
    train_victim(victim, victim_loader, device, args.epochs)
    labels, preds, logits_np, raw_h_np = embed_raw_victim(victim, test_loader, device)
    victim_acc = float(np.mean(labels == preds))

    ref_seed = args.train_seed + 101
    seed_all(ref_seed)
    rg = torch.Generator().manual_seed(ref_seed)
    ref_loader = DataLoader(train_ds, batch_size=256, shuffle=True, generator=rg, num_workers=2)
    ref = SemanticMLP().to(device)
    train_victim(ref, ref_loader, device, args.epochs)
    ref_labels, ref_preds, ref_emb_test = embed_dataset(ref, test_loader, device)
    ref_acc = float(np.mean(ref_labels == ref_preds))

    if victim_acc < 0.97 or ref_acc < 0.95:
        raise RuntimeError(f"accuracy gate failed victim={victim_acc:.4f} ref={ref_acc:.4f}")

    cal_ds = Subset(train_ds, range(CALIBRATION_N))
    cal_loader = DataLoader(cal_ds, batch_size=512, shuffle=False, num_workers=2)
    rcal_labels, rcal_preds, rcal_emb = embed_dataset(ref, cal_loader, device)
    good = rcal_labels == rcal_preds
    semantic_delta, n_sem_pairs = calibrate_delta(rcal_emb[good], rcal_labels[good])

    target_indices, feasible_counts, rejected = select_balanced_feasible_targets(
        labels, preds, ref_labels, ref_preds, ref_emb_test,
        semantic_delta, args.pool, args.targets,
    )

    # Materialize test tensors once; MNIST is small.
    test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))], dim=0)
    rows: List[dict] = []
    image_panels: List[torch.Tensor] = []

    for target_ord, t0 in enumerate(target_indices):
        t = int(t0)
        target_class = int(preds[t])
        x_t = test_x[t:t+1].to(device)
        h_t = torch.from_numpy(raw_h_np[t:t+1]).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            _, ref_h_t = reference_forward(ref, x_t)
        radius = certified_radius(victim, logits_np[t], target_class)

        legal, sem_all = legal_mask_for_target(
            t, labels, ref_labels, ref_preds, ref_emb_test, semantic_delta
        )
        eligible = np.flatnonzero(legal)
        pool_rng = np.random.default_rng(POOL_SELECTION_SEED_BASE + t * 23)
        pool_idx = pool_rng.choice(eligible, size=args.pool, replace=False)
        pool_dist = raw_distances(raw_h_np[t], raw_h_np[pool_idx])
        order = np.argsort(pool_dist, kind="stable")
        start_local = order[:args.restarts]
        start_idx = pool_idx[start_local]
        x0 = test_x[start_idx].to(device).clone()
        source_y = torch.tensor(labels[start_idx], device=device, dtype=torch.long)

        # Freeze networks; gradients flow only to x.
        for p in victim.parameters():
            p.requires_grad_(False)
        for p in ref.parameters():
            p.requires_grad_(False)

        x = x0.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=args.lr)
        best_feasible = [None for _ in range(args.restarts)]

        for step in range(args.steps + 1):
            opt.zero_grad(set_to_none=True)
            v_logits, h = victim_raw(victim, x)
            r_logits, r_h = reference_forward(ref, x)
            feat_sq = torch.mean((h - h_t.expand_as(h)) ** 2, dim=1)
            sem_d = torch.linalg.vector_norm(r_h - ref_h_t.expand_as(r_h), dim=1)
            sem_pen = F.relu(semantic_delta - sem_d) ** 2
            src_ce = F.cross_entropy(r_logits, source_y, reduction="none")
            src_mse = torch.mean((x - x0) ** 2, dim=(1, 2, 3))
            loss_each = feat_sq + args.lambda_sem * sem_pen + args.lambda_cls * src_ce + args.lambda_src * src_mse
            loss = loss_each.mean()

            with torch.no_grad():
                raw_d = torch.linalg.vector_norm(h - h_t.expand_as(h), dim=1)
                v_pred = v_logits.argmax(1)
                r_pred = r_logits.argmax(1)
                feasible = (sem_d >= semantic_delta) & (r_pred == source_y)
                for k in range(args.restarts):
                    if bool(feasible[k]):
                        record = (
                            float(raw_d[k]), step, x[k:k+1].detach().cpu().clone(),
                            int(v_pred[k]), int(r_pred[k]), float(sem_d[k]),
                        )
                        if best_feasible[k] is None or record[0] < best_feasible[k][0]:
                            best_feasible[k] = record

            if step == args.steps:
                break
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(0.0, 1.0)

        # Evaluate the best feasible point for each restart, then choose the one
        # with the smallest raw victim feature distance.
        valid = [(k, r) for k, r in enumerate(best_feasible) if r is not None]
        if not valid:
            raise RuntimeError(f"Target {t}: optimizer produced no semantically feasible iterate")
        best_k, best = min(valid, key=lambda kr: kr[1][0])
        final_d, best_step, best_x, v_pred_final, r_pred_final, sem_final = best
        source_class = int(labels[start_idx[best_k]])
        empirical_success = int(
            v_pred_final == target_class
            and r_pred_final == source_class
            and sem_final >= semantic_delta
            and source_class != target_class
        )
        certified_success = int(
            final_d < radius
            and r_pred_final == source_class
            and sem_final >= semantic_delta
            and source_class != target_class
        )
        if certified_success and not empirical_success:
            raise AssertionError("Certified second preimage did not preserve victim target label")

        init_d = float(pool_dist[start_local[best_k]])
        rows.append({
            "target_order": target_ord,
            "target_test_index": t,
            "target_class": target_class,
            "source_test_index": int(start_idx[best_k]),
            "source_class": source_class,
            "certified_radius": radius,
            "initial_raw_distance": init_d,
            "final_raw_distance": final_d,
            "distance_reduction_pct": 100.0 * (init_d - final_d) / max(init_d, 1e-12),
            "initial_to_radius_ratio": init_d / max(radius, 1e-12),
            "final_to_radius_ratio": final_d / max(radius, 1e-12),
            "best_step": int(best_step),
            "final_victim_pred": int(v_pred_final),
            "final_reference_pred": int(r_pred_final),
            "final_semantic_distance": sem_final,
            "semantic_delta": semantic_delta,
            "empirical_second_preimage_success": empirical_success,
            "certified_second_preimage_success": certified_success,
            "successful_restarts": int(sum(1 for r in best_feasible if r is not None)),
        })

        if target_ord < 10:
            image_panels.extend([x_t.cpu(), x0[best_k:best_k+1].cpu(), best_x])
        print(
            f"target={t} y={target_class} src={source_class} "
            f"raw {init_d:.3f}->{final_d:.3f} r={radius:.3f} "
            f"sem={sem_final:.3f} vpred={v_pred_final} rpred={r_pred_final} "
            f"success={empirical_success} cert={certified_success}",
            flush=True,
        )

    with (out / "per_target.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    if image_panels:
        grid = torch.cat(image_panels, dim=0)
        tvutils.save_image(grid, out / "first10_target_source_refined.png", nrow=3, padding=2)

    empirical_asr = float(np.mean([r["empirical_second_preimage_success"] for r in rows]))
    certified_asr = float(np.mean([r["certified_second_preimage_success"] for r in rows]))
    result = {
        "name": "QDSA-Experiment-3B-Continuous-Existence-Gate",
        "role": "white-box continuous-domain upper bound; not the final quantum attack",
        "victim_accuracy": victim_acc,
        "reference_accuracy": ref_acc,
        "semantic_delta": semantic_delta,
        "semantic_calibration_pairs": n_sem_pairs,
        "targets": len(rows),
        "pool_size": args.pool,
        "restarts": args.restarts,
        "steps": args.steps,
        "optimizer": "Adam on input pixels",
        "loss": "raw victim feature collision + semantic-distance barrier + independent-reference source-class CE + weak source L2",
        "empirical_second_preimage_asr": empirical_asr,
        "certified_second_preimage_asr": certified_asr,
        "targets_empirical_success": int(sum(r["empirical_second_preimage_success"] for r in rows)),
        "targets_certified_success": int(sum(r["certified_second_preimage_success"] for r in rows)),
        "mean_initial_raw_distance": float(np.mean([r["initial_raw_distance"] for r in rows])),
        "mean_final_raw_distance": float(np.mean([r["final_raw_distance"] for r in rows])),
        "median_final_to_radius_ratio": float(np.median([r["final_to_radius_ratio"] for r in rows])),
        "mean_semantic_distance_final": float(np.mean([r["final_semantic_distance"] for r in rows])),
        "mean_eligible_candidates": float(np.mean(list(feasible_counts.values()))),
        "rejected_for_capacity": int(rejected),
        "next_gate": "Only if continuous solutions exist should QCO/Jacobian-null-space tunnelling be evaluated as the attack mechanism.",
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("SUMMARY_JSON " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
