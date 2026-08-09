#!/usr/bin/env python3
"""Experiment 3C: source-neighborhood robustness of continuous second preimages.

Experiment 3B established continuous solutions but unconstrained optimization can
produce noisy images.  This experiment adds an explicit source-preservation
constraint: every refined image is projected to an L_inf ball around the original
semantic source image.  We sweep eps in {0.10, 0.20, 0.30} while keeping the
independent-reference source class and D_sem >= Delta constraints.

This remains a white-box feasibility/robustness gate.  It is not yet a claim of
quantum advantage; it defines the valid continuous attack domain for the next
Jacobian-null-space/QCO experiment.
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
from exp3b_continuous_existence import victim_raw, reference_forward

EPS_SWEEP = (0.10, 0.20, 0.30)


def project_source_ball(x: torch.Tensor, x0: torch.Tensor, eps: float) -> None:
    with torch.no_grad():
        lo = torch.clamp(x0 - eps, 0.0, 1.0)
        hi = torch.clamp(x0 + eps, 0.0, 1.0)
        x.copy_(torch.maximum(torch.minimum(x, hi), lo))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp3c_out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-seed", type=int, default=20260809)
    ap.add_argument("--targets", type=int, default=20)
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--lambda-sem", type=float, default=150.0)
    ap.add_argument("--lambda-cls", type=float, default=2.0)
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
        raise RuntimeError("accuracy gate failed")

    cal_ds = Subset(train_ds, range(CALIBRATION_N))
    cal_loader = DataLoader(cal_ds, batch_size=512, shuffle=False, num_workers=2)
    rcal_labels, rcal_preds, rcal_emb = embed_dataset(ref, cal_loader, device)
    good = rcal_labels == rcal_preds
    semantic_delta, n_sem_pairs = calibrate_delta(rcal_emb[good], rcal_labels[good])

    target_indices, feasible_counts, rejected = select_balanced_feasible_targets(
        labels, preds, ref_labels, ref_preds, ref_emb_test,
        semantic_delta, args.pool, args.targets,
    )
    test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))], dim=0)

    for p in victim.parameters(): p.requires_grad_(False)
    for p in ref.parameters(): p.requires_grad_(False)

    rows: List[dict] = []
    panels = {eps: [] for eps in EPS_SWEEP}

    for target_ord, t0 in enumerate(target_indices):
        t = int(t0)
        target_class = int(preds[t])
        x_t = test_x[t:t+1].to(device)
        h_t = torch.from_numpy(raw_h_np[t:t+1]).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            _, ref_h_t = reference_forward(ref, x_t)
        radius = certified_radius(victim, logits_np[t], target_class)

        legal, _ = legal_mask_for_target(t, labels, ref_labels, ref_preds, ref_emb_test, semantic_delta)
        eligible = np.flatnonzero(legal)
        pool_rng = np.random.default_rng(POOL_SELECTION_SEED_BASE + t * 23)
        pool_idx = pool_rng.choice(eligible, size=args.pool, replace=False)
        pool_dist = raw_distances(raw_h_np[t], raw_h_np[pool_idx])
        order = np.argsort(pool_dist, kind="stable")[:args.restarts]
        start_idx = pool_idx[order]
        x0_all = test_x[start_idx].to(device).clone()
        source_y_all = torch.tensor(labels[start_idx], device=device, dtype=torch.long)

        for eps in EPS_SWEEP:
            x0 = x0_all.clone()
            source_y = source_y_all.clone()
            x = x0.clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([x], lr=args.lr)
            best = [None for _ in range(args.restarts)]

            for step in range(args.steps + 1):
                opt.zero_grad(set_to_none=True)
                v_logits, h = victim_raw(victim, x)
                r_logits, r_h = reference_forward(ref, x)
                feat_sq = torch.mean((h - h_t.expand_as(h)) ** 2, dim=1)
                sem_d = torch.linalg.vector_norm(r_h - ref_h_t.expand_as(r_h), dim=1)
                sem_pen = F.relu(semantic_delta - sem_d) ** 2
                src_ce = F.cross_entropy(r_logits, source_y, reduction="none")
                loss = torch.mean(feat_sq + args.lambda_sem * sem_pen + args.lambda_cls * src_ce)

                with torch.no_grad():
                    raw_d = torch.linalg.vector_norm(h - h_t.expand_as(h), dim=1)
                    vp = v_logits.argmax(1); rp = r_logits.argmax(1)
                    feasible = (sem_d >= semantic_delta) & (rp == source_y)
                    for k in range(args.restarts):
                        if bool(feasible[k]):
                            rec = (
                                float(raw_d[k]), step, x[k:k+1].detach().cpu().clone(),
                                int(vp[k]), int(rp[k]), float(sem_d[k]),
                                float(torch.max(torch.abs(x[k] - x0[k]))),
                                float(torch.linalg.vector_norm((x[k] - x0[k]).flatten())),
                            )
                            if best[k] is None or rec[0] < best[k][0]:
                                best[k] = rec

                if step == args.steps: break
                loss.backward(); opt.step(); project_source_ball(x, x0, eps)

            valid = [(k, r) for k, r in enumerate(best) if r is not None]
            if not valid:
                raise RuntimeError(f"target={t} eps={eps}: no feasible iterate")
            k, rec = min(valid, key=lambda kr: kr[1][0])
            final_d, best_step, best_x, vp, rp, sem_d, linf, l2 = rec
            source_class = int(labels[start_idx[k]])
            init_d = float(pool_dist[order[k]])
            empirical = int(vp == target_class and rp == source_class and sem_d >= semantic_delta)
            certified = int(final_d < radius and rp == source_class and sem_d >= semantic_delta)
            if certified and not empirical:
                raise AssertionError("certificate violated")

            rows.append({
                "target_order": target_ord,
                "target_test_index": t,
                "target_class": target_class,
                "source_test_index": int(start_idx[k]),
                "source_class": source_class,
                "eps_linf": eps,
                "certified_radius": radius,
                "initial_raw_distance": init_d,
                "final_raw_distance": final_d,
                "distance_reduction_pct": 100.0 * (init_d-final_d)/max(init_d,1e-12),
                "final_to_radius_ratio": final_d/max(radius,1e-12),
                "final_linf_from_source": linf,
                "final_l2_from_source": l2,
                "best_step": int(best_step),
                "final_victim_pred": vp,
                "final_reference_pred": rp,
                "final_semantic_distance": sem_d,
                "semantic_delta": semantic_delta,
                "empirical_success": empirical,
                "certified_success": certified,
            })
            if target_ord < 6:
                panels[eps].extend([x_t.cpu(), x0[k:k+1].cpu(), best_x])
            print(f"t={t} eps={eps:.2f} {init_d:.2f}->{final_d:.2f} ratio={final_d/max(radius,1e-12):.2f} sem={sem_d:.3f} success={empirical} cert={certified}", flush=True)

    with (out / "per_target_eps.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    summaries=[]
    for eps in EPS_SWEEP:
        g=[r for r in rows if abs(r["eps_linf"]-eps)<1e-9]
        summaries.append({
            "eps_linf": eps,
            "targets": len(g),
            "empirical_asr": float(np.mean([r["empirical_success"] for r in g])),
            "certified_asr": float(np.mean([r["certified_success"] for r in g])),
            "mean_final_raw_distance": float(np.mean([r["final_raw_distance"] for r in g])),
            "median_final_to_radius_ratio": float(np.median([r["final_to_radius_ratio"] for r in g])),
            "mean_distance_reduction_pct": float(np.mean([r["distance_reduction_pct"] for r in g])),
            "mean_final_linf": float(np.mean([r["final_linf_from_source"] for r in g])),
            "mean_final_l2": float(np.mean([r["final_l2_from_source"] for r in g])),
            "min_final_semantic_distance": float(np.min([r["final_semantic_distance"] for r in g])),
        })
        if panels[eps]:
            tvutils.save_image(torch.cat(panels[eps],dim=0), out/f"examples_eps_{eps:.2f}.png", nrow=3, padding=2)

    with (out / "summary.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(summaries[0].keys())); w.writeheader(); w.writerows(summaries)
    protocol={
        "name":"QDSA-Experiment-3C-Source-Ball-Robustness",
        "role":"semantic robustness gate before Jacobian-null-space/quantum refinement",
        "victim_accuracy":victim_acc,
        "reference_accuracy":ref_acc,
        "semantic_delta":semantic_delta,
        "eps_sweep":list(EPS_SWEEP),
        "targets":len(target_indices),
        "pool_size":args.pool,
        "restarts":args.restarts,
        "steps":args.steps,
        "hard_constraints":["x in L_inf source ball","pixels in [0,1]","reference source class at accepted iterate","D_sem >= Delta"],
        "semantic_calibration_pairs":n_sem_pairs,
        "mean_eligible_candidates":float(np.mean(list(feasible_counts.values()))),
        "rejected_for_capacity":int(rejected),
        "summary":summaries,
    }
    (out/"summary.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    print("SUMMARY_JSON "+json.dumps(protocol,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
