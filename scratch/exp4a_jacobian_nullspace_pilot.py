#!/usr/bin/env python3
"""Experiment 4A: QDSA Jacobian semantic-nullspace refinement pilot.

This is the first direct test of the proposed *mechanism* after the source-ball
feasibility gate.  It deliberately separates two questions:

  Seed selection (fixed 128 victim-representation oracle-equivalent budget)
    - dh_bbht: standard quantum minimum finding baseline
    - random
    - pixel_nn
    - semantic_boundary

  Continuous refinement inside the SAME L_inf=0.20 source ball
    - barrier_pgd: strong ordinary gradient baseline with semantic barriers
    - semantic_nullspace: project the victim feature-collision gradient into the
      first-order null space of an independent semantic Jacobian.

The semantic Jacobian has two rows:
  c1(x) = D_sem(x, x_target)
  c2(x) = reference source-class logit margin
and the null-space direction is

  d = -[I - J^T (J J^T + lambda I)^(-1) J] grad L_v.

Thus J d ~= 0: the step reduces victim representation distance while preserving
both semantic separation and source-class evidence to first order.  A shared
backtracking feasibility check enforces D_sem >= Delta, reference source class,
pixels in [0,1], and the L_inf source ball for BOTH refiners.

This is still a white-box mechanism pilot.  DH/BBHT is not claimed as novel; the
novel hypothesis being tested is the coupling of semantic second-preimage search
with semantic-Jacobian null-space transport.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils as tvutils

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from exp1_mnist_qco import SmallCNN, embed_dataset, qco_min_find, random_min_find, seed_all, train_victim
from exp1b_mnist_strong_baselines import pixel_rank
from exp1c_qdsa_semantic_gate import CALIBRATION_N, SemanticMLP, calibrate_delta, raw_flat
from exp2_qdsa_attack_effect_v2 import POOL_SELECTION_SEED_BASE, legal_mask_for_target, select_balanced_feasible_targets
from exp3a_certified_raw_collision import embed_raw_victim, certified_radius, raw_distances
from exp3b_continuous_existence import victim_raw, reference_forward

EPS_LINF = 0.20
SEED_BUDGET = 128
REFINE_STEPS = 120
STEP_SIZE = 0.020
NULL_DAMPING = 1e-6
SEM_GUARD = 0.030
CLS_GUARD = 0.75
LAMBDA_SEM = 120.0
LAMBDA_CLS = 2.0
BACKTRACKS = 7
STOCHASTIC_REPEATS = (17, 29)
CHECKPOINTS = (0, 20, 40, 80, 120)
EPS = 1e-12


def source_margin(logits: torch.Tensor, source_y: int) -> torch.Tensor:
    mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
    mask[source_y] = False
    other = torch.max(logits[..., mask], dim=-1).values
    return logits[..., source_y] - other


def project_box(x: torch.Tensor, x0: torch.Tensor, eps: float = EPS_LINF) -> torch.Tensor:
    lo = torch.clamp(x0 - eps, 0.0, 1.0)
    hi = torch.clamp(x0 + eps, 0.0, 1.0)
    return torch.maximum(torch.minimum(x, hi), lo)


def state_metrics(
    victim: SmallCNN,
    ref: SemanticMLP,
    x: torch.Tensor,
    h_target: torch.Tensor,
    ref_h_target: torch.Tensor,
    target_class: int,
    source_class: int,
    radius: float,
    semantic_delta: float,
) -> Dict[str, float]:
    with torch.no_grad():
        vlog, h = victim_raw(victim, x)
        rlog, rh = reference_forward(ref, x)
        raw_d = float(torch.linalg.vector_norm(h - h_target))
        sem_d = float(torch.linalg.vector_norm(rh - ref_h_target))
        margin = float(source_margin(rlog, source_class))
        vp = int(vlog.argmax(1).item())
        rp = int(rlog.argmax(1).item())
    feasible = bool(sem_d >= semantic_delta and rp == source_class)
    empirical = bool(feasible and vp == target_class and source_class != target_class)
    certified = bool(feasible and raw_d < radius and source_class != target_class)
    return {
        "raw_distance": raw_d,
        "semantic_distance": sem_d,
        "source_margin": margin,
        "victim_pred": vp,
        "reference_pred": rp,
        "feasible": int(feasible),
        "empirical_success": int(empirical),
        "certified_success": int(certified),
    }


def normalized_step(direction: torch.Tensor, step_size: float) -> torch.Tensor:
    scale = torch.max(torch.abs(direction)).clamp_min(1e-12)
    return step_size * direction / scale


def feasible_candidate(
    ref: SemanticMLP,
    cand: torch.Tensor,
    ref_h_target: torch.Tensor,
    source_class: int,
    semantic_delta: float,
) -> bool:
    with torch.no_grad():
        rlog, rh = reference_forward(ref, cand)
        sem_d = float(torch.linalg.vector_norm(rh - ref_h_target))
        rp = int(rlog.argmax(1).item())
    return bool(sem_d >= semantic_delta and rp == source_class)


def refine_one(
    victim: SmallCNN,
    ref: SemanticMLP,
    x0: torch.Tensor,
    h_target: torch.Tensor,
    ref_h_target: torch.Tensor,
    target_class: int,
    source_class: int,
    radius: float,
    semantic_delta: float,
    mode: str,
    steps: int,
    step_size: float,
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]], torch.Tensor]:
    """Return best feasible state, checkpoints, and best image."""
    x = x0.clone().detach()
    best = state_metrics(victim, ref, x, h_target, ref_h_target, target_class, source_class, radius, semantic_delta)
    best_x = x.clone().detach()
    checkpoints: Dict[int, Dict[str, float]] = {0: dict(best)}
    total_backtracks = 0
    rejected_steps = 0
    null_rank_sum = 0.0

    for step in range(1, steps + 1):
        xv = x.clone().detach().requires_grad_(True)
        vlog, h = victim_raw(victim, xv)
        rlog, rh = reference_forward(ref, xv)
        vloss = 0.5 * torch.sum((h - h_target) ** 2)
        sem_d = torch.linalg.vector_norm(rh - ref_h_target)
        cls_margin = source_margin(rlog, source_class)

        if mode == "barrier_pgd":
            sem_pen = F.relu((semantic_delta + SEM_GUARD) - sem_d) ** 2
            cls_pen = F.relu(CLS_GUARD - cls_margin) ** 2
            loss = vloss + LAMBDA_SEM * sem_pen + LAMBDA_CLS * cls_pen
            grad = torch.autograd.grad(loss, xv, retain_graph=False)[0]
            direction = -grad
            null_rank = 0
        elif mode == "semantic_nullspace":
            g_v = torch.autograd.grad(vloss, xv, retain_graph=True)[0].reshape(-1)
            g_sem = torch.autograd.grad(sem_d, xv, retain_graph=True)[0].reshape(-1)
            g_cls = torch.autograd.grad(cls_margin, xv, retain_graph=False)[0].reshape(-1)
            J = torch.stack([g_sem, g_cls], dim=0)
            # Remove numerically degenerate semantic rows before solving.
            norms = torch.linalg.vector_norm(J, dim=1)
            keep = norms > 1e-10
            J = J[keep]
            null_rank = int(J.shape[0])
            if null_rank:
                gram = J @ J.T + NULL_DAMPING * torch.eye(null_rank, device=J.device, dtype=J.dtype)
                rhs = J @ g_v
                coeff = torch.linalg.solve(gram, rhs)
                projected = g_v - J.T @ coeff
            else:
                projected = g_v
            direction = (-projected).reshape_as(xv)
        else:
            raise ValueError(mode)

        null_rank_sum += null_rank
        base_delta = normalized_step(direction, step_size)
        accepted = False
        for bt in range(BACKTRACKS + 1):
            frac = 0.5 ** bt
            cand = project_box(x + frac * base_delta, x0)
            if feasible_candidate(ref, cand, ref_h_target, source_class, semantic_delta):
                x = cand.detach()
                total_backtracks += bt
                accepted = True
                break
        if not accepted:
            rejected_steps += 1

        m = state_metrics(victim, ref, x, h_target, ref_h_target, target_class, source_class, radius, semantic_delta)
        if m["feasible"] and m["raw_distance"] < best["raw_distance"] - 1e-10:
            best = dict(m)
            best_x = x.clone().detach()

        if step in CHECKPOINTS:
            checkpoints[step] = dict(best)

    best.update({
        "total_backtracks": total_backtracks,
        "rejected_steps": rejected_steps,
        "mean_nullspace_constraint_rank": null_rank_sum / max(steps, 1),
    })
    return best, checkpoints, best_x


def choose_seed(
    method: str,
    distances: np.ndarray,
    sem_pool: np.ndarray,
    target_pix: np.ndarray,
    pool_pix: np.ndarray,
    budget: int,
    rng_seed: int,
) -> Tuple[int, int]:
    rng = np.random.default_rng(rng_seed)
    if method == "dh_bbht":
        q = qco_min_find(distances, budget, rng)
        return int(q.best_index), int(q.strict_queries)
    if method == "random":
        idx, _, used = random_min_find(distances, budget, rng)
        return int(idx), int(used)
    if method == "pixel_nn":
        order = pixel_rank(target_pix, pool_pix)
        queried = order[: min(budget, len(order))]
        idx = int(queried[np.argmin(distances[queried])])
        return idx, int(len(queried))
    if method == "semantic_boundary":
        order = np.argsort(sem_pool, kind="stable")
        queried = order[: min(budget, len(order))]
        idx = int(queried[np.argmin(distances[queried])])
        return idx, int(len(queried))
    raise ValueError(method)


def aggregate(rows: List[dict]) -> List[dict]:
    out: List[dict] = []
    seed_methods = sorted({r["seed_method"] for r in rows})
    refine_modes = sorted({r["refine_mode"] for r in rows})
    for sm in seed_methods:
        for rm in refine_modes:
            cell = [r for r in rows if r["seed_method"] == sm and r["refine_mode"] == rm]
            targets = sorted({r["target_test_index"] for r in cell})
            target_emp, target_cert, target_dist, target_seed, target_bt, target_rej = [], [], [], [], [], []
            cp_rates: Dict[int, List[float]] = {c: [] for c in CHECKPOINTS}
            for t in targets:
                g = [r for r in cell if r["target_test_index"] == t]
                target_emp.append(float(np.mean([r["best_empirical_success"] for r in g])))
                target_cert.append(float(np.mean([r["best_certified_success"] for r in g])))
                target_dist.append(float(np.mean([r["best_raw_distance"] for r in g])))
                target_seed.append(float(np.mean([r["seed_raw_distance"] for r in g])))
                target_bt.append(float(np.mean([r["total_backtracks"] for r in g])))
                target_rej.append(float(np.mean([r["rejected_steps"] for r in g])))
                for c in CHECKPOINTS:
                    cp_rates[c].append(float(np.mean([r[f"success_at_{c}"] for r in g])))
            rec = {
                "seed_method": sm,
                "refine_mode": rm,
                "n_independent_targets": len(targets),
                "target_averaged_asr": float(np.mean(target_emp)),
                "target_averaged_certified_asr": float(np.mean(target_cert)),
                "mean_seed_raw_distance": float(np.mean(target_seed)),
                "mean_best_raw_distance": float(np.mean(target_dist)),
                "mean_backtracks": float(np.mean(target_bt)),
                "mean_rejected_steps": float(np.mean(target_rej)),
            }
            for c in CHECKPOINTS:
                rec[f"asr_at_step_{c}"] = float(np.mean(cp_rates[c]))
            out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp4a_out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-seed", type=int, default=20260809)
    ap.add_argument("--targets", type=int, default=10)
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--seed-budget", type=int, default=SEED_BUDGET)
    ap.add_argument("--steps", type=int, default=REFINE_STEPS)
    ap.add_argument("--step-size", type=float, default=STEP_SIZE)
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
    pix = raw_flat(test_x)
    for p in victim.parameters(): p.requires_grad_(False)
    for p in ref.parameters(): p.requires_grad_(False)

    rows: List[dict] = []
    example_panels: List[torch.Tensor] = []
    seed_methods = ("dh_bbht", "random", "pixel_nn", "semantic_boundary")
    refine_modes = ("barrier_pgd", "semantic_nullspace")

    for target_ord, t0 in enumerate(target_indices):
        t = int(t0)
        target_class = int(preds[t])
        x_t = test_x[t:t+1].to(device)
        h_t = torch.from_numpy(raw_h_np[t:t+1]).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            _, ref_h_t = reference_forward(ref, x_t)
        radius = certified_radius(victim, logits_np[t], target_class)

        legal, sem_all = legal_mask_for_target(t, labels, ref_labels, ref_preds, ref_emb_test, semantic_delta)
        eligible = np.flatnonzero(legal)
        pool_rng = np.random.default_rng(POOL_SELECTION_SEED_BASE + t * 23)
        pool_idx = pool_rng.choice(eligible, size=args.pool, replace=False)
        distances = raw_distances(raw_h_np[t], raw_h_np[pool_idx])
        sem_pool = sem_all[pool_idx]
        pool_pix = pix[pool_idx]

        for sm in seed_methods:
            repeats = STOCHASTIC_REPEATS if sm in ("dh_bbht", "random") else (0,)
            for rep in repeats:
                seed_rng = int(500000 + t * 131 + rep * 1009 + sum(ord(ch) for ch in sm))
                local_idx, seed_queries = choose_seed(
                    sm, distances, sem_pool, pix[t], pool_pix,
                    args.seed_budget, seed_rng,
                )
                global_idx = int(pool_idx[local_idx])
                x0 = test_x[global_idx:global_idx+1].to(device)
                source_class = int(labels[global_idx])
                seed_raw = float(distances[local_idx])
                seed_sem = float(sem_pool[local_idx])

                # The legal pool construction guarantees semantic legality at step 0.
                with torch.no_grad():
                    r0, _ = reference_forward(ref, x0)
                    if int(r0.argmax(1).item()) != source_class or seed_sem < semantic_delta:
                        raise AssertionError("illegal seed")

                for rm in refine_modes:
                    best, cps, best_x = refine_one(
                        victim, ref, x0, h_t, ref_h_t,
                        target_class, source_class, radius, semantic_delta,
                        rm, args.steps, args.step_size,
                    )
                    row = {
                        "target_order": target_ord,
                        "target_test_index": t,
                        "target_class": target_class,
                        "seed_method": sm,
                        "seed_repeat": rep,
                        "refine_mode": rm,
                        "seed_test_index": global_idx,
                        "source_class": source_class,
                        "seed_queries": seed_queries,
                        "seed_raw_distance": seed_raw,
                        "seed_semantic_distance": seed_sem,
                        "certified_radius": radius,
                        "best_raw_distance": best["raw_distance"],
                        "best_semantic_distance": best["semantic_distance"],
                        "best_source_margin": best["source_margin"],
                        "best_victim_pred": best["victim_pred"],
                        "best_reference_pred": best["reference_pred"],
                        "best_empirical_success": best["empirical_success"],
                        "best_certified_success": best["certified_success"],
                        "total_backtracks": best["total_backtracks"],
                        "rejected_steps": best["rejected_steps"],
                        "mean_nullspace_constraint_rank": best["mean_nullspace_constraint_rank"],
                    }
                    for c in CHECKPOINTS:
                        cp = cps.get(c, best)
                        row[f"success_at_{c}"] = cp["empirical_success"]
                        row[f"raw_distance_at_{c}"] = cp["raw_distance"]
                    rows.append(row)

                    if target_ord < 4 and sm == "dh_bbht" and rep == STOCHASTIC_REPEATS[0]:
                        example_panels.extend([x_t.cpu(), x0.cpu(), best_x.cpu()])

                print(
                    f"target={t} seed={sm}/{rep} src={source_class} "
                    f"seed_d={seed_raw:.2f} q={seed_queries}", flush=True,
                )

    with (out / "trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = aggregate(rows)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    if example_panels:
        tvutils.save_image(torch.cat(example_panels, dim=0), out / "qco_examples_target_seed_refined.png", nrow=3, padding=2)

    # Direct paired mechanism contrast: null-space minus barrier at target level,
    # averaging stochastic seed repeats first.
    contrasts = []
    for sm in seed_methods:
        diffs_asr, diffs_dist = [], []
        for t in sorted({r["target_test_index"] for r in rows}):
            a = [r for r in rows if r["target_test_index"] == t and r["seed_method"] == sm and r["refine_mode"] == "semantic_nullspace"]
            b = [r for r in rows if r["target_test_index"] == t and r["seed_method"] == sm and r["refine_mode"] == "barrier_pgd"]
            diffs_asr.append(float(np.mean([r["best_empirical_success"] for r in a]) - np.mean([r["best_empirical_success"] for r in b])))
            diffs_dist.append(float(np.mean([r["best_raw_distance"] for r in a]) - np.mean([r["best_raw_distance"] for r in b])))
        contrasts.append({
            "seed_method": sm,
            "nullspace_minus_barrier_asr": float(np.mean(diffs_asr)),
            "nullspace_minus_barrier_raw_distance": float(np.mean(diffs_dist)),
            "targets_nullspace_asr_better": int(np.sum(np.asarray(diffs_asr) > 0)),
            "targets_barrier_asr_better": int(np.sum(np.asarray(diffs_asr) < 0)),
            "targets_nullspace_distance_better": int(np.sum(np.asarray(diffs_dist) < -1e-9)),
            "targets_barrier_distance_better": int(np.sum(np.asarray(diffs_dist) > 1e-9)),
        })
    with (out / "mechanism_contrasts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(contrasts[0].keys())); w.writeheader(); w.writerows(contrasts)

    result = {
        "name": "QDSA-Experiment-4A-Jacobian-Nullspace-Pilot",
        "role": "mechanism pilot; standard DH/BBHT seed search + proposed semantic-nullspace continuous transport",
        "victim_accuracy": victim_acc,
        "reference_accuracy": ref_acc,
        "semantic_delta": semantic_delta,
        "eps_linf": EPS_LINF,
        "seed_budget_strict_oracle_equiv": args.seed_budget,
        "refine_steps": args.steps,
        "step_size": args.step_size,
        "targets": int(len(target_indices)),
        "target_class_balance": {str(c): int(np.sum(labels[target_indices] == c)) for c in range(10)},
        "pool_size": args.pool,
        "seed_methods": list(seed_methods),
        "refine_modes": list(refine_modes),
        "semantic_jacobian_constraints": ["D_sem to target", "independent-reference source-class logit margin"],
        "hard_feasibility": ["D_sem >= Delta", "reference source class", "L_inf source ball", "pixels [0,1]"],
        "quantum_novelty_claim": "none for DH/BBHT; hypothesis concerns coupling with semantic-Jacobian null-space transport",
        "semantic_calibration_pairs": n_sem_pairs,
        "mean_eligible_candidates": float(np.mean(list(feasible_counts.values()))),
        "rejected_for_capacity": int(rejected),
        "summary": summary,
        "mechanism_contrasts": contrasts,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("SUMMARY_JSON " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
