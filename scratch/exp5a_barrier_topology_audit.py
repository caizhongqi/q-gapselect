#!/usr/bin/env python3
"""Experiment 5A: semantic-fiber barrier / basin-topology audit.

Motivation
----------
The 50-target Experiment 4C killed the apparent QCO x null-space interaction.
The stable quantum-side signal is only seed-search efficiency.  Before inventing
any quantum-tunnelling mechanism, this experiment tests whether the continuous
QDSA landscape actually contains the kind of separated legal basins for which a
quantum tunnelling argument could be scientifically meaningful.

Protocol
--------
- MNIST SmallCNN victim and independent SemanticMLP reference, unchanged.
- Strict semantic second-preimage gate D_sem >= Delta and reference source class.
- Fixed source L_inf ball eps=0.20.
- Seed: one pre-registered DH/BBHT run at strict budget 128.
- Deterministic strong constrained Adam from that seed.
- Eight stochastic initial restarts inside the SAME source ball, optimized with
  the SAME constrained Adam objective.  Restarts are an existence/topology audit,
  not a query-matched attack comparison.

For each target we ask:
1) Does deterministic Adam fail while another legal restart succeeds?  This is a
   basin-dependence / rescue event.
2) Are the deterministic and rescued endpoints separated by an observed energy
   rise along a direct interpolation or a piecewise path through the legal source
   seed?  We record legality fractions and barrier height/width only when the
   sampled path is entirely semantically legal.

Important limitation
--------------------
A positive barrier along a sampled feasible path is NOT a proof of the minimum
energy barrier: it is an observed path barrier (an upper bound on a minimax path
barrier).  This gate only decides whether a tunnelling-style quantum core is worth
formalizing; it does not itself claim a quantum speedup.
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

from exp1_mnist_qco import SmallCNN, embed_dataset, seed_all, train_victim
from exp1c_qdsa_semantic_gate import CALIBRATION_N, SemanticMLP, calibrate_delta, raw_flat
from exp2_qdsa_attack_effect_v2 import POOL_SELECTION_SEED_BASE, legal_mask_for_target, select_balanced_feasible_targets
from exp3a_certified_raw_collision import embed_raw_victim, certified_radius, raw_distances
from exp3b_continuous_existence import victim_raw, reference_forward
from exp4a_jacobian_nullspace_pilot import EPS_LINF, choose_seed, source_margin
from exp4b_strong_refiner_audit import ADAM_LR, ADAM_LAMBDA_SEM, ADAM_LAMBDA_CLS, metrics

SEED_BUDGET = 128
QCO_REPEAT = 17
RESTART_NOISE = 0.05
PATH_POINTS = 101
EPS = 1e-12


def project_to_source_ball(x: torch.Tensor, center: torch.Tensor, eps: float = EPS_LINF) -> torch.Tensor:
    lo = torch.clamp(center - eps, 0.0, 1.0)
    hi = torch.clamp(center + eps, 0.0, 1.0)
    return torch.maximum(torch.minimum(x, hi), lo)


def is_semantically_legal(ref, x, ref_h_t, source_class: int, delta: float) -> bool:
    with torch.no_grad():
        rlog, rh = reference_forward(ref, x)
        sem_d = float(torch.linalg.vector_norm(rh - ref_h_t))
        rp = int(rlog.argmax(1).item())
    return bool(sem_d >= delta and rp == source_class)


def constrained_objective(victim, ref, x, h_t, ref_h_t, source_class: int, delta: float):
    _, h = victim_raw(victim, x)
    rlog, rh = reference_forward(ref, x)
    feat = torch.mean((h - h_t) ** 2)
    sem_d = torch.linalg.vector_norm(rh - ref_h_t)
    sem_pen = F.relu(delta - sem_d) ** 2
    src_ce = F.cross_entropy(rlog, torch.tensor([source_class], device=x.device))
    return feat + ADAM_LAMBDA_SEM * sem_pen + ADAM_LAMBDA_CLS * src_ce


def run_adam_from_init(
    victim, ref,
    init_x: torch.Tensor,
    ball_center: torch.Tensor,
    h_t: torch.Tensor,
    ref_h_t: torch.Tensor,
    target_class: int,
    source_class: int,
    radius: float,
    delta: float,
    steps: int,
    lr: float,
) -> Tuple[Dict[str, float], torch.Tensor, Dict[str, float]]:
    """Strong Adam with a fixed source-ball center independent of initialization."""
    x = project_to_source_ball(init_x.clone().detach(), ball_center).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    best = metrics(victim, ref, x, h_t, ref_h_t, target_class, source_class, radius, delta)
    best_x = x.detach().clone()
    feasible_count = int(best["feasible"])
    success_first_step = 0 if best["empirical_success"] else -1

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = constrained_objective(victim, ref, x, h_t, ref_h_t, source_class, delta)
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.copy_(project_to_source_ball(x, ball_center))

        m = metrics(victim, ref, x, h_t, ref_h_t, target_class, source_class, radius, delta)
        feasible_count += int(m["feasible"])
        if success_first_step < 0 and m["empirical_success"]:
            success_first_step = step
        if m["feasible"] and m["raw_distance"] < best["raw_distance"] - 1e-10:
            best = dict(m)
            best_x = x.detach().clone()

    diag = {
        "feasible_iterate_fraction": feasible_count / float(steps + 1),
        "first_success_step": success_first_step,
    }
    return best, best_x, diag


def sample_legal_restart(
    ref,
    center: torch.Tensor,
    ref_h_t: torch.Tensor,
    source_class: int,
    delta: float,
    rng: np.random.Generator,
    noise_amp: float,
    max_attempts: int = 64,
) -> Tuple[torch.Tensor, int, float]:
    for attempt in range(1, max_attempts + 1):
        noise = torch.from_numpy(rng.uniform(-noise_amp, noise_amp, size=center.shape).astype(np.float32)).to(center.device)
        cand = project_to_source_ball(center + noise, center)
        if is_semantically_legal(ref, cand, ref_h_t, source_class, delta):
            linf = float(torch.max(torch.abs(cand - center)))
            return cand.detach(), attempt, linf
    return center.clone().detach(), max_attempts, 0.0


@torch.inference_mode()
def path_profile(
    victim,
    ref,
    points: torch.Tensor,
    h_t: torch.Tensor,
    ref_h_t: torch.Tensor,
    source_class: int,
    delta: float,
) -> Dict[str, float]:
    energies: List[float] = []
    sems: List[float] = []
    source_valid: List[int] = []
    for i in range(points.shape[0]):
        x = points[i:i+1]
        _, h = victim_raw(victim, x)
        rlog, rh = reference_forward(ref, x)
        e = 0.5 * float(torch.sum((h - h_t) ** 2))
        sd = float(torch.linalg.vector_norm(rh - ref_h_t))
        rp = int(rlog.argmax(1).item())
        energies.append(e)
        sems.append(sd)
        source_valid.append(int(rp == source_class))

    e = np.asarray(energies, dtype=float)
    sem = np.asarray(sems, dtype=float)
    src = np.asarray(source_valid, dtype=int)
    legal = (sem >= delta) & (src == 1)
    endpoint_level = max(float(e[0]), float(e[-1]))
    raw_barrier = max(0.0, float(e.max()) - endpoint_level)
    if raw_barrier > 0:
        half_level = endpoint_level + 0.5 * raw_barrier
        above = np.flatnonzero(e >= half_level)
        barrier_width = float((above[-1] - above[0]) / max(1, len(e) - 1)) if len(above) else 0.0
    else:
        barrier_width = 0.0
    fully_legal = bool(np.all(legal))
    return {
        "path_points": int(len(e)),
        "path_legal_fraction": float(np.mean(legal)),
        "path_source_class_fraction": float(np.mean(src)),
        "path_min_semantic_distance": float(sem.min()),
        "path_energy_start": float(e[0]),
        "path_energy_end": float(e[-1]),
        "path_energy_max": float(e.max()),
        "path_raw_barrier_height": raw_barrier,
        "path_barrier_width_halfheight": barrier_width,
        "path_fully_semantic_legal": int(fully_legal),
        "path_feasible_barrier_height": raw_barrier if fully_legal else float("nan"),
        "path_feasible_barrier_width": barrier_width if fully_legal else float("nan"),
    }


def interpolate(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    ts = torch.linspace(0.0, 1.0, n, device=a.device).reshape(n, 1, 1, 1)
    return (1.0 - ts) * a + ts * b


def via_center_path(a: torch.Tensor, center: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    n1 = n // 2 + 1
    n2 = n - n1 + 1
    p1 = interpolate(a, center, n1)
    p2 = interpolate(center, b, n2)
    return torch.cat([p1, p2[1:]], dim=0)


def bootstrap_ci(values: np.ndarray, seed: int = 2026080951, reps: int = 10000):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return [float("nan"), float("nan")]
    sims = np.empty(reps, dtype=float)
    for i in range(reps):
        sims[i] = float(np.mean(values[rng.integers(0, n, size=n)]))
    return [float(np.quantile(sims, 0.025)), float(np.quantile(sims, 0.975))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp5a_out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-seed", type=int, default=20260809)
    ap.add_argument("--targets", type=int, default=20)
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--seed-budget", type=int, default=SEED_BUDGET)
    ap.add_argument("--det-steps", type=int, default=120)
    ap.add_argument("--restart-steps", type=int, default=180)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--restart-noise", type=float, default=RESTART_NOISE)
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
        raise RuntimeError(f"accuracy gate failed victim={victim_acc:.4f} ref={ref_acc:.4f}")

    cal_ds = Subset(train_ds, range(CALIBRATION_N))
    cal_loader = DataLoader(cal_ds, batch_size=512, shuffle=False, num_workers=2)
    rcal_labels, rcal_preds, rcal_emb = embed_dataset(ref, cal_loader, device)
    good = rcal_labels == rcal_preds
    delta, n_sem_pairs = calibrate_delta(rcal_emb[good], rcal_labels[good])

    target_indices, feasible_counts, rejected = select_balanced_feasible_targets(
        labels, preds, ref_labels, ref_preds, ref_emb_test,
        delta, args.pool, args.targets,
    )
    test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))], dim=0)
    pix = raw_flat(test_x)
    for p in victim.parameters(): p.requires_grad_(False)
    for p in ref.parameters(): p.requires_grad_(False)

    target_rows: List[dict] = []
    restart_rows: List[dict] = []
    path_rows: List[dict] = []
    panels: List[torch.Tensor] = []

    for target_ord, t0 in enumerate(target_indices):
        t = int(t0)
        target_class = int(preds[t])
        x_t = test_x[t:t+1].to(device)
        h_t = torch.from_numpy(raw_h_np[t:t+1]).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            _, ref_h_t = reference_forward(ref, x_t)
        radius = certified_radius(victim, logits_np[t], target_class)

        legal, sem_all = legal_mask_for_target(t, labels, ref_labels, ref_preds, ref_emb_test, delta)
        eligible = np.flatnonzero(legal)
        pool_rng = np.random.default_rng(POOL_SELECTION_SEED_BASE + t * 23)
        pool_idx = pool_rng.choice(eligible, size=args.pool, replace=False)
        distances = raw_distances(raw_h_np[t], raw_h_np[pool_idx])
        sem_pool = sem_all[pool_idx]
        pool_pix = pix[pool_idx]

        qco_rng_seed = int(500000 + t * 131 + QCO_REPEAT * 1009 + sum(ord(ch) for ch in "dh_bbht"))
        local_idx, seed_queries = choose_seed(
            "dh_bbht", distances, sem_pool, pix[t], pool_pix,
            args.seed_budget, qco_rng_seed,
        )
        seed_idx = int(pool_idx[local_idx])
        source_class = int(labels[seed_idx])
        x0 = test_x[seed_idx:seed_idx+1].to(device)
        seed_raw = float(distances[local_idx])
        seed_sem = float(sem_pool[local_idx])
        if not is_semantically_legal(ref, x0, ref_h_t, source_class, delta):
            raise AssertionError("QCO seed is not semantically legal")

        det_best, det_x, det_diag = run_adam_from_init(
            victim, ref, x0, x0, h_t, ref_h_t,
            target_class, source_class, radius, delta,
            args.det_steps, ADAM_LR,
        )

        best_restart = None
        best_restart_x = None
        restart_successes = 0
        restart_cert_successes = 0
        endpoint_vectors: List[np.ndarray] = []

        for r in range(args.restarts):
            rrng = np.random.default_rng(8_000_000 + t * 1009 + r * 97)
            init, attempts, init_linf = sample_legal_restart(
                ref, x0, ref_h_t, source_class, delta,
                rrng, args.restart_noise,
            )
            rb, rx, rdiag = run_adam_from_init(
                victim, ref, init, x0, h_t, ref_h_t,
                target_class, source_class, radius, delta,
                args.restart_steps, ADAM_LR,
            )
            restart_successes += int(rb["empirical_success"])
            restart_cert_successes += int(rb["certified_success"])
            endpoint_vectors.append(rx.detach().cpu().numpy().reshape(-1))
            restart_rows.append({
                "target_order": target_ord,
                "target_test_index": t,
                "target_class": target_class,
                "source_class": source_class,
                "restart": r,
                "init_sampling_attempts": attempts,
                "init_linf_from_seed": init_linf,
                "best_raw_distance": rb["raw_distance"],
                "best_semantic_distance": rb["semantic_distance"],
                "best_source_margin": rb["source_margin"],
                "best_victim_pred": rb["victim_pred"],
                "best_reference_pred": rb["reference_pred"],
                "empirical_success": rb["empirical_success"],
                "certified_success": rb["certified_success"],
                "feasible_iterate_fraction": rdiag["feasible_iterate_fraction"],
                "first_success_step": rdiag["first_success_step"],
            })
            # Prefer successful endpoints; within same success status choose lower energy/distance.
            key = (int(rb["empirical_success"]), -float(rb["raw_distance"]))
            if best_restart is None:
                best_restart, best_restart_x, best_key = dict(rb), rx.clone(), key
            elif key > best_key:
                best_restart, best_restart_x, best_key = dict(rb), rx.clone(), key

        if best_restart is None or best_restart_x is None:
            raise RuntimeError("No restart result")

        # Basin diversity: endpoint pixel-space spread across stochastic restarts.
        if len(endpoint_vectors) >= 2:
            mat = np.stack(endpoint_vectors, axis=0)
            pair_d = []
            for i in range(len(mat)):
                for j in range(i + 1, len(mat)):
                    pair_d.append(float(np.linalg.norm(mat[i] - mat[j])))
            mean_endpoint_pair_l2 = float(np.mean(pair_d))
            max_endpoint_pair_l2 = float(np.max(pair_d))
        else:
            mean_endpoint_pair_l2 = 0.0
            max_endpoint_pair_l2 = 0.0

        rescued = int((not det_best["empirical_success"]) and bool(best_restart["empirical_success"]))
        any_restart_success = int(restart_successes > 0)

        direct_points = interpolate(det_x, best_restart_x, PATH_POINTS)
        via_points = via_center_path(det_x, x0, best_restart_x, PATH_POINTS)
        direct = path_profile(victim, ref, direct_points, h_t, ref_h_t, source_class, delta)
        via = path_profile(victim, ref, via_points, h_t, ref_h_t, source_class, delta)
        for path_name, prof in (("direct", direct), ("via_seed", via)):
            path_rows.append({
                "target_order": target_ord,
                "target_test_index": t,
                "target_class": target_class,
                "source_class": source_class,
                "deterministic_success": det_best["empirical_success"],
                "best_restart_success": best_restart["empirical_success"],
                "rescued_failure": rescued,
                "path_type": path_name,
                **prof,
            })

        target_rows.append({
            "target_order": target_ord,
            "target_test_index": t,
            "target_class": target_class,
            "source_test_index": seed_idx,
            "source_class": source_class,
            "seed_queries_strict": seed_queries,
            "seed_raw_distance": seed_raw,
            "seed_semantic_distance": seed_sem,
            "certified_radius": radius,
            "det_best_raw_distance": det_best["raw_distance"],
            "det_success": det_best["empirical_success"],
            "det_certified_success": det_best["certified_success"],
            "det_feasible_iterate_fraction": det_diag["feasible_iterate_fraction"],
            "restart_success_count": restart_successes,
            "restart_certified_success_count": restart_cert_successes,
            "restart_success_rate": restart_successes / float(args.restarts),
            "best_restart_raw_distance": best_restart["raw_distance"],
            "best_restart_success": best_restart["empirical_success"],
            "best_restart_certified_success": best_restart["certified_success"],
            "rescued_failure": rescued,
            "mean_restart_endpoint_pair_l2": mean_endpoint_pair_l2,
            "max_restart_endpoint_pair_l2": max_endpoint_pair_l2,
            "direct_path_fully_legal": direct["path_fully_semantic_legal"],
            "direct_path_legal_fraction": direct["path_legal_fraction"],
            "direct_path_barrier_height": direct["path_feasible_barrier_height"],
            "direct_path_barrier_width": direct["path_feasible_barrier_width"],
            "via_seed_path_fully_legal": via["path_fully_semantic_legal"],
            "via_seed_path_legal_fraction": via["path_legal_fraction"],
            "via_seed_path_barrier_height": via["path_feasible_barrier_height"],
            "via_seed_path_barrier_width": via["path_feasible_barrier_width"],
        })

        if target_ord < 10:
            panels.extend([x_t.cpu(), x0.cpu(), det_x.cpu(), best_restart_x.cpu()])

        print(
            f"target={t} y={target_class} src={source_class} q={seed_queries} "
            f"det={det_best['empirical_success']} restart={best_restart['empirical_success']} "
            f"rescued={rescued} restarts_ok={restart_successes}/{args.restarts} "
            f"d={det_best['raw_distance']:.2f}->{best_restart['raw_distance']:.2f} "
            f"direct_legal={direct['path_fully_semantic_legal']} via_legal={via['path_fully_semantic_legal']}",
            flush=True,
        )

    with (out / "per_target.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(target_rows[0].keys())); w.writeheader(); w.writerows(target_rows)
    with (out / "restarts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(restart_rows[0].keys())); w.writeheader(); w.writerows(restart_rows)
    with (out / "path_profiles.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(path_rows[0].keys())); w.writeheader(); w.writerows(path_rows)
    if panels:
        tvutils.save_image(torch.cat(panels, dim=0), out / "first10_target_seed_det_best_restart.png", nrow=4, padding=2)

    det = np.asarray([r["det_success"] for r in target_rows], dtype=float)
    anyr = np.asarray([r["best_restart_success"] for r in target_rows], dtype=float)
    rescue = np.asarray([r["rescued_failure"] for r in target_rows], dtype=float)
    seed_q = np.asarray([r["seed_queries_strict"] for r in target_rows], dtype=float)
    det_d = np.asarray([r["det_best_raw_distance"] for r in target_rows], dtype=float)
    rest_d = np.asarray([r["best_restart_raw_distance"] for r in target_rows], dtype=float)

    det_fail = det == 0
    direct_legal_rescue = [r for r in target_rows if r["rescued_failure"] and r["direct_path_fully_legal"]]
    via_legal_rescue = [r for r in target_rows if r["rescued_failure"] and r["via_seed_path_fully_legal"]]
    direct_barriers = np.asarray([r["direct_path_barrier_height"] for r in direct_legal_rescue], dtype=float)
    via_barriers = np.asarray([r["via_seed_path_barrier_height"] for r in via_legal_rescue], dtype=float)

    summary = {
        "name": "QDSA-Experiment-5A-Semantic-Fiber-Barrier-Topology-Audit",
        "role": "pre-tunnelling scientific gate; no quantum tunnelling claim",
        "victim_accuracy": victim_acc,
        "reference_accuracy": ref_acc,
        "semantic_delta": delta,
        "eps_linf": EPS_LINF,
        "targets": int(len(target_rows)),
        "pool_size": args.pool,
        "qco_seed_budget": args.seed_budget,
        "mean_qco_strict_queries_used": float(seed_q.mean()),
        "deterministic_adam_steps": args.det_steps,
        "restart_adam_steps": args.restart_steps,
        "restarts_per_target": args.restarts,
        "restart_noise_linf": args.restart_noise,
        "deterministic_asr": float(det.mean()),
        "any_restart_asr": float(anyr.mean()),
        "deterministic_failure_count": int(np.sum(det_fail)),
        "rescued_failure_count": int(np.sum(rescue)),
        "rescue_rate_among_deterministic_failures": float(np.sum(rescue) / max(1, np.sum(det_fail))),
        "rescue_rate_bootstrap_95ci_all_targets": bootstrap_ci(rescue),
        "mean_deterministic_best_raw_distance": float(det_d.mean()),
        "mean_best_restart_raw_distance": float(rest_d.mean()),
        "mean_restart_minus_deterministic_raw_distance": float(np.mean(rest_d - det_d)),
        "mean_restart_endpoint_pair_l2": float(np.mean([r["mean_restart_endpoint_pair_l2"] for r in target_rows])),
        "rescued_with_fully_legal_direct_path": int(len(direct_legal_rescue)),
        "rescued_with_fully_legal_via_seed_path": int(len(via_legal_rescue)),
        "mean_direct_observed_feasible_barrier_height_on_rescues": float(np.mean(direct_barriers)) if len(direct_barriers) else None,
        "median_direct_observed_feasible_barrier_height_on_rescues": float(np.median(direct_barriers)) if len(direct_barriers) else None,
        "mean_via_seed_observed_feasible_barrier_height_on_rescues": float(np.mean(via_barriers)) if len(via_barriers) else None,
        "median_via_seed_observed_feasible_barrier_height_on_rescues": float(np.median(via_barriers)) if len(via_barriers) else None,
        "semantic_calibration_pairs": n_sem_pairs,
        "mean_eligible_candidates": float(np.mean(list(feasible_counts.values()))),
        "rejected_for_capacity": int(rejected),
        "decision_rule": {
            "support_tunnelling_followup": "nontrivial deterministic-failure rescue rate plus separated legal endpoints with observed legal-path energy rises; then compare thermal/annealing and only afterwards formulate a semantic-fiber Hamiltonian",
            "reject_tunnelling_story": "restarts do not rescue failures, or ordinary restarts solve nearly everything without evidence of separated basins/barriers",
        },
        "limitation": "sampled-path barrier heights are observed path barriers, not proofs of the minimum feasible minimax barrier",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY_JSON " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
