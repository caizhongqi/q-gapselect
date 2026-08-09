#!/usr/bin/env python3
"""Experiment 1: MNIST + SmallCNN + Quantum Collision Operator (QCO).

Purpose
-------
Mechanism-only test: under the same oracle-query budget, compare Random Search
against a statevector-simulated quantum minimum-finding routine built from BBHT
(Grover) steps. The objective is the L2 distance between L2-normalized penultimate
representations h(x) and h(x0). This experiment does NOT claim an end-to-end
hardware speedup and does NOT yet measure attack success.

Important accounting
--------------------
The full candidate distance table is computed once only so that a classical
statevector simulator can emulate coherent phase marking. Those precomputations
are explicitly outside the reported query count. The primary budget uses a
conservative victim-evaluation equivalent:

  Grover phase-mark iterate = 2 victim-oracle calls (compute + uncompute)
  measured-candidate verification = 1 victim-oracle call
  Random Search candidate check = 1 victim-oracle call

We also record the less conservative logical predicate-query count.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


BUDGETS = (16, 32, 64, 128, 256)
SIM_SEEDS = (11, 22, 33, 44, 55)
DEFAULT_POOL = 512
DEFAULT_TARGETS = 20
BBHT_LAMBDA = 6.0 / 5.0
EPS = 1e-12


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SmallCNN(nn.Module):
    """Victim fixed to the latest Experiment-1 design."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor, return_feature: bool = False):
        z = self.features(x)
        h = self.fc(z.flatten(1))
        logits = self.classifier(h)
        if return_feature:
            return logits, F.normalize(h, p=2, dim=1)
        return logits


def train_victim(model: nn.Module, loader: DataLoader, device: torch.device, epochs: int) -> None:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        model.train()
        n = good = 0
        loss_sum = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            n += len(y)
            good += int((logits.argmax(1) == y).sum())
            loss_sum += float(loss.detach()) * len(y)
        print(f"epoch={ep+1} train_loss={loss_sum/n:.5f} train_acc={good/n:.4f}", flush=True)


@torch.inference_mode()
def embed_dataset(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ys, preds, hs = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits, h = model(x, return_feature=True)
        ys.append(y.numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        hs.append(h.cpu().numpy().astype(np.float64))
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(hs)


def l2_distances(h0: np.ndarray, h: np.ndarray) -> np.ndarray:
    # h and h0 are already unit-normalized.
    return np.linalg.norm(h - h0[None, :], axis=1)


def grover_measure(marked: np.ndarray, iterations: int, rng: np.random.Generator) -> int:
    """Exact statevector simulation on the candidate-index register."""
    n = int(marked.size)
    psi = np.full(n, 1.0 / math.sqrt(n), dtype=np.float64)
    for _ in range(iterations):
        # QCO phase marking: candidates closer than the current incumbent get -1.
        psi[marked] *= -1.0
        # Grover diffusion about the mean.
        psi = 2.0 * psi.mean() - psi
    probs = psi * psi
    probs /= probs.sum()
    return int(rng.choice(n, p=probs))


@dataclass
class QCOResult:
    best_index: int
    best_distance: float
    strict_queries: int
    logical_queries: int
    improvements: int
    phase_iterations: int
    verification_queries: int


def qco_min_find(distances: np.ndarray, budget: int, rng: np.random.Generator) -> QCOResult:
    """Durr-Hoyer-style adaptive minimum finding using BBHT search subroutines.

    The budget is enforced on strict victim-evaluation-equivalent queries:
    2 per Grover phase-mark iterate + 1 per measured-candidate verification.
    """
    n = len(distances)
    init = int(rng.integers(n))
    best_idx = init
    best = float(distances[init])
    strict_q = 1
    logical_q = 1
    verifications = 1
    phase_iters = 0
    improvements = 0

    while strict_q < budget:
        marked = distances < (best - EPS)
        if not np.any(marked):
            break

        m = 1.0
        improved_this_round = False
        while strict_q < budget:
            remaining = budget - strict_q
            # r Grover iterations + one verification must fit strict budget:
            # 2*r + 1 <= remaining.
            max_r_budget = max(0, (remaining - 1) // 2)
            if remaining < 1:
                break
            width = max(1, int(math.ceil(m)))
            max_r = min(width - 1, max_r_budget)
            if max_r < 0:
                break
            r = int(rng.integers(0, max_r + 1)) if max_r > 0 else 0

            idx = grover_measure(marked, r, rng)
            strict_q += 2 * r + 1
            logical_q += r + 1
            phase_iters += r
            verifications += 1

            d = float(distances[idx])
            if d < best - EPS:
                best = d
                best_idx = idx
                improvements += 1
                improved_this_round = True
                break

            m = min(BBHT_LAMBDA * m, math.sqrt(n))
            if strict_q >= budget:
                break

        if not improved_this_round and strict_q >= budget:
            break
        if not improved_this_round and m >= math.sqrt(n):
            # Continue only if there is budget; BBHT may still fail stochastically.
            continue

    return QCOResult(
        best_index=best_idx,
        best_distance=best,
        strict_queries=strict_q,
        logical_queries=logical_q,
        improvements=improvements,
        phase_iterations=phase_iters,
        verification_queries=verifications,
    )


def random_min_find(distances: np.ndarray, budget: int, rng: np.random.Generator) -> Tuple[int, float, int]:
    q = min(int(budget), len(distances))
    idxs = rng.choice(len(distances), size=q, replace=False)
    local = int(np.argmin(distances[idxs]))
    idx = int(idxs[local])
    return idx, float(distances[idx]), q


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def summarize(rows: List[dict]) -> List[dict]:
    out = []
    keys = sorted({(r["condition"], r["budget"]) for r in rows})
    for condition, budget in keys:
        cell = [r for r in rows if r["condition"] == condition and r["budget"] == budget]
        q = np.array([r["qco_best_distance"] for r in cell], float)
        c = np.array([r["random_best_distance"] for r in cell], float)
        opt = np.array([r["global_best_distance"] for r in cell], float)
        wins = int(np.sum(q < c - EPS))
        losses = int(np.sum(c < q - EPS))
        ties = len(cell) - wins - losses
        out.append({
            "condition": condition,
            "budget": budget,
            "n_trials": len(cell),
            "qco_mean_best_l2": float(q.mean()),
            "random_mean_best_l2": float(c.mean()),
            "qco_median_best_l2": float(np.median(q)),
            "random_median_best_l2": float(np.median(c)),
            "mean_relative_improvement_pct": float(100.0 * np.mean((c - q) / np.maximum(c, 1e-12))),
            "paired_qco_wins": wins,
            "paired_random_wins": losses,
            "ties": ties,
            "qco_win_rate": float(wins / len(cell)),
            "two_sided_sign_p": exact_sign_p(wins, losses),
            "qco_exact_global_min_rate": float(np.mean(np.isclose(q, opt, atol=1e-12))),
            "random_exact_global_min_rate": float(np.mean(np.isclose(c, opt, atol=1e-12))),
            "qco_mean_opt_gap": float(np.mean(q - opt)),
            "random_mean_opt_gap": float(np.mean(c - opt)),
            "qco_mean_strict_queries": float(np.mean([r["qco_strict_queries"] for r in cell])),
            "qco_mean_logical_queries": float(np.mean([r["qco_logical_queries"] for r in cell])),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp1_qco_out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-seed", type=int, default=20260809)
    ap.add_argument("--targets", type=int, default=DEFAULT_TARGETS)
    ap.add_argument("--pool", type=int, default=DEFAULT_POOL)
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
    test_acc = float(np.mean(labels == preds))
    print(f"test_accuracy={test_acc:.5f}", flush=True)
    if test_acc < 0.97:
        raise RuntimeError(f"Victim accuracy gate failed: {test_acc:.4f} < 0.97")

    correct = np.flatnonzero(labels == preds)
    target_rng = np.random.default_rng(2026080903)
    target_indices = target_rng.choice(correct, size=min(args.targets, len(correct)), replace=False)

    protocol = {
        "name": "Experiment-1-MNIST-SmallCNN-QCO",
        "objective": "minimize L2 distance between normalized penultimate embeddings under matched query budgets",
        "victim": "SmallCNN conv32-conv64-fc128",
        "dataset": "MNIST",
        "candidate_source": "MNIST test set; no perturbation synthesis",
        "pool_size": args.pool,
        "targets": int(len(target_indices)),
        "conditions": ["all_candidates", "cross_semantic"],
        "budgets_strict_victim_eval_equiv": list(BUDGETS),
        "simulation_seeds": list(SIM_SEEDS),
        "qco": {
            "type": "statevector BBHT / adaptive minimum finding",
            "bbht_lambda": BBHT_LAMBDA,
            "phase_mark_rule": "distance < incumbent distance",
            "phase_iter_strict_cost": 2,
            "verification_strict_cost": 1,
            "note": "distance table is precomputed only to emulate coherent marking; it is excluded from conditional query count",
        },
        "random": {"type": "uniform without-replacement candidate evaluation", "query_cost_per_candidate": 1},
        "attack_success_measured": False,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    rows: List[dict] = []
    all_indices = np.arange(len(test_ds))

    for target_ord, t in enumerate(target_indices):
        h0 = emb[t]
        y0 = int(labels[t])
        for condition in ("all_candidates", "cross_semantic"):
            mask = all_indices != int(t)
            if condition == "cross_semantic":
                mask &= labels != y0
            eligible = all_indices[mask]
            if len(eligible) < args.pool:
                raise RuntimeError("Not enough eligible candidates for fixed pool")

            # Pool is fixed per target/condition, independent of the method and budget.
            pool_rng = np.random.default_rng(50_000 + int(t) * 17 + (0 if condition == "all_candidates" else 1))
            pool_idx = pool_rng.choice(eligible, size=args.pool, replace=False)
            distances = l2_distances(h0, emb[pool_idx])
            global_best = float(np.min(distances))
            global_best_local = int(np.argmin(distances))

            for sim_seed in SIM_SEEDS:
                for budget in BUDGETS:
                    # Independent but reproducible RNG streams for paired methods.
                    base_seed = int(sim_seed * 1_000_003 + int(t) * 97 + budget * 13 + (0 if condition == "all_candidates" else 7))
                    q_rng = np.random.default_rng(base_seed)
                    r_rng = np.random.default_rng(base_seed + 1)

                    qres = qco_min_find(distances, budget, q_rng)
                    ridx, rbest, rq = random_min_find(distances, budget, r_rng)
                    rows.append({
                        "target_order": target_ord,
                        "target_test_index": int(t),
                        "target_label": y0,
                        "condition": condition,
                        "pool_size": args.pool,
                        "budget": budget,
                        "sim_seed": sim_seed,
                        "global_best_distance": global_best,
                        "global_best_test_index": int(pool_idx[global_best_local]),
                        "qco_best_distance": qres.best_distance,
                        "qco_best_test_index": int(pool_idx[qres.best_index]),
                        "qco_best_label": int(labels[pool_idx[qres.best_index]]),
                        "qco_strict_queries": qres.strict_queries,
                        "qco_logical_queries": qres.logical_queries,
                        "qco_improvements": qres.improvements,
                        "qco_phase_iterations": qres.phase_iterations,
                        "qco_verification_queries": qres.verification_queries,
                        "random_best_distance": rbest,
                        "random_best_test_index": int(pool_idx[ridx]),
                        "random_best_label": int(labels[pool_idx[ridx]]),
                        "random_queries": rq,
                    })

    trial_fields = list(rows[0].keys())
    with (out / "trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=trial_fields)
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows)
    summary_fields = list(summary[0].keys())
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary)

    result = {
        "test_accuracy": test_acc,
        "n_targets": int(len(target_indices)),
        "pool_size": args.pool,
        "summary": summary,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), out / "smallcnn.pt")

    print("SUMMARY_JSON " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
