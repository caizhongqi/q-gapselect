#!/usr/bin/env python3
"""Experiment 6A: semantic representation near-claw feasibility gate.

Why this experiment
-------------------
The learned-surrogate audit (5QC) falsified the idea that standard target-fixed
DH/BBHT minimum finding is the right quantum core: a classical online surrogate
can learn that scalar objective very effectively.  6A returns to the collision
primitive itself.

We form two semantically distinct *domains* of inputs rather than one fixed target.
Each domain is a public, semantic-preserving L_inf=0.20 perturbation cloud around a
correctly classified natural center.  For adjacent class pairs c -> (c+1) mod 10,
we search offline for cross-domain representation claws.

Attack-side representation fingerprint
--------------------------------------
The victim raw penultimate vector h in R^128 is standardized using a fixed training
calibration subset, projected by a fixed random 6-D Gaussian projection, and grid
quantized.  The quantization width is calibrated on TRAIN cross-class pairs only
to a rare fingerprint-collision rate near 1e-4.  Equal fingerprints are therefore
an attacker-defined exact claw predicate; they are not claimed to be an internal
"neural hash" implemented by the victim.

Every fingerprint claw is then verified using the full raw victim representation.
We report:
  - semantic validity D_sem >= Delta,
  - raw representation distance,
  - hard-label aliasing,
  - a certified second-preimage condition: the target-domain point is classified
    as its target semantic class and the source-domain point falls inside that
    target point's linear-head certified raw-feature radius.

If certified claws do not exist with useful density, a claw-finding quantum walk
has no attack object and this direction should be rejected before implementation.
If they do exist, 6B can compare classical learned collision discovery with a
proper claw/relation-finding quantum walk.  6A itself makes no quantum speedup or
novel-algorithm claim.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
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
from exp1c_qdsa_semantic_gate import CALIBRATION_N, SemanticMLP, calibrate_delta
from exp3a_certified_raw_collision import embed_raw_victim
from exp3b_continuous_existence import reference_forward, victim_raw

EPS_LINF = 0.20
PROJ_DIM = 6
TARGET_CALIBRATION_COLLISION_RATE = 1e-4
CALIBRATION_PAIR_SAMPLES = 300_000
WIDTH_GRID = (0.35, 0.50, 0.70, 0.90, 1.10, 1.35, 1.65, 2.00)
N_PREFIXES = (128, 256, 512, 1024)
EPS = 1e-12


def pairwise_raw_min(a: np.ndarray, b: np.ndarray) -> Tuple[int, int, float]:
    an = np.sum(a * a, axis=1, keepdims=True)
    bn = np.sum(b * b, axis=1, keepdims=True).T
    d2 = an + bn - 2.0 * (a @ b.T)
    np.maximum(d2, 0.0, out=d2)
    flat = int(np.argmin(d2))
    i, j = np.unravel_index(flat, d2.shape)
    return int(i), int(j), float(math.sqrt(float(d2[i, j])))


def certified_radii(model: SmallCNN, logits: np.ndarray, expected_class: int) -> np.ndarray:
    w = model.classifier.weight.detach().cpu().numpy().astype(np.float64)
    denom = np.linalg.norm(w[expected_class][None, :] - w, axis=1)
    out = np.zeros(len(logits), dtype=np.float64)
    for i, z in enumerate(logits):
        if int(np.argmax(z)) != expected_class:
            continue
        vals = []
        for j in range(w.shape[0]):
            if j == expected_class:
                continue
            margin = float(z[expected_class] - z[j])
            if margin <= 0 or denom[j] <= EPS:
                vals = []
                break
            vals.append(margin / float(denom[j]))
        if vals:
            out[i] = min(vals)
    return out


def choose_centers(
    labels: np.ndarray,
    preds: np.ndarray,
    ref_preds: np.ndarray,
    ref_emb: np.ndarray,
    delta: float,
) -> Dict[int, int]:
    """One deterministic center per class, ensuring adjacent class-pair separation."""
    good = (labels == preds) & (labels == ref_preds)
    pools = {c: np.flatnonzero(good & (labels == c)) for c in range(10)}
    rng = np.random.default_rng(2026080960)
    for c in pools:
        pools[c] = rng.permutation(pools[c])

    # Greedy choose one per class, then verify c -> c+1 semantic separation.
    for attempt in range(200):
        centers = {c: int(pools[c][attempt % min(len(pools[c]), 200)]) for c in range(10)}
        ok = True
        for c in range(10):
            d = float(np.linalg.norm(ref_emb[centers[c]] - ref_emb[centers[(c + 1) % 10]]))
            if d < delta:
                ok = False
                break
        if ok:
            return centers
    # More flexible deterministic pairwise greedy fallback.
    centers = {}
    centers[0] = int(pools[0][0])
    for c in range(1, 10):
        prev = centers[c - 1]
        found = None
        for idx in pools[c][:500]:
            if float(np.linalg.norm(ref_emb[prev] - ref_emb[int(idx)])) >= delta:
                found = int(idx); break
        if found is None:
            raise RuntimeError(f"Could not find separated center for class {c}")
        centers[c] = found
    if float(np.linalg.norm(ref_emb[centers[9]] - ref_emb[centers[0]])) < delta:
        # Replace class 0 if necessary while preserving 0->1 and 9->0.
        for idx in pools[0][:1000]:
            idx = int(idx)
            if (float(np.linalg.norm(ref_emb[centers[9]] - ref_emb[idx])) >= delta
                and float(np.linalg.norm(ref_emb[idx] - ref_emb[centers[1]])) >= delta):
                centers[0] = idx; break
        else:
            raise RuntimeError("Could not close semantic class ring")
    return centers


def generate_cloud(
    center_x: torch.Tensor,
    semantic_class: int,
    victim: SmallCNN,
    ref: SemanticMLP,
    n: int,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    xs: List[torch.Tensor] = []
    raw_h: List[np.ndarray] = []
    logits: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    ref_h: List[np.ndarray] = []
    attempts = 0

    while sum(len(x) for x in xs) < n:
        batch = 2048
        noise = torch.from_numpy(rng.normal(size=(batch, 1, 7, 7)).astype(np.float32)).to(device)
        noise = F.interpolate(noise, size=(28, 28), mode="bilinear", align_corners=False)
        scale = noise.abs().flatten(1).amax(1).clamp_min(1e-8).view(-1, 1, 1, 1)
        noise = noise / scale
        amp = torch.from_numpy(rng.uniform(0.02, EPS_LINF, size=(batch, 1, 1, 1)).astype(np.float32)).to(device)
        cand = torch.clamp(center_x + amp * noise, 0.0, 1.0)
        with torch.no_grad():
            rlog, rh = reference_forward(ref, cand)
            keep = rlog.argmax(1) == semantic_class
        if torch.any(keep):
            kept = cand[keep]
            with torch.no_grad():
                vlog, vh = victim_raw(victim, kept)
                _, rrh = reference_forward(ref, kept)
            xs.append(kept.detach().cpu())
            raw_h.append(vh.detach().cpu().numpy().astype(np.float64))
            logits.append(vlog.detach().cpu().numpy().astype(np.float64))
            preds.append(vlog.argmax(1).detach().cpu().numpy())
            ref_h.append(rrh.detach().cpu().numpy().astype(np.float64))
        attempts += batch
        if attempts > n * 100:
            raise RuntimeError(f"Cloud generation acceptance too low for class {semantic_class}")

    x = torch.cat(xs, dim=0)[:n]
    h = np.concatenate(raw_h, axis=0)[:n]
    z = np.concatenate(logits, axis=0)[:n]
    p = np.concatenate(preds, axis=0)[:n]
    rh = np.concatenate(ref_h, axis=0)[:n]
    return {
        "x": x,
        "raw_h": h,
        "logits": z,
        "preds": p,
        "ref_h": rh,
        "attempted": attempts,
        "acceptance_rate": float(n / attempts),
    }


def projected_values(h: np.ndarray, mu: np.ndarray, sd: np.ndarray, proj: np.ndarray) -> np.ndarray:
    return ((h - mu[None, :]) / sd[None, :]) @ proj.T


def hash_codes(v: np.ndarray, width: float, offset_frac: np.ndarray) -> np.ndarray:
    offset = offset_frac[None, :] * width
    return np.floor((v + offset) / width).astype(np.int32)


def estimate_hash_collision_rate(codes: np.ndarray, labels: np.ndarray, pairs: int, seed: int) -> Tuple[float, int]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    hits = total = 0
    while total < pairs:
        m = min(100000, pairs - total)
        a = rng.integers(0, n, size=m)
        b = rng.integers(0, n, size=m)
        keep = (a != b) & (labels[a] != labels[b])
        if not np.any(keep):
            continue
        aa, bb = a[keep], b[keep]
        eq = np.all(codes[aa] == codes[bb], axis=1)
        hits += int(np.sum(eq))
        total += int(len(eq))
    return hits / float(total), hits


def calibrate_width(v: np.ndarray, labels: np.ndarray, offset_frac: np.ndarray) -> Tuple[float, List[dict]]:
    stats = []
    for width in WIDTH_GRID:
        c = hash_codes(v, width, offset_frac)
        rate, hits = estimate_hash_collision_rate(c, labels, CALIBRATION_PAIR_SAMPLES, 2026080961 + int(width * 1000))
        stats.append({"width": width, "cross_class_rate": rate, "hits": hits})
    # Compare in log space with a half-count floor so zero-hit widths remain ordered.
    floor = 0.5 / CALIBRATION_PAIR_SAMPLES
    best = min(stats, key=lambda r: abs(math.log(max(r["cross_class_rate"], floor)) - math.log(TARGET_CALIBRATION_COLLISION_RATE)))
    return float(best["width"]), stats


def matching_pairs(ca: np.ndarray, cb: np.ndarray) -> List[Tuple[int, int]]:
    buckets: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for j, row in enumerate(cb):
        buckets[tuple(int(x) for x in row)].append(j)
    out: List[Tuple[int, int]] = []
    for i, row in enumerate(ca):
        js = buckets.get(tuple(int(x) for x in row))
        if js:
            out.extend((i, j) for j in js)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp6a_out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-seed", type=int, default=20260809)
    ap.add_argument("--cloud-size", type=int, default=max(N_PREFIXES))
    args = ap.parse_args()
    if args.cloud_size < max(N_PREFIXES):
        raise ValueError("cloud-size must cover N_PREFIXES")

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
    labels, preds, logits_test, raw_h_test = embed_raw_victim(victim, test_loader, device)
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

    # Training calibration for semantic Delta AND representation fingerprint.
    cal_ds = Subset(train_ds, range(CALIBRATION_N))
    cal_loader = DataLoader(cal_ds, batch_size=512, shuffle=False, num_workers=2)
    cal_labels, cal_preds, _, cal_h = embed_raw_victim(victim, cal_loader, device)
    rcal_labels, rcal_preds, rcal_emb = embed_dataset(ref, cal_loader, device)
    rgood = rcal_labels == rcal_preds
    delta, n_sem_pairs = calibrate_delta(rcal_emb[rgood], rcal_labels[rgood])

    good = (cal_labels == cal_preds) & (rcal_labels == rcal_preds)
    hcal = cal_h[good]
    ycal = cal_labels[good]
    mu = hcal.mean(axis=0)
    sd = hcal.std(axis=0)
    sd[sd < 1e-6] = 1.0
    prng = np.random.default_rng(2026080962)
    proj = prng.normal(size=(PROJ_DIM, hcal.shape[1])).astype(np.float64)
    proj /= np.linalg.norm(proj, axis=1, keepdims=True)
    offset_frac = prng.random(PROJ_DIM)
    vcal = projected_values(hcal, mu, sd, proj)
    width, width_stats = calibrate_width(vcal, ycal, offset_frac)

    centers = choose_centers(labels, preds, ref_preds, ref_emb_test, delta)
    test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))], dim=0)
    for p in victim.parameters(): p.requires_grad_(False)
    for p in ref.parameters(): p.requires_grad_(False)

    clouds: Dict[int, Dict[str, object]] = {}
    for c in range(10):
        idx = centers[c]
        clouds[c] = generate_cloud(
            test_x[idx:idx+1].to(device), c, victim, ref,
            args.cloud_size, 2026080970 + c * 1009, device,
        )
        clouds[c]["proj"] = projected_values(clouds[c]["raw_h"], mu, sd, proj)
        clouds[c]["codes"] = hash_codes(clouds[c]["proj"], width, offset_frac)
        clouds[c]["radii"] = certified_radii(victim, clouds[c]["logits"], c)
        print(f"cloud class={c} center={idx} accept={clouds[c]['acceptance_rate']:.4f}", flush=True)

    rows: List[dict] = []
    pair_details: List[dict] = []
    panels: List[torch.Tensor] = []

    for source_class in range(10):
        target_class = (source_class + 1) % 10
        A = clouds[source_class]
        B = clouds[target_class]
        center_a = centers[source_class]
        center_b = centers[target_class]
        center_sem = float(np.linalg.norm(ref_emb_test[center_a] - ref_emb_test[center_b]))

        for n in N_PREFIXES:
            ca = A["codes"][:n]; cb = B["codes"][:n]
            pairs = matching_pairs(ca, cb)
            valid = alias = target_alias = certified = 0
            best_claw_d = float("inf")
            best_pair = None
            verified = []
            for i, j in pairs:
                sem_d = float(np.linalg.norm(A["ref_h"][i] - B["ref_h"][j]))
                if sem_d < delta:
                    continue
                valid += 1
                raw_d = float(np.linalg.norm(A["raw_h"][i] - B["raw_h"][j]))
                same_label = int(A["preds"][i]) == int(B["preds"][j])
                if same_label:
                    alias += 1
                is_target_alias = same_label and int(B["preds"][j]) == target_class
                if is_target_alias:
                    target_alias += 1
                is_cert = bool(B["radii"][j] > 0 and raw_d < B["radii"][j])
                if is_cert:
                    certified += 1
                    if not (int(A["preds"][i]) == target_class and int(B["preds"][j]) == target_class):
                        raise AssertionError("Certified claw did not preserve target hard label")
                if raw_d < best_claw_d:
                    best_claw_d = raw_d; best_pair = (i, j, sem_d, is_target_alias, is_cert)
                verified.append(raw_d)

            gi, gj, global_d = pairwise_raw_min(A["raw_h"][:n], B["raw_h"][:n])
            global_sem = float(np.linalg.norm(A["ref_h"][gi] - B["ref_h"][gj]))
            global_same = int(A["preds"][gi]) == int(B["preds"][gj])
            global_target_alias = int(global_same and int(B["preds"][gj]) == target_class)
            global_cert = int(B["radii"][gj] > 0 and global_d < B["radii"][gj] and global_sem >= delta)

            rows.append({
                "source_class": source_class,
                "target_class": target_class,
                "source_center_test_index": center_a,
                "target_center_test_index": center_b,
                "center_semantic_distance": center_sem,
                "domain_size_each": n,
                "cross_pairs": n * n,
                "fingerprint_claw_pairs": len(pairs),
                "semantic_valid_fingerprint_claws": valid,
                "semantic_valid_claw_density": valid / float(n * n),
                "hard_label_alias_claws": alias,
                "target_label_alias_claws": target_alias,
                "certified_second_preimage_claws": certified,
                "certified_claw_density": certified / float(n * n),
                "fingerprint_precision_target_alias": target_alias / float(max(valid, 1)),
                "fingerprint_precision_certified": certified / float(max(valid, 1)),
                "best_fingerprint_claw_raw_distance": best_claw_d if valid else float("nan"),
                "median_fingerprint_claw_raw_distance": float(np.median(verified)) if verified else float("nan"),
                "global_cross_domain_min_raw_distance": global_d,
                "global_min_semantic_distance": global_sem,
                "global_min_target_alias": global_target_alias,
                "global_min_certified": global_cert,
            })

            if n == max(N_PREFIXES) and best_pair is not None:
                i, j, sem_d, is_alias, is_cert = best_pair
                pair_details.append({
                    "source_class": source_class,
                    "target_class": target_class,
                    "source_local_index": int(i),
                    "target_local_index": int(j),
                    "raw_distance": best_claw_d,
                    "semantic_distance": sem_d,
                    "source_victim_pred": int(A["preds"][i]),
                    "target_victim_pred": int(B["preds"][j]),
                    "target_certified_radius": float(B["radii"][j]),
                    "target_alias": int(is_alias),
                    "certified": int(is_cert),
                })
                panels.extend([
                    test_x[center_a:center_a+1],
                    A["x"][i:i+1],
                    test_x[center_b:center_b+1],
                    B["x"][j:j+1],
                ])
        print(f"pair {source_class}->{target_class} done", flush=True)

    with (out / "pair_prefix_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if pair_details:
        with (out / "best_claw_details.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pair_details[0].keys())); w.writeheader(); w.writerows(pair_details)
    if panels:
        tvutils.save_image(torch.cat(panels, dim=0), out / "classpair_center_and_best_claw.png", nrow=4, padding=2)

    aggregate = []
    for n in N_PREFIXES:
        g = [r for r in rows if r["domain_size_each"] == n]
        aggregate.append({
            "domain_size_each": n,
            "class_pairs": len(g),
            "pairs_with_any_semantic_valid_fingerprint_claw": int(sum(r["semantic_valid_fingerprint_claws"] > 0 for r in g)),
            "pairs_with_target_label_alias_claw": int(sum(r["target_label_alias_claws"] > 0 for r in g)),
            "pairs_with_certified_second_preimage_claw": int(sum(r["certified_second_preimage_claws"] > 0 for r in g)),
            "total_semantic_valid_fingerprint_claws": int(sum(r["semantic_valid_fingerprint_claws"] for r in g)),
            "total_target_label_alias_claws": int(sum(r["target_label_alias_claws"] for r in g)),
            "total_certified_second_preimage_claws": int(sum(r["certified_second_preimage_claws"] for r in g)),
            "mean_semantic_valid_claw_density": float(np.mean([r["semantic_valid_claw_density"] for r in g])),
            "mean_certified_claw_density": float(np.mean([r["certified_claw_density"] for r in g])),
            "mean_best_fingerprint_claw_raw_distance": float(np.nanmean([r["best_fingerprint_claw_raw_distance"] for r in g])),
            "mean_global_cross_domain_min_raw_distance": float(np.mean([r["global_cross_domain_min_raw_distance"] for r in g])),
        })
    with (out / "aggregate.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(aggregate[0].keys())); w.writeheader(); w.writerows(aggregate)

    result = {
        "name": "QDSA-Experiment-6A-Semantic-NearClaw-Feasibility",
        "role": "collision-native feasibility gate after target-fixed minimum search lost to learned classical surrogate",
        "victim_accuracy": victim_acc,
        "reference_accuracy": ref_acc,
        "semantic_delta": delta,
        "eps_linf_cloud": EPS_LINF,
        "projection_dim": PROJ_DIM,
        "fingerprint_width": width,
        "fingerprint_width_grid": list(WIDTH_GRID),
        "target_train_cross_class_fingerprint_collision_rate": TARGET_CALIBRATION_COLLISION_RATE,
        "fingerprint_calibration_stats": width_stats,
        "fingerprint_projection": "fixed Gaussian random projection of standardized raw victim penultimate representation",
        "fingerprint_warning": "attacker-defined fingerprint, not an internal neural hash",
        "class_pairs": 10,
        "pairing": "c -> (c+1) mod 10",
        "domain_prefixes": list(N_PREFIXES),
        "cloud_size": args.cloud_size,
        "center_indices": {str(k): int(v) for k, v in centers.items()},
        "semantic_calibration_pairs": n_sem_pairs,
        "aggregate": aggregate,
        "decision_rule": {
            "continue_to_6B": "certified cross-semantic claws appear in multiple class pairs and grow with domain size without requiring test-tuned fingerprint width",
            "reject_claw_attack": "fingerprint claws are absent or almost never yield target-label/certified second-preimage relations",
        },
        "quantum_novelty_claim": "none; 6A only establishes whether a collision/claw attack object exists",
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("SUMMARY_JSON " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
