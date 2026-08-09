#!/usr/bin/env python3
"""Stage 9 F-v1 v2: independent training-order control + matched holdout baselines."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import stage9_fv1_arch_sweep as v1


def enumerate_cross_pairs(labels, valid, codes, cap=50000):
    buckets = defaultdict(list)
    for i in np.flatnonzero(valid):
        buckets[int(codes[i])].append(int(i))
    pair_counts = Counter()
    sampled_pairs = []
    for idxs in buckets.values():
        by = defaultdict(list)
        for i in idxs:
            by[int(labels[i])].append(i)
        labs = sorted(by)
        for ai, a in enumerate(labs):
            for b in labs[ai + 1 :]:
                pair_counts[(a, b)] += len(by[a]) * len(by[b])
                if len(sampled_pairs) >= cap:
                    continue
                room = cap - len(sampled_pairs)
                for i in by[a]:
                    for j in by[b]:
                        if room <= 0:
                            break
                        sampled_pairs.append((i, j))
                        room -= 1
                    if room <= 0:
                        break
    return pair_counts, sampled_pairs


def pair_metrics(hold_bits, pairs):
    if not pairs:
        return np.array([], dtype=float), np.array([], dtype=bool)
    agreement = np.fromiter(
        (np.mean(hold_bits[i] == hold_bits[j]) for i, j in pairs),
        dtype=float,
        count=len(pairs),
    )
    exact = np.fromiter(
        (np.all(hold_bits[i] == hold_bits[j]) for i, j in pairs),
        dtype=bool,
        count=len(pairs),
    )
    return agreement, exact


def matched_random_baseline(labels, valid, hold_bits, collision_pairs, seed, reps=500):
    if not collision_pairs:
        return {
            "matched_random_holdout_mean": float("nan"),
            "matched_random_holdout_sd": float("nan"),
            "holdout_uplift_abs": float("nan"),
            "holdout_uplift_relative": float("nan"),
            "holdout_uplift_empirical_p": float("nan"),
            "matched_random_exact_rate": float("nan"),
            "collision_exact_rate": float("nan"),
            "exact_uplift_empirical_p": float("nan"),
        }

    observed_ag, observed_ex = pair_metrics(hold_bits, collision_pairs)
    sampled_counts = Counter()
    for i, j in collision_pairs:
        a, b = sorted((int(labels[i]), int(labels[j])))
        sampled_counts[(a, b)] += 1

    valid_by_class = {
        c: np.flatnonzero(valid & (labels == c)) for c in np.unique(labels)
    }
    rng = np.random.default_rng(99173 + int(seed))
    random_means = []
    random_exact_rates = []
    for _ in range(reps):
        agreements = []
        exact_count = 0
        total = 0
        for (a, b), n in sampled_counts.items():
            ia = rng.choice(valid_by_class[a], size=n, replace=True)
            ib = rng.choice(valid_by_class[b], size=n, replace=True)
            eq = hold_bits[ia] == hold_bits[ib]
            agreements.append(eq.mean(axis=1))
            exact_count += int(np.all(eq, axis=1).sum())
            total += n
        arr = np.concatenate(agreements)
        random_means.append(float(arr.mean()))
        random_exact_rates.append(float(exact_count / total))

    random_means = np.asarray(random_means)
    random_exact_rates = np.asarray(random_exact_rates)
    obs_mean = float(observed_ag.mean())
    obs_exact_rate = float(observed_ex.mean())
    baseline_mean = float(random_means.mean())
    return {
        "matched_random_holdout_mean": baseline_mean,
        "matched_random_holdout_sd": float(random_means.std(ddof=1)),
        "holdout_uplift_abs": float(obs_mean - baseline_mean),
        "holdout_uplift_relative": float(obs_mean / baseline_mean - 1.0) if baseline_mean else float("nan"),
        "holdout_uplift_empirical_p": float((np.sum(random_means >= obs_mean) + 1) / (reps + 1)),
        "matched_random_exact_rate": float(random_exact_rates.mean()),
        "collision_exact_rate": obs_exact_rate,
        "exact_uplift_empirical_p": float((np.sum(random_exact_rates >= obs_exact_rate) + 1) / (reps + 1)),
    }


def code_metrics(bits, valid, codes):
    valid_codes = codes[valid]
    counts = Counter(map(int, valid_codes))
    n = len(valid_codes)
    if n:
        probs = np.asarray(list(counts.values()), dtype=float) / n
        entropy = float(-(probs * np.log2(probs)).sum())
        unique_rate = float(len(counts) / n)
        max_bucket = int(max(counts.values()))
    else:
        entropy = float("nan")
        unique_rate = float("nan")
        max_bucket = 0
    weights = bits.sum(axis=1)
    return {
        "valid_fraction": float(valid.mean()),
        "unique_code_rate": unique_rate,
        "empirical_code_entropy_bits": entropy,
        "max_code_bucket": max_bucket,
        "mean_response_weight": float(weights.mean()),
        "sd_response_weight": float(weights.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--models", nargs="+", default=["mlp", "tinycnn", "lenet5", "resnet18", "tinyvit"])
    ap.add_argument("--out", default="stage9_out")
    ap.add_argument("--baseline-reps", type=int, default=500)
    args = ap.parse_args()

    v1.seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    tf = transforms.ToTensor()
    train_set = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_set = datasets.MNIST("data", train=False, download=True, transform=tf)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2)
    images = torch.cat([x for x, _ in test_loader])
    labels = torch.cat([y for _, y in test_loader]).numpy()
    search = v1.make_probe_bank("search")
    hold = v1.make_probe_bank("holdout")

    protocol = {
        "name": "F-v1",
        "training_order_seed": args.seed,
        "search": {
            "shift_pixels": v1.SEARCH_SHIFT,
            "directions": v1.DIRECTIONS,
            "occlusion_size": v1.SEARCH_OCC,
            "occlusion_positions": v1.SEARCH_OCC_POS,
            "noise_eps": v1.SEARCH_NOISE_EPS,
            "noise_seed": v1.SEARCH_NOISE_SEED,
        },
        "holdout": {
            "shift_pixels": v1.HOLDOUT_SHIFT,
            "occlusion_size": v1.HOLDOUT_OCC,
            "occlusion_positions": v1.HOLDOUT_OCC_POS,
            "noise_eps": v1.HOLDOUT_NOISE_EPS,
            "noise_seed": v1.HOLDOUT_NOISE_SEED,
        },
        "weight_window": [v1.WEIGHT_MIN, v1.WEIGHT_MAX],
        "matched_random_baseline_reps": args.baseline_reps,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    rows = []
    for name in args.models:
        print(f"=== MODEL {name} seed={args.seed} ===", flush=True)
        v1.seed_all(args.seed)
        train_gen = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            train_set,
            batch_size=256,
            shuffle=True,
            generator=train_gen,
            num_workers=2,
        )
        model = v1.build_model(name)
        v1.train_model(model, train_loader, device, args.epochs)
        model.eval()

        preds, bits = v1.response_codes(model, images, device, search)
        _, hold_bits = v1.response_codes(model, images, device, hold)
        valid, codes, stats = v1.collision_stats(labels, preds, bits, hold_bits)
        _, collision_pairs = enumerate_cross_pairs(labels, valid, codes)
        stats.update(code_metrics(bits, valid, codes))
        stats.update(
            matched_random_baseline(
                labels,
                valid,
                hold_bits,
                collision_pairs,
                seed=args.seed,
                reps=args.baseline_reps,
            )
        )
        stats.update(
            {
                "architecture": name,
                "seed": args.seed,
                "training_order_seed": args.seed,
                "epochs": args.epochs,
                "test_accuracy": float(np.mean(preds == labels)),
            }
        )
        print("RESULT " + json.dumps(stats, sort_keys=True), flush=True)
        rows.append(stats)
        np.savez_compressed(
            out / f"{name}_seed{args.seed}_codes.npz",
            labels=labels,
            preds=preds,
            search_bits=bits,
            holdout_bits=hold_bits,
            valid=valid,
            codes=codes,
        )

    fields = [
        "architecture", "seed", "training_order_seed", "epochs", "test_accuracy",
        "valid_nondegenerate", "valid_fraction", "cross_semantic_claws", "collision_density",
        "class_pairs_with_claws", "unique_code_rate", "empirical_code_entropy_bits",
        "max_code_bucket", "mean_response_weight", "sd_response_weight",
        "pair_1_4_claws", "pair_0_6_claws", "mean_holdout_agreement",
        "matched_random_holdout_mean", "matched_random_holdout_sd", "holdout_uplift_abs",
        "holdout_uplift_relative", "holdout_uplift_empirical_p", "holdout_pair_sample_n",
        "holdout_exact_in_sample", "collision_exact_rate", "matched_random_exact_rate",
        "exact_uplift_empirical_p", "top_class_pairs",
    ]
    with (out / "stage9_fv1_v2_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r.get(k) for k in fields} for r in rows])
    print("SUMMARY_JSON " + json.dumps(rows, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
