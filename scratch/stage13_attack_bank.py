#!/usr/bin/env python3
"""Stage 13: stronger unseen perturbation bank for zero-target-query transfer.

Search collisions remain defined only by frozen F-v1.  This script adds a disjoint,
stronger attack bank and saves its relative hard-label flip bits for downstream
source->target transfer analysis.  No attack-bank bit participates in collision
selection.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import stage9_fv1_arch_sweep as s9

ATTACK_SHIFT = 4
ATTACK_OCC = 9
ATTACK_OCC_POS = (1, 7, 13, 19)
ATTACK_NOISE_1_EPS = 0.25
ATTACK_NOISE_1_SEED = 2026081301
ATTACK_NOISE_2_EPS = 0.35
ATTACK_NOISE_2_SEED = 2026081302


def make_attack_bank():
    probes = []
    for dy, dx in s9.DIRECTIONS:
        probes.append(("shift", dy * ATTACK_SHIFT, dx * ATTACK_SHIFT))
    for y in ATTACK_OCC_POS:
        for x in ATTACK_OCC_POS:
            probes.append(("occ", y, x, ATTACK_OCC))
    m1 = s9.fixed_sign_masks(ATTACK_NOISE_1_SEED)
    m2 = s9.fixed_sign_masks(ATTACK_NOISE_2_SEED)
    for i in range(8): probes.append(("noise", m1[i], ATTACK_NOISE_1_EPS))
    for i in range(8): probes.append(("noise", m2[i], ATTACK_NOISE_2_EPS))
    assert len(probes) == 40
    return probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mlp","tinycnn","lenet5","resnet18","tinyvit"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--out", default="stage13_out")
    args = ap.parse_args()

    s9.seed_all(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    tf = transforms.ToTensor()
    train_set = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_set = datasets.MNIST("data", train=False, download=True, transform=tf)
    train_gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, generator=train_gen, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2)
    images = torch.cat([x for x,_ in test_loader])
    labels = torch.cat([y for _,y in test_loader]).numpy()

    model = s9.build_model(args.model)
    s9.train_model(model, train_loader, device, args.epochs)
    model.eval()
    search = s9.make_probe_bank("search")
    attack = make_attack_bank()
    preds, search_bits = s9.response_codes(model, images, device, search)
    _, attack_bits = s9.response_codes(model, images, device, attack)
    weight = search_bits.sum(1)
    valid = (preds == labels) & (weight >= s9.WEIGHT_MIN) & (weight <= s9.WEIGHT_MAX)
    codes = s9.pack_bits(search_bits)
    attack_weight = attack_bits.sum(1)
    np.savez_compressed(
        out / f"{args.model}_seed{args.seed}_attack_bank.npz",
        labels=labels, preds=preds, search_bits=search_bits, attack_bits=attack_bits,
        valid=valid, codes=codes, attack_weight=attack_weight,
    )
    print(
        "SUMMARY",
        args.model, args.seed,
        "acc", float(np.mean(preds == labels)),
        "valid", int(valid.sum()),
        "attack_flip_mean_all", float(attack_weight.mean()),
        "attack_flip_mean_valid", float(attack_weight[valid].mean()) if valid.any() else float("nan"),
        flush=True,
    )

if __name__ == "__main__":
    main()
