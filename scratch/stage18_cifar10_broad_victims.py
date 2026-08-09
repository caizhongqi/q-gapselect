#!/usr/bin/env python3
"""Stage 18: broad pretrained CIFAR-10 victim replication.

This stage deliberately does not retune the attack after observing Stage-17
outcomes. It reuses the frozen F-C10-v1 32-bit public hard-label response code,
the response-weight window [10,22], the independent 24-bit cross-family secret
bank, and the class-pair-matched random control from Stage 17.

The only changed variable is victim architecture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import stage17_cifar10_pretrained as s17


MODELS = (
    "cifar10_resnet56",
    "cifar10_vgg16_bn",
    "cifar10_mobilenetv2_x1_0",
    "cifar10_shufflenetv2_x1_0",
    "cifar10_repvgg_a0",
    "cifar10_vit_b16",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--out", default="stage18_out")
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print("loading pretrained", args.model, flush=True)
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        args.model,
        pretrained=True,
        trust_repo=True,
        force_reload=False,
    ).to(device).eval()

    ds = datasets.CIFAR10(
        "data", train=False, download=True, transform=transforms.ToTensor()
    )
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=2)
    images = torch.cat([x for x, _ in loader])
    labels = torch.cat([y for _, y in loader]).numpy()

    public_bank = s17.public_bank()
    secret_bank = s17.secret_bank()

    preds, public_bits = s17.response_bits(
        model, images, device, public_bank, s17.apply_public, "public"
    )
    _, secret_bits = s17.response_bits(
        model, images, device, secret_bank, s17.apply_secret, "secret"
    )

    valid, codes, stats = s17.analyze(labels, preds, public_bits, secret_bits)
    stats.update(
        {
            "stage": 18,
            "dataset": "CIFAR10",
            "model": args.model,
            "public_protocol": "frozen F-C10-v1",
            "public_bits": 32,
            "public_noise_eps": s17.PUBLIC_NOISE_EPS,
            "weight_window": [s17.WEIGHT_MIN, s17.WEIGHT_MAX],
            "secret_protocol": "frozen 24-bit cross-family bank",
            "retuned_after_stage17": False,
            "test_examples": int(len(labels)),
        }
    )

    print("SUMMARY " + json.dumps(stats, sort_keys=True), flush=True)
    np.savez_compressed(
        out / f"{args.model}.npz",
        labels=labels,
        preds=preds,
        public_bits=public_bits,
        secret_bits=secret_bits,
        valid=valid,
        codes=codes,
    )
    (out / f"{args.model}.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
