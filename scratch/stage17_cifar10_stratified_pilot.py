#!/usr/bin/env python3
"""Fast, label-only stratified pilot for Stage 17.

Selects the first 200 CIFAR-10 test examples from each ground-truth class by
original dataset index (2000 examples total). Selection is fixed from labels and
indices only; no model response enters selection. Reuses the exact frozen
F-C10-v1 public/secret protocol and analysis from stage17_cifar10_pretrained.py.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets,transforms
import stage17_cifar10_pretrained as s17


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,choices=['cifar10_resnet20','cifar10_mobilenetv2_x0_5','cifar10_vgg11_bn']);ap.add_argument('--out',default='stage17_pilot_out');args=ap.parse_args()
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);dev=torch.device('cpu')
    model=torch.hub.load('chenyaofo/pytorch-cifar-models',args.model,pretrained=True,trust_repo=True,force_reload=False).to(dev).eval()
    ds=datasets.CIFAR10('data',train=False,download=True,transform=transforms.ToTensor())
    labels_all=np.asarray(ds.targets); keep=[]
    for c in range(10): keep.extend(np.flatnonzero(labels_all==c)[:200].tolist())
    keep=np.asarray(sorted(keep),dtype=int)
    images=torch.stack([ds[int(i)][0] for i in keep]);labels=labels_all[keep]
    preds,pub=s17.response_bits(model,images,dev,s17.public_bank(),s17.apply_public,'public')
    _,sec=s17.response_bits(model,images,dev,s17.secret_bank(),s17.apply_secret,'secret')
    valid,codes,stats=s17.analyze(labels,preds,pub,sec);stats.update({'dataset':'CIFAR10','model':args.model,'pilot_n':len(keep),'selection':'first_200_per_ground_truth_class_by_original_index'})
    print('SUMMARY '+json.dumps(stats,sort_keys=True),flush=True)
    np.savez_compressed(out/f'{args.model}_pilot.npz',indices=keep,labels=labels,preds=preds,public_bits=pub,secret_bits=sec,valid=valid,codes=codes)
    (out/f'{args.model}_pilot.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
if __name__=='__main__':main()
