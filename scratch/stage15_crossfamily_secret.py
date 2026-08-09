#!/usr/bin/env python3
"""Stage 15: cross-family secret challenges for functional-fingerprint impersonation.

Public collision construction remains frozen F-v1 (shift/occlusion/fixed-noise).
This script evaluates a disjoint secret bank consisting only of transformation
families absent from F-v1: rotation, blur, photometric transforms, and shear.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

import stage9_fv1_arch_sweep as s9


def secret_bank():
    p=[]
    for a in (-15.0,-10.0,-5.0,5.0,10.0,15.0): p.append(("rotate",a))
    for sig in (0.5,0.8,1.1,1.4): p.append(("blur",sig))
    for v in (0.65,0.82,1.18,1.35): p.append(("contrast",v))
    for v in (0.72,0.86,1.14,1.28): p.append(("brightness",v))
    for v in (0.70,0.85,1.15,1.30): p.append(("gamma",v))
    for sh in (-12.0,12.0): p.append(("shear",sh))
    assert len(p)==24
    return p


def apply_secret(x,p):
    k=p[0];v=p[1]
    if k=="rotate": return TF.rotate(x,v,interpolation=TF.InterpolationMode.BILINEAR,fill=0.0)
    if k=="blur": return TF.gaussian_blur(x,[5,5],[v,v])
    if k=="contrast": return TF.adjust_contrast(x,v)
    if k=="brightness": return TF.adjust_brightness(x,v)
    if k=="gamma": return TF.adjust_gamma(x,v)
    if k=="shear": return TF.affine(x,angle=0.0,translate=[0,0],scale=1.0,shear=[v,0.0],interpolation=TF.InterpolationMode.BILINEAR,fill=0.0)
    raise KeyError(k)

@torch.inference_mode()
def response(model,images,device,bank):
    base=s9.predict(model,images,device);bits=[]
    for j,p in enumerate(bank):
        yp=s9.predict(model,apply_secret(images,p),device)
        bits.append((yp!=base).numpy().astype(np.uint8));print(f"secret {j+1}/{len(bank)} {p}",flush=True)
    return base.numpy(),np.stack(bits,1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,choices=['mlp','tinycnn','lenet5','resnet18','tinyvit']);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--epochs',type=int,default=2);ap.add_argument('--out',default='stage15_out');args=ap.parse_args()
    s9.seed_all(args.seed);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);dev=torch.device('cpu');tf=transforms.ToTensor()
    tr=datasets.MNIST('data',train=True,download=True,transform=tf);te=datasets.MNIST('data',train=False,download=True,transform=tf)
    g=torch.Generator().manual_seed(args.seed);trl=DataLoader(tr,batch_size=256,shuffle=True,generator=g,num_workers=2);tel=DataLoader(te,batch_size=512,shuffle=False,num_workers=2)
    images=torch.cat([x for x,_ in tel]);labels=torch.cat([y for _,y in tel]).numpy();model=s9.build_model(args.model);s9.train_model(model,trl,dev,args.epochs);model.eval()
    search=s9.make_probe_bank('search');preds,search_bits=s9.response_codes(model,images,dev,search);_,secret_bits=response(model,images,dev,secret_bank())
    w=search_bits.sum(1);valid=(preds==labels)&(w>=s9.WEIGHT_MIN)&(w<=s9.WEIGHT_MAX);codes=s9.pack_bits(search_bits)
    np.savez_compressed(out/f'{args.model}_seed{args.seed}_crossfamily.npz',labels=labels,preds=preds,search_bits=search_bits,secret_bits=secret_bits,valid=valid,codes=codes)
    print('SUMMARY',args.model,args.seed,'acc',float(np.mean(preds==labels)),'valid',int(valid.sum()),'secret_flip_mean_valid',float(secret_bits[valid].sum(1).mean()) if valid.any() else float('nan'),flush=True)
if __name__=='__main__': main()
