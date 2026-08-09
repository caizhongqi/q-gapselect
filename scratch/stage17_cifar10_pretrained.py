#!/usr/bin/env python3
"""Stage 17: functional collisions on natural RGB CIFAR-10 with pretrained victims.

The attack remains hard-label only.  Images are kept in raw [0,1] pixel space for
all challenges, then normalized with the victim authors' CIFAR-10 evaluation
statistics immediately before inference.

Primary public protocol F-C10-v1 is frozen before observing collision outcomes:
  8 zero-fill shifts of 2 px
  16 black 8x8 occlusions on a 4x4 grid
  8 fixed RGB sign-noise probes at eps=8/255
The direct-transfer nondegenerate window remains response weight 10..22.

The 24-bit secret bank uses transformation families absent from the public bank:
rotation, Gaussian blur, contrast, brightness, gamma, and shear.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

MEAN = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).view(1,3,1,1)
STD = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).view(1,3,1,1)
DIRECTIONS = ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))
PUBLIC_SHIFT = 2
PUBLIC_OCC = 8
PUBLIC_OCC_POS = (0,8,16,24)
PUBLIC_NOISE_EPS = 8.0/255.0
PUBLIC_NOISE_SEED = 2026081701
WEIGHT_MIN, WEIGHT_MAX = 10, 22


def shift_zero(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    out = torch.zeros_like(x)
    h,w=x.shape[-2:]
    ys0,ys1=max(0,-dy),min(h,h-dy); xs0,xs1=max(0,-dx),min(w,w-dx)
    yd0,yd1=max(0,dy),min(h,h+dy); xd0,xd1=max(0,dx),min(w,w+dx)
    out[...,yd0:yd1,xd0:xd1]=x[...,ys0:ys1,xs0:xs1]
    return out


def fixed_rgb_sign_masks(seed: int, n: int=8) -> torch.Tensor:
    g=torch.Generator().manual_seed(seed)
    z=torch.randint(0,2,(n,3,32,32),generator=g,dtype=torch.float32)
    return z.mul_(2).sub_(1)


def public_bank():
    p=[]
    for dy,dx in DIRECTIONS: p.append(("shift",dy*PUBLIC_SHIFT,dx*PUBLIC_SHIFT))
    for y in PUBLIC_OCC_POS:
        for x in PUBLIC_OCC_POS: p.append(("occ",y,x,PUBLIC_OCC))
    m=fixed_rgb_sign_masks(PUBLIC_NOISE_SEED)
    for i in range(8): p.append(("noise",m[i],PUBLIC_NOISE_EPS))
    assert len(p)==32
    return p


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


def apply_public(x,p):
    if p[0]=="shift": return shift_zero(x,p[1],p[2])
    if p[0]=="occ":
        out=x.clone(); y,z,s=p[1],p[2],p[3]; out[...,y:y+s,z:z+s]=0.0; return out
    return torch.clamp(x + p[2]*p[1].to(x.device),0.0,1.0)


def apply_secret(x,p):
    k,v=p
    if k=="rotate": return TF.rotate(x,v,interpolation=TF.InterpolationMode.BILINEAR,fill=0.0)
    if k=="blur": return TF.gaussian_blur(x,[5,5],[v,v])
    if k=="contrast": return torch.clamp(TF.adjust_contrast(x,v),0.0,1.0)
    if k=="brightness": return torch.clamp(TF.adjust_brightness(x,v),0.0,1.0)
    if k=="gamma": return torch.clamp(TF.adjust_gamma(x,v),0.0,1.0)
    if k=="shear": return TF.affine(x,angle=0.0,translate=[0,0],scale=1.0,shear=[v,0.0],interpolation=TF.InterpolationMode.BILINEAR,fill=0.0)
    raise KeyError(k)


@torch.inference_mode()
def predict(model, images, device, chunk=512):
    out=[]; mean=MEAN.to(device); std=STD.to(device)
    for s in range(0,len(images),chunk):
        x=images[s:s+chunk].to(device)
        x=(x-mean)/std
        out.append(model(x).argmax(1).cpu())
    return torch.cat(out)


@torch.inference_mode()
def response_bits(model, images, device, bank, apply_fn, prefix):
    base=predict(model,images,device); bits=[]
    for j,p in enumerate(bank):
        yp=predict(model,apply_fn(images,p),device)
        bits.append((yp!=base).numpy().astype(np.uint8))
        print(f"{prefix} {j+1}/{len(bank)} {p[0]}",flush=True)
    return base.numpy(),np.stack(bits,axis=1)


def pack_bits(bits):
    powers=(1<<np.arange(bits.shape[1],dtype=np.uint64))
    return (bits.astype(np.uint64)*powers).sum(1)


def collision_pairs(labels, valid, codes, cap=50000):
    buckets=defaultdict(list)
    for i in np.flatnonzero(valid): buckets[int(codes[i])].append(int(i))
    pair_counts=Counter(); pairs=[]; total=0
    for ids in buckets.values():
        by=defaultdict(list)
        for i in ids: by[int(labels[i])].append(i)
        labs=sorted(by)
        for ai,a in enumerate(labs):
            for b in labs[ai+1:]:
                n=len(by[a])*len(by[b]); total+=n; pair_counts[(a,b)]+=n
                if len(pairs)<cap:
                    room=cap-len(pairs)
                    for i in by[a]:
                        for j in by[b]:
                            if room<=0: break
                            pairs.append((i,j)); room-=1
                        if room<=0: break
    class_n=Counter(int(labels[i]) for i in np.flatnonzero(valid))
    denom=sum(class_n[a]*class_n[b] for a in class_n for b in class_n if a<b)
    return pairs,pair_counts,total,float(total/denom if denom else 0.0)


def pair_agreement(bits,pairs,mask=None):
    if not pairs: return float("nan")
    if mask is None: mask=np.ones(bits.shape[1],dtype=bool)
    if not np.any(mask): return float("nan")
    return float(np.mean([np.mean(bits[i,mask]==bits[j,mask]) for i,j in pairs]))


def matched_random(labels,valid,secret_bits,pairs,seed=1701,reps=200,mask=None):
    if not pairs: return float("nan"),float("nan"),float("nan")
    if mask is None: mask=np.ones(secret_bits.shape[1],dtype=bool)
    counts=Counter(tuple(sorted((int(labels[i]),int(labels[j])))) for i,j in pairs)
    pools={c:np.flatnonzero(valid&(labels==c)) for c in np.unique(labels)}
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(reps):
        s=0.0;nall=0
        for (a,b),n in counts.items():
            ia=rng.choice(pools[a],size=n,replace=True); ib=rng.choice(pools[b],size=n,replace=True)
            s += float((secret_bits[ia][:,mask]==secret_bits[ib][:,mask]).mean(axis=1).sum()); nall+=n
        vals.append(s/nall)
    vals=np.asarray(vals)
    return float(vals.mean()),float(vals.std(ddof=1)),float((np.sum(vals>=0)+0)) # p computed outside


def analyze(labels,preds,public_bits,secret_bits):
    correct=preds==labels; weight=public_bits.sum(1); valid=correct&(weight>=WEIGHT_MIN)&(weight<=WEIGHT_MAX); codes=pack_bits(public_bits)
    pairs,pair_counts,total,density=collision_pairs(labels,valid,codes)
    obs=pair_agreement(secret_bits,pairs)
    # matched random, retain individual replicates here to compute empirical p
    counts=Counter(tuple(sorted((int(labels[i]),int(labels[j])))) for i,j in pairs)
    pools={c:np.flatnonzero(valid&(labels==c)) for c in np.unique(labels)}
    rng=np.random.default_rng(2026081702); reps=[]
    if pairs:
        for _ in range(200):
            ss=0.0; nn=0
            for (a,b),n in counts.items():
                ia=rng.choice(pools[a],size=n,replace=True); ib=rng.choice(pools[b],size=n,replace=True)
                ss += float((secret_bits[ia]==secret_bits[ib]).mean(axis=1).sum()); nn+=n
            reps.append(ss/nn)
    base=float(np.mean(reps)) if reps else float("nan")
    p=float((np.sum(np.asarray(reps)>=obs)+1)/(len(reps)+1)) if reps else float("nan")
    marginal=secret_bits[valid].mean(0) if valid.any() else np.zeros(secret_bits.shape[1])
    info=(marginal>=.05)&(marginal<=.95)
    obs_info=pair_agreement(secret_bits,pairs,info)
    reps_info=[]
    if pairs and info.any():
        rng=np.random.default_rng(2026081703)
        for _ in range(200):
            ss=0.0;nn=0
            for (a,b),n in counts.items():
                ia=rng.choice(pools[a],size=n,replace=True); ib=rng.choice(pools[b],size=n,replace=True)
                ss += float((secret_bits[ia][:,info]==secret_bits[ib][:,info]).mean(axis=1).sum()); nn+=n
            reps_info.append(ss/nn)
    base_info=float(np.mean(reps_info)) if reps_info else float("nan")
    p_info=float((np.sum(np.asarray(reps_info)>=obs_info)+1)/(len(reps_info)+1)) if reps_info else float("nan")
    hist=np.bincount(weight[correct].astype(int),minlength=33)
    q=np.quantile(weight[correct],[0,.1,.25,.5,.75,.9,1]).tolist() if correct.any() else []
    return valid,codes,{
        "accuracy":float(correct.mean()),"correct":int(correct.sum()),"valid":int(valid.sum()),"valid_fraction_of_correct":float(valid.sum()/correct.sum()) if correct.sum() else 0.0,
        "strict_claws":int(total),"class_pairs":int(len(pair_counts)),"collision_density":density,
        "secret_agreement":obs,"matched_random_secret_agreement":base,"secret_uplift":float(obs-base) if pairs else float("nan"),"secret_empirical_p":p,
        "info_secret_bits":int(info.sum()),"info_secret_agreement":obs_info,"info_matched_random":base_info,"info_secret_uplift":float(obs_info-base_info) if pairs and info.any() else float("nan"),"info_empirical_p":p_info,
        "public_weight_quantiles_correct":q,"public_weight_hist_correct":hist.tolist(),
        "top_pairs":[[list(k),int(v)] for k,v in pair_counts.most_common(10)],
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True,choices=['cifar10_resnet20','cifar10_mobilenetv2_x0_5','cifar10_vgg11_bn']); ap.add_argument('--out',default='stage17_out'); args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);device=torch.device('cpu')
    print('loading pretrained',args.model,flush=True)
    model=torch.hub.load('chenyaofo/pytorch-cifar-models',args.model,pretrained=True,trust_repo=True,force_reload=False).to(device).eval()
    ds=datasets.CIFAR10('data',train=False,download=True,transform=transforms.ToTensor()); loader=DataLoader(ds,batch_size=512,shuffle=False,num_workers=2)
    images=torch.cat([x for x,_ in loader]); labels=torch.cat([y for _,y in loader]).numpy()
    pb=public_bank(); sb=secret_bank(); preds,public_bits=response_bits(model,images,device,pb,apply_public,'public'); _,secret_bits=response_bits(model,images,device,sb,apply_secret,'secret')
    valid,codes,stats=analyze(labels,preds,public_bits,secret_bits); stats.update({'dataset':'CIFAR10','model':args.model,'public_noise_eps':PUBLIC_NOISE_EPS,'weight_window':[WEIGHT_MIN,WEIGHT_MAX]})
    print('SUMMARY '+json.dumps(stats,sort_keys=True),flush=True)
    np.savez_compressed(out/f'{args.model}.npz',labels=labels,preds=preds,public_bits=public_bits,secret_bits=secret_bits,valid=valid,codes=codes)
    (out/f'{args.model}.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')

if __name__=='__main__': main()
