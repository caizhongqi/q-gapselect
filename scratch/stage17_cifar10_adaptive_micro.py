#!/usr/bin/env python3
"""Exact adaptive-prefix Stage-17 collision micro test.

This computes the same strict 32-bit F-C10-v1 cross-semantic collisions as a
full public-code evaluation on the fixed 100-per-class CIFAR-10 subset, but
stops querying any sample as soon as either:
  (1) its current response-prefix bucket contains only one ground-truth class,
      because prefix refinement can never merge buckets later; or
  (2) its partial response weight can no longer finish inside [10,22].

Thus pruning changes cost, not the exact final collision set under the frozen
protocol. It also measures the real mixed-prefix component-query savings that
LPC exploits.
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms
import stage17_cifar10_pretrained as s17


def mixed_indices(indices: np.ndarray, labels: np.ndarray, prefix: np.ndarray) -> np.ndarray:
    buckets=defaultdict(list)
    for i in indices: buckets[int(prefix[i])].append(int(i))
    keep=[]
    for ids in buckets.values():
        if len({int(labels[i]) for i in ids}) >= 2:
            keep.extend(ids)
    return np.asarray(sorted(keep),dtype=np.int64)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True,choices=['cifar10_resnet20','cifar10_mobilenetv2_x0_5','cifar10_vgg11_bn'])
    ap.add_argument('--out',default='stage17_adaptive_out')
    args=ap.parse_args()
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);dev=torch.device('cpu')
    model=torch.hub.load('chenyaofo/pytorch-cifar-models',args.model,pretrained=True,trust_repo=True,force_reload=False).eval()
    ds=datasets.CIFAR10('data',train=False,download=True,transform=transforms.ToTensor())
    la=np.asarray(ds.targets); keep=[]
    for c in range(10): keep.extend(np.flatnonzero(la==c)[:100].tolist())
    keep=np.asarray(sorted(keep),dtype=int)
    images=torch.stack([ds[int(i)][0] for i in keep]); labels=la[keep]
    base=s17.predict(model,images,dev); correct=(base.numpy()==labels)
    n=len(labels); prefix=np.zeros(n,dtype=np.uint64); weight=np.zeros(n,dtype=np.int16)
    active=np.flatnonzero(correct).astype(np.int64)
    bank=s17.public_bank(); active_curve=[]; component_calls=0
    for j,p in enumerate(bank):
        before=int(len(active)); component_calls += before
        if before==0:
            active_curve.append({'probe':j+1,'kind':p[0],'queried':0,'after_weight':0,'after_mixed':0})
            continue
        xp=s17.apply_public(images[active],p)
        yp=s17.predict(model,xp,dev).numpy()
        bits=(yp!=base.numpy()[active]).astype(np.uint8)
        prefix[active] |= (bits.astype(np.uint64) << np.uint64(j))
        weight[active] += bits.astype(np.int16)
        remaining=32-(j+1)
        feasible=(weight[active] <= s17.WEIGHT_MAX) & ((weight[active]+remaining) >= s17.WEIGHT_MIN)
        feasible_idx=active[feasible]
        active=mixed_indices(feasible_idx,labels,prefix)
        active_curve.append({'probe':j+1,'kind':p[0],'queried':before,'after_weight':int(len(feasible_idx)),'after_mixed':int(len(active))})
        print('PREFIX',j+1,p[0],'queried',before,'weight_feasible',len(feasible_idx),'mixed',len(active),flush=True)
    final=active[(weight[active]>=s17.WEIGHT_MIN)&(weight[active]<=s17.WEIGHT_MAX)]
    buckets=defaultdict(list)
    for i in final:buckets[int(prefix[i])].append(int(i))
    pc=Counter(); claws=0
    for ids in buckets.values():
        by=Counter(int(labels[i]) for i in ids); ls=sorted(by)
        for ai,a in enumerate(ls):
            for b in ls[ai+1:]:
                z=by[a]*by[b]; claws+=z; pc[(a,b)]+=z
    brute_public_calls=int(correct.sum())*32
    result={
        'dataset':'CIFAR10','model':args.model,'pilot_n':n,
        'accuracy':float(correct.mean()),'correct':int(correct.sum()),
        'strict_claws':int(claws),'class_pairs':int(len(pc)),
        'final_mixed_valid_samples':int(len(final)),
        'adaptive_component_calls':int(component_calls),
        'brute_component_calls_on_correct':int(brute_public_calls),
        'component_call_fraction':float(component_calls/brute_public_calls) if brute_public_calls else 0.0,
        'component_call_saving':float(1-component_calls/brute_public_calls) if brute_public_calls else 0.0,
        'top_pairs':[[list(k),int(v)] for k,v in pc.most_common(10)],
        'active_curve':active_curve,
        'selection':'first_100_per_ground_truth_class_by_original_index',
        'protocol':'frozen F-C10-v1, exact adaptive prefix pruning'
    }
    print('SUMMARY '+json.dumps(result,sort_keys=True),flush=True)
    (out/f'{args.model}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(out/f'{args.model}.npz',indices=keep,labels=labels,base=base.numpy(),correct=correct,prefix=prefix,weight=weight,final=final)

if __name__=='__main__':main()
