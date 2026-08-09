#!/usr/bin/env python3
"""Stage 16: frozen F-v1 functional collisions across 28x28 datasets.

Datasets: FashionMNIST and KMNIST. Victims: MLP, ResNet18-like, TinyViT.
Public F-v1 and cross-family secret responses are generated from the same frozen
victim instance in one job, preventing cross-stage retraining drift.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

import stage9_fv1_arch_sweep as s9
import stage15_crossfamily_secret as s15

DATASETS={'fashion':datasets.FashionMNIST,'kmnist':datasets.KMNIST}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=sorted(DATASETS));ap.add_argument('--model',required=True,choices=['mlp','resnet18','tinyvit']);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--epochs',type=int,default=3);ap.add_argument('--out',default='stage16_out');args=ap.parse_args()
 s9.seed_all(args.seed);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);dev=torch.device('cpu');tf=transforms.ToTensor();DS=DATASETS[args.dataset]
 tr=DS('data',train=True,download=True,transform=tf);te=DS('data',train=False,download=True,transform=tf);g=torch.Generator().manual_seed(args.seed)
 trl=DataLoader(tr,batch_size=256,shuffle=True,generator=g,num_workers=2);tel=DataLoader(te,batch_size=512,shuffle=False,num_workers=2);images=torch.cat([x for x,_ in tel]);labels=torch.cat([y for _,y in tel]).numpy()
 model=s9.build_model(args.model);s9.train_model(model,trl,dev,args.epochs);model.eval();search=s9.make_probe_bank('search');preds,search_bits=s9.response_codes(model,images,dev,search);_,secret_bits=s15.response(model,images,dev,s15.secret_bank())
 w=search_bits.sum(1);correct=(preds==labels);valid=correct&(w>=s9.WEIGHT_MIN)&(w<=s9.WEIGHT_MAX);codes=s9.pack_bits(search_bits)
 # fixed-window strict cross-semantic claw count
 from collections import defaultdict,Counter
 buckets=defaultdict(list)
 for i in np.flatnonzero(valid):buckets[int(codes[i])].append(int(i))
 claws=0;cp=Counter()
 for ids in buckets.values():
  by=defaultdict(int)
  for i in ids:by[int(labels[i])]+=1
  ls=sorted(by)
  for ia,a in enumerate(ls):
   for b in ls[ia+1:]:n=by[a]*by[b];claws+=n;cp[(a,b)]+=n
 summary={'dataset':args.dataset,'model':args.model,'seed':args.seed,'epochs':args.epochs,'accuracy':float(correct.mean()),'valid':int(valid.sum()),'valid_fraction':float(valid.mean()),'claws':int(claws),'class_pairs':int(len(cp)),'secret_flip_mean_valid':float(secret_bits[valid].sum(1).mean()) if valid.any() else float('nan'),'top_pairs':[[list(k),int(v)] for k,v in cp.most_common(8)]}
 np.savez_compressed(out/f'{args.dataset}_{args.model}_seed{args.seed}.npz',labels=labels,preds=preds,search_bits=search_bits,secret_bits=secret_bits,correct=correct,valid=valid,codes=codes)
 (out/f'{args.dataset}_{args.model}_seed{args.seed}.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print('SUMMARY '+json.dumps(summary,sort_keys=True),flush=True)
if __name__=='__main__':main()
