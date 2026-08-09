#!/usr/bin/env python3
"""Stage 17 micro: first 100 test examples per CIFAR-10 class, public F-C10-v1 only."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets,transforms
import stage17_cifar10_pretrained as s17


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,choices=['cifar10_resnet20','cifar10_mobilenetv2_x0_5','cifar10_vgg11_bn']);ap.add_argument('--out',default='stage17_micro_out');args=ap.parse_args()
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);dev=torch.device('cpu')
    model=torch.hub.load('chenyaofo/pytorch-cifar-models',args.model,pretrained=True,trust_repo=True,force_reload=False).eval()
    ds=datasets.CIFAR10('data',train=False,download=True,transform=transforms.ToTensor());la=np.asarray(ds.targets);keep=[]
    for c in range(10):keep.extend(np.flatnonzero(la==c)[:100].tolist())
    keep=np.asarray(sorted(keep),dtype=int);images=torch.stack([ds[int(i)][0] for i in keep]);labels=la[keep]
    preds,bits=s17.response_bits(model,images,dev,s17.public_bank(),s17.apply_public,'public');correct=preds==labels;w=bits.sum(1);valid=correct&(w>=s17.WEIGHT_MIN)&(w<=s17.WEIGHT_MAX);codes=s17.pack_bits(bits)
    _,pc,total,density=s17.collision_pairs(labels,valid,codes);hist=np.bincount(w[correct].astype(int),minlength=33);q=np.quantile(w[correct],[0,.1,.25,.5,.75,.9,1]).tolist() if correct.any() else []
    r={'model':args.model,'n':len(keep),'accuracy':float(correct.mean()),'correct':int(correct.sum()),'valid':int(valid.sum()),'valid_fraction_of_correct':float(valid.sum()/correct.sum()) if correct.sum() else 0.0,'strict_claws':int(total),'class_pairs':int(len(pc)),'collision_density':density,'weight_quantiles_correct':q,'weight_hist_correct':hist.tolist(),'top_pairs':[[list(k),int(v)] for k,v in pc.most_common(10)]}
    print('SUMMARY '+json.dumps(r,sort_keys=True),flush=True);(out/f'{args.model}.json').write_text(json.dumps(r,indent=2),encoding='utf-8');np.savez_compressed(out/f'{args.model}.npz',indices=keep,labels=labels,preds=preds,public_bits=bits,valid=valid,codes=codes)
if __name__=='__main__':main()
