#!/usr/bin/env python3
"""Stage 9 pilot: reproducible external functional-collision census on MNIST.

This is intentionally isolated on a scratch branch. It does not modify the q-gapselect
research line. The old Stage-6 probe coordinates were not retained, so this script
freezes a new fully specified F-v1 protocol and reruns old + new victim families under
the same oracle. No hidden activations, logits, gradients, or weights enter the attack
predicate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SEARCH_SHIFT = 2
SEARCH_OCC = 7
SEARCH_OCC_POS = (0, 7, 14, 21)
SEARCH_NOISE_EPS = 0.20
SEARCH_NOISE_SEED = 2026080901
HOLDOUT_SHIFT = 3
HOLDOUT_OCC = 6
HOLDOUT_OCC_POS = ((2,2),(2,10),(2,18),(10,2),(10,10),(10,18),(18,2),(18,10),(18,18),(6,6),(6,16),(16,11))
HOLDOUT_NOISE_EPS = 0.15
HOLDOUT_NOISE_SEED = 2026080902
WEIGHT_MIN, WEIGHT_MAX = 10, 22
DIRECTIONS = ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def shift_zero(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    out = torch.zeros_like(x)
    h, w = x.shape[-2:]
    ys0, ys1 = max(0, -dy), min(h, h-dy)
    xs0, xs1 = max(0, -dx), min(w, w-dx)
    yd0, yd1 = max(0, dy), min(h, h+dy)
    xd0, xd1 = max(0, dx), min(w, w+dx)
    out[..., yd0:yd1, xd0:xd1] = x[..., ys0:ys1, xs0:xs1]
    return out


def fixed_sign_masks(seed: int, n: int = 8) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = torch.randint(0, 2, (n, 1, 28, 28), generator=g, dtype=torch.float32)
    return z.mul_(2).sub_(1)


def make_probe_bank(kind: str):
    probes = []
    if kind == "search":
        for dy, dx in DIRECTIONS:
            probes.append(("shift", dy*SEARCH_SHIFT, dx*SEARCH_SHIFT))
        for y in SEARCH_OCC_POS:
            for x in SEARCH_OCC_POS:
                probes.append(("occ", y, x, SEARCH_OCC))
        masks = fixed_sign_masks(SEARCH_NOISE_SEED)
        for i in range(8): probes.append(("noise", masks[i], SEARCH_NOISE_EPS))
        assert len(probes) == 32
    else:
        for dy, dx in DIRECTIONS:
            probes.append(("shift", dy*HOLDOUT_SHIFT, dx*HOLDOUT_SHIFT))
        for y, x in HOLDOUT_OCC_POS:
            probes.append(("occ", y, x, HOLDOUT_OCC))
        masks = fixed_sign_masks(HOLDOUT_NOISE_SEED)
        for i in range(8): probes.append(("noise", masks[i], HOLDOUT_NOISE_EPS))
        assert len(probes) == 28
    return probes


def apply_probe(x: torch.Tensor, p) -> torch.Tensor:
    if p[0] == "shift": return shift_zero(x, p[1], p[2])
    if p[0] == "occ":
        out = x.clone(); y, z, s = p[1], p[2], p[3]
        out[..., y:y+s, z:z+s] = 0.0; return out
    mask, eps = p[1].to(x.device), p[2]
    return torch.clamp(x + eps * mask, 0.0, 1.0)


class MLP(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Flatten(), nn.Linear(784,256), nn.ReLU(), nn.Linear(256,64), nn.ReLU(), nn.Linear(64,10))
    def forward(self,x): return self.net(x)

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__(); self.c1=nn.Conv2d(1,8,3,padding=1); self.c2=nn.Conv2d(8,16,3,padding=1); self.fc=nn.Linear(16*7*7,10)
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.c1(x)),2); x=F.max_pool2d(F.relu(self.c2(x)),2); return self.fc(x.flatten(1))

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__(); self.c1=nn.Conv2d(1,6,5,padding=2); self.c2=nn.Conv2d(6,16,5); self.f1=nn.Linear(16*5*5,120); self.f2=nn.Linear(120,84); self.f3=nn.Linear(84,10)
    def forward(self,x):
        x=F.avg_pool2d(torch.tanh(self.c1(x)),2); x=F.avg_pool2d(torch.tanh(self.c2(x)),2); x=x.flatten(1); return self.f3(torch.tanh(self.f2(torch.tanh(self.f1(x)))))

class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__(); self.c1=nn.Conv2d(cin,cout,3,stride,padding=1,bias=False); self.b1=nn.BatchNorm2d(cout); self.c2=nn.Conv2d(cout,cout,3,padding=1,bias=False); self.b2=nn.BatchNorm2d(cout)
        self.skip = nn.Identity() if stride==1 and cin==cout else nn.Sequential(nn.Conv2d(cin,cout,1,stride,bias=False), nn.BatchNorm2d(cout))
    def forward(self,x): return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.skip(x))

class ResNet18MNIST(nn.Module):
    def __init__(self):
        super().__init__(); self.stem=nn.Sequential(nn.Conv2d(1,16,3,padding=1,bias=False),nn.BatchNorm2d(16),nn.ReLU()); layers=[]; cin=16
        for cout,stride in [(16,1),(16,1),(32,2),(32,1),(64,2),(64,1),(128,2),(128,1)]: layers.append(Block(cin,cout,stride)); cin=cout
        self.body=nn.Sequential(*layers); self.fc=nn.Linear(128,10)
    def forward(self,x): x=self.body(self.stem(x)); return self.fc(F.adaptive_avg_pool2d(x,1).flatten(1))

class TinyViT(nn.Module):
    def __init__(self, dim=64, depth=2, heads=4):
        super().__init__(); self.patch=nn.Conv2d(1,dim,4,4); self.cls=nn.Parameter(torch.zeros(1,1,dim)); self.pos=nn.Parameter(torch.randn(1,50,dim)*0.02)
        enc=nn.TransformerEncoderLayer(dim,heads,dim*4,batch_first=True,norm_first=True,dropout=0.0,activation="gelu"); self.enc=nn.TransformerEncoder(enc,depth); self.norm=nn.LayerNorm(dim); self.fc=nn.Linear(dim,10)
    def forward(self,x):
        x=self.patch(x).flatten(2).transpose(1,2); c=self.cls.expand(x.size(0),-1,-1); x=torch.cat([c,x],1); x=x+self.pos[:,:x.size(1)]; return self.fc(self.norm(self.enc(x)[:,0]))


def build_model(name):
    return {"mlp":MLP,"tinycnn":TinyCNN,"lenet5":LeNet5,"resnet18":ResNet18MNIST,"tinyvit":TinyViT}[name]()


def train_model(model, loader, device, epochs):
    model.to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3); model.train()
    for ep in range(epochs):
        n=0; good=0; total_loss=0.0
        for x,y in loader:
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True); z=model(x); loss=F.cross_entropy(z,y); loss.backward(); opt.step()
            total_loss += float(loss)*len(y); good += int((z.argmax(1)==y).sum()); n += len(y)
        print(f"epoch={ep+1} train_loss={total_loss/n:.5f} train_acc={good/n:.4f}", flush=True)

@torch.inference_mode()
def predict(model,x,device,chunk=1024):
    out=[]
    for s in range(0,len(x),chunk): out.append(model(x[s:s+chunk].to(device)).argmax(1).cpu())
    return torch.cat(out)

@torch.inference_mode()
def response_codes(model, images, device, probes):
    base=predict(model,images,device); bits=[]
    for j,p in enumerate(probes):
        yp=predict(model,apply_probe(images,p),device)
        bits.append((yp!=base).numpy().astype(np.uint8)); print(f"probe {j+1}/{len(probes)}",flush=True)
    return base.numpy(), np.stack(bits,1)


def pack_bits(bits):
    powers=(1 << np.arange(bits.shape[1],dtype=np.uint64)); return (bits.astype(np.uint64)*powers).sum(1)


def collision_stats(labels,preds,bits,hold_bits):
    correct=preds==labels; weight=bits.sum(1); valid=correct & (weight>=WEIGHT_MIN) & (weight<=WEIGHT_MAX); codes=pack_bits(bits)
    buckets=defaultdict(list)
    for i in np.flatnonzero(valid): buckets[int(codes[i])].append(int(i))
    pair_counts=Counter(); pairs=[]; cross=0
    for idxs in buckets.values():
        by=defaultdict(list)
        for i in idxs: by[int(labels[i])].append(i)
        labs=sorted(by)
        for ai,a in enumerate(labs):
            for b in labs[ai+1:]:
                n=len(by[a])*len(by[b]); cross+=n; pair_counts[(a,b)]+=n
                if len(pairs)<50000:
                    room=50000-len(pairs)
                    for i in by[a]:
                        for j in by[b]:
                            if room<=0: break
                            pairs.append((i,j)); room-=1
                        if room<=0: break
    class_n=Counter(int(labels[i]) for i in np.flatnonzero(valid)); denom=sum(class_n[a]*class_n[b] for a in class_n for b in class_n if a<b)
    if pairs:
        ag=np.array([np.mean(hold_bits[i]==hold_bits[j]) for i,j in pairs],float); ex=np.array([np.all(hold_bits[i]==hold_bits[j]) for i,j in pairs],bool)
        hold_mean=float(ag.mean()); hold_exact=int(ex.sum())
    else: hold_mean=float("nan"); hold_exact=0
    return valid,codes,{
        "valid_nondegenerate":int(valid.sum()),"cross_semantic_claws":int(cross),"collision_density":float(cross/denom if denom else 0.0),
        "class_pairs_with_claws":int(sum(v>0 for v in pair_counts.values())),"pair_1_4_claws":int(pair_counts[(1,4)]),"pair_0_6_claws":int(pair_counts[(0,6)]),
        "holdout_pair_sample_n":len(pairs),"mean_holdout_agreement":hold_mean,"holdout_exact_in_sample":hold_exact,
        "top_class_pairs":";".join(f"{a}-{b}:{n}" for (a,b),n in pair_counts.most_common(10))
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--epochs",type=int,default=3); ap.add_argument("--seed",type=int,default=7); ap.add_argument("--models",nargs="+",default=["mlp","tinycnn","lenet5","resnet18","tinyvit"]); ap.add_argument("--out",default="stage9_out"); args=ap.parse_args()
    seed_all(args.seed); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); device=torch.device("cpu")
    tf=transforms.ToTensor(); tr=datasets.MNIST("data",train=True,download=True,transform=tf); te=datasets.MNIST("data",train=False,download=True,transform=tf)
    g=torch.Generator().manual_seed(args.seed); tr_loader=DataLoader(tr,batch_size=256,shuffle=True,generator=g,num_workers=2); te_loader=DataLoader(te,batch_size=512,shuffle=False,num_workers=2)
    images=torch.cat([x for x,_ in te_loader]); labels=torch.cat([y for _,y in te_loader]).numpy(); search=make_probe_bank("search"); hold=make_probe_bank("holdout")
    protocol={"name":"F-v1","search":{"shift_pixels":SEARCH_SHIFT,"directions":DIRECTIONS,"occlusion_size":SEARCH_OCC,"occlusion_positions":SEARCH_OCC_POS,"noise_eps":SEARCH_NOISE_EPS,"noise_seed":SEARCH_NOISE_SEED},"holdout":{"shift_pixels":HOLDOUT_SHIFT,"occlusion_size":HOLDOUT_OCC,"occlusion_positions":HOLDOUT_OCC_POS,"noise_eps":HOLDOUT_NOISE_EPS,"noise_seed":HOLDOUT_NOISE_SEED},"weight_window":[WEIGHT_MIN,WEIGHT_MAX]}
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    rows=[]
    for name in args.models:
        print(f"=== MODEL {name} ===",flush=True); seed_all(args.seed); model=build_model(name); train_model(model,tr_loader,device,args.epochs); model.eval()
        preds,bits=response_codes(model,images,device,search); _,hold_bits=response_codes(model,images,device,hold); valid,codes,stats=collision_stats(labels,preds,bits,hold_bits); acc=float(np.mean(preds==labels)); stats.update({"architecture":name,"seed":args.seed,"epochs":args.epochs,"test_accuracy":acc})
        print("RESULT "+json.dumps(stats,sort_keys=True),flush=True); rows.append(stats)
        np.savez_compressed(out/f"{name}_seed{args.seed}_codes.npz",labels=labels,preds=preds,search_bits=bits,holdout_bits=hold_bits,valid=valid,codes=codes)
    fields=["architecture","seed","epochs","test_accuracy","valid_nondegenerate","cross_semantic_claws","collision_density","class_pairs_with_claws","pair_1_4_claws","pair_0_6_claws","mean_holdout_agreement","holdout_pair_sample_n","holdout_exact_in_sample","top_class_pairs"]
    with (out/"stage9_fv1_summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows([{k:r.get(k) for k in fields} for r in rows])
    print("SUMMARY_JSON "+json.dumps(rows,sort_keys=True),flush=True)

if __name__=="__main__": main()
