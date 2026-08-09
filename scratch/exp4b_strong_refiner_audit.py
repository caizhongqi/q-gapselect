#!/usr/bin/env python3
"""Experiment 4B: strong-refiner audit of QDSA semantic-nullspace transport.

Experiment 4A showed a large gap over a feasibility-backtracked barrier PGD, but
that baseline rejected many steps.  This audit replaces it with the stronger
source-ball Adam used in Experiment 3C.  Both refiners start from exactly the same
seed for each target/method/repeat and receive the same refinement-step budget.

Refiners
--------
1. constrained_adam:
   Adam minimizes raw victim-feature collision + semantic distance barrier +
   independent-reference source-class CE, projecting every iterate to the same
   L_inf=0.20 source ball.  The best semantically legal iterate is reported.
2. semantic_nullspace:
   Experiment-4A semantic-Jacobian null-space transport, with hard feasibility
   backtracking.

Seed methods retain the fixed 128 representation-oracle-equivalent budget:
DH/BBHT, random, pixel-NN, semantic-boundary.

This remains a controlled white-box mechanism experiment, not a black-box attack
claim and not a claim that DH/BBHT itself is novel.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils as tvutils

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from exp1_mnist_qco import SmallCNN, embed_dataset, seed_all, train_victim
from exp1c_qdsa_semantic_gate import CALIBRATION_N, SemanticMLP, calibrate_delta, raw_flat
from exp2_qdsa_attack_effect_v2 import POOL_SELECTION_SEED_BASE, legal_mask_for_target, select_balanced_feasible_targets
from exp3a_certified_raw_collision import embed_raw_victim, certified_radius, raw_distances
from exp3b_continuous_existence import victim_raw, reference_forward
from exp4a_jacobian_nullspace_pilot import (
    EPS_LINF, SEED_BUDGET, STEP_SIZE, STOCHASTIC_REPEATS, CHECKPOINTS,
    choose_seed, refine_one, source_margin,
)

ADAM_LR = 0.03
ADAM_LAMBDA_SEM = 150.0
ADAM_LAMBDA_CLS = 2.0
EPS = 1e-12


def project_source_ball_(x: torch.Tensor, x0: torch.Tensor, eps: float = EPS_LINF) -> None:
    with torch.no_grad():
        lo = torch.clamp(x0 - eps, 0.0, 1.0)
        hi = torch.clamp(x0 + eps, 0.0, 1.0)
        x.copy_(torch.maximum(torch.minimum(x, hi), lo))


def metrics(victim, ref, x, h_t, ref_h_t, target_class, source_class, radius, delta):
    with torch.no_grad():
        vlog, h = victim_raw(victim, x)
        rlog, rh = reference_forward(ref, x)
        d = float(torch.linalg.vector_norm(h - h_t))
        sem = float(torch.linalg.vector_norm(rh - ref_h_t))
        margin = float(source_margin(rlog, source_class))
        vp = int(vlog.argmax(1).item()); rp = int(rlog.argmax(1).item())
    feasible = sem >= delta and rp == source_class
    empirical = feasible and vp == target_class and source_class != target_class
    certified = feasible and d < radius and source_class != target_class
    return {
        "raw_distance": d,
        "semantic_distance": sem,
        "source_margin": margin,
        "victim_pred": vp,
        "reference_pred": rp,
        "feasible": int(feasible),
        "empirical_success": int(empirical),
        "certified_success": int(certified),
    }


def constrained_adam(
    victim, ref, x0, h_t, ref_h_t, target_class, source_class, radius, delta,
    steps: int, lr: float = ADAM_LR,
):
    x = x0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    best = metrics(victim, ref, x, h_t, ref_h_t, target_class, source_class, radius, delta)
    best_x = x.detach().clone()
    cps = {0: dict(best)}
    infeasible_iterates = 0

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        vlog, h = victim_raw(victim, x)
        rlog, rh = reference_forward(ref, x)
        feat = torch.mean((h - h_t) ** 2)
        sem_d = torch.linalg.vector_norm(rh - ref_h_t)
        sem_pen = F.relu(delta - sem_d) ** 2
        src_ce = F.cross_entropy(rlog, torch.tensor([source_class], device=x.device))
        loss = feat + ADAM_LAMBDA_SEM * sem_pen + ADAM_LAMBDA_CLS * src_ce
        loss.backward()
        opt.step()
        project_source_ball_(x, x0)

        m = metrics(victim, ref, x, h_t, ref_h_t, target_class, source_class, radius, delta)
        if not m["feasible"]:
            infeasible_iterates += 1
        if m["feasible"] and m["raw_distance"] < best["raw_distance"] - 1e-10:
            best = dict(m); best_x = x.detach().clone()
        if step in CHECKPOINTS:
            cps[step] = dict(best)

    best.update({"infeasible_iterates": infeasible_iterates, "total_backtracks": 0, "rejected_steps": 0})
    return best, cps, best_x


def aggregate(rows: List[dict]) -> List[dict]:
    out=[]
    for sm in sorted({r['seed_method'] for r in rows}):
        for rm in sorted({r['refine_mode'] for r in rows}):
            cell=[r for r in rows if r['seed_method']==sm and r['refine_mode']==rm]
            targets=sorted({r['target_test_index'] for r in cell})
            vals={k:[] for k in ['asr','cert','seed','dist','infeasible','rejected']}
            cp={c:[] for c in CHECKPOINTS}
            for t in targets:
                g=[r for r in cell if r['target_test_index']==t]
                vals['asr'].append(np.mean([r['best_empirical_success'] for r in g]))
                vals['cert'].append(np.mean([r['best_certified_success'] for r in g]))
                vals['seed'].append(np.mean([r['seed_raw_distance'] for r in g]))
                vals['dist'].append(np.mean([r['best_raw_distance'] for r in g]))
                vals['infeasible'].append(np.mean([r['infeasible_iterates'] for r in g]))
                vals['rejected'].append(np.mean([r['rejected_steps'] for r in g]))
                for c in CHECKPOINTS:
                    cp[c].append(np.mean([r[f'success_at_{c}'] for r in g]))
            rec={
                'seed_method':sm,'refine_mode':rm,'n_independent_targets':len(targets),
                'target_averaged_asr':float(np.mean(vals['asr'])),
                'target_averaged_certified_asr':float(np.mean(vals['cert'])),
                'mean_seed_raw_distance':float(np.mean(vals['seed'])),
                'mean_best_raw_distance':float(np.mean(vals['dist'])),
                'mean_infeasible_iterates':float(np.mean(vals['infeasible'])),
                'mean_rejected_steps':float(np.mean(vals['rejected'])),
            }
            for c in CHECKPOINTS: rec[f'asr_at_step_{c}']=float(np.mean(cp[c]))
            out.append(rec)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='exp4b_out')
    ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--train-seed',type=int,default=20260809)
    ap.add_argument('--targets',type=int,default=10)
    ap.add_argument('--pool',type=int,default=512)
    ap.add_argument('--seed-budget',type=int,default=SEED_BUDGET)
    ap.add_argument('--steps',type=int,default=120)
    args=ap.parse_args()

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cpu'); tf=transforms.ToTensor()
    train_ds=datasets.MNIST('data',train=True,download=True,transform=tf)
    test_ds=datasets.MNIST('data',train=False,download=True,transform=tf)

    seed_all(args.train_seed)
    vg=torch.Generator().manual_seed(args.train_seed)
    victim_loader=DataLoader(train_ds,batch_size=256,shuffle=True,generator=vg,num_workers=2)
    test_loader=DataLoader(test_ds,batch_size=512,shuffle=False,num_workers=2)
    victim=SmallCNN().to(device); train_victim(victim,victim_loader,device,args.epochs)
    labels,preds,logits_np,raw_h_np=embed_raw_victim(victim,test_loader,device)
    victim_acc=float(np.mean(labels==preds))

    ref_seed=args.train_seed+101; seed_all(ref_seed)
    rg=torch.Generator().manual_seed(ref_seed)
    ref_loader=DataLoader(train_ds,batch_size=256,shuffle=True,generator=rg,num_workers=2)
    ref=SemanticMLP().to(device); train_victim(ref,ref_loader,device,args.epochs)
    ref_labels,ref_preds,ref_emb_test=embed_dataset(ref,test_loader,device)
    ref_acc=float(np.mean(ref_labels==ref_preds))
    if victim_acc<0.97 or ref_acc<0.95: raise RuntimeError('accuracy gate failed')

    cal_ds=Subset(train_ds,range(CALIBRATION_N)); cal_loader=DataLoader(cal_ds,batch_size=512,shuffle=False,num_workers=2)
    rcal_labels,rcal_preds,rcal_emb=embed_dataset(ref,cal_loader,device); good=rcal_labels==rcal_preds
    delta,n_sem_pairs=calibrate_delta(rcal_emb[good],rcal_labels[good])
    target_indices,feasible_counts,rejected=select_balanced_feasible_targets(labels,preds,ref_labels,ref_preds,ref_emb_test,delta,args.pool,args.targets)

    test_x=torch.stack([test_ds[i][0] for i in range(len(test_ds))],dim=0); pix=raw_flat(test_x)
    for p in victim.parameters(): p.requires_grad_(False)
    for p in ref.parameters(): p.requires_grad_(False)

    rows=[]; panels=[]
    seed_methods=('dh_bbht','random','pixel_nn','semantic_boundary')
    for target_ord,t0 in enumerate(target_indices):
        t=int(t0); target_class=int(preds[t]); x_t=test_x[t:t+1].to(device)
        h_t=torch.from_numpy(raw_h_np[t:t+1]).to(device=device,dtype=torch.float32)
        with torch.no_grad(): _,ref_h_t=reference_forward(ref,x_t)
        radius=certified_radius(victim,logits_np[t],target_class)
        legal,sem_all=legal_mask_for_target(t,labels,ref_labels,ref_preds,ref_emb_test,delta)
        eligible=np.flatnonzero(legal); pool_rng=np.random.default_rng(POOL_SELECTION_SEED_BASE+t*23)
        pool_idx=pool_rng.choice(eligible,size=args.pool,replace=False)
        distances=raw_distances(raw_h_np[t],raw_h_np[pool_idx]); sem_pool=sem_all[pool_idx]; pool_pix=pix[pool_idx]

        for sm in seed_methods:
            repeats=STOCHASTIC_REPEATS if sm in ('dh_bbht','random') else (0,)
            for rep in repeats:
                seed_rng=int(500000+t*131+rep*1009+sum(ord(ch) for ch in sm))
                li,seed_queries=choose_seed(sm,distances,sem_pool,pix[t],pool_pix,args.seed_budget,seed_rng)
                gi=int(pool_idx[li]); x0=test_x[gi:gi+1].to(device); source_class=int(labels[gi])
                seed_raw=float(distances[li]); seed_sem=float(sem_pool[li])

                for rm in ('constrained_adam','semantic_nullspace'):
                    if rm=='constrained_adam':
                        best,cps,best_x=constrained_adam(victim,ref,x0,h_t,ref_h_t,target_class,source_class,radius,delta,args.steps)
                    else:
                        best,cps,best_x=refine_one(victim,ref,x0,h_t,ref_h_t,target_class,source_class,radius,delta,'semantic_nullspace',args.steps,STEP_SIZE)
                        best['infeasible_iterates']=0
                    row={
                        'target_order':target_ord,'target_test_index':t,'target_class':target_class,
                        'seed_method':sm,'seed_repeat':rep,'refine_mode':rm,'seed_test_index':gi,
                        'source_class':source_class,'seed_queries':seed_queries,'seed_raw_distance':seed_raw,
                        'seed_semantic_distance':seed_sem,'certified_radius':radius,
                        'best_raw_distance':best['raw_distance'],'best_semantic_distance':best['semantic_distance'],
                        'best_source_margin':best['source_margin'],'best_victim_pred':best['victim_pred'],
                        'best_reference_pred':best['reference_pred'],'best_empirical_success':best['empirical_success'],
                        'best_certified_success':best['certified_success'],'infeasible_iterates':best.get('infeasible_iterates',0),
                        'total_backtracks':best.get('total_backtracks',0),'rejected_steps':best.get('rejected_steps',0),
                    }
                    for c in CHECKPOINTS:
                        cp=cps.get(c,best); row[f'success_at_{c}']=cp['empirical_success']; row[f'raw_distance_at_{c}']=cp['raw_distance']
                    rows.append(row)
                    if target_ord<4 and sm=='dh_bbht' and rep==STOCHASTIC_REPEATS[0]: panels.extend([x_t.cpu(),x0.cpu(),best_x.cpu()])
        print(f'target={t} class={target_class} done',flush=True)

    with (out/'trials.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary=aggregate(rows)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    contrasts=[]
    for sm in seed_methods:
        da=[]; dd=[]
        for t in sorted({r['target_test_index'] for r in rows}):
            n=[r for r in rows if r['target_test_index']==t and r['seed_method']==sm and r['refine_mode']=='semantic_nullspace']
            a=[r for r in rows if r['target_test_index']==t and r['seed_method']==sm and r['refine_mode']=='constrained_adam']
            da.append(float(np.mean([r['best_empirical_success'] for r in n])-np.mean([r['best_empirical_success'] for r in a])))
            dd.append(float(np.mean([r['best_raw_distance'] for r in n])-np.mean([r['best_raw_distance'] for r in a])))
        contrasts.append({
            'seed_method':sm,'nullspace_minus_adam_asr':float(np.mean(da)),
            'nullspace_minus_adam_raw_distance':float(np.mean(dd)),
            'targets_nullspace_asr_better':int(np.sum(np.asarray(da)>0)),
            'targets_adam_asr_better':int(np.sum(np.asarray(da)<0)),
            'targets_nullspace_distance_better':int(np.sum(np.asarray(dd)<-1e-9)),
            'targets_adam_distance_better':int(np.sum(np.asarray(dd)>1e-9)),
        })
    with (out/'strong_refiner_contrasts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(contrasts[0].keys())); w.writeheader(); w.writerows(contrasts)
    if panels: tvutils.save_image(torch.cat(panels,dim=0),out/'qco_adam_vs_nullspace_examples.png',nrow=3,padding=2)

    result={
        'name':'QDSA-Experiment-4B-Strong-Refiner-Audit','role':'strong classical refiner audit before scaling',
        'victim_accuracy':victim_acc,'reference_accuracy':ref_acc,'semantic_delta':delta,'eps_linf':EPS_LINF,
        'seed_budget_strict_oracle_equiv':args.seed_budget,'refine_steps':args.steps,'targets':int(len(target_indices)),
        'pool_size':args.pool,'seed_methods':list(seed_methods),'refiners':['constrained_adam','semantic_nullspace'],
        'adam_lr':ADAM_LR,'adam_lambda_sem':ADAM_LAMBDA_SEM,'adam_lambda_cls':ADAM_LAMBDA_CLS,
        'hard_final_semantic_validity':['D_sem >= Delta','reference source class','L_inf source ball','pixels [0,1]'],
        'quantum_novelty_claim':'none for DH/BBHT','semantic_calibration_pairs':n_sem_pairs,
        'mean_eligible_candidates':float(np.mean(list(feasible_counts.values()))),'rejected_for_capacity':int(rejected),
        'summary':summary,'strong_refiner_contrasts':contrasts,
    }
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('SUMMARY_JSON '+json.dumps(result,sort_keys=True),flush=True)

if __name__=='__main__': main()
