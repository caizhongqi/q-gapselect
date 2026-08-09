#!/usr/bin/env python3
"""Experiment 5QB: strong classical-structure audit for QDSA seed search.

Experiment 5Q found a large-N crossover for fixed-accounting DH/BBHT, but its
classical structured controls used pixel and semantic rankings separately.  This
experiment attacks that weakness before any quantum-advantage claim.

Classical controls
------------------
1. hybrid_min_rank:
   free public ranking by the minimum of pixel-rank and semantic-rank.
2. hybrid_borda:
   free public ranking by the sum of pixel-rank and semantic-rank.
3. graph_best_first:
   an adaptive classical search on a kNN graph built from the independent public
   semantic embedding.  It warms up with 8 hybrid-min candidates, then expands
   neighbors of queried candidates with the smallest observed victim collision
   distance; every 16th query is a hybrid-rank exploration step.  Only queried
   victim distances are used adaptively.

Quantum control
---------------
Fixed-budget DH/BBHT from Experiment 5Q: no free empty-marked-set/global-optimum
check, 2 oracle calls per Grover phase iteration and 1 per verification.

All methods receive exactly the same victim-representation oracle budget.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from exp1_mnist_qco import SmallCNN, embed_dataset, seed_all, train_victim
from exp1c_qdsa_semantic_gate import CALIBRATION_N, SemanticMLP, calibrate_delta, raw_flat
from exp2_qdsa_attack_effect_v2 import legal_mask_for_target, select_balanced_feasible_targets
from exp3a_certified_raw_collision import embed_raw_victim, raw_distances
from exp5q_query_scaling_fixed_accounting import qco_min_find_fixed_budget, quality_metrics

N_VALUES = (512, 1024, 2048)
BUDGETS = (64, 128, 256, 512)
SIM_SEEDS = (11, 22, 33, 44, 55)
KNN_K = 16
WARM_QUERIES = 8
GLOBAL_EXPLORE_PERIOD = 16
EPS = 1e-12


def pixel_order(target_pix: np.ndarray, pool_pix: np.ndarray) -> np.ndarray:
    d2 = np.sum((pool_pix - target_pix[None, :]) ** 2, axis=1)
    return np.argsort(d2, kind="stable")


def semantic_order(target_ref: np.ndarray, pool_ref: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(pool_ref - target_ref[None, :], axis=1)
    return np.argsort(d, kind="stable")


def fusion_orders(p_order: np.ndarray, s_order: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(p_order)
    rp = np.empty(n, dtype=np.int64); rp[p_order] = np.arange(n)
    rs = np.empty(n, dtype=np.int64); rs[s_order] = np.arange(n)
    min_rank = np.minimum(rp, rs)
    sum_rank = rp + rs
    # deterministic tie-breaking with the complementary score, then index
    idx = np.arange(n)
    order_min = np.lexsort((idx, sum_rank, min_rank))
    order_sum = np.lexsort((idx, min_rank, sum_rank))
    return order_min.astype(int), order_sum.astype(int)


def master_semantic_distance_matrix(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float32, copy=False)
    norms = np.sum(z * z, axis=1, keepdims=True)
    d2 = norms + norms.T - 2.0 * (z @ z.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def knn_from_submatrix(d2: np.ndarray, n: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    sub = d2[:n, :n].copy()
    np.fill_diagonal(sub, np.inf)
    kk = min(k, n - 1)
    nbr = np.argpartition(sub, kth=kk - 1, axis=1)[:, :kk]
    row = np.arange(n)[:, None]
    local_order = np.argsort(sub[row, nbr], axis=1, kind="stable")
    nbr = nbr[row, local_order]
    edge = np.sqrt(sub[row, nbr])
    return nbr.astype(int), edge.astype(np.float64)


def best_from_order(distances: np.ndarray, order: np.ndarray, budget: int) -> Tuple[int, float, int]:
    q = min(int(budget), len(distances))
    queried = order[:q]
    li = int(queried[np.argmin(distances[queried])])
    return li, float(distances[li]), q


def graph_best_first(
    distances: np.ndarray,
    fusion_order: np.ndarray,
    nbr: np.ndarray,
    edge: np.ndarray,
    budget: int,
) -> Tuple[int, float, int]:
    """Adaptive public-graph search using only already queried victim distances."""
    n = len(distances)
    qmax = min(int(budget), n)
    queried = np.zeros(n, dtype=bool)
    heap: List[Tuple[float, float, int, int]] = []
    best_idx = -1
    best = float("inf")
    queries = 0
    fusion_cursor = 0

    def query_idx(i: int) -> None:
        nonlocal best_idx, best, queries
        if queried[i] or queries >= qmax:
            return
        queried[i] = True
        queries += 1
        y = float(distances[i])
        if y < best - EPS:
            best = y; best_idx = i
        for j, ed in zip(nbr[i], edge[i]):
            j = int(j)
            if not queried[j]:
                # Parent objective first, then public edge length.  No fitted
                # hyperparameter or test-label information is used.
                heapq.heappush(heap, (y, float(ed), j, i))

    # Warm-up from hybrid public geometry.
    while queries < min(WARM_QUERIES, qmax):
        while fusion_cursor < n and queried[int(fusion_order[fusion_cursor])]:
            fusion_cursor += 1
        if fusion_cursor >= n:
            break
        i = int(fusion_order[fusion_cursor]); fusion_cursor += 1
        query_idx(i)

    while queries < qmax:
        # Periodic global exploration prevents a single local component/frontier
        # from monopolizing the entire query budget.
        use_global = (queries % GLOBAL_EXPLORE_PERIOD == 0)
        chosen = None
        if not use_global:
            while heap:
                _, _, j, _ = heapq.heappop(heap)
                if not queried[j]:
                    chosen = j
                    break
        if chosen is None:
            while fusion_cursor < n and queried[int(fusion_order[fusion_cursor])]:
                fusion_cursor += 1
            if fusion_cursor < n:
                chosen = int(fusion_order[fusion_cursor]); fusion_cursor += 1
            else:
                unseen = np.flatnonzero(~queried)
                if len(unseen) == 0:
                    break
                chosen = int(unseen[0])
        query_idx(chosen)

    if best_idx < 0:
        raise RuntimeError("graph search made no query")
    return best_idx, best, queries


def summarize(rows: List[dict]) -> List[dict]:
    out=[]
    for n in N_VALUES:
        for b in [x for x in BUDGETS if x <= n]:
            for m in sorted({r['method'] for r in rows}):
                cell=[r for r in rows if r['pool_size']==n and r['budget']==b and r['method']==m]
                if not cell: continue
                targets=sorted({r['target_test_index'] for r in cell})
                exact=[]; near=[]; regret=[]; dist=[]; rank=[]; queries=[]
                for t in targets:
                    g=[r for r in cell if r['target_test_index']==t]
                    exact.append(np.mean([r['exact_global_hit'] for r in g]))
                    near.append(np.mean([r['within_5pct_global'] for r in g]))
                    regret.append(np.mean([r['normalized_regret'] for r in g]))
                    dist.append(np.mean([r['best_distance'] for r in g]))
                    rank.append(np.mean([r['rank_fraction'] for r in g]))
                    queries.append(np.mean([r['queries'] for r in g]))
                out.append({
                    'pool_size':n,'budget':b,'method':m,'n_independent_targets':len(targets),
                    'exact_global_hit_rate':float(np.mean(exact)),
                    'within_5pct_global_rate':float(np.mean(near)),
                    'mean_normalized_regret':float(np.mean(regret)),
                    'mean_best_distance':float(np.mean(dist)),
                    'mean_rank_fraction':float(np.mean(rank)),
                    'mean_queries':float(np.mean(queries)),
                })
    return out


def paired_bootstrap(rows: List[dict], n: int, b: int, classical: str, reps: int = 10000) -> Dict[str, float]:
    targets=sorted({r['target_test_index'] for r in rows})
    q=[]; c=[]; qr=[]; cr=[]
    for t in targets:
        qg=[r for r in rows if r['target_test_index']==t and r['pool_size']==n and r['budget']==b and r['method']=='qco_fixed']
        cg=[r for r in rows if r['target_test_index']==t and r['pool_size']==n and r['budget']==b and r['method']==classical]
        q.append(np.mean([r['exact_global_hit'] for r in qg])); c.append(np.mean([r['exact_global_hit'] for r in cg]))
        qr.append(np.mean([r['normalized_regret'] for r in qg])); cr.append(np.mean([r['normalized_regret'] for r in cg]))
    d=np.asarray(q)-np.asarray(c); dr=np.asarray(qr)-np.asarray(cr)
    rng=np.random.default_rng(2026080952+n+b+sum(ord(x) for x in classical))
    be=np.empty(reps); br=np.empty(reps)
    for i in range(reps):
        ix=rng.integers(0,len(d),size=len(d)); be[i]=d[ix].mean(); br[i]=dr[ix].mean()
    return {
        'pool_size':n,'budget':b,'classical_method':classical,
        'qco_minus_classical_exact_hit':float(d.mean()),
        'exact_bootstrap_lo':float(np.quantile(be,.025)),'exact_bootstrap_hi':float(np.quantile(be,.975)),
        'qco_minus_classical_normalized_regret':float(dr.mean()),
        'regret_bootstrap_lo':float(np.quantile(br,.025)),'regret_bootstrap_hi':float(np.quantile(br,.975)),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='exp5qb_out')
    ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--train-seed',type=int,default=20260809)
    ap.add_argument('--targets',type=int,default=20)
    ap.add_argument('--max-pool',type=int,default=2048)
    args=ap.parse_args()

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cpu'); tf=transforms.ToTensor()
    train_ds=datasets.MNIST('data',train=True,download=True,transform=tf)
    test_ds=datasets.MNIST('data',train=False,download=True,transform=tf)

    seed_all(args.train_seed); vg=torch.Generator().manual_seed(args.train_seed)
    victim_loader=DataLoader(train_ds,batch_size=256,shuffle=True,generator=vg,num_workers=2)
    test_loader=DataLoader(test_ds,batch_size=512,shuffle=False,num_workers=2)
    victim=SmallCNN().to(device); train_victim(victim,victim_loader,device,args.epochs)
    labels,preds,_,raw_h=embed_raw_victim(victim,test_loader,device)
    victim_acc=float(np.mean(labels==preds))

    ref_seed=args.train_seed+101; seed_all(ref_seed); rg=torch.Generator().manual_seed(ref_seed)
    ref_loader=DataLoader(train_ds,batch_size=256,shuffle=True,generator=rg,num_workers=2)
    ref=SemanticMLP().to(device); train_victim(ref,ref_loader,device,args.epochs)
    ref_labels,ref_preds,ref_emb=embed_dataset(ref,test_loader,device); ref_acc=float(np.mean(ref_labels==ref_preds))
    if victim_acc<.97 or ref_acc<.95: raise RuntimeError('accuracy gate failed')

    cal_ds=Subset(train_ds,range(CALIBRATION_N)); cal_loader=DataLoader(cal_ds,batch_size=512,shuffle=False,num_workers=2)
    rcal_labels,rcal_preds,rcal_emb=embed_dataset(ref,cal_loader,device); good=rcal_labels==rcal_preds
    delta,n_sem_pairs=calibrate_delta(rcal_emb[good],rcal_labels[good])
    targets,feasible_counts,rejected=select_balanced_feasible_targets(labels,preds,ref_labels,ref_preds,ref_emb,delta,args.max_pool,args.targets)
    test_x=torch.stack([test_ds[i][0] for i in range(len(test_ds))],dim=0); pix=raw_flat(test_x)

    rows=[]
    for target_ord,t0 in enumerate(targets):
        t=int(t0); legal,sem_all=legal_mask_for_target(t,labels,ref_labels,ref_preds,ref_emb,delta); eligible=np.flatnonzero(legal)
        master_rng=np.random.default_rng(12_000_000+t*31); master_pool=master_rng.choice(eligible,size=args.max_pool,replace=False)
        master_ref=ref_emb[master_pool]
        master_d2=master_semantic_distance_matrix(master_ref)

        for n in N_VALUES:
            pool_idx=master_pool[:n]; distances=raw_distances(raw_h[t],raw_h[pool_idx])
            pord=pixel_order(pix[t],pix[pool_idx]); sord=semantic_order(ref_emb[t],ref_emb[pool_idx])
            fmin,fborda=fusion_orders(pord,sord); nbr,edge=knn_from_submatrix(master_d2,n,KNN_K)
            for b in [x for x in BUDGETS if x<=n]:
                for ss in SIM_SEEDS:
                    base=int(ss*1_000_003+t*97+n*17+b*13)
                    qres=qco_min_find_fixed_budget(distances,b,np.random.default_rng(base))
                    qm=quality_metrics(qres.best_distance,distances)
                    rows.append({'target_order':target_ord,'target_test_index':t,'target_class':int(labels[t]),'pool_size':n,'budget':b,'sim_seed':ss,'method':'qco_fixed','queries':qres.strict_queries,'selected_local_index':qres.best_index,**qm})
                for m,order in [('hybrid_min_rank',fmin),('hybrid_borda',fborda)]:
                    li,best,q=best_from_order(distances,order,b); qm=quality_metrics(best,distances)
                    rows.append({'target_order':target_ord,'target_test_index':t,'target_class':int(labels[t]),'pool_size':n,'budget':b,'sim_seed':-1,'method':m,'queries':q,'selected_local_index':li,**qm})
                li,best,q=graph_best_first(distances,fmin,nbr,edge,b); qm=quality_metrics(best,distances)
                rows.append({'target_order':target_ord,'target_test_index':t,'target_class':int(labels[t]),'pool_size':n,'budget':b,'sim_seed':-1,'method':'graph_best_first','queries':q,'selected_local_index':li,**qm})
        print(f'target={t} done',flush=True)

    with (out/'trials.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary=summarize(rows)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    contrasts=[]
    for n in N_VALUES:
        for b in [x for x in BUDGETS if x<=n]:
            for c in ('hybrid_min_rank','hybrid_borda','graph_best_first'):
                contrasts.append(paired_bootstrap(rows,n,b,c))
    with (out/'paired_contrasts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(contrasts[0].keys())); w.writeheader(); w.writerows(contrasts)

    result={
        'name':'QDSA-Experiment-5QB-Strong-Classical-Structure-Audit',
        'role':'stress-test fixed-accounting DH/BBHT scaling against fused and adaptive classical public-geometry search',
        'victim_accuracy':victim_acc,'reference_accuracy':ref_acc,'semantic_delta':delta,
        'targets':int(len(targets)),'pool_sizes':list(N_VALUES),'budgets':list(BUDGETS),'sim_seeds':list(SIM_SEEDS),
        'methods':['qco_fixed','hybrid_min_rank','hybrid_borda','graph_best_first'],
        'graph_knn_k':KNN_K,'graph_warm_queries':WARM_QUERIES,'graph_global_explore_period':GLOBAL_EXPLORE_PERIOD,
        'query_accounting':'all methods receive the same victim-representation query budget; QCO has no free optimum detection',
        'coherent_oracle_assumption':'QCO assumes coherent victim raw-representation distance predicate access',
        'semantic_calibration_pairs':n_sem_pairs,'mean_eligible_candidates':float(np.mean(list(feasible_counts.values()))),'rejected_for_capacity':int(rejected),
        'summary':summary,'paired_contrasts':contrasts,
    }
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('SUMMARY_JSON '+json.dumps(result,sort_keys=True),flush=True)

if __name__=='__main__': main()
