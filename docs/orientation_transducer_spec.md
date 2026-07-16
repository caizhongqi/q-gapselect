# Orientation-aware multiscale Top-k relation transducer

Status: **stopped in its present form after the mandatory composition audit**.

This document fixes the next quantum-theory target after the adaptive-phase
execution baseline.  It deliberately excludes application-specific reward
oracles, QNNs, and any boundary routine that already reveals Top-k membership.

## 1. Access model and output relation

The input is coherent canonical block access to unknown angles
`theta_1, ..., theta_n` through `B_theta` and its inverse.  The algorithm knows
`n`, `k`, and a failure target `delta`; it is not given:

- the sorted order;
- the Top-k boundary;
- a numerical separating threshold;
- selected/rejected membership; or
- arm-dependent gaps.

It must output the exact Top-k set with probability at least `1-delta`.  A
certificate must be independently checkable under the same charged oracle.

The present `QBoundaryEstimator` does not satisfy this interface: its
certificate already contains selected/rejected membership.  It remains an
execution and falsification control only.

## 2. Multiscale three-state relation

Let `epsilon_r = 2^-r` be an angular resolution scale and let `beta_r` be a
reversibly represented boundary interval, not a classically materialized
membership certificate.  A level classifier must implement the three-state
relation

```text
IN          theta_i is certified above the boundary interval by epsilon_r
OUT         theta_i is certified below the boundary interval by epsilon_r
UNRESOLVED  neither statement is yet certified
```

The bounded-error classifier must expose a unitary stopping flag, preserve the
arm index, and uncompute phase-estimation workspace.  Reusing a measured
classical classification table is outside the target model.

Define `A_r` as the coherently active arms at level `r`.  For orientation
`b in {selected, rejected}`, let `N_r = |A_r|` and let `M_(r,b)` be the number
of newly certifiable outputs on that orientation.  The candidate accounting
functional is

```text
H_b = sum_r sqrt(N_r * (M_(r,b) + 1)) / epsilon_r
H_orient = min(H_selected, H_rejected)
```

This expression is a conjecture.  The repository must not call it a runtime
bound until all construction and lower-bound obligations below are discharged.

## 3. Candidate constructive architecture

1. **Unknown-boundary transducer.** Maintain a coherent interval/order
   relation sufficient for three-state comparison without returning a full
   selected/rejected table.
2. **Variable-precision classifier.** Stop easy arms at coarse levels and carry
   only `UNRESOLVED` arms forward.  All stop flags and garbage must be reversible.
3. **Activity-history unitary.** Support coherent membership in `A_r` without
   rebuilding an `O(nr)` classical table or assuming free QRAM.
4. **Known batch extraction.** Use charged all-marked extraction only after the
   level predicate is defined.  Batch extraction itself is prior work.
5. **Two orientation workers.** Dovetail selected and rejected certificate
   relations and accept the first independently verified complete certificate.
6. **Final verifier.** Charge every verification query and reject unresolved,
   duplicate, or non-separating outputs.  Budget exhaustion never proves
   absence.

The current adaptive-phase scheduler implements only a preliminary component:
it chooses one QPE precision from a measured scalar margin before search.  It
does not execute coherent multilevel stopping and does not satisfy step 1.

The direct unknown-oracle multi-output executor is also narrower than this
specification: it sequentially repeats full-workspace BBHT with measured
exclusion and fresh verification.  It is an executable baseline, not a direct
one-shot multi-output theorem.

## 4. Mandatory prior-composition audit

Before claiming a new upper bound, instantiate the strongest known generic
composition results against this exact relation:

- gapped phase/amplitude discrimination;
- variable-time amplitude amplification and estimation;
- quantum best-arm identification;
- weighted/thrifty quantum subroutine composition;
- all-marked extraction; and
- approximate and exact quantum k-minimum finding.

In particular, compute the outer query weights required by Jeffery-style
subroutine composition.  If an existing composition yields
`soft-O(H_orient)` or better under the same access and output model, the
candidate novelty is falsified.

Relevant primary sources include:

- [Childs, Kothari, and Somma, 2017](https://arxiv.org/abs/1511.02306)
- [Wang et al., quantum best-arm identification, 2021](https://arxiv.org/abs/2007.07049)
- [Jeffery, quantum subroutine composition](https://arxiv.org/abs/2209.14146)
- [Jeffery et al., loop composition, 2026](https://arxiv.org/abs/2605.07518)
- [Belovs, Jeffery, and Yolcu, 2024](https://arxiv.org/abs/2311.15873)
- [Ambainis, Kokainis, and Vihrovs, 2023](https://arxiv.org/abs/2302.06749)
- [van Apeldoorn, Gribling, and Nieuwboer, 2024](https://arxiv.org/abs/2302.10244)
- [Gao, Ji, and Wang, 2025](https://arxiv.org/abs/2412.16586)
- [Wang, Xu, and Zhang, 2026 preprint](https://arxiv.org/abs/2601.13195)

## 5. Separation and lower-bound targets

One falsification family uses `n=3m`, `k=m`, random arm identities, an unknown
center `beta`, all selected arms at `beta+gamma_m`, one rejected arm at
`beta-gamma_m`, and the remaining rejected arms at constant angular distance,
with `gamma_m=m^-2`.

The original target was to prove, under identical access assumptions,

```text
candidate rejected-orientation:  soft-O(sqrt(m) / gamma_m)
    strongest audited black boxes:    soft-O(m / gamma_m)
```

This comparison is false.  Constant-precision coarse grouping followed by
sign-reversed strong-oracle best-arm identification on the remaining `m+1`
near-boundary arms costs

```text
soft-O(m + sqrt(m) / gamma_m) = soft-O(sqrt(m) / gamma_m),
```

which matches the candidate scale.  The family is therefore a negative audit
case, not a separation witness.  A replacement lower bound must cover all
adaptive quantum algorithms, justify cross-level composition, and match a
full same-interface algorithm rather than an arithmetic proxy.

## 6. Termination gates

The candidate is stopped or renamed as an application baseline if any of the
following occurs:

- generic composition already matches the proposed functional;
- activity history introduces a hidden linear reconstruction cost;
- bounded-error reuse forces worst-gap precision on all active arms;
- the explicit separation family collapses under a stronger black-box method;
- no all-algorithms matching lower bound can be established; or
- the natural-purification transport requires a free clean-rotation conversion.

The generic-composition and explicit-family gates have now fired.  The only
remaining high-value successor problem is a new same-interface construction
for an unknown Top-k boundary and coherent unresolved-history access without
free QRAM, together with a different hard family and all-algorithms lower
bound.  The present `H_orient` outer sum is not itself a new quantum core.
The active successor specification is
`docs/unknown_boundary_history_spec.md`.
