# Unified theorem-closure audit

Audit date: 2026-07-15.

Status: **the current Q-GapSelect candidate does not have a closed quantum-
advantage theorem**.  This audit replaces the former practice of comparing an
upper proxy, a composition proxy, and a lower proxy that merely shared the
parameter name `m`.

The machine-readable implementation is
`src/qgapselect/theorem_closure_audit.py`; its invariants are exercised by
`tests/test_theorem_closure_audit.py`.

The stable programmatic entry points are:

```python
audit = build_unified_theorem_closure_audit(m)
status = machine_readable_status_map(audit)
```

`status` exposes `family_ids`, `interface_ids`, `upper`, `composition`,
`lower_bound`, `obligation_statuses`, `theorem_claimable`, and
`ccf_a_quantum_advantage_claimable` as JSON-safe fields.

## 1. One contract, not three count profiles

Every quantity in one comparison record now carries both:

- `family_id`: a digest of the actual input-family schema and information
  contract; and
- `interface_id`: a digest of the oracle, output relation, failure target,
  boundary access, activity-history access, QRAM access, partition visibility,
  and stopping-time visibility.

The common oracle is the canonical controlled block rotation `B_theta` (or its
inverse), charged one call at a time.  The common output relation is the exact
strict-gap Top-k index set with success probability at least `1-delta`.
Neither variant supplies the numeric Top-k boundary, an active-history table,
or history QRAM.

The input family is now concrete.  At level `r`, a block of `A_r` arms contains
exactly `M_r` high arms at angle `beta + epsilon_r`; the other block arms have
angle `beta - epsilon_r`.  All far arms have angle zero.  The common center
`beta` is hidden in a declared interval around `pi/4`.  Therefore the exact
Top-k output is precisely the union of the high arms across the blocks.  The
choices of high indices and `beta` parameterize the input set; they are not a
single deterministic vector.

This construction instantiates the same default scale profile used by the
charged-history candidate:

```text
n = m^3,  R = k = m,  A_r = m,  M_r = 1,
epsilon_r = m^-2 (r+1)^0.25.
```

## 2. The information dichotomy

There are two possible contracts, and neither currently supports the desired
claim.

### Public static partition

If the level blocks are a public, input-independent partition, a legal
composition runs one gap-resolved quantum selection procedure per block.  Up
to the same suppressed logarithmic/confidence factors used by the candidate,
its charged proxy is

```text
C_partition = sum_r sqrt(A_r M_r) c_r,
```

where `c_r` is the finite-QPE checker cost.  This procedure needs neither a
numeric global boundary nor an active-history QRAM table.  It follows from
known quantum best-arm/minimum and marked-extraction machinery; it is not a new
primitive.  Relevant primary sources are:

- [quantum best-arm identification](https://arxiv.org/abs/2007.07049);
- [quantum approximate k-minimum finding](https://arxiv.org/abs/2412.16586);
- [optimal all-marked extraction](https://arxiv.org/abs/2302.10244).

The current candidate proxy is

```text
C_candidate = C_boundary + C_history + C_direct.
```

On every audited default point, the public-partition composition is within the
declared `1.2` matching tolerance and is numerically smaller than the candidate:

| m | n | candidate | partition composition | composition / candidate |
|---:|---:|---:|---:|---:|
| 4 | 64 | 427.362 | 248.000 | 0.580 |
| 8 | 512 | 3,900.996 | 2,873.682 | 0.737 |
| 16 | 4,096 | 35,653.817 | 31,680.000 | 0.889 |
| 32 | 32,768 | 308,005.186 | 272,072.062 | 0.883 |
| 64 | 262,144 | 2,922,927.544 | 2,588,160.000 | 0.885 |
| 128 | 2,097,152 | 29,612,315.574 | 26,505,575.252 | 0.895 |
| 256 | 16,777,216 | 312,299,749.261 | 283,635,712.000 | 0.908 |

These are analytic query proxies, not hardware runs.  Their role is
falsification: the advertised strict separation does not survive this legal
public-partition composition.

The failure is asymptotic, not a finite-grid accident.  The QPE schedule obeys
`c_r < 4/epsilon_r`, so the partitioned proxy is

```text
O(sqrt(m) m^2 sum_{r<=m} r^-1/4) = O(m^13/4).
```

The candidate proxy is at least its declared boundary term
`sqrt(m^3) m^2 = m^7/2`.  Thus
`C_partition / C_candidate = O(m^-1/4)`: on the public-partition
instantiation, the known-composition proxy asymptotically beats rather than
merely matches the candidate proxy.  The code records this as
`proved_local_proxy_asymptotic_lemma`; it is still a proxy comparison, not a
hardware or end-to-end algorithm theorem.

### Hidden instance-dependent partition

If the blocks are hidden, the partitioned baseline is correctly marked
invalid.  But the candidate also loses the very information needed to apply
its `A_r`-only history and extraction formulas.  It must coherently discover,
use, and erase the hidden partition without materializing a free history list.
No such construction is presently proved.

This prevents an interface switch in either direction:

```text
public partition  -> candidate is matched by known composition;
hidden partition  -> candidate upper bound is unproved.
```

## 3. Variable-time stopping cannot be the novelty by itself

For known stopping times, the `sqrt(sum_i t_i^2)` search scale is prior work.
[Ambainis--Kokainis--Vihrovs](https://arxiv.org/abs/2302.06749) summarize the
known-times upper and matching lower bound and prove an
`Omega(sqrt(T log T))` lower bound for an unknown-times family.  The later
[computation-tree preprint](https://arxiv.org/abs/2505.22405) reports an
`O(sqrt(T log min(n,t_max)))` unknown-times upper bound matching that logarithmic
barrier in the relevant regime.

Consequently, a universal hidden-stopping-time `O(sqrt(T))` claim is false.  A
Q-GapSelect proof would have to state a special structural promise and prove
that the published unknown-time lower-bound family violates that promise.
Simply renaming a stopping register “activity history” is insufficient.

Generic composition is also a mandatory boundary, not a baseline that can be
represented by one guessed scalar.  The relevant primary works are
[quantum subroutine composition](https://arxiv.org/abs/2209.14146),
[transducer-based time composition](https://arxiv.org/abs/2311.15873), and the
2026 preprint on [loop composition](https://arxiv.org/abs/2605.07518), which
explicitly recovers prior variable-time-search behaviour by accounting for
loops.

## 4. What is actually proved locally

The audit contains two modest, valid lemmas.

1. **Contract identity.** Canonical serialization and SHA-256 fingerprints
   ensure that a comparison cannot silently change the oracle, output
   relation, information access, or hard family.  A public and hidden
   partition deliberately receive different `family_id` values even though
   their numeric angle profiles share one `base_family_id`.
2. **Constant-component maximum.** For positive `x_1,...,x_s`,
   `sum_j x_j <= s max_j x_j`.  Thus lower bounds for every component would
   match a constant-size additive upper bound *only if all component bounds
   hold on the same family and interface*.  The lemma cannot combine lower
   bounds proved on different witness families.

Neither lemma is a new quantum algorithm result.

## 5. Lower-bound boundary

Independent homogeneous search blocks have a strong direct-product barrier;
see [Klauck--Spalek--de Wolf](https://arxiv.org/abs/quant-ph/0402123), the
search-specific adversary proof of
[Ambainis](https://arxiv.org/abs/quant-ph/0508200), and the general
[Lee--Roland theorem](https://arxiv.org/abs/1104.4468).  The code exposes only
the largest homogeneous-bucket floor directly licensed by those results.

The desired weighted expression

```text
sum_r sqrt(A_r M_r) c_r
```

is **not** labelled as proved.  A valid lower bound must construct a weighted
adversary matrix for the canonical angle-rotation oracle, preserve the exact
Top-k relation, and show that small angular separations contribute the claimed
`c_r` factor.  It must also handle an algorithm that queries blocks in
superposition.  A counteralgorithm below the weighted sum or a loss in the
angle-oracle reduction falsifies this target.

## 6. Closure result and next admissible direction

The dependency ledger ends in:

```text
CF-PUBLIC   = falsified (known composition matches)
UB-HIDDEN   = proof obligation (history discovery/cleanup missing)
LB-WEIGHTED = proof obligation (weighted canonical adversary missing)
CLOSE       = blocked
```

Therefore the current family cannot support a CCF-A quantum-advantage claim.
The next admissible research move is not to tune experiments for a larger
ratio.  It is to define a *special hidden-history promise* that simultaneously:

1. does not reveal a static block partition;
2. excludes the published general unknown-time lower-bound witness only for a
   mathematically stated reason;
3. admits a compiled coherent discovery-and-cleanup transducer; and
4. supports a weighted adversary lower bound against all algorithms.

If the promise makes the history efficiently computable from public indices,
known composition returns and the novelty gate fails.  If it does not, the
discovery cost must appear in the upper bound.  That is the current theorem
frontier.
