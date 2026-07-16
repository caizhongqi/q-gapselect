# CCF-A mainline status after history certification

## Mainline

The algorithmic mainline remains quantum Top-k selection with heterogeneous
stopping histories. The eventual application is surrogate search and transfer
for LLM attack policies, but the current evidence tier is deliberately
algorithm-only: no LLM, commercial API, security, hardware, or application
claim is made here.

## Closed in this iteration

The previous candidate used a fresh fixed-precision all-arm verification pass.
On the clustered hard family, 472 of 500 maximum-cap outputs were exact but
only one was certified. The new history certificate:

- allocates selection risk by a summable schedule with total at most
  `delta/2`;
- deterministically replays every interval classification and quota-closure
  transition;
- checks zero compute--phase--uncompute residuals;
- issues no fresh verification query.

The complete 37,500-attempt ablation records the following maximum-cap result:

| Method | Certified exact | Mean queries |
|---|---:|---:|
| History-certificate candidate | 0.9887 | 297,543 |
| Strongest k-only reference | 0.9973 | 270,387 |

For the candidate, equal-gap and dyadic families reach 1.000 and the clustered
family reaches 0.966. This closes the certification-loss failure but does not
establish empirical superiority.

## Active CCF-A blockers

The evidence gate remains false for eight explicit reasons. The decisive
algorithmic blockers are now:

1. the analytic executor still charges serial per-arm IAE calls rather than an
   executed coherent-index stopping unitary;
2. the candidate remains slightly less accurate and more query-expensive than
   the strongest same-information reference;
3. the same-interface composition frontier is not separated;
4. no new variable-time upper bound or matching lower bound is proved;
5. state preparation, gates, depth, qubits, and cleanup are not closed at the
   scalable level.

## Unknown-boundary protocol and legal semantic baseline

The next campaign is frozen in
`docs/coherent_unknown_boundary_experiment_protocol.md`. It separates exact
circuit semantics, matched fixed-cap performance, resource scaling, published
composition instantiation, and the matching lower-bound proof into independent
gates. Supplied schedules are now stronger-information regression controls.

The first new legal semantic comparator is a measured-QPE exhaustive rank
baseline. It receives only the canonical oracle, `k`, and public precision and
shot limits. In the clean-provenance 26-record panel, all 22 strict-boundary
attempts return the exact complete membership mask and all four tie controls
fail closed. Query formulas and rank cleanup reconcile exactly, with no QRAM.
Its certified-exact rate is nevertheless zero: per-arm measurement/reset is
not end-to-end coherent and the modal finite-shot rank has no confidence
theorem. It is therefore a correctness baseline, not evidence for the proposed
advantage.

The mandatory composition frontier now explicitly includes MIQAE per-arm
sort, coherent estimation with a rounding-promise audit, approximate
`k`-minimum, repeated QBAI, optimal unknown-time search, loop/transducer
composition, Tunable VTAA, and generated-predicate all-marked extraction.

## Next theorem target

The next core object is a replay-preserving coherent frontier unitary. For each
arm `i`, let `t_i` be the first history level at which its membership becomes
fixed. The implementation must coherently apply only the scheduled prefix,
write the stop/output/history registers, copy the durable output, and uncompute
all transient work while preserving the classical replay invariant.

The target theorem is intentionally open: derive a charged variable-time cost
in terms of the stopping profile `(t_i)` that is strictly below every legal
same-interface composition baseline on an explicit family, then prove a
matching oracle lower bound. If the resulting bound is matched by coarse
partition plus best-arm identification, the direction is falsified rather
than promoted as a CCF-A contribution.
