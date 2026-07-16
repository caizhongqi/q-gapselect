# Selection-history certificate soundness

## Scope

This note closes the finite-state certificate obligation for the executable
activity-history reference. It does not prove coherent-index execution,
variable-time composition, a new quantum upper bound, or a matching lower
bound.

Assume a strict Top-k boundary, so that the k-th and (k+1)-st means differ.
At level `l`, every active arm `i` receives a confidence interval
`I[l,i] = [L[l,i], U[l,i]]`. Let `E` be the event that every interval generated
by the selection history contains its true arm mean.

## Risk ledger

The per-call selection risk is

\[
\delta_{l,i}=\frac{3\delta}{\pi^2 n(l+1)^2}.
\]

Consequently,

\[
\sum_{l\ge 0}\sum_{i=1}^{n}\delta_{l,i}
=\frac{3\delta}{\pi^2}\sum_{l\ge0}\frac{1}{(l+1)^2}
=\frac{\delta}{2}.
\]

Thus `Pr(E) >= 1-delta/2` by a union bound, without requiring independence
between levels or arms.

## Replay lemma

**Lemma.** Conditional on `E`, every arm accepted by `_classify` belongs to the
remaining Top-q active set, every rejected arm lies outside it, and each
deterministic quota-closure transition preserves those statements. Therefore,
if replay ends with exactly `k` selected arms and no unresolved arm, the
selected set is the unique Top-k set.

**Proof.** For an accepted arm `i`, fewer than `q` other active intervals have
upper endpoint at least `L[l,i]`. Under `E`, fewer than `q` other active means
can be at least `mu_i`; hence `i` is in the remaining Top-q set. For a rejected
arm `i`, at least `q` other active intervals have lower endpoint strictly above
`U[l,i]`. Under `E`, those `q` means are strictly greater than `mu_i`, so `i`
cannot be in the remaining Top-q set.

Induct on levels. Previously accepted and rejected arms are safe by the
induction hypothesis. When accepted arms fill the global Top-k quota, all
remaining arms are safely rejected. When the number of not-yet-rejected arms
equals the remaining quota, every remaining arm is safely accepted. These are
exactly the two quota-closure rules replayed by the verifier. A complete replay
therefore returns the unique Top-k set. QED.

## Executable obligations

The implementation issues a history certificate only when all of the following
hold:

1. the output contains exactly `k` selected arms and no unresolved arm;
2. replay recomputes every confidence decision and quota closure and matches
   every recorded birth set;
3. every compute--phase--uncompute layer has zero predicate and phase residual;
4. the accumulated selection-risk upper bound is at most `delta/2`.

The certificate performs no fresh oracle query. Its empirical success must
still be reported as `certified AND exact`; the lemma is a probabilistic
soundness statement, not permission to relabel an incorrect output as correct.

## Remaining CCF-A blockers

The lemma removes redundant all-arm verification from the candidate but does
not address the central quantum claims. The remaining core obligations are:

- a genuine coherent-index stopping unitary implementing the same replayable
  transitions;
- an information- and query-matched composition frontier;
- a variable-time upper bound that beats that frontier on an explicit family;
- a matching lower bound under the same oracle interface;
- gate, depth, qubit, state-preparation, and cleanup accounting.
