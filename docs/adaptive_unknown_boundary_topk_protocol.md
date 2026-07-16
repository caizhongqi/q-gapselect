# Adaptive unknown-boundary Top-k diagnostic protocol

## Scope

`adaptive_unknown_boundary_topk.py` is the fail-closed S3 bridge from the S2
single-precision circuit to an adaptive experiment campaign.  It accepts an
opaque canonical oracle, `k`, consecutive public precision bounds, one public
diagnostic-error target, and hard resource caps.  It cannot accept an answer,
gap, boundary, family label, precision schedule, activity history, or QRAM
list.

Each executed precision is the full S2 coherent
compute-rank-copy-uncompute-inverse-QPE circuit.  The controller then reads its
exact statevector diagnostics.  This read is simulation introspection, not a
free physical observable, so the controller and its stop bits are classical.
The result schema makes this limitation machine-checkable.

## Consecutive precision policy

For public bounds `m_min` and `m_max`, the implementation constructs

\[
m=m_{\min},m_{\min}+1,\ldots,m_{\max}
\]

internally.  Before level `m`, it checks the exact S2 statevector dimension and
the canonical query cost

\[
Q_m=2n(2^{m+1}-1).
\]

The level is never executed if it would exceed either hard cap.  Every
executed level reconciles the observed forward, inverse, controlled-forward,
and controlled-inverse counts against this formula; QRAM queries remain zero.

## Diagnostic stop rule

Let `p_dom` be the exact-state probability of the dominant durable mask,
`p_strict` the discrete strict-boundary probability, and `p_garbage` the
executed transient nonzero probability after S2 replay.  The simulator latches
a diagnostic stop only when:

1. the dominant mask has exactly `k` bits;
2. all canonical query counts reconcile;
3. the cleanup collision and purity identities pass numerical tolerance; and
4. `max(1-p_dom, 1-p_strict, p_garbage)` is at most the public target.

This maximum is an exact simulator mass relative to the dominant mask.  It is
not a Top-k correctness or confidence bound.  A selected set is returned only
under the explicit `DIAGNOSTIC_MASK` label, and `certificate.issued` is always
false.  If no level passes, or a hard cap blocks the next level, output is
`INCONCLUSIVE`.

## S3 panel

The preregistered configuration includes:

- an exact-grid first-level success;
- an off-grid instance stopping at three phase qubits;
- a harder off-grid instance stopping at four phase qubits;
- an exact tie that remains inconclusive; and
- a 40-query cap that blocks the required third level.

The off-grid outputs are useful diagnostics for choosing the next coherent
construction, but they are not evidence of a coherent adaptive controller.

## Claim boundary

The panel supports exact query accounting, resource-cap behavior, durable
level output diagnostics, and fail-closed orchestration.  It does not support
a single variable-time unitary, coherent stopping-history cleanup, an
observable stop estimator, generic off-grid correctness, a new upper bound,
composition separation, a matching lower bound, quantum advantage, or a
CCF-A claim.
