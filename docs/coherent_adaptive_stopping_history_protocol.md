# Tiny true-coherent stopping-history protocol

## Circuit implemented

`coherent_adaptive_stopping_history.py` is a separate true-coherent reference
kernel for exactly two arms and `k=1`.  It does not accept a precision
schedule.  The public policy is fixed in code:

- level 0 uses two phase qubits per arm;
- level 1 uses three phase qubits per arm; and
- a branch stops when the folded discrete phase ranks differ by at least two
  bins.

One 19-qubit exact statevector contains all of the following registers:

| Register | Dimension | Role |
|---|---:|---|
| shared phase pack | 64 | QPE estimates for both arms |
| compiled indices | 4 | ordinary arm basis labels |
| rewards | 4 | canonical reward qubits |
| stopping history | 4 | two coherent stop bits |
| scratch mask | 4 | first accepted direct membership mask |
| durable mask | 4 | output surviving replay |
| rank/stop work | 4 | winner bit and stop bit |
| query control | 2 | coherent active-and-phase conjunction |

The forward transducer applies the first level, coherently negates the first
history bit to obtain the level-1 active flag, and controls every level-1
reward preparation and Grover-oracle invocation on that flag.  The stop
relation latches the history bit and scratch membership mask.  After both
levels, the circuit XOR-copies scratch to the durable mask and applies the two
level kernels in reverse order.  This is one unitary; the history is not a
classical controller.

## Query ledger

One forward history computation has fixed level costs 28 and 60 canonical
queries.  Complete durable-copy replay executes both level kernels twice.  The
worst-case circuit ledger is therefore:

| Kind | Count |
|---|---:|
| forward | 4 |
| inverse | 4 |
| controlled forward | 84 |
| controlled inverse | 84 |
| total coherent | 176 |
| QRAM | 0 |

Controlled calls count even when the active branch has zero amplitude.  The
executed query total is always 176.

These level totals are not accepted from constants alone.  Every oracle call
is tagged by phase level and arm.  The runtime artifact sums
`QuerySnapshot.by_tag` and reconciles full replay as 56 queries for level 0 and
120 for level 1.  Because each level kernel is invoked exactly twice, the
runtime-derived one-way costs must be 28 and 60, including the per-query-type
split.  A mismatch closes the budget-valid gate.

For history probabilities `p0` (stop at level 0), `p1` (stop at level 1), and
`pu` (unresolved), the artifact also records

\[
T_{\mathrm{RMS}}
=\sqrt{p_0 28^2+(p_1+p_u)88^2}.
\]

This is only a target for a future variable-time theorem.  It is not an
executed saving and does not replace the 176-query ledger.  A valid theorem
still needs a scalable stopping-unitary construction, cleanup/resource bound,
and compatible variable-time amplification or estimation argument.

## Cleanup and fail-closed rule

After reverse replay, every phase, index, reward, history, scratch, rank-work,
and query-control register is audited against zero.  The durable output is
excluded.  Independently, the implementation checks

\[
P_{\mathrm{garbage}}=1-\sum_y p_y^2
\]

and the corresponding output-purity identity.  A mask is exposed only when
cleanup passes, the history is resolved with probability one to numerical
tolerance, and the durable mask is deterministic with one selected bit.

The exact-grid first-stop and second-stop fixtures clean and output mask `01`.
An arm-1 fixture separately checks durable mask `10`.
The exact tie cleans but remains unresolved.  The generic off-grid fixture
creates mixed stopping histories and durable-output entanglement, so it returns
`INCONCLUSIVE`.

The inactive level-1 kernel is an identity only on its documented clean-work
subspace: phase, index, reward, scratch, rank-stop, and query-control work must
start at zero.  The formal panel executes a clean inactive basis witness and a
negative control with nonzero rank-stop work.  The clean witness has negligible
identity residual; the dirty control changes history/scratch and has a nonzero
residual.  This prevents the active flag from being misreported as a global
identity theorem.  The audit is still a basis witness, not a proof for the
whole clean subspace.

## Claim boundary

This kernel establishes tiny exact-state semantics for a coherent history,
active-controlled later level, direct output copy, full replay, and exact
worst-case query accounting.  It does not establish generic finite-QPE
correctness, an adaptive confidence certificate, branch-RMS query savings, a
scalable heterogeneous variable-time algorithm, a new upper bound, a matching
lower bound, hardware performance, quantum advantage, or CCF-A readiness.
