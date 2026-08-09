# Stage 11 — Real DELTA-STABILIZE and LCP-Max Reversible Compiler

## E50 — Real-code DELTA-STABILIZE verification

The Stage-8 DELTA-STABILIZE rule was executed on real F-v1 MNIST response matrices rather than synthetic codes.

For every sampled Johnson-product edge, the simulator was allowed to:

1. read components already present in the old canonical cache;
2. call the component oracle only when the target canonical recursion required a missing bit;
3. erase every old-only cached bit by one call to the same XOR component oracle.

Across **3,100 real edges** at N/side = 128 and 256:

- forward canonical-cache failures: 0;
- component-call identity failures: 0;
- reverse-recovery failures: 0.

For every edge,

\[
Q_{\Delta}=|\mathcal C(S)\triangle\mathcal C(S')|,
\]

and applying the reverse-edge DELTA-STABILIZE restored the exact old cache with the same component-call count. This establishes the k=1 **basis-branch component-oracle schedule** on the sampled real response geometry. It does not by itself establish a low-depth fault-tolerant implementation.

Observed branch time T=4+Q_delta:

| N/side | edges | mean T | RMS T | p95 | max |
|---:|---:|---:|---:|---:|---:|
| 128 | 2100 | 39.78 | 44.23 | 77 | 147 |
| 256 | 1000 | 42.68 | 48.23 | 86.05 | 201 |

Fresh-query and erase-query contributions are balanced around 50/50 in the median edge.

### Power-of-two clock stress

Rounding each branch stopping time upward to the next power of two gives an RMS time ratio of approximately:

- 1.459 at N=128;
- 1.440 at N=256.

This is a **clock/gate-time stress factor, not an extra component-oracle-query factor**. If it is conservatively multiplied into the transition term anyway, the controlled-one-claw N=256 median advantage falls to about 1.08 and only 60% of cells remain above one. In the real natural multi-claw regime, a 1.44 transition multiplier still gives median advantages about 1.38 at N=128 and 1.82 at N=256.

Therefore the query-level k=1 claim survives exact real branch scheduling, while physical-time margin remains tight in the sparse one-claw regime.

## E51 — LCP-max characterization of the canonical certificate

For a cross-side pair define

\[
\lambda_{ij}=\operatorname{LCP}(F(a_i),F(b_j))\in\{0,\ldots,m\}.
\]

For every active A item define

\[
L_i=\max_j\lambda_{ij},
\]

and analogously for every B item. Then the number of response components in the canonical LPC for item i is

\[
\ell_i=\begin{cases}
 m,&L_i=m,\\
 L_i+1,&L_i<m.
\end{cases}
\]

The whole certificate therefore satisfies

\[
|\mathcal C(S)|=\sum_{i\in S_A}\ell_i+\sum_{j\in S_B}\ell_j,
\]

and the subset is marked iff

\[
\max_{i,j}\lambda_{ij}=m.
\]

This follows directly from the mixed-prefix definition: item i is queried at depth d iff at least one opposite-side item shares its first d response bits.

The identity was checked on **1,550 real Johnson states**:

- per-item prefix-length failures: 0;
- certificate-size identity failures: 0;
- marked-flag equivalence failures: 0.

Observed mean queried prefix length per active item:

- N=128, r=25: 7.25 bits;
- N=256, r=40: 7.76 bits.

Thus the real certificate is far from a 32-bit full load and can be represented as a fixed per-slot response cache plus a 6-bit prefix-length register, without a variable-address sparse trie.

## E52 — Clean reversible LCP primitive

A concrete reversible LCP_XOR microcircuit was constructed from X, CNOT, and Toffoli gates.

For m-bit strings a,b it uses:

- m XOR-difference scratch bits;
- m+1 live-prefix bits;
- m first-mismatch bits;
- ceil(log2(m+1)) output bits.

At each position k it computes a one-hot first-mismatch condition from the live prefix, XORs the binary first-mismatch index into the output register, encodes m when no mismatch exists, and then reverses all live/first/difference work.

Exhaustive truth-table simulation for m=1,...,8 covered all 2^(2m) input pairs at every width. Results:

- wrong LCP output: 0;
- dirty scratch states: 0.

A clean LCP_XOR uses exactly

\[
4m
\]

Toffolis. At m=32 the primitive resource count is:

- 128 Toffolis;
- 209 CNOTs;
- 130 X gates;
- 97 reusable scratch qubits;
- 6 output qubits.

## E53 — Streaming LCP/max compiler resource skeleton

A conservative uniform compiler can avoid the recursive trie logic:

1. allocate 32 fixed cache qubits and a 6-bit prefix-length register per active Johnson slot;
2. for each A slot, compute its r pairwise LCP values into a reusable r-by-6 work row, reversibly reduce them to a maximum, copy the resulting canonical prefix length, then uncompute the row;
3. repeat for every B slot;
4. query only cache positions required by the new lengths;
5. after all LCP/max work is uncomputed, XOR-erase old cache positions beyond the new canonical lengths.

The last two steps retain the exact DELTA-STABILIZE component-query identity. The LCP/max network contributes non-oracle gates.

Using a conservative 6-bit reversible compare/swap upper bound for the max reductions gives the following logical resource skeleton:

| N/side | r | naive prefix logic qubits | naive prefix Toffoli | streaming LCP qubits | streaming LCP Toffoli |
|---:|---:|---:|---:|---:|---:|
| 32 | 10 | 1068 | 213477 | 1057 | 57920 |
| 64 | 16 | 1792 | 550077 | 1649 | 148736 |
| 128 | 25 | 3034 | 1348197 | 2558 | 363800 |
| 256 | 40 | 5488 | 3460317 | 4097 | 932480 |
| 512 | 64 | 10394 | 8872701 | 6604 | 2388992 |
| 640 | 74 | 12896 | 11866341 | 7764 | 3194432 |

The streaming LCP representation reduces this conservative non-oracle Toffoli upper bound by about **73%** across the tested r range and reduces the logical-qubit skeleton increasingly as r grows.

These counts are not transpiled Clifford+T or fault-tolerant physical resources. They deliberately separate:

- component-oracle calls, which are the query-complexity resource;
- LCP/max/control logic, which is a reversible-gate overhead;
- physical implementation of the victim hard-label component oracle, which remains a separate unresolved cost.

## Main-line implication

Stage 11 materially strengthens the k=1 query-level mechanism:

- real DELTA-STABILIZE has now been checked forward and backward on thousands of real-code Johnson edges;
- canonical LPC has an exact LCP-max representation;
- the core LCP operation has an explicit clean reversible microcircuit;
- a fixed-slot streaming compiler avoids a variable-address sparse-trie requirement and gives an explicit logical resource envelope.

However, the sparse one-claw **physical-time** margin remains narrow after clock synchronization stress. The next gate is therefore not another query curve. It is a complete small-r coherent transition circuit with the subset update, LCP/max work registers, cache queries, marked flag, and clock register present simultaneously, followed by statevector/permutation validation and a gate-count comparison against this compiler skeleton.
