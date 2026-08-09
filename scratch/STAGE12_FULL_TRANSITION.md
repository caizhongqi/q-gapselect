# Stage 12 — Full-Cache Coherent Transition Validation

A small but complete outer transition state was built using **real MLP seed-7 F-v1 codes** rather than synthetic codes.

Configuration:

- one-claw real-code universe: N=4 per semantic side;
- Johnson subset size: r=2;
- response width: m=32;
- state contains the A/B subsets, real cache-value register, per-universe-item canonical prefix lengths, marked flag, and remove/insert edge labels for both sides.

For each source state the transition uses a fixed-depth event clock. The branch's real DELTA event list consists of:

1. all new-only canonical cache components, each toggled by the corresponding XOR component oracle;
2. all old-only components, toggled once to erase them;
3. idle clock slots after the branch has stopped;
4. a final outer relabel from the source subset/edge to the target subset/reverse-edge, with the canonical target prefix lengths and marked flag.

All valid subset-edge basis states were exhaustively enumerated:

\[
\binom{4}{2}^2\,[2(4-2)]^2=576.
\]

Results:

- valid input states: 576;
- unique target states: 576;
- invalid targets: 0;
- transition bijective: true;
- W^2=I failures: 0;
- maximum real component-event count: 69;
- mean event count: 31.75;
- RMS event count: 42.743;
- p95 event count: 69;
- random complex-superposition input norm: 1.0;
- output norm: 0.9999999999999999;
- ||W^2 psi - psi||_2: 0.

The real event-count histogram over the 576 source basis states was:

- 8 calls: 144 states;
- 10 calls: 72 states;
- 11 calls: 144 states;
- 68 calls: 72 states;
- 69 calls: 144 states.

This verifies that the complete **outer cache/subset/edge state conversion** can be represented as a fixed-depth unitary permutation with idle padding while preserving the real DELTA oracle schedule.

Boundary: the event selector in this exhaustive simulator is evaluated as a classical reversible control function on basis states; this experiment does not yet provide a fault-tolerant decomposition of the selector, nor the complete MNRS phase-detection/search circuit. Those remain separate from the proven outer permutation.
