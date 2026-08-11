# Q-COLLIDE practical utility / decision-impact protocol

This document defines the paper's practical-utility table. The purpose is not to list possible applications; every row must correspond to a real operational decision and a prospective measurable endpoint.

# Table 17 — Practical Utility and Decision Impact

| Practical use | Neural system | Real operational decision | Conventional baseline | Q-COLLIDE output used | Prospective validation | Primary utility metric | Success gate |
|---|---|---|---|---|---|---|---|
| **Model selection before deployment** | vision / text / audio / graph | choose between models with matched task quality | task score, calibration, robustness score | `K_C/N`, basin density, persistence | select lower-collision model without using held-out failure set, then evaluate both on held-out collision/failure candidates | held-out independent failure capacity; selection regret | lower-`K_C` choice must show lower held-out failure capacity at matched task score |
| **Guard selection for an LLM stack** | prompt-injection guard + downstream LM | choose which guard to deploy at a fixed benign-acceptance operating point | benign acceptance, injection acceptance, AUROC/F1 | guard-specific `K_C`, nonzero-basin count, `K_C@Q` | freeze guard choice on calibration fixtures; evaluate on disjoint prompt sources / fixture seeds | held-out injection collision capacity and missed-mode count | collision-informed guard choice must reduce held-out `K_C` without materially degrading benign acceptance |
| **Red-team budget prioritization** | vision / text / Guard / multimodal | decide which candidate pairs/cases to inspect under a fixed query/evaluation budget | random search, nearest-control, uncertainty, strongest information-matched classical ranker | `K_C@Q`, basin-aware candidate ranking, capacity certificate | equal-query comparison on a hidden candidate pool | fraction of full `K_C` recovered at 0.25/0.5/1/2/5/10% query budget | Q-COLLIDE must recover more independent failure modes at the same information and query budget |
| **Targeted model repair** | trainable vision / text / policy models | decide which representation directions/modules to modify | random-null, variance-matched, generic regularization, full adversarial retraining | basin contribution, dangerous subspace, layer-localized `K_C` | apply targeted closure on training/calibration data and test on disjoint held-out failures | `-Delta K_C / |Delta task score|`, held-out recurrence rate | targeted repair must remove more held-out collision capacity per unit task-score loss than controls |
| **Where to repair a deep network** | ResNet / ViT / BERT / wav2vec / GNN | select layer/block for intervention instead of modifying the entire model | gradient norm, activation norm, attention magnitude | layerwise `K_C(l)`, basin formation, persistence onset | choose intervention layer from calibration topology; evaluate repair effectiveness downstream | repair efficiency per modified parameter / layer | collision-localized layer repair must match or exceed whole-model/random-layer repair with less task degradation or fewer modified parameters |
| **Release / acceptance gate for a model** | any atlas system | decide whether a candidate model is acceptable for release under a fixed control specification | task accuracy + conventional robustness checklist | capacity upper certificate, persistence, maximum overlap degree `Delta` | define threshold before evaluating a held-out release set | false-negative release rate for hidden collision failures | certificate-based gate must reject high-risk models that pass conventional quality gates, without excessive false rejection |
| **Distribution-shift early warning** | vision / audio / time series / graph | decide whether a deployed model has entered a representation regime requiring retraining or review | task error, confidence, entropy, calibration drift | shift in `K_C`, basin density, persistence fingerprint | monitor corruption/noise/seasonal/graph shift before severe task degradation | lead time or AUC for predicting later failure degradation | topology shift must predict future/held-out task or constraint failure beyond conventional confidence/calibration drift |
| **Cross-model transfer-risk forecasting** | models sharing the same input/task space | decide whether failures found on model A justify testing model B | architecture similarity, logit agreement, embedding similarity | collision-witness transfer matrix and basin overlap | predict transfer on held-out witnesses before evaluating target model | transfer-success prediction AUC / rank correlation | collision-topology overlap must predict transfer better than task-output similarity baselines |
| **Multimodal alignment diagnosis** | CLIP / BLIP-style systems | identify image-text regions where alignment representation looks similar but downstream behavior diverges | cosine similarity / retrieval rank only | multimodal `K_C`, collision basins, behavior divergence | evaluate held-out retrieval/caption/VQA mismatches in discovered basins | independent mismatch modes per query and transfer across prompts/images | collision-guided discovery must reveal independent alignment failures missed by similarity-only ranking |
| **Policy/control safety diagnosis** | PPO / recurrent policy / Decision Transformer | identify state/control regions with similar control representation but divergent action/return/constraint behavior | return, entropy, value error, state coverage | policy collision capacity and basin persistence | evaluate held-out trajectories / offline replay from discovered basins | independent unsafe/low-return behavior modes, constraint violation rate | high-`K_C` basins must concentrate future control failures and targeted repair must reduce them |
| **Benchmarking representation bottlenecks** | all domains | compare whether a control/monitoring bottleneck is sufficiently informative | probe accuracy / R2 alone | `K_C` conditioned on bottleneck rank, capacity-vs-rank spectrum | vary bottleneck rank under matched task performance | rank required to drive capacity below a preregistered threshold | Q-COLLIDE must expose cases where probe score is high yet independent collision capacity remains large |
| **Quantum search triage at large candidate scale** | real neural collision graphs | decide which collision-search problems are worth compiling into a quantum oracle/resource study | classical all-pairs / nearest-neighbour query accounting | real `N`, `K_C`, `Delta`, capacity certificate, analytic quantum query scale | feed real graph statistics into oracle/resource accounting; no hardware claim unless compiled | query reduction versus strongest same-oracle classical algorithm, plus logical resource crossover | only systems with a verified same-oracle query advantage and plausible resource crossover proceed to hardware-oriented claims |

## Why this table is different from an application list

The table is decision-oriented. A practical claim is admitted only if the topology changes an actual choice and that choice improves a held-out endpoint. For example, saying that `K_C` correlates with vulnerability is not enough for the model-selection row. The experiment must choose a model using `K_C` **before** inspecting the held-out failure set, then demonstrate lower held-out failure capacity than a task-score-only choice.

# Planned paper presentation

The final paper should report a compact version of Table 17 with one row per validated use, plus a status column:

- `Validated`: prospective held-out decision experiment completed;
- `Partial`: structural evidence exists, but prospective decision validation is incomplete;
- `Planned`: no empirical utility claim yet.

Expected initial status before the new utility experiments run:

| Practical use | Current evidence status |
|---|---|
| Guard selection | **Partial** — two real guards already show sharply different collision regimes, but prospective held-out guard-choice validation remains to be run |
| Red-team budget prioritization | **Partial** — query-budget discovery curves exist for the Guard study; broader cross-domain matched-baseline validation remains |
| Targeted repair | **Partial** — earlier targeted-vs-random closure evidence exists in visual experiments; the new atlas needs prospective held-out repair validation |
| Model selection | Planned |
| Layer-localized repair | Planned |
| Release gate | Planned |
| Distribution-shift early warning | Planned |
| Transfer-risk forecasting | Planned |
| Multimodal alignment diagnosis | Planned |
| Policy/control diagnosis | Planned |
| Bottleneck benchmarking | Partial — rank-spectrum evidence exists in vision, cross-domain replication remains |
| Quantum search triage | Partial — synthetic overlap scaling and real Guard query proxies exist; real-atlas graph resource accounting remains open |

# Core practical-utility metrics

The practical section should standardize the following decision metrics where applicable:

1. `Utility@Q = K_C@Q / K_C_full`: fraction of independent failure capacity found within budget.
2. `Repair efficiency = -Delta K_C / max(|Delta task score|, epsilon)`: collision capacity removed per unit task degradation.
3. `Selection regret`: held-out failure capacity of the model selected by a criterion minus the minimum held-out failure capacity among matched-quality candidates.
4. `Collision-risk lift`: failure probability inside discovered collision basins divided by the background failure probability.
5. `Transfer prediction AUC`: ability of source topology overlap to predict target-model transfer.
6. `Early-warning lead`: shift interval between a topology alarm and a conventional task-quality alarm.
7. `Certificate width`: uncertainty of the capacity interval used for release or quantum-triage decisions.

# Fail-closed rules

- A correlation between `K_C` and a failure metric is not by itself a validated practical use.
- A utility experiment must separate the data used to construct/estimate topology from the data used to judge the decision.
- Model/guard selection must be made before reading the held-out failure set.
- Query-efficiency claims require identical candidate information and query accounting across baselines.
- Repair claims require matched intervention budgets and report task-quality loss.
- A `K_C=0` analytic quantum proxy cell cannot support a quantum discovery-advantage claim.
- No practical quantum-runtime claim is allowed without explicit oracle/resource accounting.
