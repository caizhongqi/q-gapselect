# Q-GapAttack experiment-design audit

- Design: `qgapattack-ccfa-preregistered-matrix-v1`
- Frozen on: `2026-07-15`
- Claim status: `preregistered_experiment_design_no_empirical_superiority_claim`
- Design valid: `true`
- Empirically ready: `false`
- CCF-A claimable: `false`

## Baselines

| ID | Track | Stage | Family | Primary | Status |
|---|---|---|---|---:|---|
| qgapselect | quantum_core | complete_topk_selection | proposed | true | implemented_scaffold_theorem_open |
| classical_clucb_topk | quantum_core | complete_topk_selection | classical_fixed_confidence | true | planned |
| classical_successive_rejects_topk | quantum_core | complete_topk_selection | classical_fixed_confidence | true | planned |
| q_uniform_qae_sort | quantum_core | complete_topk_selection | independent_quantum | true | partially_implemented |
| q_adaptive_independent_qae_sort | quantum_core | complete_topk_selection | independent_quantum | true | partially_implemented |
| q_repeat_qbai | quantum_core | complete_topk_selection | quantum_composition | true | analytic_proxy_only |
| q_coarse_boundary_qbai | quantum_core | complete_topk_selection | quantum_composition | true | analytic_proxy_only |
| q_known_time_vts_ae | quantum_core | complete_topk_selection | quantum_composition | true | planned |
| q_unknown_time_vts_ae | quantum_core | complete_topk_selection | quantum_composition | true | planned |
| q_subroutine_composition | quantum_core | complete_topk_selection | quantum_composition | true | manual_theorem_instantiation_required |
| q_all_marked_generated_predicate | quantum_core | complete_topk_selection | quantum_composition | true | planned |
| oracle_durr_hoyer_minimum | quantum_core | oracle_aided_lower_envelope | input_model_diagnostic | false | planned_analytic |
| oracle_kminima | quantum_core | oracle_aided_lower_envelope | input_model_diagnostic | false | planned_analytic |
| oracle_all_marked | quantum_core | oracle_aided_lower_envelope | input_model_diagnostic | false | planned_analytic |
| diag_tunable_vtaa | quantum_core | theory_diagnostic | input_model_diagnostic | false | manual_theorem_instantiation_required |
| invalid_free_history_qram | quantum_core | negative_control | invalid_negative_control | false | implemented_negative_control |
| qgapattack | attack_application | end_to_end | proposed | true | pipeline_scaffold_only |
| attack_clean_control | attack_application | control | clean_control | true | implemented_in_replay_schema |
| attack_random_selector | attack_application | source_selection | random_selector | true | planned |
| attack_best_of_n | attack_application | end_to_end | random_selector | true | planned |
| attack_exhaustive_rate_selector | attack_application | source_selection | exhaustive_selector | true | partially_implemented |
| attack_successive_halving | attack_application | source_selection | classical_adaptive_selector | true | planned |
| attack_clucb_topk | attack_application | source_selection | classical_adaptive_selector | true | planned |
| attack_cost_aware_racing | attack_application | source_selection | classical_adaptive_selector | true | planned |
| attack_xoxo_gcgs_selector | attack_application | source_selection | classical_adaptive_selector | true | external_reproduction_required |
| attack_independent_qae_sort | attack_application | source_selection | independent_quantum_selector | true | planned_emulation |
| attack_qgapselect_selector | attack_application | source_selection | proposed | true | adapter_only |
| attack_insec | attack_application | end_to_end | code_security_attack | true | external_reproduction_required |
| attack_xoxo | attack_application | end_to_end | code_security_attack | true | external_reproduction_required |
| attack_codelmsec | attack_application | end_to_end | code_security_attack | true | external_reproduction_required |
| attack_deceptprompt | attack_application | end_to_end | code_security_attack | true | external_reproduction_required |
| attack_tpia | attack_application | extended_threat_model | code_security_attack | true | external_reproduction_required |
| attack_hackode | attack_application | extended_threat_model | code_security_attack | true | external_reproduction_required |
| attack_gcg | attack_application | end_to_end | general_llm_attack | true | external_reproduction_required |
| attack_autodan | attack_application | end_to_end | general_llm_attack | true | external_reproduction_required |
| attack_pair | attack_application | end_to_end | general_llm_attack | true | external_reproduction_required |
| attack_tap | attack_application | end_to_end | general_llm_attack | true | external_reproduction_required |
| attack_victim_oracle_rank | attack_application | diagnostic | oracle_upper_bound | false | planned_diagnostic |

## Experiment panels

| ID | Track | Baselines | Benchmarks | Repetitions | Status |
|---|---|---:|---:|---:|---|
| Q1-correctness-calibration | quantum_core | 5 | 2 | 500 | planned_not_executed |
| Q2-composition-frontier | quantum_core | 9 | 1 | 500 | planned_not_executed |
| Q3-oracle-lower-envelopes | quantum_core | 7 | 1 | 500 | planned_not_executed |
| Q4-history-resource-ablation | quantum_core | 6 | 2 | 500 | planned_not_executed |
| A1-frozen-pool-selector | attack_application | 8 | 3 | 10 | planned_not_executed |
| A2-code-security-end-to-end | attack_application | 8 | 5 | 10 | planned_not_executed |
| A3-general-attack-stress-test | attack_application | 6 | 2 | 10 | planned_not_executed |
| A4-transfer-generalization | attack_application | 10 | 4 | 10 | planned_not_executed |
| A5-budget-efficiency | attack_application | 11 | 2 | 10 | planned_not_executed |
| A6-ablation-resources | attack_application | 7 | 2 | 10 | planned_not_executed |
| A7-extended-context-threats | attack_application | 5 | 3 | 10 | planned_not_executed |

## Open blockers

- At least one primary baseline is not implemented or frozen for reproduction.
- Experiment panels are preregistered but not all have immutable executed artifacts.
- Execution requirement remains open: all_primary_baselines_executable.
- Execution requirement remains open: dataset_manifests_frozen.
- Execution requirement remains open: independent_reproduction_complete.
- Execution requirement remains open: model_revisions_frozen.
- Execution requirement remains open: raw_records_persisted.
- Execution requirement remains open: validator_revisions_frozen.
- The Layer-P reversible LLM reward sampler and its cleanup/resource theorem are open.
- The Q-GapSelect upper bound, composition separation, and matching lower bound are open.
- Closed victim APIs can only provide classical transfer evidence, not coherent queries.
