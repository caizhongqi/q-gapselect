.PHONY: install install-datasets install-uv test test-quantum lint check experiment quantum-core \
	scaling reference coherent quantum quantum-history unknown-boundary-grid \
	coherent-statevector-history charged-history variable-time-charged \
	stopping-transducer theorem-scaffold replay-coherent-frontier coherent-rank-baseline \
	coherent-unknown-boundary-topk hidden-frontier-fixtures strong-composition-registry \
	composition-frontier theorem-closure-audit lower-bound proof-ledger research-gap \
	attack attack-design \
	frozen-selector-benchmark frozen-quantum-reference frozen-anchor-calibration \
	ccfa-matched-benchmark download-uci uci-classifier-benchmark paper clean

install:
	python -m pip install -e '.[dev,plots,datasets]'

install-datasets:
	python -m pip install -e '.[datasets]'

install-uv:
	UV_CACHE_DIR=.uv-cache uv sync --frozen --all-extras

test:
	python -m pytest

test-quantum:
	python -m pytest \
		tests/test_activity_history_transducer.py \
		tests/test_charged_activity_history.py \
		tests/test_charged_activity_history_script.py \
		tests/test_variable_time_charged_history.py \
		tests/test_variable_time_charged_history_script.py \
		tests/test_stopping_time_transducer.py \
		tests/test_stopping_time_transducer_script.py \
		tests/test_stopping_time_theorem.py \
		tests/test_stopping_unitary_theorem_script.py \
		tests/test_replay_coherent_frontier.py \
		tests/test_replay_coherent_frontier_script.py \
		tests/test_coherent_rank_baseline.py \
		tests/test_coherent_rank_baseline_script.py \
		tests/test_coherent_unknown_boundary_topk.py \
		tests/test_coherent_unknown_boundary_topk_script.py \
		tests/test_hidden_frontier_fixtures.py \
		tests/test_hidden_frontier_fixture_manifest_script.py \
		tests/test_strong_composition_registry.py \
		tests/test_strong_composition_registry_script.py \
		tests/test_composition_frontier.py \
		tests/test_composition_frontier_script.py \
		tests/test_lower_bound_program.py \
		tests/test_lower_bound_program_script.py \
		tests/test_proof_ledger.py \
		tests/test_proof_ledger_script.py \
		tests/test_unknown_boundary_history.py \
		tests/test_composition_audit.py \
		tests/test_adaptive_phase.py \
		tests/test_direct_topk.py \
		tests/test_quantum_benchmarks_script.py \
		tests/test_unknown_boundary_grid_script.py \
		tests/test_research_gap_audit.py
	python -m pytest \
		tests/test_coherent_activity_history_core.py \
		tests/test_coherent_activity_history_statevector.py \
		tests/test_coherent_statevector_history_script.py \
		tests/test_matched_quantum_baselines.py \
		tests/test_ccfa_matched_benchmarking.py \
		tests/test_ccfa_matched_benchmarks_script.py \
		tests/test_ccfa_evidence_gate.py \
		tests/test_fixed_fixture_calibration.py \
		tests/test_theorem_closure_audit.py \
		tests/test_theorem_closure_audit_script.py

lint:
	python -m ruff check .

check: lint test

experiment: scaling coherent quantum-core

quantum-core: quantum unknown-boundary-grid charged-history variable-time-charged stopping-transducer theorem-scaffold replay-coherent-frontier coherent-rank-baseline coherent-unknown-boundary-topk hidden-frontier-fixtures strong-composition-registry composition-frontier lower-bound proof-ledger research-gap

scaling:
	python scripts/run_scaling.py --output artifacts/scaling.json

reference:
	python scripts/run_reference.py --output artifacts/reference_results.json

coherent:
	python scripts/run_coherent.py --output artifacts/coherent_results.json

coherent-statevector-history:
	python scripts/run_coherent_statevector_history.py \
		--config configs/coherent_statevector_history.json \
		--output artifacts/coherent_statevector_history.json

replay-coherent-frontier:
	python scripts/run_replay_coherent_frontier.py \
		--config configs/replay_coherent_frontier.json \
		--output artifacts/replay_coherent_frontier.json

coherent-rank-baseline:
	python scripts/run_coherent_rank_baseline.py \
		--config configs/coherent_rank_baseline.json \
		--output artifacts/coherent_rank_baseline.json

coherent-unknown-boundary-topk:
	python scripts/run_coherent_unknown_boundary_topk.py \
		--config configs/coherent_unknown_boundary_topk.json \
		--output artifacts/coherent_unknown_boundary_topk.json

hidden-frontier-fixtures:
	python scripts/run_hidden_frontier_fixture_manifest.py \
		--config configs/hidden_frontier_fixture_manifest.json \
		--output artifacts/hidden_frontier_fixture_manifest.json

strong-composition-registry:
	python scripts/run_strong_composition_registry.py \
		--config configs/strong_composition_registry.json \
		--output artifacts/strong_composition_registry.json

quantum:
	python scripts/run_quantum_benchmarks.py \
		--config configs/quantum_benchmarks.json \
		--output artifacts/quantum_benchmark_diagnostic.json

quantum-history:
	python scripts/run_quantum_benchmarks.py \
		--config configs/quantum_benchmarks.json \
		--output artifacts/unknown_boundary_history_diagnostic.json \
		--suite unknown_boundary_history

unknown-boundary-grid:
	python scripts/run_unknown_boundary_grid.py \
		--config configs/unknown_boundary_grid.json \
		--output artifacts/unknown_boundary_grid.json

charged-history:
	python scripts/run_charged_activity_history.py \
		--config configs/charged_activity_history.json \
		--output artifacts/charged_activity_history.json

variable-time-charged:
	python scripts/run_variable_time_charged_history.py \
		--config configs/variable_time_charged_history.json \
		--output artifacts/variable_time_charged_history.json

stopping-transducer:
	python scripts/run_stopping_time_transducer.py \
		--config configs/stopping_time_transducer.json \
		--output artifacts/stopping_time_transducer.json

theorem-scaffold:
	python scripts/run_stopping_unitary_theorem.py \
		--config configs/stopping_unitary_theorem.json \
		--output artifacts/stopping_unitary_theorem.json \
		--markdown docs/stopping_unitary_theorem.md

composition-frontier:
	python scripts/run_composition_frontier.py \
		--config configs/composition_frontier.json \
		--output artifacts/composition_frontier.json

theorem-closure-audit:
	python scripts/run_theorem_closure_audit.py \
		--config configs/theorem_closure_audit.json \
		--output artifacts/theorem_closure_audit.json

lower-bound:
	python scripts/run_lower_bound_program.py \
		--config configs/lower_bound_program.json \
		--output artifacts/lower_bound_program.json \
		--markdown docs/lower_bound_program.md

proof-ledger:
	python scripts/run_proof_ledger.py \
		--stopping-artifact artifacts/stopping_unitary_theorem.json \
		--composition-artifact artifacts/composition_frontier.json \
		--lower-bound-artifact artifacts/lower_bound_program.json \
		--output artifacts/proof_ledger.json \
		--markdown docs/proof_ledger.md

research-gap:
	python scripts/run_research_gap_audit.py \
		--quantum-artifact artifacts/quantum_benchmark_diagnostic.json \
		--grid-artifact artifacts/unknown_boundary_grid.json \
		--charged-artifact artifacts/charged_activity_history.json \
		--variable-time-artifact artifacts/variable_time_charged_history.json \
		--stopping-artifact artifacts/stopping_time_transducer.json \
		--theorem-artifact artifacts/stopping_unitary_theorem.json \
		--composition-artifact artifacts/composition_frontier.json \
		--lower-bound-artifact artifacts/lower_bound_program.json \
		--proof-ledger-artifact artifacts/proof_ledger.json \
		--output artifacts/research_gap_audit.json \
		--markdown docs/research_gap_audit.md

attack:
	python scripts/run_attack_study.py --config configs/attack_study.json \
		--output artifacts/attack_study_results.json \
		--raw-output artifacts/attack_study_raw.jsonl

attack-design:
	python scripts/run_qgapattack_experiment_design.py \
		--config configs/qgapattack_experiments.json \
		--output artifacts/qgapattack_experiment_design.json \
		--markdown docs/qgapattack_experiment_design_audit.md \
		--strict-design

frozen-selector-benchmark:
	python scripts/run_frozen_selector_benchmarks.py \
		--config configs/frozen_selector_benchmarks.json \
		--output artifacts/frozen_selector_benchmark_diagnostic.json

frozen-quantum-reference:
	python scripts/run_frozen_quantum_reference_benchmarks.py \
		--config configs/frozen_quantum_reference_benchmarks.json \
		--output artifacts/frozen_quantum_reference_diagnostic.json

frozen-anchor-calibration:
	python scripts/run_frozen_anchor_calibration.py \
		--config configs/frozen_anchor_calibration.json \
		--mixture-artifact artifacts/frozen_quantum_reference_diagnostic.json \
		--output artifacts/frozen_anchor_calibration.json.gz

ccfa-matched-benchmark:
	python scripts/run_ccfa_matched_benchmarks.py \
		--config configs/ccfa_matched_benchmarks.json \
		--mixture-artifact artifacts/frozen_quantum_reference_diagnostic.json \
		--theory-artifact artifacts/theorem_closure_audit.json \
		--output artifacts/ccfa_matched_benchmark_diagnostic.json.gz

download-uci:
	python scripts/download_uci_benchmarks.py letter optdigits covertype \
		--output data/uci

uci-classifier-benchmark:
	python scripts/run_uci_classifier_benchmarks.py \
		--config configs/uci_classifier_benchmarks.json \
		--data-root data/uci \
		--theory-artifact artifacts/theorem_closure_audit.json \
		--output artifacts/uci_classifier_benchmark_diagnostic.json.gz

paper:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex

clean:
	latexmk -C -cd paper/main.tex
