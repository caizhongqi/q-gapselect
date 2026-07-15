.PHONY: install test lint check experiment scaling reference coherent quantum attack paper clean

install:
	python -m pip install -e '.[dev,plots]'

test:
	python -m pytest

lint:
	python -m ruff check .

check: lint test

experiment: scaling coherent quantum attack

scaling:
	python scripts/run_scaling.py --output artifacts/scaling.json

reference:
	python scripts/run_reference.py --output artifacts/reference_results.json

coherent:
	python scripts/run_coherent.py --output artifacts/coherent_results.json

quantum:
	python scripts/run_quantum_benchmarks.py \
		--config configs/quantum_benchmarks.json \
		--output artifacts/quantum_benchmark_diagnostic.json

attack:
	python scripts/run_attack_study.py --config configs/attack_study.json \
		--output artifacts/attack_study_results.json \
		--raw-output artifacts/attack_study_raw.jsonl

paper:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex

clean:
	latexmk -C -cd paper/main.tex
