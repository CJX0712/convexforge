# ConvexForge Makefile — one-click reproducible workflow.
# Determinism: pin BLAS/LAPACK threads to 1 before any NumPy import.

PY := python
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export NUMEXPR_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1

.PHONY: help install dev lint format test selftest demo ci clean

help:
	@echo "ConvexForge targets:"
	@echo "  install   install package"
	@echo "  dev       install dev deps (pytest, ruff)"
	@echo "  lint      ruff check + format --check"
	@echo "  test      run unit tests"
	@echo "  selftest  run the 10 invariant gates"
	@echo "  demo      run benchmark -> benchmark.json"
	@echo "  ci        full local CI closure"
	@echo "  clean     remove caches"

install:
	pip install -e .

dev:
	pip install -r requirements.txt

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

test:
	pytest -q

selftest:
	$(PY) -m convexforge.cli selftest

demo:
	$(PY) -m convexforge.cli demo --out benchmark.json

ci: lint selftest test demo
	@echo "CI closure OK"

clean:
	rm -rf .pytest_cache .ruff_cache .coverage
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
