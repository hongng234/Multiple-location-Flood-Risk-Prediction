# Flood Risk Prediction — common tasks
# Windows: install make via `choco install make` / `scoop install make`, or run from Git Bash.

ifeq ($(OS),Windows_NT)
	PY := .venv/Scripts/python
else
	PY := .venv/bin/python
endif

.PHONY: help venv install run refresh synthetic predict test cov clean clean-outputs

help:            ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

venv:            ## Create the virtual environment
	python -m venv .venv

install:         ## Install all dependencies (runtime + tests)
	$(PY) -m pip install -r requirements.txt

run:             ## Train + benchmark (uses cached raw data if valid)
	$(PY) main.py

refresh:         ## Re-download data from Open-Meteo, then train + benchmark
	$(PY) main.py --refresh

predict:         ## Live next-day forecast for all locations (needs internet + trained model)
	$(PY) predict.py

test:            ## Run the offline pytest suite
	$(PY) -m pytest

cov:             ## Run tests with coverage report
	$(PY) -m pytest --cov=src --cov=main --cov=predict --cov-report=term-missing

clean:           ## Remove Python caches
	$(PY) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if '.venv' not in p.parts]; shutil.rmtree('.pytest_cache', ignore_errors=True)"

clean-outputs:   ## Delete generated figures, metrics and model
	$(PY) -c "import shutil; shutil.rmtree('outputs', ignore_errors=True)"
