.PHONY: help install test train demo clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:
	pip install -e .
	pip install -e ".[dev]"

data:
	python scripts/download_data.py

train:
	python scripts/train_pipeline.py

train-cv:
	python scripts/train_pipeline.py --cross-validate

demo:
	python scripts/inference_demo.py

test:
	pytest tests/ -v --tb=short

notebook:
	jupyter notebook notebooks/

docker-build:
	docker build -f deploy/Dockerfile -t predictive-maintenance:latest .

docker-run:
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/models:/app/models predictive-maintenance:latest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf experiments/mlruns
