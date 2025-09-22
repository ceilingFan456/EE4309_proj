# EE4309 ObjDet Project Makefile

.PHONY: help setup train eval infer test clean samples

MODEL ?=
MODEL_NORMALIZED := $(shell printf '%s' "$(if $(MODEL),$(MODEL),vit)" | tr '[:upper:]' '[:lower:]')
OUTPUT ?= runs/$(MODEL_NORMALIZED)_voc07
EPOCHS ?= 1
BATCH_SIZE ?= 1
WEIGHTS ?= runs/$(MODEL_NORMALIZED)_voc07/best.pt
IMAGE_SET ?= test
SUBSET_SIZE ?= 500
INPUT ?= data/sample_images

help:  ## Show this help message
	@echo "EE4309 ObjDet Project"
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies and setup the project in editable mode
	conda env create -f env.yml || conda env update -f env.yml
	conda run -n ee4309-objdet pip install -r requirements.txt
	conda run -n ee4309-objdet pip install -e .
	@echo "Setup complete. Please run 'conda activate ee4309-objdet'"

	

samples:  ## Download sample images for inference testing
	bash scripts/download.sh

train:  ## Train the model (override MODEL=resnet50 etc.)
	bash scripts/train.sh "$(OUTPUT)" "$(EPOCHS)" "$(BATCH_SIZE)" "$(MODEL_NORMALIZED)"

eval:  ## Evaluate the model (override WEIGHTS=... MODEL=...)
	bash scripts/evaluate.sh "$(WEIGHTS)" "$(IMAGE_SET)" "$(SUBSET_SIZE)" "$(MODEL_NORMALIZED)"

infer:  ## Run inference (override INPUT=... MODEL=...; run 'make samples' first)
	bash scripts/infer.sh "$(WEIGHTS)" "$(INPUT)" "$(MODEL_NORMALIZED)"

test:  ## Run tests
	python3 -m pytest tests/ -v

clean:  ## Clean up runs directory
	rm -rf runs/
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	rm -rf data/

lint:  ## Run code linting
	python3 -m flake8 src/ --max-line-length=120 --ignore=E203,W503

format:  ## Format code
	python3 -m black src/ --line-length=120

check:  ## Run all checks
	make lint
	make test

submit:  ## Create a submission
	bash scripts/submit.sh
