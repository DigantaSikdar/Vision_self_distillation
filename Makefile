# CAAD — common entrypoints. Everything reads from a YAML; pass CFG/RUN/etc.
.DEFAULT_GOAL := help
SHELL := /bin/bash

CFG  ?= configs/train/caad_lora_qwen25vl7b.yaml
ACC  ?= configs/accelerate/zero2.yaml
SUITE ?= configs/eval/video_suite.yaml
GPUS ?= 0

.PHONY: help install train eval smoke plots lint clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## editable install (ARM-clean base + data/viz/logging/dev; torch from uenv)
	pip install -e ".[data,viz,logging,dev]"

install-scale:  ## add multi-GPU + vLLM extras (x86 / where they build)
	pip install -e ".[viz,logging,dev,deepspeed,rollout]"

train:  ## CFG=configs/train/<exp>.yaml make train   (output_dir set in the recipe)
	CFG=$(CFG) ACC=$(ACC) bash scripts/train.sh

eval:  ## RUN=<output_dir> GPUS=0,1 make eval
	RUN=$(RUN) SUITE=$(SUITE) GPUS=$(GPUS) bash scripts/eval.sh

plots:  ## RUN=<output_dir> make plots
	python -m caad.viz.plots --run $(RUN)

smoke:  ## fast import + config-resolution sanity check (no GPU)
	python -c "import caad; from caad.utils.config import load_config; \
	  c=load_config('$(CFG)'); print('OK', c['method'], '->', c['output_dir'])"
	python -c "from caad.eval.tasks.registry import available; print('tasks:', available())"

lint:  ## ruff check
	ruff check src

clean:  ## remove caches (NOT outputs/)
	find . -name __pycache__ -type d -prune -exec rm -rf {} + ; \
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
