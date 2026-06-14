# Vision Self-Distillation — CAAD

**CAAD** (Corruption-Aware Adaptive Distillation): a self-distillation method
that improves a video VLM's robustness to spatio-temporal corruptions. A student
answers from a *corrupted* view while an EMA teacher scores those same tokens
under the *clean* view; an adaptive per-token divergence + a visual-feature
alignment loss + an anchor KL pull the student toward the clean-view behavior.

See [`docs/design.md`](docs/design.md) for the method and the cross-input
alignment invariant.

## Layout
```
src/caad/            importable package (pip install -e .)
  trainers/          one class per algorithm; variants = config flags
    base.py          shared rollout->step->backward->EMA loop
    caad.py          the CAAD per-sample step
  losses/divergence.py   gated FKL/JSD, anchor KL, visual L2
  data/              corruption engine (pure fn) + manifest dataset
  models/model_utils.py  dual-adapter LoRA plumbing + visual features
  rollout.py         vLLM generation worker (per-cluster integration seam)
  eval/              run_task (atomic) -> orchestrate (fan-out); metrics ONCE
  viz/               plots read outputs/, write outputs/
  utils/             config loading, logging, seeding
configs/             EXPERIMENTS LIVE HERE — self-contained YAML recipes, one per
  train/             run (+ RUN_PLAN.md); eval/ + accelerate/ alongside
scripts/             thin parameterized launchers (+ slurm/ templates)
<output_dir>/        ALL artifacts (git-ignored): frozen config + ckpts + results + logs
docs/  paper/        design notes, manuscript
```

## Setup
On Clariden (GH200): start the uenv, then a `--system-site-packages` venv so the
uenv's PyTorch is inherited (don't pip-reinstall torch on aarch64):
```bash
uenv start pytorch/v2.9.1:v2 --view=default
python -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -e ".[viz,logging,dev]"   # or: make install  (ARM-clean base; torch from uenv)
make smoke                             # import + config resolution check
pytest -q                              # unit tests (corruption/divergence/metrics/config)
```
Scale-up extras (x86 / where they build): `make install-scale` adds `deepspeed`
(multi-GPU ZeRO) and `vllm` (the `vllm` rollout backend). On ARM, start with the
`hf` rollout backend and single-GPU `accelerate` config.

## Smoke test the pipeline (no real data, 1 GPU)
Verify data → loop → loss → checkpoint end-to-end before a real run. Uses tiny
synthetic clips and a **mock** rollout (no vLLM):
```bash
# 1) fabricate ~8 tiny synthetic clips + a manifest
python data/prepare_corruptions.py synthetic --n 8 \
  --clean-dir data/smoke/clean --out-dir data/smoke/corrupted \
  --manifest data/smoke/manifest.jsonl --frames 8 --hw 128
# 2) run 2 optimizer steps on one GPU
CFG=configs/train/smoke.yaml ACC=configs/accelerate/single_gpu.yaml bash scripts/train.sh
```

## Prepare real data
Render corrupted copies of your clean clips and write the training manifest
(`data/dataset.py` documents the row schema):
```bash
python data/prepare_corruptions.py real \
  --source data/raw/clean.jsonl --out-dir data/corrupted \
  --manifest data/train_manifest.jsonl --styles fog,night,rain --severity 3
```

## Train (one command per experiment)
```bash
# everything (incl. output_dir) reads from the recipe YAML
CFG=configs/train/caad_lora_qwen25vl7b.yaml bash scripts/train.sh
# or
make train CFG=configs/train/caad_lora_qwen25vl7b.yaml
```
A new experiment is a **new self-contained YAML under `configs/train/`** (copy a
recipe, change a few keys — incl. its `output_dir` — everything for a run lives in
one file), not a new `.py` or submit script. CLI overrides work too (give the
variant its own `output_dir`):
```bash
CFG=configs/train/caad_lora_qwen25vl7b.yaml OUT=outputs/caad_lora_7b_l2hi \
  bash scripts/train.sh caad.lambda_l2=1.0
```

## Evaluate
```bash
RUN=outputs/caad_lora_qwen25vl7b GPUS=0,1,2,3 bash scripts/eval.sh   # RUN = the recipe's output_dir
```
`orchestrate` fans out one `run_task` per (checkpoint, task) cell and writes
`results/summary.json`; metrics (pass@k, maj@k, avg-pass@1) come from the single
`eval/metrics.py`.

## Conventions
1. **One atomic unit, fanned out** — `eval/run_task.py` does exactly one
   (checkpoint, task); `orchestrate.py` is the only thing that loops.
2. **Variants are config, not forked files** — one trainer; behavior is gated on
   `cfg`. `_old`/`_logps` files go to git history, never to disk.
3. **Explicit output_dir per recipe** — each recipe sets `output_dir:`; that one
   directory holds the frozen `config.yaml` + checkpoints + results + logs. Point
   it at scratch on a cluster. (Reproducibility = frozen config + git SHA.)
4. **Metrics defined once** — `eval/metrics.py`, never re-implemented per script.
5. **Self-contained recipes** — each `configs/train/config_*.yaml` is complete on
   its own (no inheritance/merging); a new run = copy one and edit it.
6. **Strict git boundary** — `src/ configs/ scripts/ docs/ paper/` are versioned;
   `outputs/ logs/ wandb/ data/` and weights are ignored. Reproducibility =
   frozen `config.yaml` + git SHA.

## Rollout backends (`rollout.backend`)
- `mock` — canned text, no model. Pipeline smoke tests / CI.
- `hf` — generate with the **live training model**. Correct on a single GPU
  (always-fresh weights, no sync needed); slower. Good for an initial 1-GPU run.
- `vllm` — dedicated GPU-0 worker (production, fast). **Seam:** the worker holds a
  separate weight copy, so `RolloutEngine._sync_vllm` must push the student
  weights for your vLLM version — until then it warns and rollouts are stale, so
  prefer `hf` for correct results while scaling up.

## Other integration seams
- `caad/eval/run_task.py::_default_sampler` — checkpoint-backed sampler for eval.
- Multi-rank vLLM (1 shared worker, N training clients) is not wired — single
  process / `hf` backend is the runnable path today.
