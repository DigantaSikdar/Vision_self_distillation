#!/usr/bin/env bash
# scripts/vllm_serve.sh — start the dedicated vLLM rollout server on ONE GPU.
# Reserves a GPU (default GPU 0) for generation; training runs on the others.
# Runtime LoRA updating is enabled so the trainer can hot-load student adapters.
#
#   MODEL=Qwen/Qwen2.5-VL-7B-Instruct GPU=0 PORT=8000 bash scripts/vllm_serve.sh
#
# Prereqs: vLLM installed (check for a CSCS `vllm` uenv first — the pip build on
# aarch64/GH200 is rough). Validate the multimodal/LoRA endpoints against your
# vLLM version; flags below are the common ones.
set -euo pipefail

MODEL="${MODEL:?set MODEL=<hf model id>}"
GPU="${GPU:-0}"
PORT="${PORT:-8000}"
MEM_UTIL="${MEM_UTIL:-0.85}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
MAX_PIXELS="${MAX_PIXELS:-200704}"

echo ">> vLLM server  model=$MODEL  gpu=$GPU  port=$PORT"
export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1   # lets /v1/load_lora_adapter work

exec vllm serve "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization "$MEM_UTIL" \
  --enable-lora --max-lora-rank "$MAX_LORA_RANK" \
  --limit-mm-per-prompt video=1 \
  --mm-processor-kwargs "{\"max_pixels\": $MAX_PIXELS}" \
  --trust-remote-code
