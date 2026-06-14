#!/usr/bin/env bash
# scripts/train_rl.sh — multi-GPU training launcher.
#
# Two modes:
#   vLLM server (default): GPU 0 runs the vLLM rollout server, GPUs 1..N-1 train.
#   NO_VLLM=true         : no server; train on ALL GPUs with the hf rollout backend
#                          (correct, no vLLM needed — use this until vLLM is validated).
#
#   # safe first run (no vLLM):
#   NO_VLLM=true NUM_GPUS=4 CFG=configs/train/caad_lora_qwen25vl7b_rl.yaml \
#     ACC=configs/accelerate/ddp.yaml bash scripts/train_rl.sh data.manifest=...
#
#   # full RL (vLLM server on GPU 0):
#   NUM_GPUS=4 CFG=configs/train/caad_lora_qwen25vl7b_rl.yaml \
#     ACC=configs/accelerate/zero2.yaml bash scripts/train_rl.sh data.manifest=...
set -euo pipefail

CFG="${CFG:?set CFG=configs/train/<rl recipe>.yaml}"
ACC="${ACC:-configs/accelerate/zero2.yaml}"
SERVER_GPU="${SERVER_GPU:-0}"
PORT="${PORT:-8000}"
NO_VLLM="${NO_VLLM:-false}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-0}}"

gpu_list() {  # args: total, exclude(-1 for none)
  python - "$1" "$2" <<'PY'
import sys
n, excl = int(sys.argv[1]), int(sys.argv[2])
print(",".join(str(g) for g in range(n) if g != excl) if n else "")
PY
}

OVERRIDE=""
if [ "$NO_VLLM" = true ]; then
  echo ">> NO_VLLM=true — hf rollout, training on ALL $NUM_GPUS GPUs (no server)"
  GPULIST="$(gpu_list "$NUM_GPUS" -1)"
  OVERRIDE="rollout.backend=hf"
else
  MODEL="$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['model']['name'])")"
  MAX_PIXELS="$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['model']['max_pixels'])")"
  echo ">> [1/3] launching vLLM server on GPU $SERVER_GPU (port $PORT)"
  mkdir -p outputs/_vllm_logs
  MODEL="$MODEL" GPU="$SERVER_GPU" PORT="$PORT" MAX_PIXELS="$MAX_PIXELS" \
    bash scripts/vllm_serve.sh > outputs/_vllm_logs/server.log 2>&1 &
  SERVER_PID=$!
  trap 'echo ">> stopping vLLM server ($SERVER_PID)"; kill $SERVER_PID 2>/dev/null || true' EXIT
  echo ">> [2/3] waiting for /health (tail outputs/_vllm_logs/server.log)"
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "   server healthy"; break; }
    kill -0 $SERVER_PID 2>/dev/null || { echo "!! server died — see outputs/_vllm_logs/server.log"; exit 1; }
    sleep 5
  done
  GPULIST="$(gpu_list "$NUM_GPUS" "$SERVER_GPU")"
fi

GPULIST="${GPULIST:-0}"
NTRAIN="$(awk -F, '{print NF}' <<< "$GPULIST")"
echo ">> training on GPUs [$GPULIST] (num_processes=$NTRAIN)  cfg=$CFG  acc=$ACC"
CUDA_VISIBLE_DEVICES="$GPULIST" \
  accelerate launch --config_file "$ACC" --num_processes "$NTRAIN" \
    -m caad.train --config "$CFG" $OVERRIDE "$@"
