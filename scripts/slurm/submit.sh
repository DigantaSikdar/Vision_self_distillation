#!/bin/bash
# ============================================================
# SLURM submission — CAAD multi-GPU RL (dedicated vLLM server)
# Cluster: CSCS Clariden (GH200 120GB). 1 node, 4 GPUs:
#   GPU 0      : vLLM rollout server (scripts/vllm_serve.sh)
#   GPUs 1..N-1: accelerate + ZeRO training
#
# Usage:
#   sbatch scripts/slurm/submit.sh                      # LoRA + ZeRO-2 (default)
#   sbatch --export=ALL,MODE=full scripts/slurm/submit.sh   # full-FT + ZeRO-3
#   sbatch --export=ALL,CFG=...,DATA=... scripts/slurm/submit.sh
# ============================================================
#SBATCH --job-name=caad-rl
#SBATCH --account=a168
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --partition=normal
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --uenv=pytorch/v2.9.1:v2     # runs inside the uenv — no manual `uenv start`
#SBATCH --view=default
set -euo pipefail

# ─── paths / env ────────────────────────────────────────────
PROJECT_DIR="${HOME}/projects/Vision_self_distillation"
cd "$PROJECT_DIR"; mkdir -p logs
source "${PROJECT_DIR}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# ─── NCCL / GH200-Clariden tuning (from your unsupervised-rl submit.sh) ──
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5 NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=13 NCCL_NVLS_ENABLE=0
export PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=18 TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0                 # V0 safer for shared-GPU / LoRA

# ─── HF / W&B ───────────────────────────────────────────────
export HF_HOME="${HF_HOME:-$SCRATCH/hf}"
# HF_TOKEN: `huggingface-cli login` OR `export HF_TOKEN=...` before sbatch. NEVER hardcode.
export WANDB_PROJECT="${WANDB_PROJECT:-caad}"

# ─── MODE -> recipe + ZeRO config (config-driven LoRA <-> full-FT) ──
MODE="${MODE:-lora}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}"   # total GPUs (GPU 0 = server)
CFG="${CFG:-configs/train/caad_lora_qwen25vl7b_rl.yaml}"
DATA="${DATA:-data/videor1/train_manifest.jsonl}"
if [ "$MODE" = "full" ]; then
  ACC="${ACC:-configs/accelerate/zero3.yaml}"
  EXTRA="lora.enabled=false rollout.sync_mode=full ${EXTRA:-}"
else
  ACC="${ACC:-configs/accelerate/zero2.yaml}"
  EXTRA="${EXTRA:-}"
fi

echo "============================================================"
echo "  job=${SLURM_JOB_ID:-?}  node=$(hostname)  mode=$MODE  gpus=$NUM_GPUS"
echo "  cfg=$CFG  acc=$ACC  data=$DATA"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# ─── hand off to the 2-phase orchestrator (server on GPU0 + training) ──
CFG="$CFG" ACC="$ACC" NUM_GPUS="$NUM_GPUS" SERVER_GPU=0 \
  bash scripts/train_rl.sh "data.manifest=$DATA" $EXTRA
