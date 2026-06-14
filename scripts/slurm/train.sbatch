#!/usr/bin/env bash
#SBATCH --job-name=caad-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x-%j.out   # cluster scheduler log only;
#SBATCH --error=outputs/slurm/%x-%j.err    # per-run logs live in the recipe's output_dir/logs/
#
# Cluster header/template ONLY. Reads CFG (+ optional ACC/OUT) from the
# environment and delegates to scripts/train.sh — no experiment logic lives here.
# output_dir is set in the recipe (point it at scratch).
#
#   sbatch --export=ALL,CFG=configs/train/caad_lora_qwen25vl7b.yaml scripts/slurm/train.sbatch
set -euo pipefail

mkdir -p outputs/slurm
# source your env here, e.g.:  source ~/.bashrc && conda activate caad
bash scripts/train.sh
