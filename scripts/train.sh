#!/usr/bin/env bash
# scripts/train.sh — one parameterized launcher, NOT one file per run.
# Everything (incl. output_dir) comes from the recipe YAML; this only wires accelerate.
#
#   CFG=configs/train/caad_lora_qwen25vl7b.yaml bash scripts/train.sh
#   CFG=configs/train/caad_fullft_qwen25vl7b.yaml ACC=configs/accelerate/zero3.yaml bash scripts/train.sh
#
# Optional: OUT=/path overrides the recipe's output_dir.
# Extra args pass through as dotted overrides:
#   CFG=... bash scripts/train.sh caad.lambda_l2=1.0 train.max_steps=4000
set -euo pipefail

CFG="${CFG:?set CFG=configs/train/<exp>.yaml}"
ACC="${ACC:-configs/accelerate/zero2.yaml}"
OUT="${OUT:-}"                  # optional: override the recipe's output_dir

echo ">> train  cfg=$CFG  acc=$ACC  out=${OUT:-<recipe output_dir>}"
accelerate launch --config_file "$ACC" -m caad.train \
  --config "$CFG" ${OUT:+--output_dir "$OUT"} "$@"
