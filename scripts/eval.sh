#!/usr/bin/env bash
# scripts/eval.sh — fan out a run's checkpoints over an eval suite.
#
#   RUN=outputs/qwen25vl7b_caad_klcov05_20260614 GPUS=0,1,2,3 bash scripts/eval.sh
set -euo pipefail

RUN="${RUN:?set RUN=<output_dir> (the recipe's output_dir)}"
SUITE="${SUITE:-configs/eval/video_suite.yaml}"
GPUS="${GPUS:-0}"

echo ">> eval  run=$RUN  suite=$SUITE  gpus=$GPUS"
python -m caad.eval.orchestrate --suite "$SUITE" --run "$RUN" --gpus "$GPUS"
python -m caad.viz.plots --run "$RUN" || true
