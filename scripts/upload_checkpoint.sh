#!/usr/bin/env bash
# scripts/upload_checkpoint.sh — push a run's checkpoint(s) to the HF hub.
# Hub target comes from the run's frozen config (<output_dir>/config.yaml::hub),
# so the upload is reproducible from the run, not from ad-hoc flags.
#
#   RUN=outputs/caad_lora_qwen25vl7b bash scripts/upload_checkpoint.sh
#   RUN=... CKPT=step_2000 REPO=user/caad-7b bash scripts/upload_checkpoint.sh
set -euo pipefail

RUN="${RUN:?set RUN=<output_dir> (the recipe's output_dir)}"
CKPT="${CKPT:-final}"
SRC="$RUN/checkpoints/$CKPT"
[[ -d "$SRC" ]] || { echo "no checkpoint at $SRC"; exit 1; }

# repo id: explicit REPO wins, else hub.model_id from the frozen config
REPO="${REPO:-$(python -c "import sys,yaml; print(yaml.safe_load(open('$RUN/config.yaml')).get('hub',{}).get('model_id',''))")}"
[[ -n "$REPO" ]] || { echo "set REPO=user/name or hub.model_id in the recipe"; exit 1; }

echo ">> upload  $SRC  ->  hf.co/$REPO"
huggingface-cli upload "$REPO" "$SRC" "$CKPT" --repo-type model
