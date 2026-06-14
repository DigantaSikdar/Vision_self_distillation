# Training recipes — run plan

Each `config_*.yaml` here is a **self-contained** recipe (everything for that run
in one file; no inheritance). A new experiment = copy a recipe, change a few
keys. The launcher reads everything from the YAML:

```bash
CFG=configs/train/caad_lora_qwen25vl7b.yaml bash scripts/train.sh
```

Each recipe sets its own `output_dir:` — that directory holds the frozen
`config.yaml`, checkpoints, results, and logs for the run. Point it at scratch on
a cluster. Override for a one-off with `OUT=/path bash scripts/train.sh`.

| recipe                          | model            | mode     | notes                  | status  |
|---------------------------------|------------------|----------|------------------------|---------|
| caad_lora_qwen25vl7b.yaml       | Qwen2.5-VL-7B    | LoRA     | default run            | planned |
| caad_fullft_qwen25vl7b.yaml     | Qwen2.5-VL-7B    | full-FT  | needs ZeRO-3 + offload | planned |
| caad_lora_qwen25vl3b.yaml       | Qwen2.5-VL-3B    | LoRA     | fast iteration         | planned |

## Common sweeps (copy a recipe, change one key)
- `caad.anchor_beta`: 0.0 / 0.05 / 0.1   (anti-collapse strength)
- `caad.lambda_l2`: 0.0 / 0.5 / 1.0      (visual-alignment weight)
- `caad.fkl_quantile`: 0.05 / 0.10 / 0.20

Or override at launch without a new file (give the variant its own output_dir):
```bash
CFG=configs/train/caad_lora_qwen25vl7b.yaml OUT=outputs/caad_lora_7b_l2hi \
  bash scripts/train.sh caad.lambda_l2=1.0
```
