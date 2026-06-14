"""
caad/train.py
=============
Training entrypoint. ONE command, everything from the YAML:

    accelerate launch -m caad.train --config configs/train/caad_lora_qwen25vl7b.yaml

The script: resolves the config (self-contained recipe + CLI overrides), creates
the recipe's output_dir tree, freezes the exact config there, constructs the
model / rollout engine / data / trainer, and hands off to TrainLoop. A new
experiment is a new YAML, never an edit here.
"""

from __future__ import annotations

import argparse

from .trainers.base import TrainLoop
from .trainers.caad import CAADTrainer
from .utils.config import freeze_config, load_config
from .utils.logging import MetricsLogger, run_dir, setup_logging
from .utils.seeding import seed_everything

# method -> trainer class. Add a variant here, select it via the recipe's `method`.
TRAINERS = {"caad": CAADTrainer}


def parse_args():
    p = argparse.ArgumentParser(description="CAAD training")
    p.add_argument("--config", required=True, help="path to a self-contained train recipe (YAML)")
    p.add_argument("--output_dir", default=None,
                   help="override the recipe's output_dir (where this run is written)")
    p.add_argument("overrides", nargs="*", help="dotted overrides: caad.lambda_l2=0.5")
    return p.parse_args()


def _total_optimizer_steps(cfg, dataset_len):
    """Total optimizer steps: train.max_steps if set, else derived from
    num_train_epochs over the dataset."""
    t = cfg["train"]
    if t.get("max_steps", 0) and t["max_steps"] > 0:
        return t["max_steps"]
    accum = t.get("per_device_train_batch_size", 1) * t["gradient_accumulation_steps"]
    per_epoch = max(1, -(-dataset_len // accum))      # ceil(dataset_len / accum)
    return t.get("num_train_epochs", 1) * per_epoch


def _build_scheduler(optimizer, cfg, total_steps):
    """LR schedule from config, warmup_ratio of total_steps as warmup.
    Honors train.lr_scheduler_type."""
    from transformers import get_scheduler
    t = cfg["train"]
    warmup = int(t.get("warmup_ratio", 0.0) * total_steps)
    return get_scheduler(t.get("lr_scheduler_type", "cosine"),
                         optimizer=optimizer, num_warmup_steps=warmup,
                         num_training_steps=total_steps)


def build_run(args):
    cfg = load_config(args.config, overrides=args.overrides)
    # output location is the recipe's output_dir (override with --output_dir).
    output_dir = args.output_dir or cfg["output_dir"]
    path = run_dir(output_dir)                 # creates checkpoints/ results/ plots/ logs/
    run_id = path.name                         # run identity for logs / W&B fallback
    freeze_config(cfg, path)
    return cfg, run_id, path


def main():
    import logging
    args = parse_args()
    cfg, run_id, path = build_run(args)
    level = getattr(logging, str(cfg.get("log_level", "info")).upper(), logging.INFO)
    log = setup_logging(path, level=level)
    log.info("run_id=%s  ->  %s", run_id, path)
    seed_everything(cfg["seed"], deterministic=cfg.get("deterministic", False))

    # --- heavy deps imported lazily so --help / config resolution stay fast ---
    from accelerate import Accelerator
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    from .data.dataset import VideoManifestDataset
    from .models.model_utils import load_base_model
    from .rollout import RolloutEngine  # vLLM wrapper; see module docstring

    acc = Accelerator(gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"])
    model, processor = load_base_model(cfg)
    if cfg["model"].get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=cfg["model"].get(
                "gradient_checkpointing_kwargs", {"use_reentrant": False}))
    dataset = VideoManifestDataset(cfg["data"]["manifest"],
                                   video_root=cfg["data"].get("video_root"))
    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        collate_fn=lambda b: b[0])
    opt = AdamW((p for p in model.parameters() if p.requires_grad),
                lr=cfg["train"]["learning_rate"])
    sched = _build_scheduler(opt, cfg, _total_optimizer_steps(cfg, len(dataset)))
    model, opt, loader = acc.prepare(model, opt, loader)

    if cfg["method"] not in TRAINERS:
        raise SystemExit(f"unknown method '{cfg['method']}'; known: {sorted(TRAINERS)}")
    trainer = TRAINERS[cfg["method"]](model, processor, cfg, acc)
    rollout = RolloutEngine(cfg, processor, model=model)
    metrics = MetricsLogger(path, wandb_cfg=cfg.get("wandb"))

    loop = TrainLoop(trainer, rollout, loader, opt, acc, cfg, path,
                     scheduler=sched, metrics_logger=metrics)
    try:
        loop.run()
    finally:
        metrics.close()
    log.info("done: %s", run_id)


if __name__ == "__main__":
    main()
