"""
caad/utils/logging.py
=====================
Output-dir layout and a thin metrics logger.

Every run owns one directory: the recipe's output_dir, holding a frozen config
copy, checkpoints, results, plots, and this run's logs — no global logs/ dump.
The run identity used in logs / as the W&B fallback name is that dir's basename.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

_LOG = logging.getLogger("caad")


def run_dir(output_dir) -> Path:
    """Create and return the run's output directory. ALL artifacts for the run
    live here: the frozen config.yaml plus checkpoints/ results/ plots/ logs/.
    output_dir comes straight from the recipe (output_dir:), TRL-style — point it
    wherever you want (e.g. a cluster scratch path)."""
    d = Path(output_dir)
    for sub in ("checkpoints", "results", "plots", "logs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def setup_logging(run_path, level=logging.INFO) -> logging.Logger:
    _LOG.setLevel(level)
    if not _LOG.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        _LOG.addHandler(sh)
        if run_path is not None:
            fh = logging.FileHandler(Path(run_path) / "logs" / "train.log")
            fh.setFormatter(fmt)
            _LOG.addHandler(fh)
    return _LOG


class MetricsLogger:
    """
    Appends step metrics to results/metrics.jsonl and, if configured, mirrors
    them to Weights & Biases. Keeps the on-disk record authoritative so viz/
    can plot a run without W&B access.
    """

    def __init__(self, run_path, wandb_cfg=None):
        self.path = Path(run_path) / "results" / "metrics.jsonl"
        self._fh = self.path.open("a")
        self._wandb = None
        if wandb_cfg and wandb_cfg.get("enabled"):
            import wandb
            self._wandb = wandb
            # explicit wandb.run_name wins; else fall back to the run-id (dir name)
            name = wandb_cfg.get("run_name") or Path(run_path).name
            wandb.init(project=wandb_cfg["project"], name=name,
                       dir=str(run_path), config=wandb_cfg.get("config"))

    def log(self, step: int, metrics: dict):
        row = {"step": step, **{k: _scalar(v) for k, v in metrics.items()}}
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        if self._wandb is not None:
            self._wandb.log(row, step=step)

    def close(self):
        self._fh.close()
        if self._wandb is not None:
            self._wandb.finish()


def _scalar(v):
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return v.item()
    except ImportError:
        pass
    return v
