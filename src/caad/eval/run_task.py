"""
caad/eval/run_task.py
=====================
The ATOMIC eval unit: one (checkpoint, task) job. Generates k completions per
example, scores them with the task's scorer, computes metrics ONCE via
eval.metrics, and writes a single JSON to <output_dir>/results/.

This file never loops over checkpoints or tasks — orchestrate.py does that. Keep
the atom dumb and parallelizable.

    python -m caad.eval.run_task --checkpoint outputs/<run>/checkpoints/final \
        --task video_qa --task-config configs/eval/video_suite.yaml \
        --out outputs/<run>/results/video_qa.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import metrics as M
from .tasks.registry import get_task


def run(checkpoint, task_name, task_cfg, *, k=1, sampler=None):
    """Evaluate one checkpoint on one task. ``sampler(example, k) -> list[str]``
    yields k completions; defaults to a vLLM-backed sampler built from the
    checkpoint. Returns the results dict (also the on-disk schema)."""
    task = get_task(task_name)
    examples = task.load(task_cfg)
    sampler = sampler or _default_sampler(checkpoint, task_cfg)

    per_sample = []
    for ex in examples:
        comps = sampler(ex, k)
        correct = [task.score(ex, c) for c in comps]
        per_sample.append({"id": ex["id"], "correct": correct,
                           "pred": comps, "gold": ex.get("gold")})

    summary = {
        "avg_pass@1": M.avg_pass_at_1(per_sample),
        "pass@k": M.pass_at_k(per_sample, k),
        "maj@k": M.maj_at_k(per_sample, k) if all("gold" in s for s in per_sample) else None,
    }
    return {"checkpoint": str(checkpoint), "task": task_name, "k": k,
            "summary": summary, "samples": per_sample}


def _default_sampler(checkpoint, task_cfg):
    raise NotImplementedError(
        "pass a sampler(example, k) -> list[str], or wire a vLLM sampler here "
        "for the checkpoint (mirrors caad.rollout.RolloutEngine)")


def main():
    p = argparse.ArgumentParser(description="evaluate one (checkpoint, task)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--task-config", required=True)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import yaml
    task_cfg = yaml.safe_load(Path(args.task_config).read_text())
    result = run(args.checkpoint, args.task, task_cfg, k=args.k)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"{args.task}: {result['summary']}")


if __name__ == "__main__":
    main()
