"""
caad/eval/orchestrate.py
========================
The ONLY thing that loops / parallelizes evaluation. Given a set of checkpoints
and a suite of tasks, it fans out one run_task job per (checkpoint, task) cell
across the available GPUs and collects the per-cell JSON into a summary table.

    python -m caad.eval.orchestrate --suite configs/eval/video_suite.yaml \
        --run <output_dir> --gpus 0,1,2,3

Each job is the atomic run_task unit; this module owns scheduling only, so the
scoring logic stays in one place (run_task + metrics).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml


def discover_checkpoints(run_dir):
    ckpts = sorted((Path(run_dir) / "checkpoints").glob("*"))
    return [c for c in ckpts if c.is_dir()]


def plan_jobs(checkpoints, tasks, results_dir):
    for ckpt in checkpoints:
        for task in tasks:
            out = Path(results_dir) / f"{ckpt.name}__{task['name']}.json"
            yield ckpt, task, out


def _run_cell(ckpt, task, out, gpu, task_config_path):
    cmd = ["python", "-m", "caad.eval.run_task",
           "--checkpoint", str(ckpt), "--task", task["name"],
           "--task-config", str(task_config_path),
           "--k", str(task.get("k", 1)), "--out", str(out)]
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    subprocess.run(cmd, check=True, env={**_os_environ(), **env})
    return out


def main():
    p = argparse.ArgumentParser(description="fan out eval across checkpoints/tasks")
    p.add_argument("--suite", required=True, help="eval suite YAML (lists tasks)")
    p.add_argument("--run", required=True, help="<output_dir> dir")
    p.add_argument("--gpus", default="0", help="comma-separated GPU ids")
    args = p.parse_args()

    suite = yaml.safe_load(Path(args.suite).read_text())
    gpus = [g for g in args.gpus.split(",") if g != ""]
    results_dir = Path(args.run) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ckpts = discover_checkpoints(args.run)
    jobs = list(plan_jobs(ckpts, suite["tasks"], results_dir))
    print(f"{len(jobs)} (checkpoint, task) cells over {len(gpus)} gpu(s)")

    def submit(i_job):
        i, (ckpt, task, out) = i_job
        return _run_cell(ckpt, task, out, gpus[i % len(gpus)], args.suite)

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        list(ex.map(submit, enumerate(jobs)))

    table = _collect(results_dir)
    (results_dir / "summary.json").write_text(json.dumps(table, indent=2))
    print(f"wrote {results_dir/'summary.json'}  ({len(table)} cells)")


def _collect(results_dir):
    rows = []
    for f in sorted(Path(results_dir).glob("*__*.json")):
        d = json.loads(f.read_text())
        rows.append({"checkpoint": d["checkpoint"], "task": d["task"], **d["summary"]})
    return rows


def _os_environ():
    import os
    return dict(os.environ)


if __name__ == "__main__":
    main()
