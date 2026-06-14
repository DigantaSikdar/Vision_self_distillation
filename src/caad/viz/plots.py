"""
caad/viz/plots.py
=================
Plotting + table generation. Reads a run's on-disk records (results/metrics.jsonl,
results/summary.json) and writes figures into the SAME run's plots/ dir. Reads
from outputs/, writes to outputs/ — never couples to live training state, so any
finished run can be re-plotted offline.

    python -m caad.viz.plots --run <output_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(run_dir):
    path = Path(run_dir) / "results" / "metrics.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def plot_curves(run_dir, keys=("loss/total", "loss/visual_l2", "loss/anchor_kl")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_metrics(run_dir)
    steps = [r["step"] for r in rows]
    out_dir = Path(run_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    for k in keys:
        ys = [(r["step"], r[k]) for r in rows if k in r]
        if ys:
            ax.plot([s for s, _ in ys], [y for _, y in ys], label=k)
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend()
    ax.set_title(Path(run_dir).name)
    out = out_dir / "loss_curves.png"
    fig.tight_layout(); fig.savefig(out, dpi=120)
    return out


def main():
    p = argparse.ArgumentParser(description="plot a run's curves")
    p.add_argument("--run", required=True)
    args = p.parse_args()
    out = plot_curves(args.run)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
