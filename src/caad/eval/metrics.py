"""
caad/eval/metrics.py
====================
The single source of truth for scoring metrics. Defined ONCE here and imported
everywhere — never re-implemented per task or per script (that drift is exactly
what causes pass@{N} vs pass@k mismatches across a codebase).

Convention: each scorer takes a list of per-sample correctness records and
returns a float. ``samples`` is a list of dicts with at least ``correct`` (a
list[bool] over the k sampled completions for that example).
"""

from __future__ import annotations

from math import comb


def pass_at_k(samples, k: int) -> float:
    """
    Unbiased pass@k (Codex/HumanEval estimator) averaged over examples.
    Each example must report n = total samples and c = number correct.
    """
    vals = []
    for s in samples:
        n = len(s["correct"])
        c = sum(s["correct"])
        if n - c < k:
            vals.append(1.0)
        else:
            vals.append(1.0 - comb(n - c, k) / comb(n, k))
    return _mean(vals)


def maj_at_k(samples, k: int) -> float:
    """Majority-vote accuracy over the first k completions (needs a `pred` list
    of normalized answers and a `gold` field per example)."""
    vals = []
    for s in samples:
        preds = s["pred"][:k]
        if not preds:
            vals.append(0.0)
            continue
        winner = max(set(preds), key=preds.count)
        vals.append(float(winner == s["gold"]))
    return _mean(vals)


def avg_pass_at_1(samples) -> float:
    """Mean per-completion accuracy (a.k.a. average pass@1 over all samples)."""
    vals = [sum(s["correct"]) / len(s["correct"]) for s in samples if s["correct"]]
    return _mean(vals)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0
