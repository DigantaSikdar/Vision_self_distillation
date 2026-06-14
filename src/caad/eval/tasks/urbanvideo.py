"""
caad/eval/tasks/urbanvideo.py
=============================
UrbanVideo-Bench multiple-choice task. Manifest rows carry the question (with
options inline) and a single-letter answer (A-E). The scorer pulls the model's
chosen letter out of the completion and compares.

Set use_corrupted: true in the eval config to score on the corrupted view (the
robustness axis CAAD targets) instead of the clean clip.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .registry import register


@register("urbanvideo")
class UrbanVideo:
    @staticmethod
    def load(cfg):
        rows = [json.loads(l) for l in Path(cfg["manifest"]).read_text().splitlines() if l.strip()]
        key = "corrupted_video" if cfg.get("use_corrupted") else "clean_video"
        return [{"id": r["video_id"], "question": r["question"], "video": r[key],
                 "gold": str(r["answer"]).strip().upper()[:1]}
                for r in rows if r.get("answer")]

    @staticmethod
    def score(example, completion) -> bool:
        pred = extract_letter(completion)
        return bool(pred) and pred == example["gold"]


def extract_letter(text: str) -> str:
    """Best-effort MCQ letter extraction from a free-form completion."""
    # 1) explicit "answer is X" / "answer: X" / "the answer (X)"
    m = re.search(r"answer\s*(?:is|:)?\s*\(?\s*([A-E])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 2) a boxed answer \boxed{X}
    m = re.search(r"\\boxed\{\s*([A-E])\s*\}", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 3) last standalone capital letter A-E (final answers tend to come last)
    cands = re.findall(r"\b([A-E])\b", text)
    return cands[-1].upper() if cands else ""
