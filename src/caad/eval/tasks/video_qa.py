"""
caad/eval/tasks/video_qa.py
===========================
Example benchmark: open-ended / multiple-choice video QA from a JSONL manifest.
A real benchmark adds one file like this (loader + scorer) and registers it.

Optionally evaluates on a CORRUPTED copy of each clip (the robustness axis CAAD
targets) by re-rendering from the recorded spec — keeping eval faithful to what
training optimized.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .registry import register


@register("video_qa")
class VideoQA:
    @staticmethod
    def load(cfg):
        rows = [json.loads(l) for l in Path(cfg["manifest"]).read_text().splitlines() if l.strip()]
        key = "corrupted_video" if cfg.get("use_corrupted") else "clean_video"
        return [{"id": r["video_id"], "question": r["question"],
                 "video": r[key], "gold": _norm(r["answer"])} for r in rows]

    @staticmethod
    def score(example, completion) -> bool:
        return _norm(completion) == example["gold"]


def _norm(s: str) -> str:
    """Lowercase, strip punctuation/whitespace — shared so load() and score()
    normalize identically (mismatched normalization silently tanks accuracy)."""
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()
