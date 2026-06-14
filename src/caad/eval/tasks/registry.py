"""
caad/eval/tasks/registry.py
===========================
name -> task module mapping. A task is any object exposing:

    load(cfg) -> list[example]            # each example: {"id", "question", "video", "gold"}
    score(example, completion) -> bool    # is this completion correct?

One file per benchmark under tasks/; register it here so eval configs can select
tasks by name. Loader + scorer live together so a benchmark is added in exactly
one place.
"""

from __future__ import annotations

_TASKS = {}


def register(name):
    def deco(obj):
        _TASKS[name] = obj
        return obj
    return deco


def get_task(name):
    if name not in _TASKS:
        raise KeyError(f"unknown task '{name}'; registered: {sorted(_TASKS)}")
    return _TASKS[name]


def available():
    return sorted(_TASKS)


# Import task modules so their @register decorators run.
from . import video_qa  # noqa: E402,F401
