"""
caad/data/dataset.py
====================
Manifest-backed dataset for CAAD training.

A manifest is a JSONL file, one sample per row:

    {"video_id": "...", "question": "...",
     "clean_video": "data/clean/abc.mp4",
     "corrupted_video": "data/corrupted/abc__fog__s3.mp4",
     "corruption": {"style": "fog", "severity": 3, "seed": 12345,
                    "temporal_shuffle": false}}

The trainer only reads ``question`` / ``clean_video`` / ``corrupted_video`` (it
hands the paths straight to qwen-vl-utils). The ``corruption`` block records the
exact CorruptionSpec so a corrupted clip can be re-rendered bit-for-bit from the
clean source — see ``render_corrupted`` and scripts/ data prep.

Corruptions are rendered to disk ahead of training (not on the fly) because the
cross-input alignment invariant requires the corrupted clip to decode to the
same frame count / resolution as the clean one; pre-rendering makes that
checkable once, offline.
"""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset

from .corruption import CorruptionSpec, apply_corruption, make_spec


class VideoManifestDataset(Dataset):
    """Yields raw manifest rows; collation/tokenization happens in the trainer."""

    def __init__(self, manifest_path, *, video_root=None):
        self.rows = [json.loads(l) for l in Path(manifest_path).read_text().splitlines() if l.strip()]
        self.video_root = Path(video_root) if video_root else None

    def _abspath(self, p):
        p = Path(p)
        return str(self.video_root / p) if self.video_root and not p.is_absolute() else str(p)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = dict(self.rows[i])
        r["clean_video"] = self._abspath(r["clean_video"])
        r["corrupted_video"] = self._abspath(r["corrupted_video"])
        return r


def render_corrupted(frames, *, video_id, style, severity, global_seed,
                     temporal_shuffle=False):
    """
    Apply a deterministic corruption to clean frames and return
    (corrupted_frames, spec). Persist ``spec.to_json()`` in the manifest so the
    clip can be reproduced from the clean source on any machine.
    """
    spec = make_spec(video_id, style, severity, global_seed,
                     temporal_shuffle=temporal_shuffle)
    return apply_corruption(frames, spec), spec


def spec_from_manifest_row(row) -> CorruptionSpec:
    """Rebuild the CorruptionSpec recorded in a manifest row."""
    c = row["corruption"]
    return CorruptionSpec(
        video_id=row["video_id"], style=c["style"], severity=int(c["severity"]),
        seed=int(c["seed"]), temporal_shuffle=bool(c.get("temporal_shuffle", False)),
        extra=c.get("extra", {}),
    )
