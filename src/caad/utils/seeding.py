"""
caad/utils/seeding.py
=====================
One place to seed every RNG so runs are reproducible from config + git SHA.
Note: corruption determinism does NOT depend on this — corruptions seed from
sample identity (see data/corruption.py). This covers model init / dataloader
shuffling / dropout.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
