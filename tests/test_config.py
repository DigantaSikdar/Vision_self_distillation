"""Recipes are self-contained and expose the keys the code reads. Runs anywhere
(pyyaml only)."""
from pathlib import Path

import yaml

from caad.utils.config import load_config

RECIPES = sorted(Path("configs/train").glob("*.yaml"))
KNOWN_METHODS = {"caad"}
REQUIRED = [
    ("method",), ("seed",), ("output_dir",),
    ("model", "name"), ("model", "max_pixels"), ("model", "max_frames"),
    ("lora", "enabled"),
    ("caad", "fkl_quantile"), ("caad", "chunk_size"), ("caad", "anchor_beta"),
    ("train", "learning_rate"), ("train", "gradient_accumulation_steps"),
    ("train", "max_steps"), ("train", "ema_every"), ("train", "save_steps"),
    ("rollout", "temperature"), ("rollout", "max_completion_length"),
    ("data", "manifest"), ("data", "system_prompt"),
]


def _get(d, path):
    for k in path:
        assert k in d, f"missing {'.'.join(path)}"
        d = d[k]
    return d


def test_recipes_exist():
    assert RECIPES, "no recipes found under configs/train/"


def test_recipes_self_contained_and_complete():
    for f in RECIPES:
        raw = yaml.safe_load(f.read_text())
        assert "defaults" not in raw, f"{f.name} is not self-contained"
        cfg = load_config(f)
        for path in REQUIRED:
            _get(cfg, path)
        assert cfg["method"] in KNOWN_METHODS, f"{f.name}: unknown method {cfg['method']}"
        assert not str(cfg["output_dir"]).endswith("/")


def test_cli_override_applies():
    cfg = load_config(RECIPES[0], overrides=["caad.lambda_l2=0.123", "lora.enabled=false"])
    assert cfg["caad"]["lambda_l2"] == 0.123
    assert cfg["lora"]["enabled"] is False


def test_smoke_recipe_uses_mock_rollout():
    smoke = Path("configs/train/smoke.yaml")
    if smoke.exists():
        assert load_config(smoke)["rollout"]["backend"] == "mock"
