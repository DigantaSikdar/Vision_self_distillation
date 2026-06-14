"""Corruption engine: determinism + the shape-preservation invariant.
Needs OpenCV (skipped where cv2 is unavailable)."""
import numpy as np
import pytest

pytest.importorskip("cv2")

from caad.data.corruption import (CORRUPTION_STYLES, apply_corruption,  # noqa: E402
                                  derive_seed, make_spec)


def _clip(seed=0, T=6, H=64, W=64):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (T, H, W, 3), dtype=np.uint8)


@pytest.mark.parametrize("style", CORRUPTION_STYLES)
def test_shape_preserved(style):
    frames = _clip()
    spec = make_spec("vid0", style, 3, global_seed=0)
    out = apply_corruption(frames, spec)
    assert out.shape == frames.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("style", CORRUPTION_STYLES)
def test_bit_identical_for_same_spec(style):
    frames = _clip()
    spec = make_spec("vid0", style, 3, global_seed=0)
    a = apply_corruption(frames, spec)
    b = apply_corruption(frames, spec)
    assert np.array_equal(a, b), f"{style} not deterministic"


def test_seed_depends_on_identity_not_order():
    # derive_seed must be stable across shuffle order / workers
    s1 = derive_seed("vidA", "fog", 3, 42)
    s2 = derive_seed("vidA", "fog", 3, 42)
    s3 = derive_seed("vidB", "fog", 3, 42)
    assert s1 == s2 and s1 != s3


def test_different_severity_changes_output():
    frames = _clip()
    lo = apply_corruption(frames, make_spec("v", "fog", 1, 0))
    hi = apply_corruption(frames, make_spec("v", "fog", 5, 0))
    assert not np.array_equal(lo, hi)
