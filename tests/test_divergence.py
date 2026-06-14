"""Divergence/regularizer losses: shapes, finiteness, gradient flow.
Needs torch (skipped where unavailable)."""
import pytest

torch = pytest.importorskip("torch")

from caad.losses.divergence import (anchor_kl_loss, fkl_gate_mask,  # noqa: E402
                                    gated_divergence_loss, teacher_entropy,
                                    visual_l2_loss)

B, L, V = 2, 12, 32


def _logits(requires_grad=False):
    return torch.randn(B, L, V, requires_grad=requires_grad)


def _mask():
    m = torch.zeros(B, L)
    m[:, 4:] = 1            # completion tokens
    return m


def test_teacher_entropy_shape_and_nonneg():
    H = teacher_entropy(_logits())
    assert H.shape == (B, L)
    assert (H >= -1e-5).all()


def test_gate_mask_subset_of_valid():
    H = teacher_entropy(_logits())
    valid = _mask()
    gate = fkl_gate_mask(H, valid, q=0.5, min_tokens=4)
    assert gate.shape == (B, L)
    assert (gate.bool() & ~valid.bool()).sum() == 0   # never gates a non-completion token


def test_gated_divergence_finite_and_has_grad():
    s = _logits(requires_grad=True)
    t = _logits()
    mask = _mask()
    gate = fkl_gate_mask(teacher_entropy(t), mask, q=0.5, min_tokens=4)
    loss, stats = gated_divergence_loss(s, t, mask, gate)
    assert torch.isfinite(loss)
    loss.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert "distill/loss" in stats


def test_anchor_kl_nonneg_finite():
    s = _logits(requires_grad=True)
    i = _logits()
    loss = anchor_kl_loss(s, i, _mask())
    assert torch.isfinite(loss) and loss.item() >= -1e-4


def test_visual_l2_zero_when_identical():
    x = torch.randn(10, 8)
    assert visual_l2_loss(x, x.clone()).item() < 1e-6
