"""
caad/divergence.py
==================
Per-token adaptive divergence for CAAD.

Gate (revised proposal):
  - teacher entropy H(T_t) computed PER ROLLOUT over valid completion tokens
  - bottom `fkl_quantile` (default 10%) entropy tokens -> forward KL  D(T||S)
  - all remaining tokens                              -> JSD_0.5(S, T)
  - rollouts shorter than `min_tokens` fall back to the batch-level quantile
    (a per-rollout 10th percentile over e.g. 12 tokens is just the 2nd lowest
    token -- too granular to be meaningful)

All quantities are computed from full-vocab log-probs in CHUNKS along the
sequence axis: with a ~150k vocab, materializing teacher+student+mixture
distributions for a whole rollout at once is the main OOM risk.

Shapes: logits (B, L, V); masks (B, L) with 1 on completion tokens.
Teacher logits must be produced under no_grad upstream; gradients flow only
through student logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def teacher_entropy(teacher_logits: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """H(T_t) per position. (B, L, V) -> (B, L), float32."""
    B, L, V = teacher_logits.shape
    out = torch.empty(B, L, dtype=torch.float32, device=teacher_logits.device)
    for s in range(0, L, chunk):
        lt = teacher_logits[:, s:s + chunk].float().log_softmax(-1)
        out[:, s:s + chunk] = -(lt.exp() * lt).sum(-1)
    return out


@torch.no_grad()
def fkl_gate_mask(entropy: torch.Tensor,
                  valid: torch.Tensor,
                  q: float = 0.10,
                  min_tokens: int = 24) -> torch.Tensor:
    """
    Boolean mask (B, L): True where forward-KL should be used.

    Per-rollout quantile threshold; short rollouts use the batch quantile so
    the gate isn't dominated by order statistics of tiny samples.
    """
    B, L = entropy.shape
    valid = valid.bool()
    masked = entropy.masked_fill(~valid, float("inf"))

    # batch-level fallback threshold over all valid tokens
    all_valid = entropy[valid]
    batch_thr = (torch.quantile(all_valid.float(), q)
                 if all_valid.numel() > 0 else torch.tensor(0.0, device=entropy.device))

    gate = torch.zeros_like(valid)
    for b in range(B):
        n = int(valid[b].sum())
        if n == 0:
            continue
        if n >= min_tokens:
            thr = torch.quantile(entropy[b][valid[b]].float(), q)
        else:
            thr = batch_thr
        gate[b] = (masked[b] <= thr) & valid[b]
    return gate


def gated_divergence_loss(student_logits: torch.Tensor,
                          teacher_logits: torch.Tensor,
                          completion_mask: torch.Tensor,
                          fkl_mask: torch.Tensor,
                          jsd_beta: float = 0.5,
                          chunk: int = 64):
    """
    Sum over tokens of  [ FKL(T||S) on gated tokens, JSD_beta(S,T) elsewhere ],
    averaged over valid tokens. Gradients flow through student_logits only.

    Returns (loss, stats_dict).
    """
    B, L, V = student_logits.shape
    valid = completion_mask.bool()
    n_valid = valid.sum().clamp(min=1)

    total = student_logits.new_zeros(())
    fkl_sum = torch.zeros((), device=student_logits.device)
    jsd_sum = torch.zeros((), device=student_logits.device)

    for s in range(0, L, chunk):
        sl = slice(s, s + chunk)
        ls = student_logits[:, sl].float().log_softmax(-1)          # grad
        with torch.no_grad():
            lt = teacher_logits[:, sl].float().log_softmax(-1)       # no grad
        pt = lt.exp()

        v = valid[:, sl]
        g = fkl_mask[:, sl] & v          # forward-KL tokens
        j = v & ~g                       # JSD tokens

        if g.any():
            # D_KL(T || S) = sum_v p_T (log p_T - log p_S)
            fkl_tok = (pt * (lt - ls)).sum(-1)
            fkl_chunk = (fkl_tok * g).sum()
            total = total + fkl_chunk
            fkl_sum += fkl_chunk.detach()

        if j.any():
            # JSD_beta(S,T) = beta KL(S||M) + (1-beta) KL(T||M),
            # M = beta S + (1-beta) T, computed stably in log space.
            log_m = torch.logsumexp(
                torch.stack([ls + torch.log(torch.tensor(jsd_beta)),
                             lt + torch.log(torch.tensor(1 - jsd_beta))], 0), dim=0)
            ps = ls.exp()
            kl_s_m = (ps * (ls - log_m)).sum(-1)
            kl_t_m = (pt * (lt - log_m)).sum(-1)
            jsd_tok = jsd_beta * kl_s_m + (1 - jsd_beta) * kl_t_m
            jsd_chunk = (jsd_tok * j).sum()
            total = total + jsd_chunk
            jsd_sum += jsd_chunk.detach()

    loss = total / n_valid
    stats = {
        "distill/loss": loss.detach(),
        "distill/fkl_frac": (fkl_mask & valid).sum() / n_valid,
        "distill/fkl_mean": fkl_sum / (fkl_mask & valid).sum().clamp(min=1),
        "distill/jsd_mean": jsd_sum / (valid & ~fkl_mask).sum().clamp(min=1),
    }
    return loss, stats


def anchor_kl_loss(student_logits: torch.Tensor,
                   init_logits: torch.Tensor,
                   completion_mask: torch.Tensor,
                   chunk: int = 64) -> torch.Tensor:
    """
    KL( pi_theta || pi_init ) per token, averaged over valid tokens.
    init_logits produced under no_grad (adapter-disabled base / frozen copy).
    """
    B, L, V = student_logits.shape
    valid = completion_mask.bool()
    n_valid = valid.sum().clamp(min=1)
    total = student_logits.new_zeros(())
    for s in range(0, L, chunk):
        sl = slice(s, s + chunk)
        ls = student_logits[:, sl].float().log_softmax(-1)
        with torch.no_grad():
            li = init_logits[:, sl].float().log_softmax(-1)
        kl = (ls.exp() * (ls - li)).sum(-1)
        total = total + (kl * valid[:, sl]).sum()
    return total / n_valid


def visual_l2_loss(student_visual: torch.Tensor,
                   clean_visual: torch.Tensor) -> torch.Tensor:
    """
    L2 alignment on post-merger visual tokens.
      student_visual: (N, D) from CORRUPTED view, WITH grad
      clean_visual:   (N, D) from CLEAN view -- caller must .detach() it
                      (stop-grad on the target prevents the model satisfying
                      the loss by degrading the clean representation)
    Both sides unit-normalized so the loss measures direction, not magnitude.
    """
    assert student_visual.shape == clean_visual.shape, (
        f"visual token mismatch {student_visual.shape} vs {clean_visual.shape}; "
        "corruptions must preserve resolution/frame count")
    s = F.normalize(student_visual.float(), dim=-1)
    t = F.normalize(clean_visual.float(), dim=-1)
    return ((s - t) ** 2).sum(-1).mean()
