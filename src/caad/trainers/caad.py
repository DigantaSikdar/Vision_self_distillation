"""
caad/trainers/caad.py
=====================
The CAAD training step. Per sample, four forwards on one base model:

  rollout      vLLM (GPU 0)         y ~ pi_student( . | q, V_corrupted )
  student      train GPUs, grad     logits + post-merger visual feats, V_corrupted
  clean feats  train GPUs, no grad  post-merger visual feats, V_clean (L2 target)
  teacher      train GPUs, no grad  EMA-adapter logits on SAME tokens, V_clean
  anchor       train GPUs, no grad  adapter-disabled logits on SAME tokens, V_corr

Loss = gated_divergence + lambda_l2 * visual_L2 + anchor_beta * KL(pi||pi_init)

Cross-input alignment invariant: clean and corrupted prompts must tokenize to
the SAME length so that teacher logits at position t score the student's token
y_t under an identical prefix structure. Corruptions preserve resolution and
frame count, so visual token counts match; we assert it anyway.
"""

from __future__ import annotations

import torch

from ..losses.divergence import (anchor_kl_loss, fkl_gate_mask,
                                 gated_divergence_loss, teacher_entropy,
                                 visual_l2_loss)
from ..models.model_utils import (STUDENT, TEACHER, get_visual_features,
                                  sync_adapter, use_adapter, use_anchor)


def build_inputs(processor, system_prompt, question, video_path,
                 completion_text, device, max_pixels, max_frames):
    """
    Tokenize (prompt + completion) with the given video. Returns the batch and
    a completion mask that is 1 exactly on completion-token positions.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "video", "video": video_path,
             "max_pixels": max_pixels, "max_frames": max_frames},
            {"type": "text", "text": question},
        ]},
    ]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    # qwen-vl-utils handles video decode -> pixel_values_videos / grid_thw
    from qwen_vl_utils import process_vision_info
    _, video_inputs = process_vision_info(messages)

    prompt_batch = processor(text=[prompt_text], videos=video_inputs,
                             return_tensors="pt")
    completion_ids = processor.tokenizer(
        completion_text, return_tensors="pt", add_special_tokens=False
    ).input_ids

    input_ids = torch.cat([prompt_batch.input_ids, completion_ids], dim=1)
    attn = torch.ones_like(input_ids)
    completion_mask = torch.zeros_like(input_ids)
    completion_mask[:, prompt_batch.input_ids.shape[1]:] = 1

    batch = {
        "input_ids": input_ids.to(device),
        "attention_mask": attn.to(device),
        "pixel_values_videos": prompt_batch.pixel_values_videos.to(device),
        "video_grid_thw": prompt_batch.video_grid_thw.to(device),
    }
    return batch, completion_mask.to(device), prompt_batch.input_ids.shape[1]


def shift_for_next_token(logits, completion_mask):
    """
    Align logits with the tokens they predict: logits[:, t] predicts token t+1.
    Returns (pred_logits, target_mask) both of length L-1.
    """
    return logits[:, :-1], completion_mask[:, 1:]


class CAADTrainer:
    def __init__(self, model, processor, cfg, accelerator,
                 frozen_teacher=None, frozen_init=None):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.acc = accelerator
        self.lora = cfg["lora"]["enabled"]
        self.frozen_teacher = frozen_teacher   # full-FT mode only
        self.frozen_init = frozen_init
        c = cfg["caad"]
        self.q = c["fkl_quantile"]
        self.min_tok = c["min_tokens_for_quantile"]
        self.jsd_beta = c["jsd_beta"]
        self.lambda_l2 = c["lambda_l2"]
        self.anchor_beta = c["anchor_beta"]
        self.ema_decay = c["ema_decay"]
        self.chunk = c["chunk_size"]

    # ------------------------------------------------------------------ #
    def step(self, sample, completion_text):
        """
        One sample -> scalar loss (caller handles grad-accum / backward).
        sample: manifest row with question / clean_video / corrupted_video.
        """
        cfg_m = self.cfg["model"]
        dev = self.acc.device
        mk = lambda video: build_inputs(
            self.processor, self.cfg["data"]["system_prompt"],
            sample["question"], video, dev,
            cfg_m["max_pixels"], cfg_m["max_frames"])

        corr_batch, corr_cmask, corr_plen = mk(sample["corrupted_video"])
        clean_batch, _, clean_plen = mk(sample["clean_video"])

        # cross-input alignment invariant
        assert corr_plen == clean_plen and \
            corr_batch["input_ids"].shape == clean_batch["input_ids"].shape, (
            "clean/corrupted prompt lengths differ -- teacher/student token "
            "positions would misalign; check corruption preserves resolution")
        # teacher must score the student's actual rollout under the clean view
        clean_batch["input_ids"] = corr_batch["input_ids"].clone()
        clean_batch["attention_mask"] = corr_batch["attention_mask"].clone()

        # ---- 1) student forward: logits + visual feats, corrupted view ----
        with use_adapter(self.model, STUDENT) if self.lora else _null():
            s_out = self.model(**corr_batch)
            s_logits = s_out.logits
            s_visual = get_visual_features(
                self.model, corr_batch["pixel_values_videos"],
                corr_batch["video_grid_thw"])

        # ---- 2) clean visual feats (L2 target, stop-grad) ----
        with torch.no_grad():
            with use_adapter(self.model, STUDENT) if self.lora else _null():
                clean_visual = get_visual_features(
                    self.model, clean_batch["pixel_values_videos"],
                    clean_batch["video_grid_thw"]).detach()

        # ---- 3) teacher logits: EMA adapter, CLEAN view, same tokens ----
        with torch.no_grad():
            if self.lora:
                with use_adapter(self.model, TEACHER):
                    t_logits = self.model(**clean_batch).logits
            else:
                t_logits = self.frozen_teacher(**clean_batch).logits

        # ---- 4) anchor logits: pi_init, corrupted view, same tokens ----
        with torch.no_grad():
            with use_anchor(self.model, self.lora, self.frozen_init) as init_m:
                i_logits = init_m(**corr_batch).logits

        # ---- losses ----
        s_pred, cmask = shift_for_next_token(s_logits, corr_cmask)
        t_pred, _ = shift_for_next_token(t_logits, corr_cmask)
        i_pred, _ = shift_for_next_token(i_logits, corr_cmask)

        H = teacher_entropy(t_pred, chunk=self.chunk)
        gate = fkl_gate_mask(H, cmask, q=self.q, min_tokens=self.min_tok)

        distill, stats = gated_divergence_loss(
            s_pred, t_pred, cmask, gate,
            jsd_beta=self.jsd_beta, chunk=self.chunk)
        l2 = visual_l2_loss(s_visual, clean_visual)
        anchor = anchor_kl_loss(s_pred, i_pred, cmask, chunk=self.chunk)

        loss = distill + self.lambda_l2 * l2 + self.anchor_beta * anchor

        stats.update({
            "loss/total": loss.detach(),
            "loss/visual_l2": l2.detach(),
            "loss/anchor_kl": anchor.detach(),
            "entropy/teacher_mean": (H * cmask).sum() / cmask.sum().clamp(min=1),
            "rollout/len": cmask.sum().float(),
        })
        return loss, stats

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update_ema_teacher(self):
        if self.lora:
            sync_adapter(self.acc.unwrap_model(self.model),
                         src=STUDENT, dst=TEACHER, decay=self.ema_decay)
        else:
            student = self.acc.unwrap_model(self.model)
            for p_t, p_s in zip(self.frozen_teacher.parameters(),
                                student.parameters()):
                p_t.mul_(self.ema_decay).add_(p_s.detach(),
                                              alpha=1 - self.ema_decay)


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False
