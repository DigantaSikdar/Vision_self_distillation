"""
caad/models/model_utils.py
==========================
Model plumbing the CAAD step depends on: a single base VLM that carries two
LoRA adapters (STUDENT, trainable; TEACHER, an EMA copy) plus the ability to
run with all adapters disabled (the pi_init anchor).

Two operating modes, selected by ``cfg["lora"]["enabled"]``:

  LoRA mode (default, memory-cheap)
    One base model + two PEFT adapters on the same weights. The teacher is an
    EMA of the student adapter; the anchor is the adapter-disabled base. No
    extra full model copies are held in memory.

  Full-FT mode
    The base model *is* the student. ``frozen_teacher`` and ``frozen_init`` are
    separate full copies the caller passes into CAADTrainer; the context
    managers here become no-ops on the student and the trainer routes teacher /
    anchor forwards to those frozen copies directly.

Adapter naming is centralized here so the trainer never hard-codes strings.

Vision-tower note: ``get_visual_features`` returns *post-merger* visual tokens
(the features that enter the language model), so the visual-L2 alignment loss
operates on the representation the LLM actually consumes — not raw patch
embeddings.
"""

from __future__ import annotations

import contextlib

import torch

# Centralized adapter names — import these, never hard-code the strings.
STUDENT = "student"
TEACHER = "teacher"


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def load_base_model(cfg):
    """
    Load the base Qwen2.5-VL model + processor and, in LoRA mode, attach the
    STUDENT and TEACHER adapters.

    Returns (model, processor). In full-FT mode no adapters are attached and the
    caller is responsible for building frozen teacher / init copies.
    """
    from transformers import (AutoProcessor,
                              Qwen2_5_VLForConditionalGeneration)

    m = cfg["model"]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        m["name"],
        torch_dtype=getattr(torch, m.get("dtype", "bfloat16")),
        attn_implementation=m.get("attn_implementation", "flash_attention_2"),
    )
    processor = AutoProcessor.from_pretrained(
        m["name"],
        max_pixels=m["max_pixels"],
    )

    if cfg["lora"]["enabled"]:
        model = attach_adapters(model, cfg["lora"])
    return model, processor


def attach_adapters(model, lora_cfg):
    """Attach the STUDENT (trainable) and TEACHER (EMA) LoRA adapters."""
    from peft import LoraConfig, get_peft_model

    peft_cfg = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg, adapter_name=STUDENT)
    model.add_adapter(TEACHER, peft_cfg)
    # Teacher starts as an exact copy of the student so EMA begins from parity.
    sync_adapter(model, src=STUDENT, dst=TEACHER, decay=0.0)
    model.set_adapter(STUDENT)
    return model


# --------------------------------------------------------------------------- #
# Adapter context managers
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def use_adapter(model, name):
    """Temporarily activate a named PEFT adapter, restoring the previous one."""
    prev = getattr(model, "active_adapters", None)
    prev = prev[0] if prev else STUDENT
    model.set_adapter(name)
    try:
        yield model
    finally:
        model.set_adapter(prev)


@contextlib.contextmanager
def use_anchor(model, lora, frozen_init):
    """
    Yield the pi_init model used for the anchor KL.

    LoRA mode: disable all adapters so the base weights act as pi_init.
    Full-FT mode: yield the caller-provided frozen initial copy.
    """
    if lora:
        with model.disable_adapter():
            yield model
    else:
        yield frozen_init


# --------------------------------------------------------------------------- #
# EMA teacher update
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sync_adapter(model, src=STUDENT, dst=TEACHER, decay=0.999):
    """
    EMA update of the destination adapter toward the source adapter:

        theta_dst <- decay * theta_dst + (1 - decay) * theta_src

    ``decay=0.0`` performs a hard copy (used to initialize the teacher). Only
    LoRA parameters carrying the adapter name in their key are touched.
    """
    params = dict(model.named_parameters())
    src_tag, dst_tag = f".{src}.", f".{dst}."
    for name, p_src in params.items():
        if src_tag not in name:
            continue
        dst_name = name.replace(src_tag, dst_tag)
        p_dst = params.get(dst_name)
        if p_dst is None:
            continue
        p_dst.mul_(decay).add_(p_src.detach(), alpha=1.0 - decay)


# --------------------------------------------------------------------------- #
# Visual features
# --------------------------------------------------------------------------- #
def get_visual_features(model, pixel_values_videos, video_grid_thw):
    """
    Post-merger visual tokens for a video, shape (N, D).

    These are the features handed to the language model (after the patch-merger
    projection), so the visual-L2 loss aligns the representation the LLM
    actually sees under the clean vs corrupted view.
    """
    core = model
    # Unwrap DDP / PEFT / the CausalLM head down to the module that owns `.visual`.
    # Walk repeatedly: e.g. DDP(.module) -> PeftModel(.base_model) -> (.model) -> .visual
    for _ in range(6):
        if hasattr(core, "visual"):
            break
        for attr in ("module", "base_model", "model"):
            if hasattr(core, attr):
                core = getattr(core, attr)
                break
        else:
            break
    visual = core.visual
    dtype = next(visual.parameters()).dtype
    return visual(pixel_values_videos.to(dtype), grid_thw=video_grid_thw)
