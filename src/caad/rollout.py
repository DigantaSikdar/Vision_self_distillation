"""
caad/rollout.py
===============
Rollout engine: samples the student's completion y ~ pi_student(. | q, V_corrupted)
that the rest of the CAAD step scores. The trainer's design runs generation on a
dedicated vLLM worker (GPU 0) while the gradient forwards run on the train GPUs,
so generation never blocks the optimizer.

Sampling uses the CORRUPTED view — the student must answer from the degraded
input; the teacher then re-scores those same tokens under the clean view. The
returned value is the raw completion text; the trainer re-tokenizes it against
both views to enforce the cross-input alignment invariant.

LoRA weights drift every step, so ``sync_weights`` pushes the latest student
adapter into the vLLM worker on the EMA cadence (exact mechanism depends on the
serving setup; left as the one integration seam to wire per cluster).
"""

from __future__ import annotations


class RolloutEngine:
    def __init__(self, cfg, processor):
        self.cfg = cfg
        self.processor = processor
        self.sampling = cfg["rollout"]
        self._llm = None  # lazy: only the rollout worker constructs vLLM

    def _ensure_llm(self):
        if self._llm is None:
            from vllm import LLM, SamplingParams
            self._llm = LLM(
                model=self.cfg["model"]["name"],
                dtype=self.cfg["model"].get("dtype", "bfloat16"),
                gpu_memory_utilization=self.sampling.get("gpu_mem_util", 0.85),
                limit_mm_per_prompt={"video": 1},
                enable_lora=self.cfg["lora"]["enabled"],
            )
            self._sp = SamplingParams(
                n=self.sampling.get("num_generations", 1),
                temperature=self.sampling["temperature"],
                top_p=self.sampling.get("top_p", 1.0),
                max_tokens=self.sampling["max_completion_length"],
            )
        return self._llm

    def generate(self, sample):
        """Sample ``num_generations`` completions conditioned on the corrupted
        video + question. Returns a list[str] (length = num_generations)."""
        llm = self._ensure_llm()
        messages = [
            {"role": "system", "content": self.cfg["data"]["system_prompt"]},
            {"role": "user", "content": [
                {"type": "video", "video": sample["corrupted_video"],
                 "max_pixels": self.cfg["model"]["max_pixels"],
                 "max_frames": self.cfg["model"]["max_frames"]},
                {"type": "text", "text": sample["question"]},
            ]},
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        _, video_inputs = process_vision_info(messages)
        out = llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"video": video_inputs[0]}}],
            self._sp)
        return [o.text for o in out[0].outputs]

    def sync_weights(self, model):
        """Push the latest student adapter into the vLLM worker (per-cluster seam)."""
        raise NotImplementedError(
            "wire LoRA weight sync to the vLLM worker for your serving setup")
