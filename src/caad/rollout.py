"""
caad/rollout.py
===============
Rollout engine: samples the student's completion(s) y ~ pi_student(. | q, V_corrupted)
on the CORRUPTED view. Three backends (rollout.backend):

  mock   canned text, no model — pipeline smoke tests / CI.
  hf     generate with the LIVE training model (.generate()). Single-/multi-rank,
         always-fresh weights (no sync needed). Slower; good to start.
  vllm   DEDICATED vLLM server on a reserved GPU (your production topology:
         GPU 0 = server, the rest = training). The server holds a separate copy
         of the weights, so it must be re-synced as the student trains:
           sync_mode=lora  save the student adapter -> vLLM hot-loads it
                           (vLLM /v1/load_lora_adapter). Cheap, robust.
           sync_mode=full  push full base weights -> the OpenAI server has no
                           clean hot-reload; left as a marked seam (see _sync_full).
         Each training rank is an independent HTTP client to the one server, so
         there is no cross-rank broadcast — the server batches concurrent requests.

VALIDATE ON HARDWARE: the multimodal video field in chat() and the LoRA-admin
endpoint names depend on your installed vLLM version — both are flagged below.
"""

from __future__ import annotations

import logging
from pathlib import Path

_LOG = logging.getLogger("caad")


class RolloutEngine:
    def __init__(self, cfg, processor, model=None):
        self.cfg = cfg
        self.processor = processor
        self.sampling = cfg["rollout"]
        self.backend = self.sampling.get("backend", "vllm")
        self.n = self.sampling.get("num_generations", 1)
        self.model = model                 # hf backend uses the live model
        self._client = None                # VLLMServerClient (vllm backend)
        self._sync_count = 0
        self._active_adapter = None        # current LoRA name served by vLLM
        self._warned = False
        # sync_mode: auto -> lora if LoRA training else full
        sm = self.sampling.get("sync_mode", "auto")
        self.sync_mode = ("lora" if cfg["lora"]["enabled"] else "full") if sm == "auto" else sm

    # ------------------------------------------------------------------ #
    def generate(self, sample):
        if self.backend == "mock":
            txt = self.sampling.get("mock_text", "Let me reason. The answer is A.")
            return [txt] * self.n
        if self.backend == "hf":
            return self._generate_hf(sample)
        return self._generate_vllm(sample)

    def sync_weights(self, model, accelerator=None):
        """Make the rollout policy reflect the latest student weights.
        mock: no-op. hf: rebind the live model. vllm: push to the server."""
        if self.backend == "mock":
            return
        if self.backend == "hf":
            self.model = model
            return
        self._sync_vllm(model, accelerator)

    # ------------------------------------------------------------------ #
    # shared prompt construction
    # ------------------------------------------------------------------ #
    def _messages(self, sample):
        return [
            {"role": "system", "content": self.cfg["data"]["system_prompt"]},
            {"role": "user", "content": [
                {"type": "video", "video": sample["corrupted_video"],
                 "max_pixels": self.cfg["model"]["max_pixels"],
                 "max_frames": self.cfg["model"]["max_frames"]},
                {"type": "text", "text": sample["question"]},
            ]},
        ]

    # ------------------------------------------------------------------ #
    # hf backend (live model)
    # ------------------------------------------------------------------ #
    def _generate_hf(self, sample):
        import torch
        from qwen_vl_utils import process_vision_info
        model = self.model
        if model is None:
            raise RuntimeError("hf backend needs RolloutEngine(cfg, processor, model=...)")
        messages = self._messages(sample)
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        _, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], videos=video_inputs,
                                return_tensors="pt").to(model.device)
        temp = self.sampling["temperature"]
        with torch.no_grad():
            out = model.generate(
                **inputs, do_sample=temp > 0,
                temperature=temp if temp > 0 else None,
                top_p=self.sampling.get("top_p", 1.0),
                max_new_tokens=self.sampling["max_completion_length"],
                num_return_sequences=self.n)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.batch_decode(gen, skip_special_tokens=True)

    # ------------------------------------------------------------------ #
    # vllm backend (dedicated server)
    # ------------------------------------------------------------------ #
    def _ensure_client(self):
        if self._client is None:
            srv = self.sampling["server"]
            self._client = VLLMServerClient(
                base_url=f"http://{srv.get('host', '127.0.0.1')}:{srv.get('port', 8000)}",
                base_model=self.cfg["model"]["name"])
            self._client.wait_ready(timeout=srv.get("startup_timeout", 600))
        return self._client

    def _generate_vllm(self, sample):
        client = self._ensure_client()
        served = self._active_adapter or self.cfg["model"]["name"]
        return client.chat(self._messages(sample), model=served, n=self.n,
                           temperature=self.sampling["temperature"],
                           top_p=self.sampling.get("top_p", 1.0),
                           max_tokens=self.sampling["max_completion_length"])

    def _sync_vllm(self, model, accelerator):
        if self.sync_mode == "lora":
            self._sync_lora(model, accelerator)
        else:
            self._sync_full(model, accelerator)
        self._sync_count += 1

    def _sync_lora(self, model, accelerator):
        """Save the student adapter and hot-load it into the vLLM server. Only the
        main process writes + registers; a barrier ensures every rank sees the new
        adapter before its next generate(). The adapter name is derived from the
        sync counter, so all ranks agree without a broadcast."""
        name = f"student-{self._sync_count}"
        out_root = Path(self.sampling["server"].get("adapter_dir", "outputs/_vllm_adapters"))
        adir = out_root / name
        is_main = accelerator is None or accelerator.is_main_process
        if is_main:
            core = accelerator.unwrap_model(model) if accelerator else model
            adir.mkdir(parents=True, exist_ok=True)
            core.save_pretrained(str(adir), selected_adapters=["student"])  # student only
            self._ensure_client().load_lora(name, str(adir.resolve()))
        if accelerator is not None:
            accelerator.wait_for_everyone()
        self._active_adapter = name

    def _sync_full(self, model, accelerator):
        """Full-FT weight push. The OpenAI vLLM server has no clean base-weight
        hot-reload, so this is a SEAM. Two ways to wire it (per your vLLM setup):
          (a) periodic restart: save a full checkpoint and relaunch the server
              from it every sync_every steps (simple, slower), or
          (b) a custom weight-update server (TRL `vllm-serve` style) that receives
              gathered ZeRO-3 params over NCCL via collective_rpc.
        Until wired, rollouts stay at the server's initial weights (stale)."""
        if not self._warned:
            _LOG.warning("rollout sync_mode=full is not wired for the vLLM server: "
                         "rollouts use the server's initial weights (STALE). Use "
                         "sync_mode=lora, or wire _sync_full (restart-from-checkpoint "
                         "or a NCCL weight-update server). See docstring.")
            self._warned = True


class VLLMServerClient:
    """Thin HTTP client to a vLLM OpenAI-compatible server.
    Endpoint shapes (`/v1/chat/completions`, `/v1/load_lora_adapter`, `/health`)
    follow vLLM's documented API but VARY BY VERSION — validate against yours."""

    def __init__(self, base_url, base_model):
        self.base_url = base_url.rstrip("/")
        self.base_model = base_model

    def wait_ready(self, timeout=600, interval=5):
        import time
        import requests
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if requests.get(self.base_url + "/health", timeout=5).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(interval)
        raise RuntimeError(f"vLLM server not ready at {self.base_url} after {timeout}s")

    def load_lora(self, name, path):
        """Hot-load a LoRA adapter. Requires the server started with
        VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 and --enable-lora."""
        import requests
        r = requests.post(self.base_url + "/v1/load_lora_adapter",
                          json={"lora_name": name, "lora_path": path}, timeout=120)
        # already-loaded is fine; surface anything else
        if r.status_code >= 400 and "already" not in r.text.lower():
            r.raise_for_status()

    def chat(self, messages, model, n, temperature, top_p, max_tokens):
        import requests
        # SEAM: video content shape is vLLM-version-specific. Many builds accept
        # {"type": "video_url", "video_url": {"url": "file:///abs/path.mp4"}}.
        payload_messages = _to_openai_messages(messages)
        r = requests.post(
            self.base_url + "/v1/chat/completions",
            json={"model": model, "messages": payload_messages, "n": n,
                  "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens},
            timeout=600)
        r.raise_for_status()
        return [c["message"]["content"] for c in r.json()["choices"]]


def _to_openai_messages(messages):
    """Convert our internal message format to OpenAI chat content. The video item
    becomes a file:// video_url — adjust to your vLLM version if it expects a
    different multimodal field (e.g. base64 or a different type tag)."""
    out = []
    for m in messages:
        if isinstance(m["content"], str):
            out.append(m)
            continue
        parts = []
        for c in m["content"]:
            if c["type"] == "video":
                url = c["video"]
                if "://" not in str(url):
                    url = "file://" + str(Path(url).resolve())
                parts.append({"type": "video_url", "video_url": {"url": url}})
            elif c["type"] == "text":
                parts.append({"type": "text", "text": c["text"]})
        out.append({"role": m["role"], "content": parts})
    return out
