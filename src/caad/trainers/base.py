"""
caad/trainers/base.py
=====================
Shared training scaffolding: the rollout -> step -> backward -> EMA loop that is
common to CAAD and any future distillation variant. Algorithm-specific behavior
lives in a per-sample ``step(sample, completion_text) -> (loss, stats)`` method
(see CAADTrainer); this loop owns batching/grad-accum, optimization, LR
scheduling, gradient clipping, EMA cadence, logging, checkpointing, optional HF
hub push, and an in-training eval hook — all driven from the recipe so nothing
is hard-coded.

Counting convention (matches TRL/HF): ``max_steps``, ``save_steps``,
``ema_every``, ``logging_steps`` and ``eval.eval_steps`` are all in **optimizer
steps**, not samples.

Effective batch per optimizer step:
    per_device_train_batch_size x gradient_accumulation_steps
        x num_generations x num_train_gpus

Video VLM samples have variable visual-token counts, so we accumulate
sequentially (one sample's forwards at a time) rather than padding a true device
batch; ``per_device_train_batch_size`` folds into the accumulation count
alongside ``gradient_accumulation_steps``. ``num_generations`` is the number of
student completions sampled per (video, question) — each is a backward.
"""

from __future__ import annotations

import logging
from pathlib import Path

_LOG = logging.getLogger("caad")


class TrainLoop:
    def __init__(self, step_trainer, rollout_engine, dataloader, optimizer,
                 accelerator, cfg, run_path, scheduler=None, metrics_logger=None,
                 evaluator=None):
        self.step_trainer = step_trainer          # .step() and .update_ema_teacher()
        self.rollout = rollout_engine             # .generate(sample) -> list[str]
        self.loader = dataloader
        self.opt = optimizer
        self.sched = scheduler
        self.acc = accelerator
        self.cfg = cfg
        self.run_path = Path(run_path)
        self.metrics = metrics_logger
        self.evaluator = evaluator                # callable(model, cfg) -> dict, or None (seam)
        t = cfg["train"]
        self.per_device_bs = t.get("per_device_train_batch_size", 1)
        self.grad_accum = t["gradient_accumulation_steps"]
        self.accum_samples = self.per_device_bs * self.grad_accum
        self.max_steps = t.get("max_steps", 0)            # optimizer steps; <=0 -> use epochs
        self.num_train_epochs = t.get("num_train_epochs", 1)
        self.ema_every = t["ema_every"]
        self.save_steps = t["save_steps"]
        self.logging_steps = t.get("logging_steps", 10)
        self.logging_first_step = t.get("logging_first_step", False)
        self.max_grad_norm = t.get("max_grad_norm", 0.0)
        self.save_total_limit = t.get("save_total_limit", 0)
        self.rollout_sync_every = (cfg.get("rollout", {}) or {}).get("sync_every", 0)
        self.hub = cfg.get("hub", {}) or {}
        ev = cfg.get("eval", {}) or {}
        self.do_eval = ev.get("do_eval", False)
        self.eval_steps = ev.get("eval_steps", 0)
        self.log_completions = (cfg.get("wandb", {}) or {}).get("log_completions", False)

    def run(self):
        micro = 0          # samples since the last optimizer update
        opt_step = 0       # optimizer steps taken
        epoch = 0
        done = False
        # make the rollout policy match the starting weights before step 1
        self.rollout.sync_weights(self.step_trainer.model, self.acc)
        self.opt.zero_grad()
        while not done:
            for sample in self.loader:
                completions = self.rollout.generate(sample)    # list, len = num_generations
                g = max(len(completions), 1)
                stats, last_completion = {}, ""
                for completion in completions:
                    loss, stats = self.step_trainer.step(sample, completion)
                    self.acc.backward(loss / (self.accum_samples * g))
                    last_completion = completion
                micro += 1
                if micro % self.accum_samples != 0:
                    continue

                # ---- optimizer update ----
                if self.max_grad_norm and self.max_grad_norm > 0:
                    self.acc.clip_grad_norm_(self.step_trainer.model.parameters(),
                                             self.max_grad_norm)
                self.opt.step()
                if self.sched is not None:
                    self.sched.step()
                self.opt.zero_grad()
                opt_step += 1

                if opt_step % self.ema_every == 0:
                    self.step_trainer.update_ema_teacher()
                if self.rollout_sync_every and opt_step % self.rollout_sync_every == 0:
                    self.rollout.sync_weights(self.step_trainer.model, self.acc)
                self._maybe_log(opt_step, stats, last_completion)
                if self.do_eval and self.eval_steps > 0 and opt_step % self.eval_steps == 0:
                    self._run_eval(opt_step)
                if opt_step % self.save_steps == 0:
                    self.save_checkpoint(opt_step)
                if self.max_steps > 0 and opt_step >= self.max_steps:
                    done = True
                    break

            epoch += 1
            if self.max_steps <= 0 and epoch >= self.num_train_epochs:
                done = True

        self.save_checkpoint(opt_step, final=True)

    # ------------------------------------------------------------------ #
    def _maybe_log(self, opt_step, stats, completion):
        if self.metrics is None or not self.acc.is_main_process:
            return
        first = self.logging_first_step and opt_step == 1
        if not first and opt_step % self.logging_steps != 0:
            return
        lr = self.sched.get_last_lr()[0] if self.sched is not None else None
        row = {**stats, "lr": lr, "epoch_step": opt_step}
        if self.log_completions:
            row["rollout/text"] = completion[:1000]
        self.metrics.log(opt_step, row)

    def _run_eval(self, opt_step):
        """In-training eval hook. The standalone harness (caad.eval) is the main
        path; this fires only if an evaluator was injected and do_eval is set."""
        if self.evaluator is None:
            _LOG.warning("eval.do_eval is set but no evaluator was injected; "
                         "skipping (use the standalone caad.eval harness)")
            return
        if not self.acc.is_main_process:
            return
        res = self.evaluator(self.acc.unwrap_model(self.step_trainer.model), self.cfg)
        if self.metrics is not None:
            self.metrics.log(opt_step, {f"eval/{k}": v for k, v in res.items()})

    # ------------------------------------------------------------------ #
    def save_checkpoint(self, opt_step, final=False):
        if not self.acc.is_main_process:
            return
        name = "final" if final else f"step_{opt_step}"
        out = self.run_path / "checkpoints" / name
        out.mkdir(parents=True, exist_ok=True)
        self.acc.unwrap_model(self.step_trainer.model).save_pretrained(out)
        if not final:
            self._prune_checkpoints()
        self._maybe_push_hub(out, opt_step, final)

    def _maybe_push_hub(self, out, opt_step, final):
        if not self.hub.get("enabled"):
            return
        every = self.hub.get("push_every_n_steps", 0)
        if not final and (every <= 0 or opt_step % every != 0):
            return
        repo = self.hub.get("model_id")
        if not repo:
            _LOG.warning("hub.enabled but hub.model_id is empty; skipping push")
            return
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo, private=self.hub.get("private", True),
                            exist_ok=True)
            api.upload_folder(folder_path=str(out), repo_id=repo,
                              path_in_repo=out.name)
            _LOG.info("pushed %s -> hf.co/%s", out.name, repo)
        except Exception as e:                       # best-effort; never crash training
            _LOG.warning("hub push failed (%s): %s", out.name, e)

    def _prune_checkpoints(self):
        """Keep only the most recent ``save_total_limit`` step checkpoints."""
        if not self.save_total_limit or self.save_total_limit <= 0:
            return
        ckpts = sorted((self.run_path / "checkpoints").glob("step_*"),
                       key=lambda p: int(p.name.split("_")[1]))
        for old in ckpts[:-self.save_total_limit]:
            for f in sorted(old.rglob("*"), reverse=True):
                f.unlink() if f.is_file() else f.rmdir()
            old.rmdir()
