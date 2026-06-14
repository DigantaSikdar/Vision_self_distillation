# CAAD — design notes

## Method
**CAAD** (Corruption-Aware Adaptive Distillation) is a self-distillation scheme
for video VLMs. Per sample, one base model does four/five forwards:

| role        | view       | grad | what                                            |
|-------------|------------|------|-------------------------------------------------|
| rollout     | corrupted  | no   | sample y ~ pi_student (vLLM worker)             |
| student     | corrupted  | yes  | logits + post-merger visual features            |
| clean feats | clean      | no   | post-merger visual features (L2 target)         |
| teacher     | clean      | no   | EMA-adapter logits on the student's tokens      |
| anchor      | corrupted  | no   | pi_init logits on the same tokens               |

Loss: `gated_divergence + lambda_l2 * visual_L2 + anchor_beta * KL(pi || pi_init)`.

- **Gated divergence** (`losses/divergence.py`): low-teacher-entropy tokens use
  forward-KL `D(T||S)`; the rest use `JSD_beta(S,T)`. Per-rollout entropy
  quantile, with a batch-level fallback for short rollouts.
- **Visual L2**: aligns the corrupted-view visual tokens to the (stop-grad)
  clean-view tokens, on the post-merger representation the LLM consumes.
- **Anchor KL**: keeps the student near pi_init to prevent collapse.

## Invariant
Clean and corrupted prompts must tokenize to the **same length** so teacher
logits at position `t` score the student's token `y_t` under an identical
prefix. Corruptions therefore preserve `(T, H, W, C)`; `data/corruption.py`
asserts shape preservation and `trainers/caad.py` asserts prompt-length parity.

## Determinism
Corruptions are pure functions of `(frames, CorruptionSpec)`; all randomness
flows from `np.random.default_rng(spec.seed)` and the seed derives from sample
identity, so a corrupted clip reproduces bit-for-bit from its clean source.

## Where things live
- `trainers/` — one trainer per algorithm; variants are config flags.
- `losses/` — divergence + alignment terms.
- `models/model_utils.py` — dual-adapter LoRA plumbing + visual features.
- `rollout.py` — vLLM generation worker (the one per-cluster integration seam).
- `eval/` — `run_task` is the atomic (checkpoint, task) unit; `orchestrate`
  fans out; `metrics.py` defines pass@k / maj@k / avg-pass@1 **once**.
