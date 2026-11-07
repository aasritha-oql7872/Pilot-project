# GRPO diagnostics: why was reward variance ~zero, and is GRPO actually learning?

## 1. Instrumentation added (per training step, in `train_metrics.csv`, `param_drift.csv`, `trajectories.jsonl`)
| Quantity requested | Where it is logged |
|---|---|
| reward mean / std / distribution | TRL `reward`, `reward_std`; per-rollout rewards in `trajectories.jsonl` |
| % groups with identical rewards, % with non-zero advantages | TRL `frac_reward_zero_std`; `searchlite/group_frac_nonzero_std` (shaped) and `…_orig_nonzero_std` (original reward); advantages are non-zero iff the group std is not ~0 |
| variance created by the shaping | `searchlite/group_frac_bonus_created_variance` (shaped reward varies although the original reward does not) |
| reward components separately | `searchlite/c_correct`, `c_search_cost`, `c_invalid_cost`, `c_shaping`, `answer_supported`, `abstained` |
| distinct responses / action trajectories per group; effective group size | `searchlite/group_distinct_trajectories_per_group` (distinct (queries…, answer) tuples out of G) |
| gradient norms, whether optimizer updates occur | TRL `grad_norm`; `param_drift.csv` (LoRA L2 and max-abs distance from the initial weights every 5 steps) |
| change to trainable parameters / adapter | `ParamDrift`, `src.train_diag --init`, `src.audit_lineage` (also vs the initial adapter) |
| KL to the reference policy | **not available: β = 0 in every run, so TRL keeps no reference model.** Weight distance is the policy-change proxy; a β>0 pilot would add TRL's `kl` metric |
| correct / incorrect / partial / unsupported fractions | `src.report_categories` (accuracy, evidence-supported, recall, unsupported, false abstention) |
| pre-training signal estimate | `src.signal_probe` (sampled rollouts, no update): informative-group fraction under the original, evidence and grounded rewards |

## 2. Code review of the optimization path [code]
* Advantage: `(r − mean)/(std + 1e-4)` per group of `num_generations` consecutive rollouts. A group with identical rewards has
  std 0 → advantage 0 → no gradient (formal reason a never-sampled action is never learned).
* Trainable parameters: only LoRA (1,146,880); [test] the base model is unchanged after an optimizer step.
* Loss values are not evidence of learning: with zero advantages TRL logs `loss = 0` and `grad_norm = 0`
  ([test] `test_zero_advantage_gives_no_gradient_and_no_update`). The earlier logs show exactly that on 50 of 60 steps.
* Tool tokens are masked from the loss by TRL (environment rollouts); SFT masking is tested separately.
* Checkpoint loading: [test] reload of `init_adapter` restores weights and is trainable.

## 3. Unit tests (CPU, all passing) — `tests/test_grpo_mechanics.py`
group-constant bonus cancels; within-group variation changes advantages; evidence bonus constant when all rollouts retrieve the
same documents; group statistics; trajectory signatures; grounded reward (supported/unsupported/abstention, no gold ids);
zero-advantage → zero gradient and no update; non-zero advantage moves exactly the LoRA parameters; adapter reload; same-seed
runs with shifted rewards stay within 1e-6; signal-probe grouping. Reward arithmetic is in `tests/test_reward_evidence.py`.

## 4. Why was reward variance near zero? What is established
1. **Saturated policy.** B answered 30/30 two-hop and 59/60 one-hop on dev, and its sampled behaviour had entropy ≈ 0.04
   [Colab, pasted]. When every rollout of a prompt succeeds in the same way, all rewards are equal (e.g. 0.90 each).
2. **Training tasks too easy for B.** C/D trained on `train` + `twohop_train`, which B was SFT-trained on. The informative steps
   (10/60) are the ones where some rollout failed.
3. **Bonus cannot help when it is group-constant** (Phase 1, §2). The original evidence reward therefore adds variance only when
   rollouts differ in what they retrieve, which a policy with a fixed query habit seldom does.
4. **Temperature 0.7 and G = 4** give little diversity from a near-deterministic policy.

Not yet established [pending, needs Colab]: the measured per-run fractions of informative groups for the *new* hard-task runs, and
whether raising temperature/group size on `hard_train` raises them (`signal_probe`, gate in `run_headroom_study.sh`).

## 5. Verdict on "is evidence-aware GRPO genuinely updating the model?"
The optimizer demonstrably updates the adapter (non-zero movement, tests above). For the C/D runs the update was small and
essentially common to both variants because the shaping did not change the advantages. No claim about the *value* of the
evidence reward can be made from those runs; the hard-task runs are designed to test it where the bonus can vary within groups.
