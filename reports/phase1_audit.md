# Phase 1 audit: repository, lineage, leakage, environment consistency

Branch `experiment/headroom-grpo` (created from `experiment/evidence-aware-grpo`). Status labels: **[code]** read from the
repository this session; **[test]** a CPU unit test written this session (all pass); **[Colab, pasted]** numbers pasted
from earlier Colab output (artifacts are on Drive, not on this machine); **[pending]** needs the Drive artifacts or a GPU.
Nothing here was run on a GPU in this session.

## 1. Models and their lineage [code]
| Model | Recipe (config) | Initialization | Training data | Reward |
|---|---|---|---|---|
| base | none | pretrained Qwen3-0.6B | – | – |
| A = M1 | `configs/train.yaml` (GRPO, 1-hop) | base + new LoRA | `train` | original, default reminder |
| B = M2 | `configs/sft.yaml` (LoRA SFT, 75 steps) | base + new LoRA | 200 trajectories from `train` + `twohop_train` | – (assistant tokens only) |
| C = M3 (easy tasks) | `configs/train_sft_grpo.yaml` | `init_adapter` = B | `train` + `twohop_train` | original, typed reminder, lr 2e-5, 60 steps |
| D = M4 (easy tasks) | `configs/train_sft_grpo_evidence.yaml` | `init_adapter` = B | same | evidence (α 0.10, β 0.30) |

C and D load B through `PeftModel.from_pretrained(base, init_adapter, is_trainable=True)` and write to separate
`--output` directories; `train_grpo.py` refuses to write over an existing `adapter/`. Both used seed 42.
[test] `test_adapter_reload_for_grpo_restores_weights_and_is_trainable` checks that this reload reproduces the saved
tensors and leaves them trainable.

## 2. The identical-adapter observation
Reported [Colab, pasted]: max|B−C| = 2.64·10⁻⁴, max|B−D| = 2.64·10⁻⁴, max|C−D| = 0.0; only 10 of 60 steps had a non-zero
gradient norm; mean `frac_reward_zero_std` = 0.95.

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Wrong/shared checkpoint path (C and D read or wrote the same files) | not supported | distinct `--output` dirs and run ids [code]; both differ from B by 2.6·10⁻⁴, so neither is a copy of B. A byte-level check of the two files is in `src.audit_lineage` [pending: run on Drive] |
| Training steps never executed | ruled out | 60 optimizer steps logged for both; the trainer raises if LoRA weights do not change (they changed) |
| Optimizer cannot update the adapter | ruled out | [test] a non-zero advantage moves exactly the LoRA parameters and leaves the base frozen |
| Updates exist but were tiny | **supported** | 10 informative steps of 60; weight change 2.6·10⁻⁴ |
| The evidence bonus never changed the advantages | **mechanism established, instance pending** | see below |

**Mechanism.** TRL 1.17 computes `advantage = (r − group mean)/(group std + 1e-4)` per group of consecutive rollouts of one
prompt (`trl/trainer/grpo_trainer.py`, lines 2904–2906 in the installed copy). Adding the same constant to every reward in a
group leaves the advantage unchanged. The evidence bonus (α·first_evidence + β·complete_chain) is identical across a group
whenever all its rollouts retrieve the same documents, which is what a saturated SFT policy does (100% full chain).
[test] `test_group_constant_bonus_cancels_in_advantages` and `test_evidence_bonus_is_constant_when_all_rollouts_retrieve_the_same_documents`
verify the cancellation; `test_two_runs_with_same_seed_and_same_advantages...` shows that runs with advantages differing only
by a group-constant shift end within 1e-6 of each other. With the same seed (same sampled tasks and rollouts, as long as
the weights are identical) and gradient differences below float32 resolution of the adapter weights, bitwise-equal adapters
are plausible. **I have not shown this happened in the Colab runs**: the exact C/D equality needs the pair comparison below.

Pending, one command on Colab (cell 4 of `SearchLite_RL_Headroom_Study.ipynb`):
`python -m src.audit_lineage B=… C=… D=…` reports file hashes, tensor-level differences, configs, and the fraction of
sampled rollouts (step, task, assistant text) that are identical between C and D; `src.traj_stats` reports, per run, the
groups whose shaped reward differs while the original reward does not ("bonus created variance"). If that fraction is 0 for D,
D optimized exactly C's objective and the identical adapters are explained.

## 3. Leakage and split checks [code, test]
* SFT data comes from `train` and `twohop_train` only (asserted in `tests/test_sft.py`); gold values appear only as final answers.
* Splits have disjoint entities and separate corpora (`tests/test_data.py`); sealed `final`/`twohop_final` hashes are verified
  by `python -m src.manifest --verify-sealed` [passes].
* **Selection on dev.** The typed reminder, the headline choice (B) and the C/D comparison all used `dev`/`twohop_dev`, so those
  numbers are development numbers. They are not independent of the choices made.
* **Dev saturation.** `twohop_dev` uses the same question templates as the SFT data with new entities, so 30/30 shows entity
  generalization only. Nothing in dev can show an RL gain beyond 30/30.
* The sealed `final`/`twohop_final` sets share the same templates, so they cannot show template generalization either.
  Whether those sealed sets have been evaluated is recorded by the Drive logs (`study_sft/run_sealed_*.log`) [pending].

## 4. Environment and evaluation consistency
* **Defect found and fixed [code]:** `src/probe.py explore` ignored `--answer-format`. The sampled B/C/D probes ran with the
  default reminder while labelled typed. Fixed; probes must be re-run (notebook cell in `SearchLite_RL_Final_Study.ipynb`).
* **Reminder mismatch for M1.** A (M1) was trained with the default reminder; B/C/D with the typed one. On the hard benchmark
  all models see `typed_abstain`, which A has never seen; this is a stated limitation, not hidden.
* **Search cap.** `MAX_SEARCHES = 2` is a module constant; three-hop chains are impossible without changing the environment for
  every model, so they are documented as unsupported (reports/evaluation_design.md).
* The retriever is lexical BM25 over a per-split corpus; ties are broken by corpus order, so retrieval is deterministic.

## 5. Unresolved questions
1. Exact cause of C == D (needs `audit_lineage` on Drive).
2. Whether the sealed `final`/`twohop_final` evaluation has been run and what it shows.
3. Whether the Colab dev numbers match the saved JSONs (`src.verify_reported`).
4. Why SFT outperformed GRPO from the base model is answered in `reports/final_rl_findings.md` from measured facts.
