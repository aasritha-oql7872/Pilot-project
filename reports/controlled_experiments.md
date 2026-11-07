# Controlled experiments: M1–M4 on the hard benchmark

**Status: designed, scripted and unit-tested; NOT yet run. No result in this file is a measurement.** Runs happen on free Colab
(`SearchLite_RL_Headroom_Study.ipynb`, branch `experiment/headroom-grpo`); nothing was started without your go-ahead.

## 1. Models
| ID | Definition | Checkpoint |
|---|---|---|
| M1 | original GRPO without SFT (condition A; incompatibility noted below) | `outputs/main_v2/adapter` |
| M2 | SFT only (condition B, **frozen**) | `outputs/study_sft/B/adapter` |
| M3 | M2 + GRPO, original reward | `outputs/study_hard/M3_s{42,43}` |
| M4 | M2 + GRPO, evidence-aware reward (α 0.10, β 0.30) | `outputs/study_hard/M4_s{42,43}` |
| M4g (extra, labelled) | M2 + GRPO, grounded reward (see §3) | `outputs/study_hard/M4g_s{42,43}` |
M3, M4, M4g start from the same frozen M2 adapter and differ only in `reward.mode` and the seed. All use identical tasks
(`hard_train` + original `train` to guard one-hop), sampling settings (set once from the signal probe on `hard_train`), 100 steps,
lr 2e-5, β = 0.
**M1 caveat:** A was trained with the default reminder and never saw `typed_abstain`; on `unanswerable` tasks it cannot be
expected to abstain. M1 is reported with that limitation and is not used to rank the GRPO variants.

## 2. Protocol and gates (`run_headroom_study.sh`)
0. Integrity: hash check of hard and sealed splits, `validate_hard` (solvable, disjoint), mechanics tests.
1. **Headroom** on `hard_dev`: base, M1, M2, old C/D evaluated greedily. Proceed only if M2 ≤ 0.85 overall or some category ≤ 0.8
   (`decision_headroom.json`). Otherwise stop: "no headroom".
2. **Signal** (`src.signal_probe`, M2 on `hard_train` only, T ∈ {0.7, 1.0}, 8 samples, 48 tasks): fraction of GRPO groups with
   non-zero reward std under each reward, and the variance created by each bonus. Choose (T, G) with the most informative
   groups among settings that still answer ≥ 80% of the time. Stop if the best has < 15% informative groups
   (`decision_signal.json`): the measured cause is then "no learning signal in this configuration" and GRPO is not run.
3. **Training** (2 seeds × {M3, M4, M4g}), then `train_diag`, `traj_stats`, `audit_lineage` per run.
4. **Development evaluation** (greedy): `hard_dev`, plus regression checks on `dev` (one-hop) and `twohop_dev`. Optional sampled
   evaluation (`SAMPLED=1`): T = 0.7, three seeds.
5. Tables: `summary_hard_dev.md`, `summary_dev_regression.md`, `summary_twohop_dev_regression.md` with paired tests vs M2.
6. **Pre-registered selection rule** (written before any hard_dev result): a GRPO variant replaces M2 as the headline only if its
   mean `hard_dev` accuracy over seeds exceeds M2's by ≥ 5 points and `dev` and `twohop_dev` each drop by ≤ 3 points.
   Otherwise M2 stays the headline. The rule is applied by hand and written to `FROZEN_HARD` before the locked evaluation.
7. **Locked evaluation** (`run_hard_final.sh`): refuses to run without `FROZEN_HARD`, `summary_hard_dev.md`, and matching hashes.
   Run once. `hard_final` is never used for tuning.

## 3. Reward components (what each term is for)
| Term | Measures | Exploit risk and mitigation |
|---|---|---|
| correct (1.0) | final-answer exact match (abstention string is the gold of unanswerable tasks) | – |
| search cost (−0.05 each) | efficiency; 20× smaller than the correct reward, so necessary searches stay profitable | – |
| invalid (−0.10) | malformed tool call/answer | – |
| evidence α/β (existing) | gold documents retrieved | uses privileged ids; group-constant when all rollouts retrieve the same documents |
| **grounded penalty (−0.25)** | a non-abstaining answer that appears neither in the question nor in any retrieved record (a guess) | uses no gold ids; cannot be gamed by always abstaining (abstention scores 0, a correct answer ~0.9); supported-but-wrong answers (misreads) are not additionally punished |
The grounded term is a **separate variable (M4g)**; M4 is the existing evidence-aware reward, so M3 vs M4 is unchanged in meaning.

## 4. GPU budget (L4, free tier subject to availability; no paid API)
Stage 1 ≈ 10 min inference · stage 2 ≈ 10 min inference · stage 3 ≈ 6 runs × ≈ 10 min · stage 4 ≈ 20 evaluations × ≈ 2 min ·
total ≈ 2.5 h, resumable (outputs saved to Drive after each stage). `SEEDS="42"` halves the training/eval cost. Final evaluation
≈ 10 min. Peak VRAM observed for earlier GRPO runs: 2.8 GB; SFT: 19.8 GB (not used here).

## 5. Comparison table (to be filled from `summary_hard_dev.md` / `summary_hard_final.md`)
| Model | hard_dev acc (k/n, Wilson) | evidence-supported | unsupported | abstention (unanswerable) | searches | dev acc | twohop_dev acc | hard_final |
|---|---|---|---|---|---|---|---|---|
| base | pending | pending | pending | pending | pending | – | – | pending |
| M1 | pending | | | | | | | pending |
| M2 | pending | | | | | pending | pending | pending |
| M3 | pending | | | | | pending | pending | pending |
| M4 | pending | | | | | pending | pending | pending |
| M4g | pending | | | | | pending | pending | pending |

## 6. Limits of the design
Two training seeds; n = 10 per category on dev, 20 on final; extra `hard_train` exposure for GRPO models that M2 lacks (no SFT-on-
`hard_train` control, so a gain would show "training on hard tasks with RL helps", not "RL beats SFT on the same data"); a
single model size; M1 not environment-compatible. Differences within a few tasks are reported as such, not as improvements.
