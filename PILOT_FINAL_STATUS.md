# Pilot final status (branch experiment/evidence-aware-grpo; nothing committed, pushed, or PR'd)

## Implemented and tested (54 tests pass)
* Reward: original (untouched), evidence-aware, potential-difference (`src/reward_evidence.py`, 12 tests incl. no gold leakage).
* Diagnostics: retrieval audit, failure taxonomy, sampled-exploration and forced-second-search probes (`src/audit.py`, `src/probe.py`).
* MDP: finite-horizon Bellman reference, **oracle-state** (observes gold evidence), 3 tests.
* **SFT warm-start** (`src/sft_data.py`, `src/sft_tokens.py`, `src/train_sft.py`, `configs/sft.yaml`): trajectories from train + twohop_train only with real
  tool observations; loss only on assistant tokens (tests: 3 supervised spans per two-hop example, tool/system/user text never supervised,
  loss = mean CE over supervised tokens, LoRA gradients nonzero). GRPO can start from the SFT adapter (`init_adapter`, trainer construction verified).
* `answer_format: typed` option (justified by the answer-extraction diagnosis), `--prompt-variant`, overwrite guards in both trainers, sealed holdouts `final` / `twohop_final`.
* Pipeline: `run_sft_study.sh` + `SearchLite_RL_SFT_Study.ipynb` (gate after B, FINAL needs a FROZEN file). No `rm`/`rmtree` anywhere in the new scripts.

## Executed locally (base model, MPS) - evidence in results/audit/
* Retrieval of the full chain with two valid queries: 100% (dev 30, test 60).
* Base sampling: 0 / 80 second searches at T=0.7 and 0 / 80 at T=1.0.
* Forced second search, all 30 dev tasks: 10/30 with the default reminder (19 answered with the researcher's name, 1 sentence);
  26/30 with the typed reminder (0 wrong-field; 4 format problems).
* Real-model SFT forward/backward: 59 of 607 tokens supervised, nonzero LoRA gradients.

## Reported from earlier Colab runs (output files are not on this machine)
v1 collapse (37 -> 38%), v2 98% test (98/95/97), two-hop 0/60 for base, one-hop and mixed-GRPO models.

## Audit of earlier scripts (done)
Found and removed `rm -rf outputs` and `rm -rf` of the Drive backup in the old `SearchLite_RL_Colab.ipynb` (now timestamped copies).
`train_grpo.py` now refuses to write over an existing adapter. The sealed splits are referenced only inside the `FINAL=1` branch
(plus tests); SFT data is asserted to come from train/twohop_train.

## Not run / blocked (no GPU results exist)
* B (SFT only), C (SFT+GRPO), D (SFT+evidence GRPO), A re-evaluation on dev, MDP tables, compute/VRAM numbers: need a Colab L4.
  Local training was not attempted at scale (Mac swap 97% full).
* The 98% artifacts are unverified here. The sealed holdouts have not been evaluated.
* If the B gate fails (second-search rate < 50% or two-hop dev accuracy < 30%) the study stops; GRPO is then not run.

## Next
Run `SearchLite_RL_SFT_Study.ipynb`; paste the tail of the log and `outputs/study_sft/summary_dev.md`. Then fill REPORT.md sections 9 and the case studies.
Housekeeping: REPORT.md, PILOT_FINAL_STATUS.md are un-ignored in `.gitignore`; `results/audit/` holds small evidence files; the earlier
`run_evidence_study.sh` / `SearchLite_RL_Evidence_Study.ipynb` are superseded by the SFT study (kept, unmodified, not required).
