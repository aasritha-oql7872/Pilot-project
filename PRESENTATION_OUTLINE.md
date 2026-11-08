# Presentation outline (8-10 min) and likely questions

Use measured numbers only; mark anything without a result as "in progress".

1. **Problem.** An agent must chain two searches: Project -> investigator -> investigator's institution.
2. **RL formulation.** Observation = question + history; actions = search(query) | answer; transition = BM25 top-2;
   reward = correct - 0.05/search - 0.10/invalid; horizon 2 searches; policy = Qwen3-0.6B + LoRA; GRPO.
3. **Why original GRPO failed.** Slide with three facts: retrieval works (100% with scripted queries); base never samples
   a second search (0/160); wrong-field answers (10/30 -> 26/30 with a typed reminder). GRPO advantage = 0 when all rewards in a
   group are equal.
4. **SFT intervention.** 200 trajectories from training data, real tool observations, loss on the model's own turns only:
   two-hop dev 0/30 -> 30/30, one-hop 59/60 (dev).
5. **Partial-SFT + GRPO (E).** Result table from `summary_early_dev.md`, or "selection found no suitable checkpoint".
6. **RL-only exploration (F).** Per-condition pilot decisions (`decision_*.json`) and, where run, learning curves.
7. **Curves.** second-search rate vs training step (`traj_stats` windows), fraction of informative groups, accuracy bars with Wilson CIs.
8. **Lessons.** exploration before reward design; check that the reward separates *sampled* trajectories; separate
   behavioural discovery, reward optimization and answer extraction.

**Demo.** `python demo.py --a eval_twohop_dev_A.json --b eval_twohop_dev_B.json`: the same question where A stops after one
search and answers with the investigator's name, and B finds the institution.

## Likely questions
* *Why GRPO?* Critic-free, verifiable reward, supported by TRL with tool environments; small memory footprint for a 0.6B LoRA.
* *Why is this RL?* The agent chooses actions (search/answer/query), receives a delayed scalar reward, and the policy is updated by policy gradient; SFT is only the initialization comparison.
* *Reward?* `correct - 0.05*searches - 0.10*invalid`; shaped variants add gold-evidence bonuses inside the reward only.
* *Why didn't GRPO discover the second search?* In 0/160 sampled base rollouts it never appeared, so no group contained it; with equal rewards the advantage is zero.
* *Can reward shaping fix it?* Only if the shaped behaviour is sampled; F-pilots test that directly.
* *SFT vs RL?* SFT imitates correct trajectories (supervision of actions); RL only reinforces what it samples and what the reward distinguishes.
* *Generalization beyond templates?* Not shown: dev and sealed sets share question templates with new entities.
* *Why is SFT better than GRPO here?* It supplies the missing action; after SFT 95% of GRPO groups had zero reward variance.
* *With more compute?* More seeds, larger model, rephrased and out-of-template questions, a real corpus, a curriculum from partial SFT.

## Final-submission slide map (mandatory requirements must be visible)
| Course requirement | Slide content (all from saved files) |
|---|---|
| 1. RL framework set up and running | architecture diagram; TRL GRPO + LoRA + BM25 environment; one line: tests pass, runs resumable on Colab; training log excerpt |
| 2. Reward and policy-loss curves | `fig1` (original GRPO) and `fig2` (M2-initialised GRPO) side by side; caption: GRPO loss is not a cross-entropy, ~0 at start and exactly 0 when a group ties, so judge by reward, informative-group share and behaviour |
| 3. Case studies of decisions and tool use | six real traces from `submission/case_studies.md`: original GRPO stops after one search; SFT searches twice; GRPO fixes an M2 error; a stale-affiliation success; a reverse two-hop failure; an unanswerable question answered without support |
| Bonus | paired M2-before / after-GRPO table with CI; format control; reward-variant ablation; signal diagnostics (share of groups with reward spread) |

Headline sentence to use only after `run_final_analysis.sh` has run: it is the first paragraph of `submission/FINAL_PILOT_REPORT.md`
(generated from the saved comparison; it includes the format-control caveat and the missing SFT-on-hard-data control).
