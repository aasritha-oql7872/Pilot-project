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
