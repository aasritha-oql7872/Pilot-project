# Can GRPO Learn Multi-Hop Tool Use? Exploration, Reward Design, and Supervised Warm-Starting in Small LLM Agents

Branch `experiment/evidence-aware-grpo`. CS 496 Agent AI pilot. Every number carries one of four labels:

* **[local]**: measured on this machine; reproducible with the listed command; evidence in `results/audit/`.
* **[Colab, pasted]**: copied from Colab output pasted into the working session. The evaluation JSONs are on Google
  Drive, not in this repository. `python -m src.verify_reported outputs/study_sft` (notebook cell 3) compares these
  numbers with the saved JSONs and prints MATCH/MISMATCH; until it has been run they are *unverified*.
* **[Colab, earlier]**: reported in earlier sessions (v1/v2/seed runs); no JSON re-checked.
* **[pending]**: script written and unit-tested, GPU run not yet done. No number is given for pending items.

## 1. Research question
Can reinforcement learning teach a small LLM agent (Qwen3-0.6B) to make *sequential* retrieval decisions on a multi-hop
question: search the project, read the investigator's name, search again for that person's institution, answer? And
if plain GRPO does not, is the cause the reward, the exploration, or the model's ability to read the evidence?

## 2. MDP formulation
* **Observation / state of the policy:** the question plus the interaction history (earlier searches and the retrieved
  text). The language model never sees the simulator's hidden state (which gold documents it has held).
* **Actions:** `search(query)` (free-text query) or a final answer in `<answer>…</answer>`.
* **Transition:** a deterministic BM25 lookup over the split's corpus returns the top-2 records; the history grows by
  the query and the records.
* **Horizon:** at most 2 searches, then the answer (3 assistant turns); extra searches are invalid actions.
* **Reward:** `R = correct − 0.05·N_search − 0.10·N_invalid` (original). Optional shaping that uses gold evidence ids
  *inside the reward function only* (never in model-visible text): evidence-aware, potential-difference, and
  exploration-incentive variants (Section 3).
* **Policy:** Qwen3-0.6B (non-thinking) with a LoRA adapter (r = 8, q_proj/v_proj, 1,146,880 trainable parameters).
* **Hidden state used by the oracle baseline:** `{S0 none, S1 partial, S2 complete}` evidence held. This is privileged
  information; the Bellman baseline that uses it is an *oracle reference*, not a fair competitor.

## 3. Algorithms
**GRPO.** For each question the trainer samples a group of G rollouts, scores them, and uses
`A_i = (r_i − mean(r)) / std(r)` as the advantage of every token of rollout i, with a clipped policy-gradient update on the
LoRA weights (β = 0, no KL term). If all rewards in a group are equal, `std = 0`, every advantage is 0, and the group
contributes no gradient. This is the formal reason why a missing behaviour cannot be learned: a second search that is
never sampled never appears in the group, so no reward can single it out.

**Reward shaping** (gold evidence ids used only by the verifier):
* evidence-aware: `R_new = R_orig + α·first_evidence + β·complete_chain` (α = 0.10, β = 0.30; one-hop tasks unchanged).
* potential-difference: bonus = Σ_t Φ(s_{t+1}) − Φ(s_t), Φ = 0.4·(fraction of required documents held). Not strictly
  policy-invariant in the Ng et al. sense because Φ at termination is non-zero.
* exploration incentive (`mode: explore`): the evidence bonuses plus `γ = 0.10` for a second search that (i) follows a
  first search that left the chain incomplete and (ii) returns a different result list than the first. Repeated queries
  and one-hop/no-search tasks never receive it. It conditions on the *state before the action*, not on success, so it
  is the one variant that rewards "keep searching" itself. It uses privileged gold evidence ids.

**Exploration and rollout diversity.** Temperature, group size and the exploration incentive change which trajectories
are sampled; only the last changes what is rewarded.

**Supervised warm-start (not RL).** LoRA fine-tuning on correct trajectories built from `train` and `twohop_train`
only, loss on the model's own turns only. It is included as a baseline that supplies the missing behaviour, not as an RL
algorithm.

**Finite-horizon Bellman baseline.** `V(s,k) = max( p_ans(s), −c + Σ_{s'} P(s'|s)·V(s',k−1) )` with `c = 0.05`,
estimated from episodes. Oracle-state reference only.

## 4. Experimental conditions
| ID | Initialization | Training | Why it is needed |
|---|---|---|---|
| base | pretrained Qwen3-0.6B | none | reference |
| A | base | GRPO, one-hop tasks, original reward | the original pilot model |
| B | base | SFT (200 trajectories) | does the missing behaviour appear with supervision? (**frozen headline**) |
| C | B | GRPO, original reward | does RL add anything on top of SFT? |
| D | B | GRPO, evidence-aware reward | does the shaped reward add anything on top of SFT? |
| E0 | early SFT checkpoint (second-search rate 20–70% on dev) | none | a policy that only sometimes searches twice |
| E1 / E2 | E0 | GRPO original / evidence reward, matched seeds and budget | can GRPO improve a partially exploring policy? |
| F0 | base | GRPO original reward, mixed tasks | reproduce the failure, with exploration statistics |
| F1 | base | GRPO, higher temperature and group size (only if sampling shows second searches) | exploration diversity |
| F2 / F2p | base | GRPO evidence / potential reward | reward shaping without supervision |
| F3 | base | GRPO evidence + explicit exploration incentive | reward the continuation itself |
| F4 (optional) | A | GRPO on two-hop tasks only | RL-only curriculum |

Staged design: a 30-step pilot of each F condition; the full 100-step run only if the pilot contains at least one
two-hop rollout with a second search. All decisions are saved (`decision_*.json`) and reported, including negative ones.

Evaluation protocol, matched across B/C/D and E: greedy decoding, 384-token budget shared by generated and tool tokens,
2-search cap, typed answer reminder. Base and A are *also* evaluated under the default reminder, to separate the effect of
the answer-format fix from training.

## 5. Results

### 5.1 Development measurements (dev = 60 one-hop/no-search tasks, twohop_dev = 30 two-hop tasks) [Colab, pasted]
Accuracy is k/n; the interval is a 95% Wilson interval (the earlier bootstrap interval was degenerate at 100% and has
been replaced; it should not be read as certainty of perfect accuracy).

| Condition | Two-hop dev | One-hop dev | Second search (two-hop) | Full chain (two-hop) |
|---|---|---|---|---|
| base, default reminder | 0/30 [0.0%, 11.4%] | 30/60 [37.7%, 62.3%] | 0/30 | 0/30; 43% invalid actions |
| A, default reminder | 0/30 | 56/60 (93.3%) | 0/30 | 0/30 |
| A, typed reminder | 0/30 | 57/60 (95.0%) | 0/30; 77% answer with the investigator's name | 0/30 |
| B SFT only | 30/30 [88.6%, 100%] | 59/60 (98.3%) | 30/30 | 30/30 |
| C SFT + GRPO original | 30/30 | 60/60 | 30/30 | 30/30 |
| D SFT + GRPO evidence | 30/30 | 60/60 | 30/30 | 30/30 |

Compute (peak VRAM / wall time): B 79 s, 19.8 GB; C 448 s, 2.8 GB; D 447 s, 2.8 GB (L4).

Reading the table:
* The one-hop A→B→C/D differences (57, 59, 60 of 60) are within 1–3 tasks and not distinguishable at this n.
* Dev is saturated for B/C/D (30/30): it cannot rank them.
* Selection of the headline model used dev only (`results/frozen_selection.json`).

### 5.2 Sampled behaviour (T = 0.7, 8 samples per task, twohop_dev)
| Model | Second-search rate | Correct | Note |
|---|---|---|---|
| B | 73.8% | 41.3% | [Colab, pasted] |
| C = D | 72.1% | 38.3% | [Colab, pasted]; C and D are identical |

**Defect found and fixed:** `src/probe.py explore` did not pass `--answer-format` to the rollouts, so these three
probes ran with the *default* reminder although B/C/D were trained and evaluated with the typed one. Their absolute values
(especially "correct") are therefore confounded by a reminder mismatch. The probe is fixed (`explore(..., answer_format)`),
the old files are kept, and notebook cell 7 re-runs the three probes into `explore_typed_{B,C,D}.json`
[pending]. The base-model probes (0/80 at T = 0.7 and T = 1.0, default reminder) were unaffected.

### 5.3 Why GRPO did nothing on top of SFT [Colab, pasted]
* 95% of training groups had identical rewards (mean `frac_reward_zero_std` = 0.95); only 10 of 60 steps had a non-zero
  gradient norm.
* The adapter moved by max|B−C| = 2.6·10⁻⁴; C and D have identical weights (max|C−D| = 0), so on the 10 informative steps
  the evidence bonus never separated the sampled rollouts. D's reward was never exercised.
* Interpretation: a policy that is already near-deterministic (entropy ≈ 0.04) gives GRPO almost no variance to exploit.
  This is a statement about *this* warm-start, not about the evidence reward in general (Section 5.5).

### 5.4 Sealed holdout (final: 100 one-hop/no-search, twohop_final: 100 two-hop) [pending]
Headline model frozen as B before any sealed evaluation (`results/frozen_selection.json`, plus a timestamped `FROZEN`
file written by `run_sealed.sh`). `bash run_sealed.sh` evaluates base, A (both reminders), B, C, D with matched settings,
writes `summary_sealed.md` (k/n, Wilson interval, second-search rate, chain retrieval, wrong-field rate, invalid actions,
searches, reward), paired sign tests against B, and a failure audit. No sealed number is quoted here until that has run.
The sealed files are hash-checked against `data/manifest_sealed.json` immediately before the run.

### 5.5 Exploratory development-only experiments [pending]
Partial-SFT (E0/E1/E2): `bash run_early_study.sh`. RL-only (F0–F4): `bash run_rl_only_study.sh`. Both are
development-only; the sealed holdouts are not used for them. For every run the scripts record second-search rate over
training windows, valid vs invalid second searches, distinct documents retrieved, complete-chain rate, reward
distribution, fraction of groups with non-zero reward std (both original and shaped reward, and the fraction of groups
where the bonus *created* variance), gradient norms, weight change from the initial adapter, one-hop regression
(search rate on no-search and lookup tasks) and compute.
Results: **none yet**. Each condition's outcome, including a stopped pilot, will be added with its decision JSON.

### 5.6 Previously reported results [Colab, earlier] (not re-verified)
v1 GRPO reward-hacked into not searching (test 37%→38%, lookup 18%→0%); with the format reminder in the tool output (v2)
one-hop test accuracy 56%→98% (seeds 42/43/44: 98/95/97%); one-hop and mixed (one-hop+two-hop) GRPO models scored 0/60 on the
old two-hop test, always one search.

## 6. Why original GRPO failed
Facts established without the SFT model:
| Check | Result | Source |
|---|---|---|
| Two scripted BM25 queries retrieve both documents | 100% (twohop_dev 30, twohop 60); the question alone never retrieves the affiliation document | [local] `python -m src.audit retrieval` |
| Base, T = 0.7, 8 rollouts × 10 tasks: second search | 0/80, all rollouts search exactly once | [local] |
| Base, T = 1.0 | 0/80 (5/80 searched zero times) | [local] |
| A on twohop_dev | 30/30 stop after one search in the partial-evidence state; 77% answer with the investigator's name | [Colab, pasted] |
| Base with the second search forced, default reminder | 10/30; 19 answer the investigator's name | [local] |
| Same, typed reminder | 26/30; 0 wrong-field; 4 format errors | [local] |

* **Missing second-search trajectories:** root cause for RL. The finding is finite-sample (0/160 under the tested
  configuration); it does not prove the action can never be sampled.
* **Sparse reward:** the second search pays only through a correct final answer, which also needs the answer-extraction
  fix. Without the second record there is no reward difference between rollouts, so groups have zero variance.
* **Tool use:** not the cause (0 invalid actions for A; retrieval complete when scripted).
* **Wrong-field answers:** the default tool-output reminder says "a name, organization, city or year", and the model copied
  the investigator's name. The typed reminder (selected on twohop_dev only) raises forced-search accuracy from 10/30 to
  26/30. This is an environment fix, applied to B/C/D, and A/base are shown under both reminders.
* **Token budget / context loss:** not observed.

## 7. What SFT changed
200 correct trajectories (80 two-hop, 80 one-hop, 40 direct answers; 8% of tokens supervised) took two-hop dev accuracy
from 0/30 to 30/30 and the second-search rate from 0% to 100% in 79 s, without hurting one-hop (59/60). The observations in
the training trajectories are produced by the real environment; gold values appear only as the final answer label; a test
checks that only the model's own turns receive gradient. SFT supplied the *behaviour* GRPO could not discover. Caveat:
the dev tasks use the same question templates as the SFT data with new entities, so this shows entity generalization, not
robustness to rephrased questions; the sealed split is the first check beyond dev, and it uses the same templates too.

## 8. When GRPO helps or fails
* **Measured:** GRPO on top of a saturated SFT policy changed nothing measurable (Section 5.3).
* **Not yet measured:** whether GRPO can improve a *partial* warm-start (E1/E2), and whether any reward or exploration
  intervention lets GRPO discover the second search from the base model (F1–F4). Both are implemented with staged gates
  and will be reported as measured, including a null result.
* The scientific claim is deliberately limited: an unsampled action cannot be reinforced, and an evidence reward only
  matters if its bonus separates sampled trajectories. `src.traj_stats` measures that directly.

## 9. Limitations
Small model (0.6B); synthetic one-relation chains and fixed question templates; one training seed for B/C/D (E has two
seeds, F one); only 30 two-hop dev tasks (a 30/30 result has a 95% Wilson lower bound of 88.6%); BM25 with a small corpus;
gold evidence ids are used as privileged reward supervision in the evidence/explore variants (never shown to the
policy); results are sensitive to the tool-output reminder text and to decoding (greedy vs T = 0.7); the explore probes of
B/C/D were confounded by the reminder defect until re-run; compute is limited to free Colab; the local Mac could not
train (swap ≈ 98% full), so all GPU claims come from Colab output.

## 10. Conclusions (to be completed with measured results)
Established: (1) plain GRPO from the pretrained model, and one-hop or mixed GRPO, never produced a second search in the
sampled rollouts we measured, so the group-relative reward had nothing to reinforce; (2) the model also had a separate
answer-extraction failure that the reminder text fixes; (3) a small supervised warm-start restores two-hop behaviour on
development tasks (30/30), and further GRPO added nothing measurable because the policy was already saturated and
deterministic. Not yet established: sealed-set generalization, whether GRPO helps from a partial warm-start, and whether
any RL-only intervention can discover the second search. Those sections will be filled from the scripts below, and a
negative outcome will be stated as such.

## 11. Reproduce
```
python -m pytest -q                                 # 66 tests
python -m src.audit retrieval --split twohop_dev    # CPU
# Colab (free GPU): open SearchLite_RL_Final_Study.ipynb and run cells in order. Equivalent shell:
python -m src.verify_reported outputs/study_sft     # reported vs saved
bash run_sealed.sh                                  # once
bash run_early_study.sh ; bash run_rl_only_study.sh # development only
python demo.py --a outputs/study_sft/eval_twohop_dev_A.json --b outputs/study_sft/eval_twohop_dev_B.json
```
