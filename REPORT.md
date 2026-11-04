# Multi-step retrieval with GRPO: why a small agent stops after one search

Branch `experiment/evidence-aware-grpo`. Every number below is labelled **[local]** (measured on this machine,
reproducible with the listed command, outputs in `outputs/audit/`), **[Colab, reported]** (pasted from the Colab
runs; the output files are not on this machine, so they were not re-verified) or **[pending]** (script written,
not yet run). Nothing here is estimated or illustrative.

## 1. Question
Can GRPO teach Qwen3-0.6B to chain two searches (project -> principal investigator -> the PI's institution), and
if not, is it because it cannot retrieve, cannot answer from evidence, or chooses to stop too early?

## 2. Setup (unchanged from the pilot)
Qwen3-0.6B (non-thinking) + LoRA r=8 on q/v, TRL GRPO, BM25 top-2 over a synthetic corpus, at most 2 searches,
reward `correct - 0.05*searches - 0.10*invalid`. Splits: train/dev/test (one-hop and no-search), `twohop`
(60, already inspected), `twohop_train` (80), `twohop_dev` (30). New sealed splits `final` and `twohop_final`
(100 each, independent seed, disjoint entities, hashes in `data/manifest_sealed.json`) are untouched until FINAL=1.

## 3. Why the first GRPO run collapsed [Colab, reported]
Run v1: test accuracy 37% -> 38%, lookup accuracy 18% -> 0%. Mechanism: the base model answered after a search with
a sentence ("Project X is funded by ...") that failed exact match, so a *correct search* scored 0 - 0.05 while a
guess scored 0. GRPO therefore learned that searching is a net loss and stopped searching. The reward was
working as written; the environment made the intended behaviour unprofitable.

## 4. How the revised environment changed training [Colab, reported]
Run v2 appends one answer-format reminder to the *tool output* (value only, in `<answer>` tags). Putting it in the
system prompt made the base model stop searching, so the placement matters for a 0.6B model. With answers now
matchable, searching pays, and one-hop test accuracy rose 56% -> 98% (seeds 42/43/44: 98/95/97, mean 96.7%, sd 1.5,
paired exact test p about 1e-12, 0 losses). Most of the gain is format and query quality; the base already searched
on 90% of lookups. Caveats: synthetic data, one model, ceiling effect.

## 5. Audit of the two-hop failure
| Check | Result | Source |
|---|---|---|
| Two scripted BM25 queries retrieve both documents | 100% on twohop_dev (30) and twohop (60); the question text alone never retrieves the affiliation document | [local] `python -m src.audit retrieval` |
| Base, sampled at T=0.7, 8 rollouts x 10 dev tasks: second search | 0 / 80; every rollout searched once | [local] `src.probe explore` |
| Same at T=1.0 | 0 / 80 second searches (5 / 80 searched zero times) | [local] |
| Base, one-hop, mixed and evidence-free models, test two-hop | 0 / 60 each; always exactly 1 search, answer = the PI's name (intermediate entity) | [Colab, reported] |
| Base with a second search *forced* (scripted queries), then asked to answer | 5 / 15 correct (33%); the failures answer with a person's name or a sentence | [local] `src.probe forced` |

Failure attribution:
* Retrieval: **not the cause** (100% with two valid queries).
* Tool calling / invalid actions / token budget: **not the cause** (0 invalid, all episodes answered; 256 tokens
  were enough for the single-search episodes that occurred).
* Stopping decision: **the proximate cause**. Every observed episode stops in the *partial evidence* state.
* Answer generation: a secondary limit. Even with the second record supplied, base answers correctly only 33% of
  the time, so a policy that searched twice would still need to learn to read the second record.
* Exploration: the root cause for RL. No sampled rollout ever contains a second search, so under GRPO the group
  advantage never rewards it, whatever the reward values are. Shaping only helps if the shaped behaviour is sampled.

### 5b. Answer-extraction diagnosis [local, results/audit/]
The forced-second-search probe (both searches scripted, the model only answers) was rerun on all 30 `twohop_dev`
tasks with the reminder text of the tool output as the only variable.

| Reminder text | Correct | Failure type (all failures listed) |
|---|---|---|
| default ("a name, organization, city or year") | 10 / 30 (33%) | 19 answer the **researcher's name** (the intermediate entity), 1 a full sentence containing the right institution |
| typed ("...if it asks where someone works, answer with the institution, not a person's name") | 26 / 30 (87%) | 0 wrong-field answers; 3 correct institution with extra words ("the institution is X", "Tr Tritaveth Council"), 1 no answer |

* Wrong field of the second document: **yes, this was the dominant failure.** The affiliation record was ranked first
  in the second search in 30 / 30 cases, so the model saw the right sentence and still copied the person's name; the
  literal word "name" in the default reminder is a plausible trigger and removing it fixes the behaviour.
* Malformed final answer: minor (4 / 30 after the fix; exact-match is strict about extra words).
* Context / token budget loss: not observed (the forced answers are short; no truncation).
* Retrieval: not a factor (100% chain retrieval).
The typed reminder is therefore a *justified* environment fix, but it changes the tool text seen by one-hop tasks, so
it is applied to B/C/D as a config option (`answer_format: typed`), and A and base are evaluated under both texts.
Selected on dev (`twohop_dev`) only; the old `twohop` test set was not used for this.

## 6. Finite-horizon MDP reference
`src/mdp_baseline.py` solves `V(s,k)=max(p_ans(s), -c + sum P(s'|s)V(s',k-1))` over states
{none, partial, complete} with `P` estimated on `twohop_dev` by a scripted BM25 policy. States use gold evidence
ids, so it is an **oracle-state reference**, not a fair competitor to Qwen. With P(S1->S2)=1.0 (scripted second
query) and any answer accuracy in the complete state above the 0.05 search cost, the optimal policy searches in S0
and S1. Qwen answers in S1 in 100% of observed episodes. [pending] the per-model tables from
`run_evidence_study.sh` fill in `p_ans` from each model's own episodes.

## 7. Evidence-aware reward (implemented, tested)
`R_new = R_orig + alpha*first_evidence + beta*complete_chain` (defaults 0.10 / 0.30), in `src/reward_evidence.py`.
Credit only for unique gold documents returned by executed searches; none for no-search tasks; one-hop tasks keep
the original reward unless `onehop_bonus: true`. A potential-difference variant (`mode: potential`,
Phi = 0.4 * fraction of required documents held) is a second controlled condition; because Phi at termination is
not zero it is not strictly policy-invariant in the Ng et al. sense. 12 unit tests cover arithmetic, uniqueness,
irrelevant searches, one-hop invariance, telescoping and absence of gold ids in model-visible text.

## 8. Supervised warm-start (implemented and tested; training not yet run)
Because no sampled rollout contains a second search (0 / 160), a reward cannot create the action. `src/sft_data.py`
builds correct trajectories from `train` and `twohop_train` only (asserted by a test): question -> `search("Project X
principal investigator")` -> real BM25 observation -> `search("<person named in that observation> affiliation")` ->
real observation -> `<answer>institution</answer>`; plus 80 one-hop lookups and 40 direct-answer examples against
forgetting. Observations are produced by running `SearchEnv` (a test recomputes them), so nothing is hand-written.
`src/sft_tokens.py` renders with the training-compatible chat template and labels only the model's own turns
(tool calls, answer, `<|im_end|>`). Tests: each two-hop example has exactly 3 supervised spans; tool, system and user
text is never supervised; the loss equals the mean cross-entropy over supervised tokens; LoRA gradients are
nonzero on a tiny Qwen3. On the real model, supervised tokens are 8% of all tokens (8,178 / 99,818) and a
forward/backward step gives nonzero LoRA gradients (untrained loss about 0.9-1.3).
Honest limitation: the warm-start can learn the question templates ("lead researcher of Project X" -> two
searches) rather than a general rule about incomplete evidence; the dev set uses the same templates with new
entities, so success would show entity generalization, not template generalization.

## 9. Experimental conditions (same budget 384, greedy, 2-search cap, dev tasks)
| | Description | Status |
|---|---|---|
| A | existing one-hop GRPO (main_v2) | [Colab, reported] 98% test; dev re-evaluation under default and typed reminder pending |
| B | SFT warm-start only | code + tests done; **pending Colab**. Gate: >= 50% second-search rate and >= 30% two-hop dev accuracy, else stop and diagnose |
| C | SFT + GRPO, original reward | pending (only after B passes) |
| D | SFT + GRPO, evidence reward | pending (only after B passes) |
| (earlier) mixed GRPO without SFT | [Colab, reported] 0 / 60 two-hop | evidence/potential variants implemented, not run: superseded by the exploration finding |
| MDP | oracle-state Bellman reference | implemented/tested; uses gold evidence state, never compared to Qwen as equal-information |

SFT gains are read from B vs A; GRPO gains from C and D vs B. Local training was not attempted at scale: the Mac was at
97% swap, so the pipeline was validated with tokenization, masks, tiny-model gradients, real-model forward/backward,
and trainer construction from an adapter only.
Commands: `bash run_sft_study.sh` (stops itself if the gate fails); sealed holdouts only with `FINAL=1` and a
`outputs/study_sft/FROZEN` file.

## 10. Limitations
Synthetic one-relation chains; one 0.6B model; single seed for C/D/D2; the two-hop test set was inspected, so final
claims use the sealed split; evidence shaping uses gold ids in the reward only (never shown to the model).

## 11. Reproduce
`python -m pytest -q` (54 tests) · `python -m src.audit retrieval --split twohop_dev` ·
`python -m src.probe explore|forced --checkpoint base --split twohop_dev --output ...` · `bash run_evidence_study.sh`
on a GPU · `FINAL=1 bash run_evidence_study.sh` once, after freezing.
