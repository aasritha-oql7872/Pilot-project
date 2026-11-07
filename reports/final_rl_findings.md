# Final RL findings (current state)

**Scientific status: incomplete.** The diagnosis part is established; the question "does evidence-aware GRPO improve an SFT
agent where SFT has headroom?" has no answer yet, because the hard-benchmark experiments have not been run. This file will be
completed from `outputs/study_hard/*` and must not be read as a positive or negative result for that question.

## Answers to the required questions
**Why did SFT initially outperform GRPO?** From the pretrained model, GRPO never saw a rollout containing a second search
(0/160 sampled base rollouts at T = 0.7 and 1.0 [local]); GRPO's advantage is zero when rewards in a group are equal, so a never-
sampled action is never reinforced. A (original GRPO) therefore stops after one search on 30/30 two-hop dev tasks and answers with
the investigator's name in 77% of them [Colab, pasted]. SFT on 200 demonstrations supplies the missing action (0/30 → 30/30).
A second, independent failure was answer extraction: forced-search accuracy rose from 10/30 to 26/30 when the tool reminder
named the requested type [local]. SFT and the reminder fix are separate effects (A is shown under both reminders).

**Why did the original GRPO runs fail to show improvement?** Two causes, measured separately: (1) no behavioural discovery
(above); (2) after SFT the policy is saturated and nearly deterministic: 95% of C/D training groups had zero reward variance and
only 10/60 steps had a gradient [Colab, pasted].

**Was evidence-aware GRPO genuinely updating the model?** The optimizer updates the adapter (movement 2.6·10⁻⁴ from B; mechanics
tests). It updated it the same way as the original-reward run: the bonus is constant within a group whenever all rollouts retrieve
the same documents, so advantages are unchanged ([test], Phase 1 §2). Whether that explains the exact C == D equality is pending
`audit_lineage` on Drive.

**Did addressing reward variance improve learning?** Not tested yet. Instrumentation, a pre-training signal probe, an alternate
grounded reward and a gated protocol exist (`reports/controlled_experiments.md`); no run has happened.

**Does evidence-aware GRPO outperform SFT on harder, unseen questions?** Unknown (pending). The pre-registered rule decides the
headline; the default remains SFT.

**If not, what are the demonstrated limitations?** Pending the headroom and signal gates. Already demonstrated: dev saturation,
a template-bound benchmark, an environment where the evidence bonus is often group-constant, a reminder-sensitive answer format,
and a probe defect (fixed) that confounded sampled B/C/D numbers.

**What does this establish about RL for tool-using LLM agents?** Only this, so far: for a small model in a lexical-search
environment, RL needs the target behaviour to be sampled and the reward to differ across sampled behaviours; if either fails,
GRPO has no gradient, and supervised initialization is what supplies the behaviour. Whether RL then adds value is the open
experiment.

## Explanation for the professor (60 seconds)
We built a small search agent (Qwen3-0.6B, BM25, at most two searches). GRPO from the base model never discovered the second
search because no sampled attempt ever contained it; GRPO has nothing to compare, so it cannot reinforce it. A separate bug-like
behaviour, copying the researcher's name instead of the institution, was fixed by naming the requested type in the tool output.
A 200-example supervised warm start gave perfect development accuracy, which left GRPO nothing to improve: 95% of its training
groups had identical rewards. We therefore built a harder benchmark with decoys, a stale record that outranks the right one,
reversed questions and unanswerable questions, plus diagnostics that report how much learning signal each reward gives, and a
pre-registered rule for choosing the final model. The comparison of SFT with SFT+GRPO on that benchmark is the next run; until then
the honest conclusion is that SFT, not RL, supplied the capability.
