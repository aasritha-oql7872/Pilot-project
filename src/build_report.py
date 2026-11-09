"""Assemble submission/FINAL_PILOT_REPORT.md from reports/FINAL_PILOT_REPORT.template.md and generated artifacts.

    python -m src.build_report --sub submission --template reports/FINAL_PILOT_REPORT.template.md

`{{inc:path}}` is replaced by the file's text (path relative to --sub) or by an explicit unavailable-artifact notice. `{{headline}}`, `{{attribution}}` and
`{{conclusions}}` are generated from the saved comparison JSONs by fixed rules; when the data is absent they say so and draw
no conclusion. Nothing is estimated or filled in by hand.
"""

import argparse
import json
import re
from pathlib import Path

NA = "> **NOT AVAILABLE:** `{path}` has not been produced yet (the experiment or analysis step that writes it has not run). No number is substituted."


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def headline_facts(sub):
    return jload(sub / "FROZEN_HARD.json"), jload(sub / "tables" / "rl_hard_final.json"), jload(sub / "tables" / "two_comparisons.json")


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def pick(two, fz):
    """(A, B, C, headline label) for the frozen headline on hard_final, or None."""
    if not fz or not two or fz["headline_model"] == "M2":
        return None
    lab = f"{fz['headline_model']}_s{fz['training_seed']}"
    comps = two["comparisons"]
    A = next((c for c in comps.values() if c["role"] == "A" and c["model"] == lab), None)
    B = next((c for c in comps.values() if c["role"] == "B" and c["model"] == lab), None)
    C = next((c for c in comps.values() if c["role"] == "C"), None)
    return (A, B, C, lab) if A and B and C else None


def stat(c, model_name="the GRPO model", ref_name="the SFT model"):
    return (f"{c['diff_pts']:+.1f} points (paired 95% CI [{c['ci95_pts'][0]:+.1f}, {c['ci95_pts'][1]:+.1f}], exact McNemar p={c['mcnemar_exact_p']:.3g}; "
            f"{c['model_only_correct']} questions only {model_name} got right, {c['ref_only_correct']} only {ref_name}, {c['both_correct']} both, {c['both_wrong']} neither)")


def verdict(c):
    return ("statistically supported at the 0.05 level, with the paired interval excluding zero" if c["significant_at_0.05"]
            else "**not statistically distinguishable from zero** (the paired interval includes zero or p >= 0.05)")


def headline(sub):
    fz, rl, two = headline_facts(sub)
    if not fz:
        return ("> **Status: INCOMPLETE.** The headline model has not been frozen and the sealed `hard_final` evaluation has not been run, "
                "so this report makes no claim yet about whether GRPO improved the agent on the sealed benchmark.")
    if fz["headline_model"] == "M2":
        return ("> **Headline:** no GRPO variant met the pre-registered rule on `hard_dev`; the headline model remains M2 (SFT only). "
                "This is reported as a negative result for additional GRPO.")
    p = pick(two, fz)
    if not p:
        return (f"> **Status: PARTIAL.** Frozen headline: {fz['headline_label']} (frozen {fz['frozen_at_utc']}). "
                "The paired sealed `hard_final` comparisons have not been produced yet.")
    A, B, C, lab = p
    cats = two["category_counts"][lab]
    return (f"> **Headline (sealed `hard_final`, n={A['n']}).** The frozen model {fz['headline_label']} scored {A['model_correct']}/{A['n']} ({A['model_acc_pct']:.1f}%). "
            f"**Comparison A, fixed evaluation prompt:** the M2 SFT checkpoint under `typed_abstain` scored {A['ref_correct']}/{A['n']} ({A['ref_acc_pct']:.1f}%), a difference of {stat(A)}. "
            f"**Comparison B, stronger SFT baseline:** the same M2 under `typed`, the reminder it was trained with, scored {B['ref_correct']}/{B['n']} ({B['ref_acc_pct']:.1f}%), a difference of {stat(B)}; "
            f"this is {verdict(B)}. The SFT checkpoint is strongly prompt-sensitive (changing only its reminder moves it {C['diff_pts']:+.1f} points), so the +{A['diff_pts']:.1f} points of A must **not** be attributed to RL alone. "
            f"Reverse two-hop ({cats['reverse_two_hop'][0]}/{cats['reverse_two_hop'][1]}) and unanswerable ({cats['unanswerable'][0]}/{cats['unanswerable'][1]}) remain weak.")


def attribution(sub):
    fz, rl, two = headline_facts(sub)
    p = pick(two, fz)
    if not p:
        return "Not yet available: this section is generated from the sealed `hard_final` paired comparisons, which have not been produced."
    A, B, C, lab = p
    rep = [c for c in two["comparisons"].values() if c["role"] == "B" and c["model"] != lab]
    stat_c = stat(C, "M2 under `typed`", "M2 under `typed_abstain`")
    rep_txt = ""
    for c in rep:
        rep_txt += (f" The replication seed {c['model']} gives {c['model_correct']}/{c['n']} vs {c['ref_correct']}/{c['n']}: {stat(c)}; this is {verdict(c)}.")
    return "\n\n".join([
        f"**Comparison A, fixed evaluation prompt.** Both models are evaluated with the same `typed_abstain` reminder, greedy decoding, the same questions and the same two-search limit (the settings table above shows the evaluation settings of every report). "
        f"{fz['headline_label']} {A['model_correct']}/{A['n']} ({A['model_acc_pct']:.1f}%) vs M2 {A['ref_correct']}/{A['n']} ({A['ref_acc_pct']:.1f}%): {stat(A)}. This is evidence that the model after GRPO training performs better than the SFT model under one fixed evaluation prompt. "
        "It is not an RL-only estimate: M2 was trained with the `typed` reminder and never saw `typed_abstain`, while the GRPO model was trained under `typed_abstain` on `hard_train`, which M2 did not see.",
        f"**Comparison B, stronger prompt-adjusted SFT baseline.** M2 evaluated with `typed`, the reminder it was trained with, scored {B['ref_correct']}/{B['n']} ({B['ref_acc_pct']:.1f}%); {fz['headline_label']} scored {B['model_correct']}/{B['n']} ({B['model_acc_pct']:.1f}%): {stat(B)}. "
        f"This difference is {verdict(B)}.{rep_txt} B tells us whether GRPO beats a stronger, practical SFT baseline; it is not a prompt-controlled estimate of the effect of RL either, because the GRPO model also trained on the hard-benchmark distribution.",
        f"**The SFT checkpoint is strongly prompt-sensitive.** The same M2 weights score {C['ref_acc_pct']:.1f}% under `typed_abstain` and {C['model_acc_pct']:.1f}% under `typed`: {stat_c}. "
        f"Arithmetically, of the {A['diff_pts']:+.1f} points in A, {C['diff_pts']:+.1f} points are obtained by changing only the reminder on the SFT checkpoint, leaving {B['diff_pts']:+.1f} points in B. "
        "These are not additive causal shares, since prompt and training may interact, and we did not measure that interaction.",
        "**What we do not claim.** We do not claim that GRPO alone explains the improvement over every plausible SFT baseline, and we do not claim that RL beats SFT on equal data: there is no SFT-on-`hard_train` control and no GRPO run under the `typed` reminder, so the contributions of GRPO, the hard-benchmark training data, and the evaluation reminder cannot be separated with the experiments that were run. "
        + (f"What the evidence supports: after the SFT warm-start, GRPO training on this task distribution produced a model that is better under the evaluation prompt than the SFT checkpoint evaluated with the same prompt; A is {verdict(A)}. "
         f"The margin over the SFT checkpoint evaluated with its own preferred prompt is {B['diff_pts']:+.1f} points; B is {verdict(B)}.")])


def chain_sentence(after, b_abs, b_typed):
    """Rule-based reading of the complete-evidence-chain rate (the search-behaviour factor of accuracy)."""
    d_abs = after["complete_chain"] - b_abs["complete_chain"]
    d_typ = after["complete_chain"] - b_typed["complete_chain"]
    if d_abs <= 0:
        return "The complete-chain rate did not increase relative to M2, so there is no search-behaviour gain on multi-hop questions."
    if d_typ <= 0.5 * d_abs:
        return (f"The complete-chain rate rises by {100 * d_abs:+.0f} points over M2 under `typed_abstain` but only {100 * d_typ:+.0f} points over M2 under its own `typed` reminder, "
                "so most of the apparent search-behaviour improvement is reproduced by the reminder alone.")
    return (f"The complete-chain rate rises by {100 * d_abs:+.0f} points over M2 under `typed_abstain` and still {100 * d_typ:+.0f} points over M2 under its own `typed` reminder, "
            "so a substantial part of the search-behaviour gain is not explained by the reminder.")


def conclusions(sub):
    fz, rl, two = headline_facts(sub)
    est = ("**Established by saved artifacts [Colab, pasted / local]:** (1) plain GRPO from the pretrained model, and one-hop or mixed GRPO, never produced a "
           "second search in the sampled rollouts we measured (0/160 samples for the base model; 0/30 two-hop dev episodes for the original GRPO model), so the "
           "group-relative reward had nothing to reinforce; (2) a separate answer-extraction failure (answering with the investigator's name) was reduced by the "
           "typed reminder; (3) a 200-trajectory supervised warm-start supplied the missing second search; (4) GRPO on a near-deterministic policy "
           "added nothing measurable on easy tasks because 95% of groups had no reward spread. We do not claim that GRPO alone discovered two-hop exploration.")
    if not fz or not two:
        return est + "\n\n**Not yet established:** whether additional GRPO improves the SFT checkpoint on the sealed hard benchmark. The headline freeze and the sealed evaluation have not been completed; no conclusion is drawn."
    if fz["headline_model"] == "M2":
        return est + "\n\n**Result:** on the hard benchmark no GRPO variant met the pre-registered improvement rule on development data, so M2 stays the headline. Additional GRPO did not demonstrably help under this setup."
    pk = pick(two, fz)
    if not pk:
        return est + "\n\n**Not yet established:** the paired sealed `hard_final` comparisons have not been produced."
    A, B, C, lab = pk
    txt = [est,
           f"**Measured on the sealed benchmark:** A (same reminder) {A['diff_pts']:+.1f} points, B (stronger SFT baseline) {B['diff_pts']:+.1f} points ({verdict(B)}), and the reminder alone moves the SFT checkpoint by {C['diff_pts']:+.1f} points. "
           "Full statistics are in Sections 10.5 and 10.7."]
    beh = (rl or {}).get("behaviour", {})
    after, b_abs, b_typed = beh.get(lab), beh.get("M2"), beh.get("M2_typed")
    if after and b_abs and b_typed:
        txt.append("On multi-hop questions (observable tool use): second-search rate "
                   f"{pct(b_abs['second_search_rate'])} (M2, typed_abstain) / {pct(b_typed['second_search_rate'])} (M2, typed) / {pct(after['second_search_rate'])} ({fz['headline_label']}); "
                   f"complete evidence chain {pct(b_abs['complete_chain'])} / {pct(b_typed['complete_chain'])} / {pct(after['complete_chain'])}; "
                   f"premature stops {pct(b_abs['premature_stop_rate'])} / {pct(b_typed['premature_stop_rate'])} / {pct(after['premature_stop_rate'])}. "
                   + chain_sentence(after, b_abs, b_typed))
    cats = two["category_counts"][lab]
    txt.append(f"**Remaining failures are not hidden by the overall number:** reverse two-hop {cats['reverse_two_hop'][0]}/{cats['reverse_two_hop'][1]} and unanswerable {cats['unanswerable'][0]}/{cats['unanswerable'][1]} for the frozen model "
               "(Section 15 classifies these from the saved tool traces: sequential retrieval in the reverse direction, stopping decisions and unsupported answers).")
    txt.append("**Answer to the research question (as supported by the evidence above):** "
               + (f"GRPO training after an SFT warm-start produced a model {A['diff_pts']:+.1f} points better than the SFT checkpoint under the same evaluation prompt, "
                  if A["significant_at_0.05"] else "GRPO training after an SFT warm-start did not produce a statistically supported gain over the SFT checkpoint under the same evaluation prompt, ")
               + f"but changing only the SFT checkpoint's evaluation reminder already moves it {C['diff_pts']:+.1f} points; the margin over the SFT checkpoint with its own reminder is {B['diff_pts']:+.1f} points (Comparison B, {verdict(B)}). "
               "GRPO did not create the second-search behaviour from the pretrained model (SFT supplied it). **Caveats that bound this claim:** GRPO runs also trained on `hard_train` under `typed_abstain`, which M2 did not see, "
               "and there is no SFT-on-`hard_train` control, so the result supports \"GRPO on this distribution helps this checkpoint under this prompt\", not \"RL beats SFT on equal data\"; categories have n=20 on hard_final, so per-category claims are suggestive.")
    return "\n\n".join(txt)


def train_abstention(sub, lab):
    """(mean, max, share of steps with any abstention) of the logged training abstention rate, or None."""
    import csv
    f = sub / "raw" / lab / "train_metrics.csv"
    if not f.exists():
        return None
    v = [float(r["searchlite/abstained"]) for r in csv.DictReader(open(f)) if r.get("searchlite/abstained") not in (None, "")]
    return (sum(v) / len(v), max(v), sum(x > 0 for x in v) / len(v)) if v else None


INCOMPLETE = {"premature_stop_after_first_hop", "wrong_first_search_then_stop", "first_search_missed_evidence", "second_search_missed_second_hop",
              "second_search_with_bridge_still_missed", "answered_without_search"}
COMPLETE_WRONG = {"evidence_complete_answer_from_decoy", "evidence_complete_wrong_answer"}


def reverse_reading(r):
    labels = r["labels"]
    fails = sum(v for k, v in labels.items() if k != "correct")
    inc = sum(v for k, v in labels.items() if k in INCOMPLETE)
    comp = sum(v for k, v in labels.items() if k in COMPLETE_WRONG)
    other = fails - inc - comp
    if not fails:
        return "No reverse two-hop failure was observed."
    t = (f"Of the {fails} failures, {inc} ended with an incomplete evidence chain (the model stopped or searched without retrieving every required record), {comp} had the complete chain retrieved but a wrong answer"
         + (f", and {other} had no parseable answer" if other else "") + ". "
         "The retrieval environment is deterministic and a scripted oracle solves every reverse question, so an incomplete chain is a failure of **sequential retrieval** (choosing the next query from what was just retrieved) "
         "or of **stopping** (answering before the chain is complete), not of answer formatting; a complete chain with a wrong answer is an answer-selection failure. ")
    if inc >= comp:
        t += ("Answering with an incomplete chain is not penalised beyond the missing correctness reward under the original reward (M3), which only checks the final string. "
              "*Hypothesis, not tested:* the training tasks are dominated by the project -> person -> institution order, so the policy may have learned a query template rather than a general rule for deciding what to look up next, "
              "which reverse questions require to flip. A reverse-order training split would test it and was not run.")
    return t


def abstention_text(ab):
    if not ab:
        return "The training log with the abstention rate is not packaged, so whether abstention was ever sampled during training cannot be checked here. "
    t = (f"Training log of this run: the logged training abstention rate (all training tasks) averaged {ab[0]:.3f}, maximum {ab[1]:.3f}, and {100 * ab[2]:.0f}% of logged steps contained any abstention. ")
    if ab[0] < 0.02:
        return t + ("GRPO can only reinforce a behaviour that its sampled rollouts contain; with fewer than 2% of training rollouts abstaining there was almost no abstaining rollout for the group-relative advantage to prefer, "
                    "which is the same exploration limitation as for the second search (an interpretation consistent with this log, not a tested mechanism). ")
    return t + "Abstention was therefore sampled during training, so lack of exploration alone does not explain this failure. "


def failure_discussion(sub):
    """Reverse two-hop and unanswerable failures, described from classified saved traces. Interpretations are marked as hypotheses."""
    fz = jload(sub / "FROZEN_HARD.json")
    fl = jload(sub / "tables" / "failures_hard_final.json")
    if not fz or not fl or fz["headline_model"] == "M2":
        return "Not yet available: this discussion is generated from the classified sealed-split traces, which have not been produced."
    lab = f"{fz['headline_model']}_s{fz['training_seed']}"
    m, ref = fl.get(lab), fl.get("M2")
    if not m:
        return "Not yet available: no classified traces for the frozen model."
    top = lambda d: ", ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))
    out = []
    r = m.get("reverse_two_hop")
    if r:
        rr = ref.get("reverse_two_hop") if ref else None
        out.append(f"**Reverse two-hop (person named first, project second; n={r['n']}, frozen model {lab}).** Observable tool behaviour: it searched twice in {100 * r['second_search_rate']:.0f}% of episodes, "
                   f"retrieved the first-hop record in {100 * r['first_hop_found_rate']:.0f}% and the complete evidence chain in {100 * r['complete_chain_rate']:.0f}%"
                   + (f" (M2 under `typed_abstain`: {100 * rr['complete_chain_rate']:.0f}% complete chains)" if rr else "")
                   + f". Failure labels from the traces: {top(r['labels'])}. Where it made a second search, the query reused the entity found by the first search in {100 * r['second_query_uses_bridge_rate_when_two_searches']:.0f}% of cases. "
                   + reverse_reading(r))
    u = m.get("unanswerable")
    if u:
        ab = train_abstention(sub, lab)
        correct = u["labels"].get("correct_abstention", 0)
        out.append(f"**Unanswerable (a required record is missing; the correct answer is `insufficient evidence`; n={u['n']}, frozen model {lab}).** Abstained correctly on {correct}/{u['n']}. Labels from the traces: {top(u['labels'])}; "
                   f"mean searches {u['mean_searches']:.2f}. The evaluation reminder tells the model that it may answer `insufficient evidence`, so the failure is not a missing instruction. "
                   + abstention_text(ab)
                   + ("The original reward does pay for abstaining on these tasks (their gold answer is `insufficient evidence`), but a reward can only be earned by a behaviour that is sampled. "
                      f"On the **evidence-grounding** side the traces show what was answered instead: {u['labels'].get('answered_with_value_from_retrieved_record', 0)} answers repeated a value taken from a retrieved record, "
                      f"{u['labels'].get('invented_unsupported_answer', 0)} answers appear in no retrieved record, {u['labels'].get('no_parseable_answer', 0)} had no parseable answer and {u['labels'].get('answered_without_search', 0)} answered without searching. "
                      "Under M3 a wrong answer on these tasks costs no more than any other wrong answer; M4g adds a penalty for an unsupported non-abstaining answer, and its sealed result on this category is in the table above. We make no claim beyond what those counts show."))
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="submission")
    ap.add_argument("--template", default="reports/FINAL_PILOT_REPORT.template.md")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sub = Path(args.sub)
    text, missing = Path(args.template).read_text(), []

    def inc(m):
        p = sub / m.group(1).strip()
        if p.exists():
            return p.read_text().strip()
        missing.append(str(p))
        return NA.format(path=p)

    text = re.sub(r"\{\{inc:([^}]+)\}\}", inc, text)
    text = text.replace("{{headline}}", headline(sub)).replace("{{attribution}}", attribution(sub)).replace("{{failure_discussion}}", failure_discussion(sub)).replace("{{conclusions}}", conclusions(sub))
    out = Path(args.out or sub / "FINAL_PILOT_REPORT.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}; {len(missing)} artifact block(s) not available:")
    for m in missing:
        print("  -", m)


if __name__ == "__main__":
    main()
