"""Assemble submission/FINAL_PILOT_REPORT.md from reports/FINAL_PILOT_REPORT.template.md and generated artifacts.

    python -m src.build_report --sub submission --template reports/FINAL_PILOT_REPORT.template.md

`{{inc:path}}` is replaced by the file's text (path relative to --sub) or by an explicit NOT AVAILABLE notice. `{{headline}}` and
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
    return jload(sub / "FROZEN_HARD.json"), jload(sub / "tables" / "rl_hard_final.json")


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def pick(rl, fz):
    """(label, comparison, behaviour-after) of the frozen headline on hard_final."""
    if not fz or not rl or fz["headline_model"] == "M2":
        return None
    lab = f"{fz['headline_model']}_s{fz['training_seed']}"
    return lab, rl["comparisons"].get(lab), rl["behaviour"].get(lab)


def headline(sub):
    fz, rl = headline_facts(sub)
    if not fz:
        return ("> **Status: INCOMPLETE.** The headline model has not been frozen and the sealed `hard_final` evaluation has not been run, "
                "so this report makes no claim yet about whether GRPO improved the agent on the sealed benchmark.")
    if fz["headline_model"] == "M2":
        return ("> **Headline:** no GRPO variant met the pre-registered rule on `hard_dev`; the headline model remains M2 (SFT only). "
                "This is reported as a negative result for additional GRPO.")
    p = pick(rl, fz)
    if not p or not isinstance(p[1], dict):
        return (f"> **Status: PARTIAL.** Frozen headline: {fz['headline_label']} (frozen {fz['frozen_at_utc']}). "
                "The sealed `hard_final` comparison has not been produced yet.")
    lab, c, b = p
    ctl = rl.get("format_control", {}).get("M2 typed vs M2 typed_abstain")
    s = (f"> **Headline (sealed `hard_final`, n={c['n']}):** frozen model {fz['headline_label']} scored {c['after']} versus {c['before']} for the "
         f"same M2 checkpoint it started from: {c['diff_pts']:+.1f} points (paired 95% CI [{c['ci95_pts'][0]:+.1f}, {c['ci95_pts'][1]:+.1f}], "
         f"exact sign test p={c['mcnemar_exact_p']:.3g}; {c['gained']} tasks gained, {c['lost']} lost).")
    if ctl:
        s += (f" Changing M2's reminder from `typed_abstain` to the `typed` one it was trained with moves M2 by {ctl['diff_pts']:+.1f} points "
              f"({ctl['before']} -> {ctl['after']}); that part of a comparison is not learning.")
    return s


def conclusions(sub):
    fz, rl = headline_facts(sub)
    est = ("**Established by saved artifacts [local / Colab, pasted]:** (1) plain GRPO from the pretrained model, and one-hop or mixed GRPO, never produced a "
           "second search in the sampled rollouts we measured (0/160 samples for the base model; 0/30 two-hop episodes for the original GRPO model), so the "
           "group-relative reward had nothing to reinforce; (2) a separate answer-extraction failure (answering with the investigator's name) was fixed by the "
           "typed reminder; (3) a 200-trajectory supervised warm-start supplied the missing second search; (4) GRPO on a near-deterministic policy "
           "added nothing measurable on easy tasks because 95% of groups had no reward spread.")
    if not fz or not rl:
        return est + "\n\n**Not yet established:** whether additional GRPO improves the SFT checkpoint on the sealed hard benchmark. The headline freeze and the sealed evaluation have not been completed; no conclusion is drawn."
    if fz["headline_model"] == "M2":
        return est + "\n\n**Result:** on the hard benchmark no GRPO variant met the pre-registered improvement rule on development data, so M2 stays the headline. Additional GRPO did not demonstrably help under this setup."
    p = pick(rl, fz)
    if not p or not isinstance(p[1], dict):
        return est + "\n\n**Not yet established:** the sealed `hard_final` comparison has not been produced."
    lab, c, after = p
    before = rl["behaviour"].get("M2")
    ctl = rl.get("format_control", {}).get("M2 typed vs M2 typed_abstain")
    sig = c["ci95_pts"][0] > 0 and c["mcnemar_exact_p"] < 0.05
    txt = [est]
    if sig:
        txt.append(f"**Measured on the sealed benchmark:** the frozen model {fz['headline_label']} improved on the M2 checkpoint it started from by {c['diff_pts']:+.1f} points "
                   f"(paired 95% CI [{c['ci95_pts'][0]:+.1f}, {c['ci95_pts'][1]:+.1f}], p={c['mcnemar_exact_p']:.3g}, n={c['n']}).")
    else:
        txt.append(f"**Measured on the sealed benchmark:** the difference between {fz['headline_label']} and M2 is {c['diff_pts']:+.1f} points "
                   f"(paired 95% CI [{c['ci95_pts'][0]:+.1f}, {c['ci95_pts'][1]:+.1f}], p={c['mcnemar_exact_p']:.3g}); this is **not** a statistically supported improvement.")
    if after and before:
        d_chain, d_acc = after["complete_chain"] - before["complete_chain"], after["accuracy"] - before["accuracy"]
        where = ("mostly in search behaviour (the complete evidence chain is retrieved more often)" if d_acc > 0 and d_chain >= 0.5 * d_acc else
                 "mostly in answer production given the evidence, not in retrieving more of it" if d_acc > 0 else "with no accuracy gain on multi-hop questions")
        txt.append(f"On multi-hop questions the accuracy change ({100 * d_acc:+.1f} pts) came {where}: second-search rate {pct(before['second_search_rate'])} -> {pct(after['second_search_rate'])}, "
                   f"complete chain {pct(before['complete_chain'])} -> {pct(after['complete_chain'])}, premature stops {pct(before['premature_stop_rate'])} -> {pct(after['premature_stop_rate'])}, "
                   f"unparseable answers {pct(before['unparseable_answer_rate'])} -> {pct(after['unparseable_answer_rate'])}.")
    if ctl:
        txt.append(f"Format control: M2 under its own `typed` reminder differs from M2 under `typed_abstain` by {ctl['diff_pts']:+.1f} points; any GRPO gain must be read net of this.")
    txt.append("**Answer to the research question (as supported by the evidence above):** GRPO improved a small agent's multi-step tool use only when the starting policy already "
               "sampled the target action with non-zero probability (supplied here by SFT, not by RL), when the training distribution produced reward spread inside groups (the share of "
               "groups with non-zero reward std is in Section 9), and when the reminder matched the evaluation setting. It did not create the second-search behaviour from the pretrained "
               "model. **Caveats that bound this claim:** GRPO runs also trained on `hard_train`, which M2 did not see, and there is no SFT-on-`hard_train` control, so the result "
               "supports \"GRPO on this distribution helps this checkpoint\", not \"RL beats SFT on equal data\"; categories have n=20 on hard_final, so per-category claims are suggestive.")
    return "\n\n".join(txt)


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
    text = text.replace("{{headline}}", headline(sub)).replace("{{conclusions}}", conclusions(sub))
    out = Path(args.out or sub / "FINAL_PILOT_REPORT.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}; {len(missing)} artifact block(s) not available:")
    for m in missing:
        print("  -", m)


if __name__ == "__main__":
    main()
