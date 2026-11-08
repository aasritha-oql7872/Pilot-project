"""Did GRPO improve the SAME M2 checkpoint? Paired before/after analysis with a format control and a behaviour breakdown.

    python -m src.rl_improvement --study outputs/study_hard --split hard_dev --after M3_s42 M3_s43 M4_s42 ... --out rl.json --md rl.md

Files used (all written by run_headroom_study.sh / run_hard_final.sh):  eval_<split>_M2.json (before, typed_abstain),
eval_<split>_<variant>_s<seed>.json (after) and, if present, eval_<split>_M2_typed.json (M2 under the reminder it was
trained with; the format control).

  RL improvement = accuracy(after GRPO) - accuracy(M2), same tasks, same decoding, same reminder, paired.
  Accuracy decomposition (answerable multi-hop episodes):  P(correct) = P(complete evidence chain) * P(correct | chain).
  A gain concentrated in the first factor is a search-behaviour gain; a gain in the second is answer extraction/format.

What this can and cannot show: the GRPO models also trained on hard_train and ran a reminder (typed_abstain) that M2 never
trained with. There is no SFT-on-hard_train control, so a gain is "GRPO on this distribution helps", not "RL beats SFT
on equal data". The format control separates the reminder effect from the training effect only partly.
"""

import argparse
import json
from math import comb
from pathlib import Path

from src.report_categories import paired_boot, sign_p
from src.summarize_study import wilson
from src.trace_facts import facts, load_tasks

MULTI = {"standard", "distractor_name", "distractor_person", "stale_affiliation", "reverse_two_hop", "twohop"}


def episodes(path):
    rep = json.loads(Path(path).read_text())
    return {e["task_id"]: e for e in rep["episodes"]}


def paired(before, after):
    ids = sorted(set(before) & set(after))
    a = [float(after[i]["correct"]) for i in ids]
    b = [float(before[i]["correct"]) for i in ids]
    w, l = sum(x > y for x, y in zip(a, b)), sum(x < y for x, y in zip(a, b))
    lo, hi = paired_boot(a, b) if ids else (0, 0)
    k0, k1 = int(sum(b)), int(sum(a))
    return {"n": len(ids), "before": f"{k0}/{len(ids)}", "after": f"{k1}/{len(ids)}", "gained": w, "lost": l,
            "diff_pts": 100 * (sum(a) - sum(b)) / max(1, len(ids)), "ci95_pts": [100 * lo, 100 * hi],
            "mcnemar_exact_p": sign_p(w, l), "before_wilson": wilson(k0, len(ids)), "after_wilson": wilson(k1, len(ids))}


def behaviour(eps, split):
    tasks = load_tasks(split) if split.startswith("hard") or split == "twohop_dev" else {}
    rows = [facts(e, tasks.get(e["task_id"])) for e in eps.values()]
    multi = [r for r in rows if (r["category"] in MULTI) or (r["kind"] == "twohop")]
    if not multi:
        return None
    n = len(multi)
    chain = [r for r in multi if r["complete_chain"]]
    return {"n_multihop": n, "accuracy": sum(r["correct"] for r in multi) / n,
            "second_search_rate": sum(r["second_search"] for r in multi) / n,
            "premature_stop_rate": sum(r["searches"] < 2 and r["answered"] for r in multi) / n,
            "first_hop_found": sum(r["first_hop_found"] for r in multi) / n,
            "complete_chain": len(chain) / n,
            "correct_given_complete_chain": (sum(r["correct"] for r in chain) / len(chain)) if chain else None,
            "unparseable_answer_rate": sum(not r["answered"] for r in multi) / n,
            "invalid_rate": sum(r["invalid"] for r in multi) / n,
            "mean_searches": sum(r["searches"] for r in multi) / n,
            "mean_reward": sum(r["reward"] for r in multi) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="outputs/study_hard")
    ap.add_argument("--split", required=True)
    ap.add_argument("--after", nargs="+", required=True, help="labels like M3_s42")
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    S = Path(args.study)
    before = episodes(S / f"eval_{args.split}_M2.json")
    res = {"split": args.split, "before": "M2 (SFT only, typed_abstain)", "comparisons": {}, "behaviour": {"M2": behaviour(before, args.split)}}
    ctl = S / f"eval_{args.split}_M2_typed.json"
    if ctl.exists():
        res["format_control"] = {"M2 typed vs M2 typed_abstain": paired(before, episodes(ctl)),
                                 "meaning": "diff = accuracy(M2 under the reminder it was trained with) - accuracy(M2 under typed_abstain)"}
        res["behaviour"]["M2_typed"] = behaviour(episodes(ctl), args.split)
    for lab in args.after:
        p = S / f"eval_{args.split}_{lab}.json"
        if not p.exists():
            res["comparisons"][lab] = "missing"
            continue
        after = episodes(p)
        res["comparisons"][lab] = paired(before, after)
        if ctl.exists():
            res["comparisons"][lab + " vs M2_typed"] = paired(episodes(ctl), after)
        res["behaviour"][lab] = behaviour(after, args.split)
    lines = [f"**RL improvement on `{args.split}`** = accuracy(after GRPO) - accuracy(M2), paired by task (exact McNemar/sign test; paired bootstrap CI)\n",
             "| Comparison | before | after | gained / lost | difference [95% CI] | p |", "|---|---|---|---|---|---|"]
    for lab, c in res["comparisons"].items():
        if c == "missing":
            lines.append(f"| {lab} | missing | | | | |")
            continue
        base = "M2 (typed reminder)" if lab.endswith("vs M2_typed") else "M2"
        lines.append(f"| {lab.replace(' vs M2_typed', '')} vs {base} | {c['before']} | {c['after']} | {c['gained']} / {c['lost']} | "
                     f"{c['diff_pts']:+.1f} pts [{c['ci95_pts'][0]:+.1f}, {c['ci95_pts'][1]:+.1f}] | {c['mcnemar_exact_p']:.3g} |")
    if "format_control" in res:
        c = res["format_control"]["M2 typed vs M2 typed_abstain"]
        lines.append(f"\nFormat control: M2 itself scores {c['after']} under the `typed` reminder vs {c['before']} under `typed_abstain` "
                     f"({c['diff_pts']:+.1f} pts, p={c['mcnemar_exact_p']:.3g}). That part of any 'GRPO gain' is a reminder effect, not learning.")
    keys = ["accuracy", "second_search_rate", "premature_stop_rate", "first_hop_found", "complete_chain",
            "correct_given_complete_chain", "unparseable_answer_rate", "invalid_rate", "mean_searches", "mean_reward"]
    lines += ["\n**Behaviour on multi-hop episodes** (observable tool-use; P(correct) = P(chain) x P(correct | chain))\n",
              "| Model | n | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 2)]
    f = lambda v: "–" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    for lab, b in res["behaviour"].items():
        if b:
            lines.append(f"| {lab} | {b['n_multihop']} | " + " | ".join(f(b[k]) for k in keys) + " |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
    if args.md:
        Path(args.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
