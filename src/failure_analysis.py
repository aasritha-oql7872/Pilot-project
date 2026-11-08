"""Failure-mode breakdown from saved evaluation traces: reverse two-hop and unanswerable questions (also every other
multi-hop category). Labels are rule-based and mutually exclusive; see trace_facts.classify_*.

    python -m src.failure_analysis --split hard_final --out fail.md --json fail.json "M2=eval_M2.json" "M3=eval_M3.json"

Report only what the tool traces show (queries, retrieved ids, final answer). It makes no claim about hidden reasoning.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from src.trace_facts import classify_twohop, classify_unanswerable, facts, load_tasks

MULTIHOP = {"standard", "distractor_name", "distractor_person", "stale_affiliation", "reverse_two_hop"}


def analyse(path, split):
    rep = json.loads(Path(path).read_text())
    tasks = load_tasks(split)
    out = {}
    for ep in rep["episodes"]:
        f = facts(ep, tasks.get(ep["task_id"]))
        cat = f["category"]
        if cat in MULTIHOP:
            f["label"] = classify_twohop(f)
        elif cat == "unanswerable":
            f["label"] = classify_unanswerable(f)
        else:
            continue
        out.setdefault(cat, []).append(f)
    return out


def summarize(by_cat):
    res = {}
    for cat, rows in by_cat.items():
        n = len(rows)
        res[cat] = {"n": n, "labels": dict(Counter(r["label"] for r in rows)),
                    "mean_searches": sum(r["searches"] for r in rows) / n,
                    "second_search_rate": sum(r["second_search"] for r in rows) / n,
                    "first_hop_found_rate": sum(r["first_hop_found"] for r in rows) / n,
                    "complete_chain_rate": sum(r["complete_chain"] for r in rows) / n,
                    "reversed_order_rate": sum(r["reversed_order"] for r in rows) / n,
                    "second_query_uses_bridge_rate_when_two_searches": (
                        sum(r["second_query_uses_bridge"] for r in rows if r["second_search"]) /
                        max(1, sum(r["second_search"] for r in rows))),
                    "duplicate_query_rate": sum(r["duplicate_query"] for r in rows) / n,
                    "examples": {lab: [r["task_id"] for r in rows if r["label"] == lab][:3] for lab in set(r["label"] for r in rows)}}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entries", nargs="+", help='"label=eval.json"')
    ap.add_argument("--split", required=True, help="split the evaluations ran on (task metadata is read from data/)")
    ap.add_argument("--categories", nargs="*", default=["reverse_two_hop", "unanswerable"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    result, lines = {}, []
    for e in args.entries:
        label, path = e.split("=", 1)
        result[label] = summarize(analyse(path, args.split))
    for cat in args.categories:
        labs = sorted({l for m in result.values() for l in m.get(cat, {}).get("labels", {})})
        if not labs:
            continue
        lines += [f"\n**{cat}** (counts of episodes per observable failure label, split `{args.split}`)\n",
                  "| Model | n | " + " | ".join(labs) + " | searches | 2nd search | chain complete |", "|" + "---|" * (len(labs) + 5)]
        for label, m in result.items():
            c = m.get(cat)
            if c:
                lines.append(f"| {label} | {c['n']} | " + " | ".join(str(c["labels"].get(l, 0)) for l in labs) +
                             f" | {c['mean_searches']:.2f} | {100 * c['second_search_rate']:.0f}% | {100 * c['complete_chain_rate']:.0f}% |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
