"""Presentation demo: replay the SAME two-hop question under two evaluated models, from saved eval JSONs (no GPU, no model).

    python demo.py --a outputs/study_sft/eval_twohop_dev_A.json --b outputs/study_sft/eval_twohop_dev_B.json
    python demo.py --a ... --b ... --task twohop_dev_007

Prints Question -> search 1 -> record -> search 2 -> record -> final answer for each model, flags which evidence
documents were retrieved, and whether the answer is correct. Picks the first task where A fails and B succeeds.
"""

import argparse
import json
import textwrap
from pathlib import Path


def load(path):
    rep = json.loads(Path(path).read_text())
    return {e["task_id"]: e for e in rep["episodes"]}, rep


def show(name, ep):
    print(f"\n=== {name} ===")
    for m in ep["messages"]:
        role = m["role"]
        if role == "user":
            print("QUESTION :", m["content"].split("\n")[0] if "\n" in m["content"] else m["content"])
        elif role == "assistant" and m.get("tool_calls"):
            q = m["tool_calls"][0]["function"]["arguments"]
            print("SEARCH   :", q.get("query") if isinstance(q, dict) else q)
        elif role == "tool":
            body = m["content"].split("\n\n")[0]
            print("RECORDS  :", textwrap.shorten(body.replace("\n", " | "), 200))
        elif role == "assistant":
            print("ANSWER   :", ep["final_answer"], "(raw:", textwrap.shorten(m["content"], 80), ")")
    got = {d for ids in ep["retrieved_doc_ids"] for d in ids}
    chain = set(ep["evidence_doc_ids"]) <= got
    print(f"searches={ep['num_searches']}  full evidence chain retrieved={chain}  gold={ep['gold_answer']!r}  "
          f"correct={ep['correct']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="eval JSON of the baseline model (e.g. condition A)")
    parser.add_argument("--b", required=True, help="eval JSON of the improved model (e.g. condition B)")
    parser.add_argument("--task", default=None)
    args = parser.parse_args()
    a, _ = load(args.a); b, _ = load(args.b)
    ids = [args.task] if args.task else [t for t in a if t in b and a[t]["kind"] == "twohop" and not a[t]["correct"] and b[t]["correct"]]
    if not ids:
        raise SystemExit("no task where A fails and B succeeds in these reports")
    tid = ids[0]
    print(f"task: {tid}")
    show("baseline (" + Path(args.a).stem + ")", a[tid])
    show("improved (" + Path(args.b).stem + ")", b[tid])


if __name__ == "__main__":
    main()
