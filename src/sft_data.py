"""Supervised warm-start data: correct agent trajectories built ONLY from training splits.

Every trajectory is produced by actually running SearchEnv, so tool observations are real BM25 outputs (never
written by hand) and are masked out of the loss. The model is trained to emit only its own turns: the search
calls and the final answer. Gold values appear only as the final-answer label, as in any supervised dataset.

    python -m src.sft_data --output outputs/sft_data.jsonl --check
"""

import argparse
import json
import random
from pathlib import Path

from data.generate import DATA_DIR, read_jsonl
from src.audit import person_from
from src.environment import ANSWER_FORMATS, SearchEnv, make_prompt, run_tool_calls
from src.search_tool import Retriever

ALLOWED_SPLITS = {"train", "twohop_train"}  # dev/test/sealed splits must never feed supervised training
ATTR_WORDS = {"pi": "principal investigator", "year": "year started launched", "funder": "funding agency funded",
              "city": "city located based", "affiliation": "affiliation institution"}
Q1 = ["Project {p} principal investigator", "Project {p} lead researcher"]
Q2 = ["{p} affiliation", "{p} affiliated institution where they work"]


def call_msg(query):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": {"query": query}}}]}


def answer_msg(value):
    return {"role": "assistant", "content": f"<answer>{value}</answer>"}


def run_search(env, messages, query):
    messages.append(call_msg(query))
    messages += run_tool_calls(env, messages[-1]["tool_calls"])
    return messages[-1]["content"]


def twohop_example(task, retriever, rng, answer_format):
    env = SearchEnv(retriever, answer_format)
    env.reset(**task)
    msgs = make_prompt(task["question"])
    obs1 = run_search(env, msgs, rng.choice(Q1).format(p=task["subject"]))
    person = person_from(obs1)
    if task["evidence_doc_ids"][0] not in env.retrieved[0] or not person:
        return None
    obs2 = run_search(env, msgs, rng.choice(Q2).format(p=person))
    if task["evidence_doc_ids"][1] not in env.retrieved[1] or task["gold"] not in obs2:
        return None
    msgs.append(answer_msg(task["gold"]))
    return msgs


def lookup_example(task, retriever, rng, answer_format):
    env = SearchEnv(retriever, answer_format)
    env.reset(**task)
    msgs = make_prompt(task["question"])
    subject = task["subject"]
    query = f"{subject} {ATTR_WORDS[task['attribute']]}" if task["attribute"] == "affiliation" \
        else f"Project {subject} {ATTR_WORDS[task['attribute']]}"
    obs = run_search(env, msgs, query)
    if task["evidence_doc_ids"][0] not in env.retrieved[0] or task["gold"] not in obs:
        return None
    msgs.append(answer_msg(task["gold"]))
    return msgs


def direct_example(task):
    return make_prompt(task["question"]) + [answer_msg(task["gold"])]


def build_examples(seed=42, n_lookup=80, n_direct=40, answer_format="typed"):
    rng = random.Random(seed)
    fmt = ANSWER_FORMATS[answer_format]
    out = []
    tasks2 = read_jsonl(DATA_DIR / "twohop_train_tasks.jsonl")
    r2 = Retriever.for_split("twohop_train")
    out += [("twohop", t["task_id"], m) for t in tasks2 if (m := twohop_example(t, r2, rng, fmt))]
    tasks1 = read_jsonl(DATA_DIR / "train_tasks.jsonl")
    r1 = Retriever.for_split("train")
    lookups = [t for t in tasks1 if t["kind"] == "lookup"]
    rng.shuffle(lookups)
    out += [("lookup", t["task_id"], m) for t in lookups[:n_lookup] if (m := lookup_example(t, r1, rng, fmt))]
    directs = [t for t in tasks1 if t["kind"] == "no_search"]
    rng.shuffle(directs)
    out += [("no_search", t["task_id"], direct_example(t)) for t in directs[:n_direct]]
    assert all(tid.split("_")[0] in ALLOWED_SPLITS or tid.startswith("twohop_train") for _, tid, _ in out)
    rng.shuffle(out)
    return [{"kind": k, "task_id": tid, "messages": m} for k, tid, m in out]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--answer-format", default="typed", choices=["default", "typed"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    ex = build_examples(args.seed, answer_format=args.answer_format)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(json.dumps(e) for e in ex) + "\n")
    kinds = {k: sum(e["kind"] == k for e in ex) for k in ("twohop", "lookup", "no_search")}
    print(json.dumps({"examples": len(ex), **kinds}))
    if args.check:
        print(json.dumps(ex[0]["messages"], indent=1)[:1500])


if __name__ == "__main__":
    main()
