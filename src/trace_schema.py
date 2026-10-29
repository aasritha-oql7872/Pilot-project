"""One JSONL line per finished episode, written during training and evaluation."""

import json

REQUIRED_FIELDS = ["run_id", "task_id", "split", "kind", "model", "step", "seed", "messages",
                   "retrieved_doc_ids", "final_answer", "gold_answer", "correct",
                   "num_searches", "num_invalid", "reward"]


def make_trace(run_id, task, model, step, seed, prompt, completion, env, result):
    return {"run_id": run_id, "task_id": task["task_id"], "split": task["split"], "kind": task["kind"],
            "model": model, "step": step, "seed": seed, "messages": prompt + completion,
            "evidence_doc_ids": task["evidence_doc_ids"], "retrieved_doc_ids": env.retrieved, "gold_answer": task["gold"], **result}


def validate_trace(trace):
    missing = [f for f in REQUIRED_FIELDS if f not in trace]
    if missing:
        raise ValueError(f"trace missing fields: {missing}")
    return trace


def append_traces(path, traces):
    with open(path, "a") as f:
        for trace in traces:
            f.write(json.dumps(validate_trace(trace), default=str) + "\n")
