"""Evaluate the frozen base model or a trained adapter on identical held-out tasks.

    python -m src.evaluate --checkpoint base --split test --output outputs/main/eval_base.json
    python -m src.evaluate --checkpoint outputs/main/adapter --split test --output outputs/main/eval_trained.json
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.generate import DATA_DIR, file_hash, read_jsonl
from src.environment import make_env_factory, run_episode
from src.trace_schema import make_trace, validate_trace


def device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_policy(model_name, checkpoint):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device())
    if checkpoint != "base":
        model = PeftModel.from_pretrained(model, checkpoint)
    return model.eval(), AutoTokenizer.from_pretrained(model_name)


def summarize(traces):
    def rate(rows, fn):
        return round(sum(map(fn, rows)) / len(rows), 4) if rows else None

    lookup = [t for t in traces if t["kind"] == "lookup"]
    no_search = [t for t in traces if t["kind"] == "no_search"]
    right_decision = lambda t: (t["num_searches"] > 0) == (t["kind"] != "no_search")
    gold_retrieved = lambda t: any(t["evidence_doc_ids"][0] in ids for ids in t["retrieved_doc_ids"])
    all_evidence = lambda t: set(t["evidence_doc_ids"]) <= {d for ids in t["retrieved_doc_ids"] for d in ids}
    by_kind = {}
    for kind in sorted({t["kind"] for t in traces}):
        rows = [t for t in traces if t["kind"] == kind]
        by_kind[kind] = {"n": len(rows), "accuracy": rate(rows, lambda t: t["correct"]),
                         "mean_searches": rate(rows, lambda t: t["num_searches"]),
                         "all_evidence_retrieved": rate(rows, all_evidence) if kind != "no_search" else None}
    return {
        "n": len(traces),
        "accuracy": rate(traces, lambda t: t["correct"]),
        "accuracy_lookup": rate(lookup, lambda t: t["correct"]),
        "accuracy_no_search": rate(no_search, lambda t: t["correct"]),
        "search_decision_accuracy": rate(traces, right_decision),
        "search_decision_accuracy_lookup": rate(lookup, right_decision),
        "search_decision_accuracy_no_search": rate(no_search, right_decision),
        "mean_searches": rate(traces, lambda t: t["num_searches"]),
        "mean_searches_lookup": rate(lookup, lambda t: t["num_searches"]),
        "mean_searches_no_search": rate(no_search, lambda t: t["num_searches"]),
        "invalid_rate": rate(traces, lambda t: t["num_invalid"] > 0),
        "completion_rate": rate(traces, lambda t: t["final_answer"] is not None),
        "evidence_recall_at_2_lookup": rate(lookup, gold_retrieved),
        "mean_reward": rate(traces, lambda t: t["reward"]),
        "by_kind": by_kind,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help='"base" or a LoRA adapter directory')
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N tasks")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.manual_seed(cfg["seed"])
    model, tokenizer = load_policy(cfg["model"], args.checkpoint)
    env = make_env_factory(args.split)()
    tasks = read_jsonl(DATA_DIR / f"{args.split}_tasks.jsonl")[: args.limit]

    traces = []
    for i, task in enumerate(tasks):
        prompt, completion, result = run_episode(model, tokenizer, env, task,
                                                 cfg["grpo"]["max_completion_length"],
                                                 cfg["chat_template_kwargs"])
        trace = make_trace(f"eval_{args.checkpoint}", task, cfg["model"], None, cfg["seed"],
                           prompt, completion, env, result)
        traces.append(validate_trace(trace))
        print(f"[{i + 1}/{len(tasks)}] {task['task_id']} searches={result['num_searches']} "
              f"answer={result['final_answer']!r} gold={task['gold']!r} correct={result['correct']}")

    report = {"checkpoint": args.checkpoint, "split": args.split, "decoding": "greedy",
              "corpus_sha256": file_hash(DATA_DIR / f"{args.split}_corpus.jsonl"),
              "metrics": summarize(traces), "episodes": traces}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
