"""Model probes that separate "cannot retrieve" from "stops early" from "cannot answer from evidence".

    python -m src.probe explore --checkpoint base --split twohop_dev --samples 8 --output X.json
    python -m src.probe forced  --checkpoint base --split twohop_dev --output X.json

explore: sample rollouts at the training temperature and count how often a second search ever happens
         (if never, GRPO has no two-search trajectory to reinforce).
forced:  run the model's own first search, then *force* a scripted second search ("<person> affiliation")
         and let the model answer. Accuracy here = can it use two-hop evidence once it is supplied.
         The scripted second query uses the retrieved text only, never the gold label.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from data.generate import DATA_DIR, read_jsonl
from src.audit import person_from
from src.environment import (ANSWER_FORMATS, SearchEnv, make_env_factory, make_prompt, parse_response, rollout_chat_template,
                             run_episode, run_tool_calls, add_response_schema)
from src.evaluate import load_policy
from src.reward import score


def explore(model, tok, cfg, split, samples, temperature, limit, variant="default", answer_format="default"):
    env = make_env_factory(split, answer_format)()
    tasks = read_jsonl(DATA_DIR / f"{split}_tasks.jsonl")[:limit]
    rows = []
    for t in tasks:
        for _ in range(samples):
            _, _, r = run_episode(model, tok, env, t, cfg["grpo"]["max_completion_length"],
                                  cfg["chat_template_kwargs"], temperature=temperature, prompt_variant=variant)
            got = {d for ids in env.retrieved for d in ids}
            rows.append({"task_id": t["task_id"], "searches": r["num_searches"], "correct": r["correct"],
                         "complete_chain": set(t["evidence_doc_ids"]) <= got})
    n = len(rows)
    dist = {k: sum(r["searches"] == k for r in rows) / n for k in (0, 1, 2)}
    return {"mode": "explore", "temperature": temperature, "rollouts": n, "search_count_distribution": dist,
            "second_search_rate": dist[2], "complete_chain_rate": sum(r["complete_chain"] for r in rows) / n,
            "correct_rate": sum(r["correct"] for r in rows) / n,
            "tasks_with_any_two_search_rollout": len({r["task_id"] for r in rows if r["searches"] == 2})}


@torch.no_grad()
def forced(model, tok, cfg, split, limit, variant="default", answer_format="default"):
    if getattr(tok, "response_template", None) is None and getattr(tok, "response_schema", None) is None:
        add_response_schema(tok)
    kw = {**cfg["chat_template_kwargs"], "chat_template": rollout_chat_template(tok)}
    retriever = make_env_factory(split)().retriever
    out = []
    for t in read_jsonl(DATA_DIR / f"{split}_tasks.jsonl")[:limit]:
        env = SearchEnv(retriever, ANSWER_FORMATS[answer_format])
        env.reset(**t)
        msgs = make_prompt(t["question"], variant)
        # Scripted two-step chain (the first query is scripted too so this isolates the answering step).
        q1 = f"Project {t['subject']} principal investigator"
        completion = []
        person = None
        for step in range(2):
            q = q1 if step == 0 else (f"{person} affiliation" if person else q1)
            call = {"type": "function", "function": {"name": "search", "arguments": {"query": q}}}
            completion.append({"role": "assistant", "content": "", "tool_calls": [call]})
            tool = run_tool_calls(env, completion[-1]["tool_calls"])
            completion += tool
            if step == 0:
                top = retriever.search(q)
                person = next((person_from(d["text"]) for d in top if person_from(d["text"])), None)
        ids = tok.apply_chat_template(msgs + completion, tools=[env.search], add_generation_prompt=True,
                                      tokenize=True, return_dict=False, **kw)
        gen = model.generate(torch.tensor([ids], device=model.device), max_new_tokens=64, do_sample=False,
                             pad_token_id=tok.pad_token_id)[0, len(ids):].tolist()
        turn = parse_response(tok, gen, prefix=ids, tools=[env.search])
        res = score(completion + [turn], t["gold"], env.num_searches)
        out.append({"task_id": t["task_id"], "answer": res["final_answer"], "correct": res["correct"],
                    "observation_docs": [d["doc_id"] for d in retriever.search(f"{person} affiliation")] if person else []})
    return {"mode": "forced_two_searches", "n": len(out), "accuracy": sum(o["correct"] for o in out) / len(out),
            "episodes": out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["explore", "forced"])
    parser.add_argument("--checkpoint", default="base")
    parser.add_argument("--split", default="twohop_dev")
    parser.add_argument("--config", default="configs/train_mixed.yaml")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--prompt-variant", default="default")
    parser.add_argument("--answer-format", default="default", choices=["default", "typed"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.manual_seed(cfg["seed"])
    model, tok = load_policy(cfg["model"], args.checkpoint)
    res = (explore(model, tok, cfg, args.split, args.samples, args.temperature, args.limit, args.prompt_variant) if args.mode == "explore"
           else forced(model, tok, cfg, args.split, args.limit, args.prompt_variant, args.answer_format))
    res.update(checkpoint=args.checkpoint, split=args.split, prompt_variant=args.prompt_variant, answer_format=args.answer_format)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
