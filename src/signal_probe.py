"""How much GRPO learning signal would this policy give on this training split? (inference only, no weights change)

    python -m src.signal_probe --checkpoint outputs/study_sft/B/adapter --split hard_train --temperature 1.0 --samples 8 \
        --answer-format typed_abstain --output outputs/study_hard/signal_T1.0.json

For each task it samples `samples` rollouts with the environment used for training, scores each under the original,
evidence and grounded rewards, and reports, for group sizes 4 and 8 (consecutive rollouts of one task):
  frac_nonzero_std            groups whose rewards differ  (GRPO advantage != 0 there; 0 means no gradient)
  frac_bonus_created_variance groups where the shaped reward differs although the original reward is constant
  distinct_trajectories_per_group, accuracy, answered rate, per-category accuracy.
Use ONLY training splits here (it is a training-configuration diagnostic; never dev/final).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from data.generate import DATA_DIR, read_jsonl
from src.environment import make_env_factory, run_episode
from src.evaluate import load_policy
from src.reward_evidence import EvidenceRewardConfig, shaped_result
from src.train_grpo import group_stats, trajectory_signature

REWARDS = {"original": EvidenceRewardConfig(mode="original"), "evidence": EvidenceRewardConfig(mode="evidence"),
           "grounded": EvidenceRewardConfig(mode="grounded")}


def summarize(rollouts, samples):
    """rollouts: list (task-major, `samples` per task) of dicts holding per-reward values."""
    out = {}
    for g in (4, 8):
        if samples % g:
            continue
        per = {}
        for mode in REWARDS:
            rew = [r["reward"][mode] for r in rollouts]; orig = [r["reward"]["original"] for r in rollouts]
            sig = [r["sig"] for r in rollouts]
            # split each task's `samples` rollouts into groups of g
            gs = group_stats(rew, orig, sig, g)
            per[mode] = {k: round(v, 4) for k, v in gs.items() if k != "groups"}
        out[f"group_size_{g}"] = per
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="base")
    parser.add_argument("--split", default="hard_train")
    parser.add_argument("--config", default="configs/train_hard_grpo.yaml")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--tasks", type=int, default=40, help="tasks to sample (spread over categories)")
    parser.add_argument("--answer-format", default="typed_abstain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.split.endswith("final") or args.split.endswith("dev") or args.split in ("dev", "test"):
        raise SystemExit("signal_probe is a training diagnostic: use a training split (hard_train), never dev/final/test.")
    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.manual_seed(args.seed)
    model, tok = load_policy(cfg["model"], args.checkpoint)
    env = make_env_factory(args.split, args.answer_format)()
    tasks = read_jsonl(DATA_DIR / f"{args.split}_tasks.jsonl")
    by_cat = defaultdict(list)
    for t in tasks:
        by_cat[t.get("category", t["kind"])].append(t)
    per_cat = max(1, args.tasks // len(by_cat))
    chosen = [t for ts in by_cat.values() for t in ts[:per_cat]]
    rollouts = []
    for t in chosen:
        for _ in range(args.samples):
            _, comp, res = run_episode(model, tok, env, t, cfg["grpo"]["max_completion_length"], cfg["chat_template_kwargs"],
                                       temperature=args.temperature)
            rollouts.append({"task": t["task_id"], "category": t.get("category", t["kind"]), "correct": res["correct"],
                             "answered": res["final_answer"] is not None, "sig": trajectory_signature(comp),
                             "reward": {m: shaped_result(res, t, env.retrieved, c, env.retrieved_text)["reward"] for m, c in REWARDS.items()}})
    n = len(rollouts)
    acc_cat = {}
    for r in rollouts:
        acc_cat.setdefault(r["category"], []).append(r["correct"])
    result = {"checkpoint": args.checkpoint, "split": args.split, "temperature": args.temperature, "samples": args.samples,
              "tasks": len(chosen), "rollouts": n, "accuracy": sum(r["correct"] for r in rollouts) / n,
              "answered_rate": sum(r["answered"] for r in rollouts) / n,
              "accuracy_by_category": {c: round(sum(v) / len(v), 3) for c, v in acc_cat.items()},
              **summarize(rollouts, args.samples)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
