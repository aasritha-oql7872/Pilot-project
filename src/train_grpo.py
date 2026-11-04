"""GRPO + LoRA training of Qwen3-0.6B on the SearchLite environment.

    python -m src.train_grpo --config configs/train.yaml --smoke
    python -m src.train_grpo --config configs/train.yaml --output outputs/main
"""

import argparse
import csv
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from data.generate import DATA_DIR, file_hash, read_jsonl
from src.environment import make_env_factory, make_prompt, score_episode
from src.reward_evidence import EvidenceRewardConfig, shaped_result
from src.trace_schema import append_traces, make_trace


def apply_overrides(cfg, overrides):
    """Dotted `key.sub=value` overrides (value parsed as YAML), recorded in config_snapshot.yaml."""
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *path, leaf = key.split(".")
        for part in path:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def load_config(path, smoke, overrides=None):
    cfg = yaml.safe_load(Path(path).read_text())
    if smoke:
        cfg["grpo"].update(cfg["smoke"])
        cfg["run_id"] = "smoke"
    return apply_overrides(cfg, overrides)


def task_dataset(splits):
    """Trainer rows: the visible prompt plus metadata that only reset()/reward see."""
    rows = [r for split in splits for r in read_jsonl(DATA_DIR / f"{split}_tasks.jsonl")]
    return Dataset.from_list([{**row, "prompt": make_prompt(row["question"])} for row in rows])


def write_system_info(path, extra):
    lines = [f"python: {platform.python_version()}", f"platform: {platform.platform()}"]
    lines += [f"{pkg}: {version(pkg)}" for pkg in ["torch", "transformers", "trl", "peft", "rank-bm25"]]
    if torch.cuda.is_available():
        lines += [f"gpu: {torch.cuda.get_device_name()}", f"cuda: {torch.version.cuda}",
                  f"vram_total_gb: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}"]
        try:
            lines.append(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
        except FileNotFoundError:
            pass
    else:
        lines.append(f"gpu: none (mps={torch.backends.mps.is_available()})")
    lines += [f"{k}: {v}" for k, v in extra.items()]
    Path(path).write_text("\n".join(lines) + "\n")


class MetricsCSV(TrainerCallback):
    """Writes every trainer log entry (loss, reward, tool stats, ...) to train_metrics.csv."""

    def __init__(self, path):
        self.path, self.rows = path, []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:  # skip the final train-summary entry
            self.rows.append({"step": state.global_step, **logs})
            fields = list(dict.fromkeys(k for row in self.rows for k in row))
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.rows)


def make_reward_func(trace_path, run_id, model_name, seed, reward_cfg=EvidenceRewardConfig()):
    def searchlite_reward(completions, environments, trainer_state, log_metric, **kwargs):
        results = [shaped_result(score_episode(env, c), env.task, env.retrieved, reward_cfg)
                   for env, c in zip(environments, completions)]
        traces = [make_trace(run_id, env.task, model_name, trainer_state.global_step, seed,
                             env.task["prompt"], c, env, r)
                  for env, c, r in zip(environments, completions, results)]
        append_traces(trace_path, traces)
        n = len(results)
        log_metric("searchlite/correct", sum(r["correct"] for r in results) / n)
        log_metric("searchlite/searches_per_episode", sum(r["num_searches"] for r in results) / n)
        log_metric("searchlite/invalid_per_episode", sum(r["num_invalid"] for r in results) / n)
        multi = [r for r, env in zip(results, environments) if len(env.task["evidence_doc_ids"]) >= 2]
        if multi:  # two-hop rollouts only
            log_metric("searchlite/twohop_first_evidence", sum(r["first_evidence_found"] for r in multi) / len(multi))
            log_metric("searchlite/twohop_complete_chain", sum(r["complete_evidence_chain"] for r in multi) / len(multi))
            log_metric("searchlite/twohop_searches", sum(r["num_searches"] for r in multi) / len(multi))
            log_metric("searchlite/twohop_correct", sum(r["correct"] for r in multi) / len(multi))
        log_metric("searchlite/answered", sum(r["final_answer"] is not None for r in results) / n)
        return [r["reward"] for r in results]
    return searchlite_reward


def lora_params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="build the trainer and exit (no training)")
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    parser.add_argument("--override", nargs="*", default=[], help="dotted config overrides, e.g. grpo.temperature=1.2")
    args = parser.parse_args()

    cfg = load_config(args.config, args.smoke, args.override)
    if args.seed is not None:
        cfg["seed"] = args.seed
    out = Path(args.output or f"outputs/{cfg['run_id']}")
    if (out / "adapter").exists():
        raise SystemExit(f"{out}/adapter already exists; refusing to overwrite a trained model. Use a new --output.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    trace_path = out / "trajectories.jsonl"
    trace_path.unlink(missing_ok=True)

    train_splits = cfg.get("train_splits", ["train"])
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model, peft_config = cfg["model"], LoraConfig(task_type="CAUSAL_LM", **cfg["lora"])
    if cfg.get("init_adapter"):  # continue from a supervised warm-start adapter (same LoRA shape)
        base = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype)
        model, peft_config = PeftModel.from_pretrained(base, cfg["init_adapter"], is_trainable=True), None
    grpo_args = GRPOConfig(
        output_dir=str(out / "checkpoints"), seed=cfg["seed"], bf16=use_bf16,
        chat_template_kwargs=cfg["chat_template_kwargs"],
        model_init_kwargs=None if peft_config is None else {"dtype": dtype},
        **cfg["grpo"])
    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=task_dataset(train_splits),
        reward_funcs=make_reward_func(trace_path, out.name, cfg["model"], cfg["seed"],
                                       EvidenceRewardConfig.from_dict(cfg.get("reward"))),
        environment_factory=make_env_factory("+".join(train_splits), cfg.get("answer_format", "default")),
        peft_config=peft_config,
        callbacks=[MetricsCSV(out / "train_metrics.csv")],
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    before = lora_params(trainer.model)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "trainable_params": trainable, "lora_tensors": len(before),
                          "init_adapter": cfg.get("init_adapter")}))
        return

    t0 = time.time()
    trainer.train()
    train_seconds = round(time.time() - t0)

    # Gate: LoRA weights must have moved, otherwise no real policy update happened.
    after = lora_params(trainer.model)
    max_delta = max((after[n] - before[n]).abs().max().item() for n in before)
    trainer.save_model(str(out / "adapter"))
    extra = {"run_id": out.name, "model": cfg["model"], "trainable_params": trainable,
             "lora_tensors": len(before), "lora_max_abs_delta": max_delta,
             "optimizer_steps": trainer.state.global_step, "train_seconds": train_seconds,
             "reward_mode": cfg.get("reward", {}).get("mode", "original"),
             "train_splits": "+".join(train_splits),
             "train_corpus_sha256": file_hash(DATA_DIR / "train_corpus.jsonl")}
    if torch.cuda.is_available():
        extra["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    write_system_info(out / "system_info.txt", extra)
    print(json.dumps(extra, indent=2))
    if max_delta == 0:
        raise RuntimeError("LoRA parameters did not change: no policy update happened.")


if __name__ == "__main__":
    main()
