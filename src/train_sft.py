"""Supervised warm-start: LoRA fine-tune Qwen3-0.6B on correct trajectories (loss on assistant tokens only).

    python -m src.train_sft --config configs/sft.yaml --output outputs/study/B
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.sft_data import build_examples
from src.sft_tokens import IGNORE, encode
from src.train_grpo import write_system_info


def batches(examples, batch_size, rng):
    order = list(range(len(examples)))
    rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        yield [examples[j] for j in order[i:i + batch_size]]


def collate(batch, pad_id, device):
    n = max(len(ids) for ids, _ in batch)
    ids = torch.tensor([i + [pad_id] * (n - len(i)) for i, _ in batch], device=device)
    lab = torch.tensor([l + [IGNORE] * (n - len(l)) for _, l in batch], device=device)
    mask = torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i, _ in batch], device=device)
    return ids, lab, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true", help="tokenize and report mask statistics only")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.output)
    if (out / "adapter").exists():
        raise SystemExit(f"{out}/adapter already exists; refusing to overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    s = cfg["sft"]

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    examples = build_examples(cfg["seed"], s["n_lookup"], s["n_direct"], cfg.get("answer_format", "typed"))
    data = [encode(tok, e["messages"], cfg["chat_template_kwargs"]) for e in examples]
    data = [(i, l) for i, l in data if len(i) <= s["max_len"]]
    n_sup = sum(sum(x != IGNORE for x in l) for _, l in data)
    stats = {"examples": len(data), "tokens": sum(len(i) for i, _ in data), "supervised_tokens": n_sup}
    print(json.dumps(stats))
    if args.dry_run:
        return

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype).to(device)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]))
    trainable = [p for p in model.parameters() if p.requires_grad]
    before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}
    opt = torch.optim.AdamW(trainable, lr=s["lr"], weight_decay=0.0)
    steps_total = s["epochs"] * -(-len(data) // s["batch_size"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / 5) * max(0.05, 1 - t / steps_total))
    rng = random.Random(cfg["seed"])
    rows, step, t0 = [], 0, time.time()
    model.train()
    for epoch in range(s["epochs"]):
        for batch in batches(data, s["batch_size"], rng):
            ids, lab, mask = collate(batch, tok.pad_token_id, device)
            loss = model(input_ids=ids, attention_mask=mask, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            rows.append({"step": step, "epoch": epoch, "loss": round(loss.item(), 5)})
            print(rows[-1], flush=True)
            if s.get("save_every") and (step % s["save_every"] == 0 or step in s.get("save_steps", [])):
                model.save_pretrained(out / f"ckpt_step{step:03d}")  # intermediate adapters (never overwritten)
    delta = max((p.detach() - before[n]).abs().max().item() for n, p in model.named_parameters() if n in before)
    model.save_pretrained(out / "adapter")
    with open(out / "train_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "epoch", "loss"]); w.writeheader(); w.writerows(rows)
    extra = {"run_id": out.name, "method": "sft_warm_start", **stats, "optimizer_steps": step,
             "train_seconds": round(time.time() - t0), "lora_max_abs_delta": delta,
             "final_loss": rows[-1]["loss"], "first_loss": rows[0]["loss"]}
    if torch.cuda.is_available():
        extra["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    write_system_info(out / "system_info.txt", extra)
    print(json.dumps(extra, indent=2))
    if delta == 0:
        raise RuntimeError("LoRA parameters did not change.")


if __name__ == "__main__":
    main()
