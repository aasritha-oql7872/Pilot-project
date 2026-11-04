"""Optimization diagnostics for one GRPO run: gradient signal and how far the policy moved.

    python -m src.train_diag outputs/study_early/E1_s42 [--init outputs/study_early/sft/ckpt_step006]

Reads train_metrics.csv (written by the trainer) and, if --init is given, compares the final LoRA weights with the
initial adapter (max |delta| and relative L2 change). beta=0 in these runs (no KL term), so weight change is the
policy-change diagnostic.
"""

import argparse
import csv
import json
from pathlib import Path

from safetensors.torch import load_file


def lora_change(init_dir, final_dir):
    a = load_file(str(Path(init_dir) / "adapter_model.safetensors"))
    b = load_file(str(Path(final_dir) / "adapter_model.safetensors"))
    keys = [k for k in a if k in b]
    max_abs = max((a[k].float() - b[k].float()).abs().max().item() for k in keys)
    num = sum(((a[k].float() - b[k].float()) ** 2).sum().item() for k in keys) ** 0.5
    den = sum((a[k].float() ** 2).sum().item() for k in keys) ** 0.5
    return {"max_abs_delta": max_abs, "relative_l2_change": num / den if den else None}


def diag(run, init=None):
    rows = list(csv.DictReader(open(Path(run) / "train_metrics.csv")))
    col = lambda name: [float(r[name]) for r in rows if r.get(name) not in (None, "")]
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    third = max(1, len(rows) // 3)
    out = {"run": str(run), "logged_steps": len(rows),
           "steps_with_nonzero_grad": sum(g > 0 for g in col("grad_norm")), "mean_grad_norm": mean(col("grad_norm")),
           "max_grad_norm": max(col("grad_norm"), default=None),
           "mean_frac_reward_zero_std": mean(col("frac_reward_zero_std")), "mean_reward_std": mean(col("reward_std")),
           "mean_reward_first_third": mean([float(r["reward"]) for r in rows[:third]]),
           "mean_reward_last_third": mean([float(r["reward"]) for r in rows[-third:]])}
    if "searchlite/twohop_searches" in rows[0]:
        ts = [r for r in rows if r.get("searchlite/twohop_searches") not in (None, "")]
        out["twohop_searches_first_third"] = mean([float(r["searchlite/twohop_searches"]) for r in ts[:third]])
        out["twohop_searches_last_third"] = mean([float(r["searchlite/twohop_searches"]) for r in ts[-third:]])
    if init and (Path(run) / "adapter").exists():
        out["policy_change_vs_init"] = lora_change(init, Path(run) / "adapter")
    sysinfo = Path(run) / "system_info.txt"
    if sysinfo.exists():
        info = dict(l.split(": ", 1) for l in sysinfo.read_text().splitlines() if ": " in l)
        out["train_seconds"], out["peak_vram_gb"] = info.get("train_seconds"), info.get("peak_vram_gb")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--init", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    res = diag(args.run, args.init)
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
