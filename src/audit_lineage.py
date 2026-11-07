"""Checkpoint lineage audit: were these adapters trained independently, and how different are they really?

    python -m src.audit_lineage B=outputs/study_sft/B C=outputs/study_sft/C D=outputs/study_sft/D --out reports/lineage.json

For every run directory (needs adapter/adapter_model.safetensors; the rest is used when present):
  * sha256 of the adapter file, tensor count, adapter_config (base model, r, targets)
  * config_snapshot.yaml: init_adapter, seed, reward mode, steps, lr, temperature, num_generations
  * system_info.txt: optimizer steps, LoRA max |delta| measured by the trainer, wall time
  * train_metrics.csv: steps with a non-zero gradient, mean fraction of zero-variance groups
For every pair: byte-identical?, max |diff| and relative L2 difference of the LoRA tensors; for GRPO runs also the
difference to the initial adapter (init_adapter); and, when both trajectories.jsonl exist, the fraction of sampled
rollouts (step, task, assistant text) that are identical, which shows whether the two runs even saw different data.
"""

import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import yaml
from safetensors.torch import load_file


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def tensors(run):
    return load_file(str(Path(run) / "adapter" / "adapter_model.safetensors"))


def diff(a, b):
    keys = [k for k in a if k in b]
    mx = max((a[k].float() - b[k].float()).abs().max().item() for k in keys)
    num = sum(((a[k].float() - b[k].float()) ** 2).sum().item() for k in keys) ** 0.5
    den = sum((a[k].float() ** 2).sum().item() for k in keys) ** 0.5
    return {"max_abs_diff": mx, "relative_l2_diff": (num / den) if den else None, "shared_tensors": len(keys),
            "only_in_a": len(set(a) - set(b)), "only_in_b": len(set(b) - set(a))}


def info(run):
    run = Path(run)
    out = {"dir": str(run), "adapter_sha256": sha(run / "adapter" / "adapter_model.safetensors")}
    t = tensors(run)
    out["tensors"] = len(t)
    out["dtype"] = str(next(iter(t.values())).dtype)
    cfg = run / "adapter" / "adapter_config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        out["adapter_config"] = {k: c.get(k) for k in ("base_model_name_or_path", "r", "lora_alpha", "target_modules")}
    snap = run / "config_snapshot.yaml"
    if snap.exists():
        c = yaml.safe_load(snap.read_text())
        g = c.get("grpo") or {}
        out["config"] = {"init_adapter": c.get("init_adapter"), "seed": c.get("seed"), "reward": (c.get("reward") or {}).get("mode"),
                         "max_steps": g.get("max_steps"), "lr": g.get("learning_rate"), "temperature": g.get("temperature"),
                         "num_generations": g.get("num_generations"), "beta": g.get("beta"), "train_splits": c.get("train_splits"),
                         "sft": c.get("sft")}
    si = run / "system_info.txt"
    if si.exists():
        d = dict(l.split(": ", 1) for l in si.read_text().splitlines() if ": " in l)
        out["system_info"] = {k: d.get(k) for k in ("optimizer_steps", "lora_max_abs_delta", "train_seconds", "reward_mode", "run_id")}
    tm = run / "train_metrics.csv"
    if tm.exists():
        rows = list(csv.DictReader(open(tm)))
        g = [float(r["grad_norm"]) for r in rows if r.get("grad_norm") not in (None, "")]
        z = [float(r["frac_reward_zero_std"]) for r in rows if r.get("frac_reward_zero_std") not in (None, "")]
        out["metrics"] = {"logged_steps": len(rows), "steps_nonzero_grad": sum(x > 0 for x in g),
                          "mean_frac_reward_zero_std": round(sum(z) / len(z), 4) if z else None}
    return out


def rollout_overlap(run_a, run_b):
    pa, pb = Path(run_a) / "trajectories.jsonl", Path(run_b) / "trajectories.jsonl"
    if not (pa.exists() and pb.exists()):
        return None
    key = lambda r: (r["step"], r["task_id"], json.dumps([m.get("content") for m in r["messages"] if m["role"] == "assistant"]))
    A = [key(json.loads(l)) for l in pa.read_text().splitlines() if l.strip()]
    B = [key(json.loads(l)) for l in pb.read_text().splitlines() if l.strip()]
    n = min(len(A), len(B))
    return {"rollouts_a": len(A), "rollouts_b": len(B), "identical_fraction_in_order": sum(a == b for a, b in zip(A, B)) / n if n else None,
            "same_task_sequence": [k[:2] for k in A[:n]] == [k[:2] for k in B[:n]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="label=run_dir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    runs = dict(r.split("=", 1) for r in args.runs)
    report = {"runs": {l: info(d) for l, d in runs.items()}, "pairs": {}}
    for (la, da), (lb, db) in combinations(runs.items(), 2):
        ta, tb = tensors(da), tensors(db)
        rep = {"byte_identical": report["runs"][la]["adapter_sha256"] == report["runs"][lb]["adapter_sha256"], **diff(ta, tb)}
        ov = rollout_overlap(da, db)
        if ov:
            rep["rollouts"] = ov
        report["pairs"][f"{la} vs {lb}"] = rep
    for l, d in runs.items():  # distance travelled from the initial adapter, where one is declared
        init = report["runs"][l].get("config", {}).get("init_adapter")
        if init and (Path(init) / "adapter_model.safetensors").exists():
            report["runs"][l]["change_from_init"] = diff(load_file(str(Path(init) / "adapter_model.safetensors")), tensors(d))
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
